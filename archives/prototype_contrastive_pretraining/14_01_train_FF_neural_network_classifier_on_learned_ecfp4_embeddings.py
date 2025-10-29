from pathlib import Path
import json
import os
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler, TensorDataset
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score


# ---- Configuration ----# 
TRAIN_NPZ = "../data/train/baseline_train_ecfp4.npz"
ENCODER_CHECKPOINT = "../models/Molecular_ResNet_1024_512_100_epochs_w_weighted_loss.pt"  # update to actual filename
EMBED_SOURCE = "preproj"  # "preproj" (encoder g) or "proj" (projection z)
EMBED_BATCH_SIZE = 2048
EMBED_NUM_WORKERS = 4
epochs = 100

def load_split_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    fps = data["fps"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    return fps, labels


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.X = torch.from_numpy(embeddings.astype(np.float32)).clone()
        self.y = torch.from_numpy(labels.astype(np.float32)).view(-1, 1).clone()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class fp_CNN_Encoder(nn.Module):
    def __init__(self, fp_dim=2048, hidden_channels=(64, 128),
                 embed_dim=256, proj_dim=120, use_projection=True,
                 batchnorm_safe=True, dropout_p=0.3):
        super().__init__()
        c1, c2 = hidden_channels
        self.conv = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=5, padding=2),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(1),
        )
        self.fc = nn.Linear(c2, embed_dim)
        self.use_projection = use_projection
        if use_projection:
            self.proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p),
                nn.Linear(embed_dim, proj_dim),
            )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.conv(x).squeeze(-1)
        g = F.normalize(self.fc(h), dim=-1, eps=1e-8)
        if self.use_projection:
            z = F.normalize(self.proj(g), dim=-1, eps=1e-8)
            return g, z
        return g, g


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, stride=1, padding=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.shortcut = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))


class fp_CNN_ResNetEncoder(nn.Module):
    def __init__(self, fp_dim=2048, hidden_channels=(128, 256), embed_dim=128,
                 proj_dim=64, use_projection=True, batchnorm_safe=True,
                 dropout_p=0.2, use_proj_skip=True):
        super().__init__()
        c1, c2 = hidden_channels
        self.stem = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualBlock(c1, c1)
        self.layer2 = ResidualBlock(c1, c2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c2, embed_dim)
        self.use_projection = use_projection
        self.use_proj_skip = use_proj_skip
        if self.use_projection:
            self.proj_fc1 = nn.Linear(embed_dim, embed_dim)
            self.proj_fc2 = nn.Linear(embed_dim, proj_dim)
            self.relu = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.pool(h).squeeze(-1)
        g = F.normalize(self.fc(h), dim=-1, eps=1e-8)
        if self.use_projection:
            h_proj = self.relu(self.proj_fc1(g))
            h_proj = self.dropout(h_proj)
            if self.use_proj_skip:
                h_proj = h_proj + g
            z = F.normalize(self.proj_fc2(h_proj), dim=-1, eps=1e-8)
            return g, z
        return g, g


class TinyMLP(nn.Module):
    def __init__(self, fp_dim, embed_dim=512, proj_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(fp_dim, 256)
        self.fc2 = nn.Linear(256, embed_dim)
        self.proj = nn.Linear(embed_dim, proj_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        g = F.normalize(self.fc2(h), dim=-1, eps=1e-8)
        z = F.normalize(self.proj(g), dim=-1, eps=1e-8)
        return g, z


def build_model(model_type="resnet", fp_dim=2048, embed_dim=512, proj_dim=256):
    model_type = model_type.lower()
    if model_type == "mlp":
        return TinyMLP(fp_dim=fp_dim, embed_dim=embed_dim, proj_dim=proj_dim)
    if model_type == "cnn":
        return fp_CNN_Encoder(fp_dim=fp_dim, embed_dim=embed_dim, proj_dim=proj_dim)
    if model_type == "resnet":
        return fp_CNN_ResNetEncoder(fp_dim=fp_dim, embed_dim=embed_dim, proj_dim=proj_dim)
    raise ValueError(f"Unknown model type: {model_type}")


def load_encoder(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt.get("hyperparameters", {})
    model_type = ckpt.get("model_type", "resnet")
    embed_dim = config.get("embed_dim", 512)
    proj_dim = config.get("proj_dim", 256)

    model = build_model(model_type=model_type, fp_dim=2048, embed_dim=embed_dim, proj_dim=proj_dim)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print0("[Warning] Missing keys when loading encoder checkpoint:", missing)
    if unexpected:
        print0("[Warning] Unexpected keys when loading encoder checkpoint:", unexpected)
    model.to(device)
    model.eval()
    return model, model_type, embed_dim, proj_dim


def compute_encoder_embeddings(
    model: nn.Module,
    fps: np.ndarray,
    device: torch.device,
    batch_size: int = EMBED_BATCH_SIZE,
    num_workers: int = EMBED_NUM_WORKERS,
) -> np.ndarray:
    tensor_dataset = TensorDataset(torch.from_numpy(fps))
    loader = DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    use_proj = EMBED_SOURCE.lower() in {"proj", "z"}
    features: List[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            outputs = model(xb)
            if isinstance(outputs, tuple):
                g, z = outputs
            else:
                g, z = outputs, outputs
            feats = z if use_proj else g
            features.append(feats.cpu().numpy())
    return np.vstack(features)


class FFClassifier(nn.Module):
    def __init__(self, input_dim: int = 2048, hidden1: int = 512, hidden2: int = 256, p_drop: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden2, 1),  # output logit
        )

    def forward(self, x):
        return self.net(x)


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)


def setup_distributed() -> torch.device:
    if not dist.is_available():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rank_env = os.environ.get("RANK")
    world_env = os.environ.get("WORLD_SIZE")
    if rank_env is None or world_env is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return device


def cleanup_distributed():
    if is_dist():
        dist.destroy_process_group()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    logits_list = []
    labels_list = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        logits_list.append(logits.detach().cpu().numpy())
        labels_list.append(yb.detach().cpu().numpy())

    logits_all = np.concatenate(logits_list, axis=0)
    labels_all = np.concatenate(labels_list, axis=0).ravel()
    probs = 1.0 / (1.0 + np.exp(-logits_all.ravel()))

    # Metrics
    auprc = average_precision_score(labels_all, probs)
    try:
        auroc = roc_auc_score(labels_all, probs)
    except ValueError:
        auroc = float("nan")  # if only one class present
    preds = (probs >= 0.5).astype(np.float32)
    acc = accuracy_score(labels_all, preds)
    return auprc, auroc, acc


def main():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = setup_distributed()
    rank = get_rank()
    world = get_world_size()
    print0(f"Using device: {device} | rank={rank} | world_size={world}")

    print0("Loading ECFP4 training split...")
    train_fps, train_labels = load_split_npz(TRAIN_NPZ)
    print0(f"Training fingerprints shape: {train_fps.shape}")

    print0(f"Loading encoder checkpoint from {ENCODER_CHECKPOINT} ...")
    encoder, encoder_type, encoder_embed_dim, encoder_proj_dim = load_encoder(ENCODER_CHECKPOINT, device)
    print0(
        f"Encoder type={encoder_type}, embed_dim={encoder_embed_dim}, "
        f"proj_dim={encoder_proj_dim}, source={EMBED_SOURCE}"
    )

    print0("Computing learned embeddings...")
    train_embeddings = compute_encoder_embeddings(encoder, train_fps, device)
    print0("Finished computing embeddings for training split.")
    print0(f"Embedding shape (train): {train_embeddings.shape}")

    del train_fps, encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print0("Building dataset and DataLoader...")
    train_ds = EmbeddingDataset(train_embeddings, train_labels)
    input_dim = train_ds.X.shape[1]
    del train_embeddings

    # Choose conservative DataLoader worker count to avoid warnings on constrained systems
    def _suggest_workers(default: int) -> int:
        try:
            sct = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
            if sct > 0:
                return max(1, min(default, sct // 2))
        except Exception:
            pass
        # Fallback to 1 to prevent oversubscription warnings
        return 1

    batch_size = 4096 if device.type == "cuda" else 1024
    nw_train = _suggest_workers(4)
    pin = device.type == "cuda"
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world,
        rank=rank,
        shuffle=True,
        drop_last=False,
    ) if world > 1 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=nw_train,
        pin_memory=pin,
    )

    # Model
    print0("Initialising feed-forward classifier...")
    model = FFClassifier(input_dim=input_dim).to(device)
    if world > 1:
        print0("Wrapping classifier with DistributedDataParallel...")
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    # Class imbalance handling via pos_weight in BCEWithLogitsLoss
    pos = max(int(train_labels.sum()), 1)
    neg = max(int(len(train_labels) - pos), 1)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    print0(f"Class counts (train): pos={pos}, neg={neg}, pos_weight={pos_weight.item():.3f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    print0("Starting training loop...")
    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            # Use new autocast API to avoid deprecation warnings
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        loss_tensor = torch.tensor(
            [epoch_loss, n_batches],
            dtype=torch.float32,
            device=device,
        )
        if is_dist():
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_steps = int(loss_tensor[1].item())
        avg_loss = loss_tensor[0].item() / max(total_steps, 1)

        if get_rank() == 0:
            dt = time.time() - t0
            print0(f"Epoch {epoch:02d} | loss={avg_loss:.4f} | {dt:.1f}s")

        if is_dist():
            dist.barrier()

    if get_rank() == 0:
        print0("Training complete. Saving classifier checkpoint...")
        target_model = model if world == 1 else model.module

        out_dir = Path("../models")
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "learned_ecfp_resnet_embedding_ffnn_pks_classifier.pt"
        meta_path = out_dir / "learned_ecfp_resnet_embedding_ffnn_pks_classifier.meta.json"
        torch.save(
            {
                "state_dict": target_model.state_dict(),
                "input_dim": input_dim,
                "hidden": [512, 256],
                "pos_weight": pos_weight.item(),
            },
            model_path,
        )

        with open(meta_path, "w") as f:
            json.dump(
                {
                    "model_path": str(model_path),
                    "input_dim": input_dim,
                    "hidden": [512, 256],
                    "batch_size_per_rank": batch_size,
                    "epochs": epochs,
                    "world_size": world,
                    "device": str(device),
                    "encoder_checkpoint": ENCODER_CHECKPOINT,
                    "encoder_type": encoder_type,
                    "encoder_embed_dim": encoder_embed_dim,
                    "encoder_proj_dim": encoder_proj_dim,
                    "embedding_source": EMBED_SOURCE,
                    "class_counts": {"pos": int(pos), "neg": int(neg)},
                    "pos_weight": float(pos_weight.item()),
                },
                f,
                indent=2,
            )

        print0(f"Saved model to {model_path} and metadata to {meta_path}")

    if is_dist():
        dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
