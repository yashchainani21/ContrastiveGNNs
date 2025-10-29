
import math
import os
import datetime
import numpy as np
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
from torch import amp
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# ---- Hyperparameters ----

MODEL_TYPE = "resnet"  # "mlp" | "cnn" | "resnet"
TRAIN_NPZ = '../data/train/baseline_train_ecfp4.npz'
VAL_NPZ = '../data/val/baseline_val_ecfp4.npz'
NORMALIZE = False
BATCH_SIZE = 2048  # per-GPU micro-batch size
EPOCHS = 10
LR = 3e-4
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.05
EMBED_DIM = 2048 
PROJ_DIM = 1024
SEED = 42
SUBSET_SIZE = 4_000_000
GRAD_ACCUM_STEPS = 16
WARMUP_EPOCHS = 3

EVAL_EMBED_BATCH_SIZE = 2048
EVAL_NUM_WORKERS = 4
EVAL_EMBED_SOURCE = "preproj"  # "preproj" (g) or "proj" (z)
SILHOUETTE_MAX_SAMPLES = 20_000

model_filename = f"Molecular_ResNet_{EMBED_DIM}_{PROJ_DIM}_{EPOCHS}_epochs_deeper_resnets.pt"
model_path =f"../models/{model_filename}"

# ---- Utilities ----

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
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rank_env = os.environ.get('RANK')
    world_env = os.environ.get('WORLD_SIZE')
    if rank_env is None or world_env is None:
        # Running in single-process mode (e.g., notebook or python script)
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not dist.is_initialized():
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend, init_method='env://')

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')
    return device


def barrier():
    if is_dist():
        dist.barrier()
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---- Dataset ----

class NPZFingerprints(Dataset):
    def __init__(self, npz_path: str, normalize: bool = False, mean=None, std=None):
        z = np.load(npz_path, allow_pickle=False)
        self.fps = z['fps']
        self.labels = z['labels'].astype(np.int64)
        self.N, self.D = self.fps.shape
        self.normalize = normalize
        if normalize:
            if mean is not None and std is not None:
                self.mean = mean.astype(np.float32)
                self.std = std.astype(np.float32)
            else:
                arr = self.fps.astype(np.float32)
                self.mean = arr.mean(axis=0)
                self.std = arr.std(axis=0) + 1e-8
        else:
            self.mean = None
            self.std = None

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x = self.fps[idx].astype(np.float32)
        if self.normalize:
            x = (x - self.mean) / self.std
        y = int(self.labels[idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def compute_dataset_embeddings(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = EVAL_EMBED_BATCH_SIZE,
    num_workers: int = EVAL_NUM_WORKERS,
    embed_source: str = EVAL_EMBED_SOURCE,
    pad_to: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    use_proj = embed_source.lower() in {"proj", "z"}
    model_was_training = model.training
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            outputs = model(xb)
            if isinstance(outputs, tuple):
                g, z = outputs
            else:
                g, z = outputs, outputs
            feats = z if use_proj else g
            embeddings.append(feats.cpu().numpy())
            labels.append(yb.numpy())
    if model_was_training:
        model.train()
    if embeddings:
        X = np.concatenate(embeddings, axis=0).astype(np.float32)
        if pad_to is not None and X.shape[1] < pad_to:
            pad_width = pad_to - X.shape[1]
            X = np.pad(X, ((0, 0), (0, pad_width)), mode="constant")
        y = np.concatenate(labels, axis=0).astype(np.int64)
    else:
        dim = pad_to if pad_to is not None else (PROJ_DIM if embed_source.lower() in {"proj", "z"} else EMBED_DIM)
        X = np.empty((0, dim), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
    return X, y


# ---- Models ----

def build_model(model_type="mlp", fp_dim=2048, embed_dim=128, proj_dim=64,
                hidden_channels=(128, 256), use_projection=True, batchnorm_safe=True,
                dropout_p=0.2):
    
    if model_type.lower() == "mlp":
        return TinyMLP(fp_dim=fp_dim, embed_dim=embed_dim, proj_dim=proj_dim)
        
    elif model_type.lower() == "cnn":
        return fp_CNN_Encoder(fp_dim=fp_dim, hidden_channels=(64, 128),
                              embed_dim=embed_dim, proj_dim=proj_dim,
                              use_projection=use_projection,
                              batchnorm_safe=batchnorm_safe)
        
    elif model_type.lower() == "resnet":
        return fp_CNN_ResNetEncoder(fp_dim=fp_dim, hidden_channels=hidden_channels,
                                    embed_dim=embed_dim, proj_dim=proj_dim,
                                    use_projection=use_projection,
                                    batchnorm_safe=batchnorm_safe,
                                    dropout_p=dropout_p)
    else:
        raise ValueError(f"Unknown model_type={model_type}")



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
            norm_layer = nn.LayerNorm(embed_dim) if batchnorm_safe else nn.BatchNorm1d(embed_dim)
            self.proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p),
                nn.Linear(embed_dim, proj_dim)
            )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.conv(x).squeeze(-1)
        g = F.normalize(self.fc(h), dim=-1, eps=1e-8)
        if self.use_projection:
            z = F.normalize(self.proj(g), dim=-1, eps=1e-8)
            return g, z
        else:
            return g


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
        # Two-stage stack with doubled residual depth
        self.layer1a = ResidualBlock(c1, c1)
        self.layer1b = ResidualBlock(c1, c1)
        self.layer2a = ResidualBlock(c1, c2)
        self.layer2b = ResidualBlock(c2, c2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c2, embed_dim)
        self.use_projection = use_projection
        self.batchnorm_safe = batchnorm_safe
        self.use_proj_skip = use_proj_skip
        if self.use_projection:
            norm_layer = nn.LayerNorm(embed_dim) if self.batchnorm_safe else nn.BatchNorm1d(embed_dim)
            self.proj_fc1 = nn.Linear(embed_dim, embed_dim)
            self.proj_fc2 = nn.Linear(embed_dim, proj_dim)
            self.relu = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(p=dropout_p)
            self.norm = norm_layer

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.stem(x)
        h = self.layer1a(h)
        h = self.layer1b(h)
        h = self.layer2a(h)
        h = self.layer2b(h)
        h = self.pool(h).squeeze(-1)
        g = F.normalize(self.fc(h), dim=-1, eps=1e-8)
        if self.use_projection:
            h_proj = self.relu(self.proj_fc1(g))
            h_proj = self.dropout(h_proj)
            h_proj = self.norm(h_proj)
            if self.use_proj_skip:
                h_proj = h_proj + g
            z = F.normalize(self.proj_fc2(h_proj), dim=-1, eps=1e-8)
            return g, z
        else:
            return g


class TinyMLP(nn.Module):
    def __init__(self, fp_dim: int, embed_dim: int = EMBED_DIM, proj_dim: int = PROJ_DIM):
        super().__init__()
        self.fc1 = nn.Linear(fp_dim, 256)
        self.fc2 = nn.Linear(256, embed_dim)
        self.proj = nn.Linear(embed_dim, proj_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        g = F.normalize(self.fc2(h), dim=-1, eps=1e-6)
        z = F.normalize(self.proj(g), dim=-1, eps=1e-6)
        return g, z


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, eps: float = 1e-8, pos_weight: float = 1.5):
        super().__init__()
        self.tau = temperature
        self.eps = eps
        self.pos_weight = pos_weight

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        z = F.normalize(z.float(), dim=-1, eps=self.eps)
        sim = (z @ z.t()) / self.tau
        eye = torch.eye(B, dtype=torch.bool, device=z.device)
        labels = labels.view(-1)
        pos_mask = (labels.view(-1, 1) == labels.view(1, -1)) & (~eye)
        valid_mask = pos_mask.sum(1) > 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=z.device, requires_grad=True)
        sim = sim[valid_mask]
        pos_mask = pos_mask[valid_mask]
        sim = sim - sim.max(dim=1, keepdim=True).values
        denom = torch.logsumexp(sim, dim=1, keepdim=True) + self.eps
        log_prob = sim - denom
        pos_counts = pos_mask.sum(1).clamp_min(1)
        pos_log_prob = (pos_mask * log_prob).sum(1) / pos_counts

        # --- NEW: weight positives more heavily ---
        # class_weights = torch.where(labels.squeeze(1) == 1, self.pos_weight, 1.0)

        class_weights = torch.ones_like(labels, dtype=pos_log_prob.dtype, device=z.device)
        class_weights[labels == 1] = self.pos_weight
        class_weights = class_weights[valid_mask]

        loss = -(class_weights * pos_log_prob).mean()
        return loss
        #return -pos_log_prob.mean()


# ---- Main ----

def main():
    device = setup_distributed()
    rank = get_rank()
    world = get_world_size()
    print0(f"DDP initialized: rank={rank}, world={world}, device={device}")
    print0(f"Using {world} GPU{'s' if world != 1 else ''} with grad accumulation steps = {GRAD_ACCUM_STEPS}")
    print0(f"Using {world} GPU{'s' if world != 1 else ''} for training")

    torch.set_num_threads(1)

    base_train = NPZFingerprints(TRAIN_NPZ, normalize=False)
    base_val = NPZFingerprints(VAL_NPZ, normalize=False)
    labels_all = base_train.labels.astype(np.int64)

    if rank == 0:
        pos_idx = np.where(labels_all == 1)[0]
        neg_idx = np.where(labels_all == 0)[0]
        rng = np.random.default_rng(SEED)
        n_pos = min(SUBSET_SIZE // 2, len(pos_idx))
        n_neg = min(SUBSET_SIZE - n_pos, len(neg_idx))
        pos_sample = rng.choice(pos_idx, size=n_pos, replace=False)
        neg_sample = rng.choice(neg_idx, size=n_neg, replace=False)
        sel_idx_np = np.concatenate([pos_sample, neg_sample]).astype(np.int64)
        rng.shuffle(sel_idx_np)
        sel_len = np.array([len(sel_idx_np)], dtype=np.int64)
    else:
        sel_idx_np = None
        sel_len = np.zeros((1,), dtype=np.int64)

    if is_dist():
        sel_len_t = torch.from_numpy(sel_len)
        if device.type == 'cuda':
            sel_len_t = sel_len_t.to(device)
        dist.broadcast(sel_len_t, src=0)
        sel_len = sel_len_t.cpu().numpy()
        if rank != 0:
            sel_idx_np = np.empty((sel_len[0],), dtype=np.int64)
        sel_idx_t = torch.from_numpy(sel_idx_np)
        if device.type == 'cuda':
            sel_idx_t = sel_idx_t.to(device)
        dist.broadcast(sel_idx_t, src=0)
        sel_idx_np = sel_idx_t.cpu().numpy()

    train_ds = Subset(base_train, sel_idx_np.tolist())

    print0("Subset stats:",
       "pos =", int((labels_all[sel_idx_np] == 1).sum()),
       "neg =", int((labels_all[sel_idx_np] == 0).sum()),
       "total =", len(sel_idx_np))

    # ---- Balanced Weighted Sampler (DDP-compatible) ----
    labels_subset = labels_all[sel_idx_np]

    # Compute inverse-frequency weights for balanced sampling
    pos_weight = len(labels_subset) / (2.0 * (labels_subset == 1).sum())
    neg_weight = len(labels_subset) / (2.0 * (labels_subset == 0).sum())
    weights_np = np.where(labels_subset == 1, pos_weight, neg_weight).astype(np.float32)

    if world > 1:
        # Each rank gets its own balanced weighted sampler
        weights_split = np.array_split(weights_np, world)[rank]
        weights_tensor = torch.tensor(weights_split, dtype=torch.float32)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights_tensor, num_samples=len(weights_tensor), replacement=True
        )
    else:
        # Single-GPU: use full weighted sampler
        weights_tensor = torch.tensor(weights_np, dtype=torch.float32)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights_tensor, num_samples=len(weights_tensor), replacement=True
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    

    model = build_model(model_type=MODEL_TYPE, fp_dim=2048,
                        embed_dim=EMBED_DIM, proj_dim=PROJ_DIM).to(device)

    if world > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None)

    #criterion = SupConLoss(temperature=TEMPERATURE).to(device)
    criterion = SupConLoss(temperature=TEMPERATURE, pos_weight=1.5).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                                weight_decay=WEIGHT_DECAY, nesterov=True)

    def cosine_warmup_lambda(epoch: int) -> float:
        if WARMUP_EPOCHS > 0 and epoch < WARMUP_EPOCHS:
            return (epoch + 1) / max(1, WARMUP_EPOCHS)
        if epoch >= EPOCHS:
            return 0.0
        if EPOCHS <= WARMUP_EPOCHS:
            return 1.0
        progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
        progress = max(0.0, min(1.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_warmup_lambda)
    scaler = amp.GradScaler(device.type)

    print0("Train size:", len(train_ds))

    for epoch in range(1, EPOCHS + 1):
        #sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        steps = 0
        accum_count = 0
        optimizer.zero_grad(set_to_none=True)

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            is_sync_step = ((accum_count + 1) % GRAD_ACCUM_STEPS == 0) or (world == 1)
            sync_ctx = contextlib.nullcontext() if is_sync_step else model.no_sync()
            with sync_ctx:
                with amp.autocast(device.type, dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16):
                    outputs = model(xb)
                    if isinstance(outputs, tuple):
                        _, z = outputs
                    else:
                        z = outputs
                    #if yb.unique().numel() < 2:
                    #    loss_full = z.sum() * 0.0
                    #else:
                    #    loss_full = criterion(z, yb)
                    loss_full = criterion(z, yb)
                    loss = loss_full / GRAD_ACCUM_STEPS

                if not torch.isfinite(loss_full):
                    loss_full = z.sum() * 0.0
                    loss = loss_full / GRAD_ACCUM_STEPS

                scaler.scale(loss).backward()

            accum_count += 1
            steps += 1
            epoch_loss += loss_full.item()

            if (accum_count % GRAD_ACCUM_STEPS == 0) or (world == 1 and is_sync_step):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        if accum_count % GRAD_ACCUM_STEPS != 0 and accum_count > 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        loss_steps = torch.tensor([epoch_loss, steps], dtype=torch.float32, device=device)
        if is_dist():
            dist.all_reduce(loss_steps, op=dist.ReduceOp.SUM)

        total_steps = int(loss_steps[1].item())
        if total_steps == 0:
            print0(f"[Epoch {epoch:03d}] all batches skipped")
        else:
            mean_loss = loss_steps[0].item() / total_steps
            print0(f"[Epoch {epoch:03d}] train_supcon={mean_loss:.4f}, lr={current_lr:.3e}")

        if get_rank() == 0:
            print0("Computing validation silhouette score from current encoder...")
            eval_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
            val_embeds, val_labels_embed = compute_dataset_embeddings(
                eval_model,
                base_val,
                device,
                batch_size=EVAL_EMBED_BATCH_SIZE,
                num_workers=EVAL_NUM_WORKERS,
                embed_source=EVAL_EMBED_SOURCE,
            )
            try:
                unique_labels = np.unique(val_labels_embed)
                if unique_labels.size >= 2 and val_embeds.shape[0] >= 2:
                    pos_mask = val_labels_embed == 1
                    neg_mask = ~pos_mask
                    pos_embeds = val_embeds[pos_mask]
                    pos_labels = val_labels_embed[pos_mask]
                    neg_embeds = val_embeds[neg_mask]
                    neg_labels = val_labels_embed[neg_mask]
                    if pos_embeds.shape[0] + neg_embeds.shape[0] > SILHOUETTE_MAX_SAMPLES:
                        remaining = max(SILHOUETTE_MAX_SAMPLES - pos_embeds.shape[0], 0)
                        if remaining < neg_embeds.shape[0]:
                            rng = np.random.default_rng(SEED + epoch)
                            idx = rng.choice(neg_embeds.shape[0], size=remaining, replace=False)
                            neg_embeds = neg_embeds[idx]
                            neg_labels = neg_labels[idx]
                    emb_sample = np.concatenate([pos_embeds, neg_embeds], axis=0)
                    lbl_sample = np.concatenate([pos_labels, neg_labels], axis=0)
                    try:
                        sil_score = silhouette_score(emb_sample.astype(np.float32), lbl_sample, metric="euclidean")
                        print0(f"[Epoch {epoch:03d}] silhouette score={sil_score:.4f}")
                    except Exception as err:
                        print0(f"[Epoch {epoch:03d}] silhouette score failed: {err}")
                else:
                    print0(f"[Epoch {epoch:03d}] silhouette score skipped (need >=2 classes)")
            finally:
                del val_embeds, val_labels_embed
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if is_dist():
            dist.barrier()

    if get_rank() == 0:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_to_save = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
        checkpoint = {
            "model_state_dict": model_to_save.state_dict(),
            "model_type": MODEL_TYPE,
            "timestamp": timestamp,
            "hyperparameters": {
                "temperature": TEMPERATURE,
                "embed_dim": EMBED_DIM,
                "proj_dim": PROJ_DIM,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LR,
                "weight_decay": WEIGHT_DECAY,
                "subset_size": SUBSET_SIZE,
            },
        }
        
        torch.save(checkpoint, model_path)
        print0(f"[Checkpoint] Saved model to {model_path}")

    barrier()
    if is_dist():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
