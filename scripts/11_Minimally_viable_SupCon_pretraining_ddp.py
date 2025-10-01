import os
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# -------------------- Utilities --------------------
def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def setup_distributed() -> torch.device:
    """Initialize torch.distributed using environment variables set by torchrun."""
    if not dist.is_available():
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


def barrier():
    if is_dist():
        dist.barrier()


def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)


def visualize_tsne(model, dataset, device, max_samples=2000, seed=42, save_prefix="tsne_embeddings"):
    if get_rank() != 0:
        return
    model.eval()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=min(max_samples, len(dataset)), replace=False)
    xs, ys = [], []
    with torch.no_grad():
        for i in idx:
            x, y = dataset[i]
            xs.append(x.numpy())
            ys.append(y)
    X_raw = np.stack(xs)
    y_all = np.array(ys)

    xb = torch.from_numpy(X_raw).to(device)
    with torch.no_grad():
        g, z = model(xb)
    X_embed = g.cpu().numpy()

    tsne_raw = TSNE(n_components=2, random_state=seed, perplexity=30).fit_transform(X_raw)
    tsne_embed = TSNE(n_components=2, random_state=seed, perplexity=30).fit_transform(X_embed)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc0 = axes[0].scatter(tsne_raw[:, 0], tsne_raw[:, 1], c=y_all, cmap="coolwarm", alpha=0.7, s=8)
    axes[0].set_title("Raw Fingerprints (t-SNE)")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    sc1 = axes[1].scatter(tsne_embed[:, 0], tsne_embed[:, 1], c=y_all, cmap="coolwarm", alpha=0.7, s=8)
    axes[1].set_title("Learned Embeddings (t-SNE)")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    plt.colorbar(sc1, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1, label="Class label")
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_prefix}_{timestamp}.png"
    plt.savefig(filename, dpi=300)
    print0(f"[t-SNE] Saved plot to {os.path.abspath(filename)}")
    plt.show()


# -------------------- Hyperparameters --------------------
MODEL_TYPE = "resnet"  # "mlp" | "cnn" | "resnet"

TRAIN_NPZ = '../data/train/baseline_train_ecfp4.npz'
VAL_NPZ   = '../data/val/baseline_val_ecfp4.npz'
NORMALIZE = False
BATCH_SIZE = 64
EPOCHS = 1000
LR = 3e-4
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.05
EMBED_DIM = 512
PROJ_DIM = 256
SEED = 42
SUBSET_SIZE = 1000

np.random.seed(SEED)
torch.manual_seed(SEED)


# -------------------- Dataset --------------------
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
            self.mean = None; self.std = None

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x = self.fps[idx].astype(np.float32)
        if self.normalize:
            x = (x - self.mean) / self.std
        y = int(self.labels[idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# -------------------- Models --------------------
def evaluate_linear_probe(model, dataloader, device, max_samples=5000):
    # Run only on rank 0 for simplicity
    if get_rank() != 0:
        return None
    model.eval()
    xs, ys = [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            g, z = model(xb)
            xs.append(g.cpu().numpy())
            ys.append(yb.cpu().numpy())
            if len(xs) * xb.size(0) >= max_samples:
                break
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, y)
    y_pred = clf.predict_proba(X)[:, 1]
    auprc = average_precision_score(y, y_pred)
    return auprc


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
        self.layer1 = ResidualBlock(c1, c1)
        self.layer2 = ResidualBlock(c1, c2)
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
        h = self.layer1(h)
        h = self.layer2(h)
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
    def __init__(self, temperature: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.tau = temperature
        self.eps = eps

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        z = F.normalize(z.float(), dim=-1, eps=self.eps)
        sim = (z @ z.t()) / self.tau
        eye = torch.eye(B, dtype=torch.bool, device=z.device)
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.t()) & (~eye)
        valid_mask = pos_mask.sum(1) > 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=z.device, requires_grad=True)
        sim = sim[valid_mask]
        pos_mask = pos_mask[valid_mask]
        sim = sim - sim.max(dim=1, keepdim=True).values
        denom = torch.logsumexp(sim, dim=1, keepdim=True)
        log_prob = sim - denom
        pos_counts = pos_mask.sum(1).clamp_min(1)
        pos_log_prob = (pos_mask * log_prob).sum(1) / pos_counts
        return -pos_log_prob.mean()


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


# -------------------- Main --------------------
def main():
    device = setup_distributed()
    rank = get_rank()
    world = get_world_size()
    print0(f"DDP initialized: rank={rank}, world={world}, device={device}")

    # Dataset and balanced subset selection on rank 0, then broadcast
    base_train = NPZFingerprints(TRAIN_NPZ, normalize=False)
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
        # Broadcast length then indices
        sel_len_t = torch.from_numpy(sel_len)
        dist.broadcast(sel_len_t, src=0)
        sel_len = sel_len_t.cpu().numpy()
        if rank != 0:
            sel_idx_np = np.empty((sel_len[0],), dtype=np.int64)
        sel_idx_t = torch.from_numpy(sel_idx_np)
        dist.broadcast(sel_idx_t, src=0)
        sel_idx_np = sel_idx_t.cpu().numpy()

    train_ds = Subset(base_train, sel_idx_np.tolist())

    # Distributed sampler over the subset (balanced overall due to selection above)
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=False) if world > 1 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print0("Subset stats:",
           "pos =", int((labels_all[sel_idx_np] == 1).sum()),
           "neg =", int((labels_all[sel_idx_np] == 0).sum()),
           "total =", len(sel_idx_np))

    # Model, loss, optimizer
    torch.set_num_threads(1)
    model = build_model(model_type=MODEL_TYPE, fp_dim=2048,
                        embed_dim=EMBED_DIM, proj_dim=PROJ_DIM).to(device)

    if world > 1 and torch.cuda.is_available():
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device.index], output_device=device.index)
    elif world > 1:
        model = nn.parallel.DistributedDataParallel(model)

    criterion = SupConLoss(temperature=TEMPERATURE).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                                weight_decay=WEIGHT_DECAY, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print0("Train size:", len(train_ds))

    for epoch in range(1, EPOCHS + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        epoch_loss, steps = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if yb.unique().numel() < 2:
                continue
            g, z = model(xb)
            loss = criterion(z, yb)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1

        scheduler.step()

        if get_rank() == 0:
            if steps == 0:
                print("[Epoch %03d] all batches skipped" % epoch)
            else:
                train_loss = epoch_loss / steps
                # Small eval on rank 0
                eval_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
                auprc = evaluate_linear_probe(model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model,
                                              eval_loader, device)
                print(f"[Epoch {epoch:03d}] train_supcon={train_loss:.4f}, train_auprc={auprc:.4f}")

    # Final t-SNE on rank 0
    visualize_tsne(model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model, train_ds, device)

    barrier()
    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

