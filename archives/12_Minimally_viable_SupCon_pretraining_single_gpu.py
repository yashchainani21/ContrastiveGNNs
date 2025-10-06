import os
import math
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, Dataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# ---- Hyperparameters ----

MODEL_TYPE = "resnet"  # "mlp" | "cnn" | "resnet"

TRAIN_NPZ = '../data/train/baseline_train_ecfp4.npz'
VAL_NPZ = '../data/val/baseline_val_ecfp4.npz'
NORMALIZE = False
BATCH_SIZE = 64
EPOCHS = 1000
LR = 3e-4
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.05
EMBED_DIM = 512
PROJ_DIM = 256
SEED = 42
SUBSET_SIZE = 10_000  # limit per epoch; set None to stream entire dataset
ENSURE_POSITIVE_PER_BATCH = True
DROP_LAST = False
EVAL_MAX_SAMPLES = 5000

np.random.seed(SEED)
torch.manual_seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_DIR = os.path.join(SCRIPT_DIR, "..", "models")


def visualize_tsne(model, dataset, device, max_samples=2000, seed=42, save_prefix="tsne_embeddings"):
    """Compare raw fingerprints vs learned embeddings on a cached subset."""
    model.eval()
    rng = np.random.default_rng(seed)

    idx = rng.choice(len(dataset), size=min(max_samples, len(dataset)), replace=False)
    xs, ys = [], []
    with torch.no_grad():
        for i in idx:
            x, y = dataset[i]
            xs.append(x.numpy())
            ys.append(y.item())
    X_raw = np.stack(xs)
    y_all = np.array(ys)

    xb = torch.from_numpy(X_raw).to(device)
    with torch.no_grad():
        g, _ = model(xb)
    X_embed = g.cpu().numpy()

    tsne_raw = TSNE(n_components=2, random_state=seed, perplexity=30).fit_transform(X_raw)
    tsne_embed = TSNE(n_components=2, random_state=seed, perplexity=30).fit_transform(X_embed)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc0 = axes[0].scatter(tsne_raw[:, 0], tsne_raw[:, 1], c=y_all, cmap="coolwarm", alpha=0.7, s=8)
    axes[0].set_title("Raw Fingerprints (t-SNE)")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    sc1 = axes[1].scatter(tsne_embed[:, 0], tsne_embed[:, 1], c=y_all, cmap="coolwarm", alpha=0.7, s=8)
    axes[1].set_title("Learned Embeddings (t-SNE)")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.colorbar(sc1, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1, label="Class label")
    plt.tight_layout()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_prefix}_{timestamp}.png"
    plt.savefig(filename, dpi=300)
    print(f"[t-SNE] Saved plot to {os.path.abspath(filename)}")

    plt.show()


torch.set_num_threads(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


class StreamingNPZBatchDataset(IterableDataset):
    """Streams batches from NPZ file without loading the entire array into RAM."""

    def __init__(self, npz_path: str, batch_size: int, subset_size: int | None,
                 ensure_positive: bool = True, drop_last: bool = False,
                 shuffle: bool = True, seed: int = 0, normalize: bool = False):
        super().__init__()
        self.npz_path = npz_path
        self.batch_size = batch_size
        self.subset_size = subset_size
        self.ensure_positive = ensure_positive
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.base_seed = seed
        self.normalize = normalize
        self._epoch = 0

        with np.load(self.npz_path, allow_pickle=False, mmap_mode='r') as data:
            labels = data['labels']
            self.total_samples = labels.shape[0]
            self.fp_dim = data['fps'].shape[1]
            self.pos_indices_full = np.flatnonzero(labels == 1)
            self.neg_indices_full = np.flatnonzero(labels == 0)
        if ensure_positive and self.pos_indices_full.size == 0:
            raise ValueError("Dataset does not contain positive samples but ensure_positive=True")

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def sample_indices(self, num_samples: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(self.base_seed if seed is None else seed)
        count = min(num_samples, self.total_samples)
        return rng.choice(self.total_samples, size=count, replace=False)

    def _select_epoch_indices(self, rng: np.random.Generator) -> np.ndarray:
        if self.subset_size is not None and self.subset_size < self.total_samples:
            n_pos = min(self.subset_size // 2, len(self.pos_indices_full))
            n_neg = min(self.subset_size - n_pos, len(self.neg_indices_full))
            pos_choices = rng.choice(self.pos_indices_full, size=n_pos, replace=False) if n_pos > 0 else np.empty(0, dtype=np.int64)
            neg_choices = rng.choice(self.neg_indices_full, size=n_neg, replace=False) if n_neg > 0 else np.empty(0, dtype=np.int64)
            indices = np.concatenate([pos_choices, neg_choices])
        else:
            indices = np.arange(self.total_samples, dtype=np.int64)

        if self.shuffle:
            rng.shuffle(indices)
        return indices

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        rng = np.random.default_rng(self.base_seed + 9973 * self._epoch + worker_id)

        with np.load(self.npz_path, allow_pickle=False, mmap_mode='r') as data:
            fps = data['fps']
            labels = data['labels']

            indices = self._select_epoch_indices(rng)
            label_subset = labels[indices]
            pos_subset = indices[label_subset == 1]

            batch_indices: list[int] = []
            for idx in indices:
                batch_indices.append(int(idx))
                if len(batch_indices) == self.batch_size:
                    yield self._materialize_batch(batch_indices, fps, labels, pos_subset, rng)
                    batch_indices = []

            if batch_indices and not self.drop_last:
                yield self._materialize_batch(batch_indices, fps, labels, pos_subset, rng)

    def _materialize_batch(self, batch_indices, fps, labels, pos_subset, rng):
        if self.ensure_positive and len(pos_subset) > 0:
            if not any(labels[idx] == 1 for idx in batch_indices):
                replacement = int(pos_subset[rng.integers(len(pos_subset))])
                if len(batch_indices) == self.batch_size:
                    batch_indices[-1] = replacement
                else:
                    batch_indices.append(replacement)

        xs, ys = [], []
        for idx in batch_indices:
            fp_row = fps[idx]
            x_np = np.asarray(fp_row, dtype=np.float32)
            if not np.issubdtype(fp_row.dtype, np.float32):
                x_np = x_np.astype(np.float32, copy=False)
            x = torch.from_numpy(np.ascontiguousarray(x_np))
            if self.normalize:
                raise NotImplementedError("On-the-fly normalization not implemented for streaming dataset")
            y = torch.tensor(int(labels[idx]), dtype=torch.long)
            xs.append(x)
            ys.append(y)

        xb = torch.stack(xs, dim=0)
        yb = torch.stack(ys, dim=0)
        return xb, yb


class NPZMemmapDataset(Dataset):
    """Indexable dataset backed by memory-mapped NPZ arrays (for eval/visualization)."""

    def __init__(self, npz_path: str, indices: np.ndarray | None = None, normalize: bool = False):
        super().__init__()
        self._npz = np.load(npz_path, allow_pickle=False, mmap_mode='r')
        self.fps = self._npz['fps']
        self.labels = self._npz['labels']
        self.normalize = normalize
        self.indices = np.arange(self.labels.shape[0], dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        fp_row = self.fps[real_idx]
        x_np = np.asarray(fp_row, dtype=np.float32)
        if not np.issubdtype(fp_row.dtype, np.float32):
            x_np = x_np.astype(np.float32, copy=False)
        x = torch.from_numpy(np.ascontiguousarray(x_np))
        if self.normalize:
            raise NotImplementedError("Normalization not implemented for NPZMemmapDataset")
        y = torch.tensor(int(self.labels[real_idx]), dtype=torch.long)
        return x, y

    def __del__(self):
        try:
            self._npz.close()
        except Exception:
            pass


def evaluate_linear_probe(model, dataloader, device, max_samples=5000):
    model.eval()
    xs, ys = [], []
    total_seen = 0

    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            g, _ = model(xb)
            xs.append(g.cpu().numpy())
            ys.append(yb.cpu().numpy())
            total_seen += xb.size(0)
            if total_seen >= max_samples:
                break

    if not xs:
        return None

    X = np.concatenate(xs, axis=0)[:max_samples]
    y = np.concatenate(ys, axis=0)[:max_samples]

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, y)
    y_pred = clf.predict_proba(X)[:, 1]
    return average_precision_score(y, y_pred)


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
            return g, g


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


stream_dataset = StreamingNPZBatchDataset(
    npz_path=TRAIN_NPZ,
    batch_size=BATCH_SIZE,
    subset_size=SUBSET_SIZE,
    ensure_positive=ENSURE_POSITIVE_PER_BATCH,
    drop_last=DROP_LAST,
    shuffle=True,
    seed=SEED,
    normalize=NORMALIZE,
)

train_loader = DataLoader(
    stream_dataset,
    batch_size=None,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

approx_samples = stream_dataset.subset_size if stream_dataset.subset_size is not None else stream_dataset.total_samples
approx_samples = int(max(1, approx_samples))
approx_batches = math.ceil(approx_samples / BATCH_SIZE)
print("Approx. train batches per epoch:", approx_batches)

model = build_model(model_type=MODEL_TYPE, fp_dim=stream_dataset.fp_dim,
                    embed_dim=EMBED_DIM, proj_dim=PROJ_DIM).to(device)

criterion = SupConLoss(temperature=TEMPERATURE).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                            weight_decay=WEIGHT_DECAY, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print("Streaming from:", TRAIN_NPZ)

for epoch in range(1, EPOCHS + 1):
    stream_dataset.set_epoch(epoch)
    model.train()
    epoch_loss, steps = 0.0, 0

    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
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

    if steps == 0:
        print(f"[Epoch {epoch:03d}] all batches skipped")
        continue

    train_loss = epoch_loss / steps

    eval_indices = stream_dataset.sample_indices(EVAL_MAX_SAMPLES, seed=SEED + epoch)
    eval_dataset = NPZMemmapDataset(TRAIN_NPZ, indices=eval_indices, normalize=NORMALIZE)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    auprc = evaluate_linear_probe(model, eval_loader, device, max_samples=EVAL_MAX_SAMPLES)
    print(f"[Epoch {epoch:03d}] train_supcon={train_loss:.4f}, train_auprc={(auprc or 0.0):.4f}")


os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint = {
    "model_state_dict": model.state_dict(),
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
        "ensure_positive": ENSURE_POSITIVE_PER_BATCH,
    },
}
model_filename = f"supcon_stream_{MODEL_TYPE.lower()}_{timestamp}.pt"
model_path = os.path.abspath(os.path.join(MODEL_SAVE_DIR, model_filename))
torch.save(checkpoint, model_path)
print(f"[Checkpoint] Saved model to {model_path}")


eval_indices = stream_dataset.sample_indices(min(2000, EVAL_MAX_SAMPLES), seed=SEED + 123)
viz_dataset = NPZMemmapDataset(TRAIN_NPZ, indices=eval_indices, normalize=NORMALIZE)
visualize_tsne(model, viz_dataset, device)
