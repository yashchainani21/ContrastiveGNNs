import numpy as np
import torch
from torch import amp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset, Sampler
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import datetime
import math

# ---- Hyperparameters ----

MODEL_TYPE = "resnet"  # "mlp" | "cnn" | "resnet"
TRAIN_NPZ = '../data/train/baseline_train_ecfp4.npz'
VAL_NPZ   = '../data/val/baseline_val_ecfp4.npz'
NORMALIZE = False
BATCH_SIZE = 8192
EPOCHS = 2
LR = 3e-4
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.05   # don’t exceed 0.1
EMBED_DIM = 512
PROJ_DIM = 256
SEED = 42
SUBSET_SIZE = 4000000 # total number of datapoints to train on
USE_WEIGHTED_SAMPLER = True  # toggle to switch between weighted sampler and positive-guaranteed batches

np.random.seed(SEED)
torch.manual_seed(SEED)

def visualize_tsne(model, dataset, device, max_samples=2000, seed=42, save_prefix="tsne_embeddings"):
    """
    Compare raw fingerprints vs learned embeddings (g).
    Saves a PNG plot in the current working directory.
    """
    model.eval()
    rng = np.random.default_rng(seed)

    # Subsample dataset if large
    idx = rng.choice(len(dataset), size=min(max_samples, len(dataset)), replace=False)
    xs, ys = [], []
    with torch.no_grad():
        for i in idx:
            x, y = dataset[i]
            xs.append(x.numpy())
            ys.append(y)
    X_raw = np.stack(xs)
    y_all = np.array(ys)

    # Learned embeddings
    xb = torch.from_numpy(X_raw).to(device)
    with torch.no_grad():
        g, z = model(xb)
    X_embed = g.cpu().numpy()

    # Run t-SNE
    tsne_raw = TSNE(n_components=2, random_state=seed, perplexity=30).fit_transform(X_raw)
    tsne_embed = TSNE(n_components=2, random_state=seed, perplexity=30).fit_transform(X_embed)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sc0 = axes[0].scatter(tsne_raw[:, 0], tsne_raw[:, 1], c=y_all, cmap="coolwarm", alpha=0.7, s=8)
    axes[0].set_title("Raw Fingerprints (t-SNE)")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    sc1 = axes[1].scatter(tsne_embed[:, 0], tsne_embed[:, 1], c=y_all, cmap="coolwarm", alpha=0.7, s=8)
    axes[1].set_title("Learned Embeddings (t-SNE)")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    plt.colorbar(sc1, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1, label="Class label")
    plt.tight_layout()

    # Save with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_prefix}_{timestamp}.png"
    plt.savefig(filename, dpi=300)
    print(f"[t-SNE] Saved plot to {os.path.abspath(filename)}")

    plt.show()

torch.set_num_threads(1)
device = torch.device('cuda')
print('Using device:', device)

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
            self.mean = None; self.std = None

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x = self.fps[idx].astype(np.float32)
        if self.normalize:
            x = (x - self.mean) / self.std
        y = int(self.labels[idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
    
# ---- Models ----
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
    nn.Linear(embed_dim, proj_dim))

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

# ---- ResNet-style fingerprint CNN encoder ----
class fp_CNN_ResNetEncoder(nn.Module):
    def __init__(self, fp_dim=2048, hidden_channels=(128, 256), embed_dim=128,
                 proj_dim=64, use_projection=True, batchnorm_safe=True,
                 dropout_p=0.2, use_proj_skip=True):
        super().__init__()
        c1, c2 = hidden_channels

        # initial conv to expand channels
        self.stem = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
        )

        # residual stack
        self.layer1 = ResidualBlock(c1, c1)
        self.layer2 = ResidualBlock(c1, c2)

        # global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # encoder head
        self.fc = nn.Linear(c2, embed_dim)

        # projection head
        self.use_projection = use_projection
        self.batchnorm_safe = batchnorm_safe
        self.use_proj_skip = use_proj_skip

        if self.use_projection:
            if self.batchnorm_safe:
                norm_layer = nn.LayerNorm(embed_dim)
            else:
                norm_layer = nn.BatchNorm1d(embed_dim)

            # residual projection block
            self.proj_fc1 = nn.Linear(embed_dim, embed_dim)
            self.proj_fc2 = nn.Linear(embed_dim, proj_dim)
            self.relu = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(p=dropout_p)
            self.norm = norm_layer

    def forward(self, x):
        # x: [B, fp_dim] or [B, 1, fp_dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, fp_dim]

        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.pool(h).squeeze(-1)  # [B, c2]

        g = F.normalize(self.fc(h), dim=-1, eps=1e-8)  # [B, embed_dim]

        if self.use_projection:
            # first projection layer
            h_proj = self.relu(self.proj_fc1(g))
            h_proj = self.dropout(h_proj)
            h_proj = self.norm(h_proj)

            if self.use_proj_skip:
                # skip connection: add input g back
                h_proj = h_proj + g  

            z = F.normalize(self.proj_fc2(h_proj), dim=-1, eps=1e-8)  # [B, proj_dim]
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

# ---- SupCon loss ----
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
    
# ---- Custom samplers ----
class EnsurePositiveBatchSampler(Sampler):
    """Batch sampler that guarantees at least one positive label per batch.

    Falls back to reusing positive samples if there are fewer positives than batches.
    """

    def __init__(self, labels, batch_size, drop_last=False, seed=0):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.labels = np.asarray(labels)
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.seed = seed
        self.num_samples = len(self.labels)
        self.pos_indices = np.where(self.labels == 1)[0]
        if self.pos_indices.size == 0:
            raise ValueError("EnsurePositiveBatchSampler requires at least one positive sample.")
        self._epoch = 0

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        return math.ceil(self.num_samples / self.batch_size)

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        indices = list(rng.permutation(self.num_samples))
        idx_ptr = 0

        while idx_ptr < len(indices):
            # Form a candidate batch
            current_slice = indices[idx_ptr: idx_ptr + self.batch_size]
            if len(current_slice) < self.batch_size and self.drop_last:
                break
            if not current_slice:
                break

            # Inject a positive if missing
            if not np.any(self.labels[current_slice] == 1):
                search_idx = idx_ptr + len(current_slice)
                swapped = False
                while search_idx < len(indices):
                    candidate = indices[search_idx]
                    if self.labels[candidate] == 1:
                        indices[idx_ptr + len(current_slice) - 1], indices[search_idx] = (
                            indices[search_idx],
                            indices[idx_ptr + len(current_slice) - 1],
                        )
                        current_slice = indices[idx_ptr: idx_ptr + len(current_slice)]
                        swapped = True
                        break
                    search_idx += 1

                if not swapped:
                    replacement = int(self.pos_indices[rng.integers(len(self.pos_indices))])
                    displaced = indices[idx_ptr + len(current_slice) - 1]
                    indices[idx_ptr + len(current_slice) - 1] = replacement
                    indices.append(displaced)
                    current_slice = indices[idx_ptr: idx_ptr + len(current_slice)]

            yield list(current_slice)
            idx_ptr += len(current_slice)

# ---- Dataset & Sampler ----
base_train = NPZFingerprints(TRAIN_NPZ, normalize=False)
labels_all = base_train.labels.astype(np.int64)

pos_idx = np.where(labels_all == 1)[0]
neg_idx = np.where(labels_all == 0)[0]
rng = np.random.default_rng(SEED)

n_pos = min(SUBSET_SIZE // 2, len(pos_idx))
n_neg = min(SUBSET_SIZE - n_pos, len(neg_idx))

pos_sample = rng.choice(pos_idx, size=n_pos, replace=False)
neg_sample = rng.choice(neg_idx, size=n_neg, replace=False)

sel_idx = np.concatenate([pos_sample, neg_sample])
rng.shuffle(sel_idx)

train_ds = Subset(base_train, sel_idx)

print("Subset stats:",
      "pos =", (labels_all[sel_idx] == 1).sum(),
      "neg =", (labels_all[sel_idx] == 0).sum(),
      "total =", len(sel_idx))

labels_subset = labels_all[sel_idx]

if USE_WEIGHTED_SAMPLER:
    class_sample_count = np.array([(labels_subset == t).sum() for t in np.unique(labels_subset)])
    weights = 1.0 / class_sample_count
    sample_weights = np.array([weights[t] for t in labels_subset])
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
    )
else:
    batch_sampler = EnsurePositiveBatchSampler(
        labels=labels_subset,
        batch_size=BATCH_SIZE,
        drop_last=False,
        seed=SEED,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=0,
    )

# ---- Model builder ----
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
    
# ---- Training ----
model = build_model(model_type=MODEL_TYPE, fp_dim=2048,
                    embed_dim=EMBED_DIM, proj_dim=PROJ_DIM).to(device)

criterion = SupConLoss(temperature=TEMPERATURE).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                            weight_decay=WEIGHT_DECAY, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print("Train size:", len(train_ds))

scaler = amp.GradScaler("cuda")

for epoch in range(1, EPOCHS+1):
    if not USE_WEIGHTED_SAMPLER:
        batch_sampler = getattr(train_loader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)
    model.train()
    epoch_loss, steps = 0.0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)

        if yb.unique().numel() < 2:
            print("Batch has only one class → skipping")
            continue

        with amp.autocast("cuda", dtype=torch.float16): 
            g, z = model(xb)
            loss = criterion(z, yb)
            if not torch.isfinite(loss):
                print(f"Warning: non-finite loss at epoch {epoch}")
                continue
        
        scaler.scale(loss).backward()  
        scaler.step(optimizer) 
        scaler.update() 
        
        epoch_loss += loss.item()
        steps += 1

    scheduler.step()
    if steps == 0:
        print(f"[Epoch {epoch:03d}] all batches skipped")
    else:
        train_loss = epoch_loss / steps
        # auprc = evaluate_linear_probe(model, train_loader, device)
        print(f"[Epoch {epoch:03d}] train_supcon={train_loss:.4f}")