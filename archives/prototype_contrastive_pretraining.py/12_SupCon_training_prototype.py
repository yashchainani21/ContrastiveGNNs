import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
import os

torch.set_num_threads(1)  # keep it simple on CPU
device = torch.device('cpu')
print('Using device:', device)

TRAIN_NPZ = '../data/train/baseline_train_ecfp4.npz'
VAL_NPZ   = '../data/val/baseline_val_ecfp4.npz'
NORMALIZE = False
BATCH_SIZE = 256
EPOCHS = 10
LR = 3e-4
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.1
EMBED_DIM = 128
PROJ_DIM = 64
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

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
    
class fp_CNN_Encoder(nn.Module):
    
    def __init__(self, fp_dim = 2048, hidden_channels = (64, 128), embed_dim = 256, proj_dim = 120, use_projection = True, batchnorm_safe = True):
        super().__init__()
        c1, c2 = hidden_channels

        # convolution stack
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels = 1, out_channels = c1, kernel_size = 5, padding = 2),
            nn.BatchNorm1d(num_features = c1),
            nn.ReLU(inplace = True),
            nn.Conv1d(in_channels = c1, out_channels = c2, kernel_size = 5, padding = 2),
            nn.BatchNorm1d(num_features = c2),
            nn.ReLU(inplace = True),
            nn.AdaptiveMaxPool1d(1), # collapse length to 1
            )

        # encoder head
        self.fc = nn.Linear(in_features = c2, out_features = embed_dim)

        # projection head
        self.use_projection = use_projection
        self.batchnorm_safe = batchnorm_safe
        if self.use_projection:
            if self.batchnorm_safe:
                # LayerNorm works with batch_size=1
                norm_layer = nn.LayerNorm(embed_dim)
            else:
                # BatchNorm1d is better if you always train with batch_size > 1
                norm_layer = nn.BatchNorm1d(embed_dim)

            self.proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                norm_layer,
                nn.Linear(embed_dim, proj_dim),
            )

    def forward(self, x):
        # x: [B, fp_dim] or [B, 1, fp_dim]
        if x.dim() == 2:
            x = x.unsqueeze(1) # add channel dim, [B, 1, fp_dim]

        h = self.conv(x).squeeze(-1) # [B, c2, 1] -> [B, c2]
        g = F.normalize(self.fc(h), dim = -1, eps=1e-8) # [B, embed_dim], normalized embedding

        if self.use_projection:
            z = F.normalize(self.proj(g), dim = -1, eps=1e-8) # [B, proj_dim], normalized projection
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
        # x: [B, fp_dim]
        h = F.relu(self.fc1(x))
        g = F.normalize(self.fc2(h), dim=-1, eps=1e-6)
        z = F.normalize(self.proj(g), dim=-1, eps=1e-6)
        #print("z norms:", z.norm(dim=-1).min().item(), z.norm(dim=-1).max().item())
        return g, z
    
class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.tau = temperature
        self.eps = eps

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        z = F.normalize(z.float(), dim=-1, eps=self.eps)  # safety normalize

        sim = (z @ z.t()) / self.tau  # [B,B]
        eye = torch.eye(B, dtype=torch.bool, device=z.device)
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.t()) & (~eye)

        # remove anchors with no positives
        valid_mask = pos_mask.sum(1) > 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        # restrict to valid anchors only
        sim = sim[valid_mask]
        pos_mask = pos_mask[valid_mask]

        # log-softmax per row
        sim = sim - sim.max(dim=1, keepdim=True).values  # stability
        logits = sim.masked_fill(~torch.isfinite(sim), float('-inf'))  # guard
        denom = torch.logsumexp(logits, dim=1, keepdim=True)
        log_prob = logits - denom

        pos_counts = pos_mask.sum(1).clamp_min(1)
        pos_log_prob = (pos_mask * log_prob).sum(1) / pos_counts

        loss = -pos_log_prob.mean()
        if not torch.isfinite(loss):
            print("SupConLoss still produced NaN; returning 0.0")
            return torch.tensor(0.0, device=z.device, requires_grad=True)
        return loss

base_train = NPZFingerprints(TRAIN_NPZ, normalize=False)
labels_all = base_train.labels.astype(np.int64)

pos_idx = np.where(labels_all == 1)[0]
neg_idx = np.where(labels_all == 0)[0]

rng = np.random.default_rng(SEED)
neg_keep = min(10_000, len(neg_idx))
neg_sample = rng.choice(neg_idx, size=neg_keep, replace=False)

sel_idx = np.concatenate([pos_idx, neg_sample])
rng.shuffle(sel_idx)

print("Selected training subset:",
"pos =", (labels_all[sel_idx] == 1).sum(),
"neg =", (labels_all[sel_idx] == 0).sum(),
"total =", len(sel_idx))

if NORMALIZE:
    arr = base_train.fps[sel_idx].astype(np.float32)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0) + 1e-8
    train_ds = Subset(NPZFingerprints(TRAIN_NPZ, normalize=True, mean=mean, std=std), sel_idx)
else:
    train_ds = Subset(base_train, sel_idx)

fp_dim = base_train.D
model = TinyMLP(fp_dim, EMBED_DIM, PROJ_DIM).to(device)

# model = fp_CNN_Encoder(
#     fp_dim=base_train.D,
#     hidden_channels=(64, 128),
#     embed_dim=EMBED_DIM,
#     proj_dim=PROJ_DIM,
#     use_projection=True,
#     batchnorm_safe=True).to(device)

criterion = SupConLoss(temperature=TEMPERATURE).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

print("Train size:", len(train_ds), "| fp_dim:", fp_dim)

train_loader = DataLoader(train_ds,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=0)

for epoch in range(1, EPOCHS+1):
    model.train()
    epoch_loss, steps = 0.0, 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)

        # Check for degenerate batch
        if yb.unique().numel() < 2:
            print("Batch has only one class → skipping")
            continue

        g, z = model(xb)
        loss = criterion(z, yb)

        if not torch.isfinite(loss):
            print(f'  Warning: non-finite loss (loss={loss.item():.4f}); skipping batch')
            continue

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        steps += 1

        if steps == 0:
            print(f"[Epoch {epoch:03d}] all batches skipped")
        else:
            train_loss = epoch_loss / steps
            print(f"[Epoch {epoch:03d}] train_supcon={train_loss:.4f}")
