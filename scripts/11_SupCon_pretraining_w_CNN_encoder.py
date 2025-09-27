import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, Subset
from types import SimpleNamespace
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Tuple

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
        g = F.normalize(self.fc(h), dim = -1) # [B, embed_dim], normalized embedding

        if self.use_projection:
            z = F.normalize(self.proj(g), dim = -1)
            return g, z
        else:
            return g


class NPZFingerprints(Dataset):
    """
    Dataset for loading precomputed fingerprints from a .npz file.
    """
    def __init__(self, npz_path: str, dtype = torch.float32, normalize = False, mean=None, std=None):
        # Note: .npz members are not truly memory-mapped; for very large sets
        # consider separate .npy with mmap_mode or zarr/hdf5.
        z = np.load(npz_path, allow_pickle=False)
        self.fps = z["fps"]
        self.labels = z["labels"]
        self.N, self.D = self.fps.shape
        self.dtype = dtype
        self.normalize = normalize
        if normalize:
            if mean is not None and std is not None:
                self.mean = np.asarray(mean, dtype=np.float32)
                self.std = np.asarray(std, dtype=np.float32)
            else:
                # compute per-feature mean/std if requested
                arr = np.asarray(self.fps, dtype=np.float32)
                self.mean = arr.mean(axis=0)
                self.std = arr.std(axis=0) + 1e-8 # avoid div-by-zero

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int):
        x = np.asarray(self.fps[idx], dtype=np.float32)
        if self.normalize:
            x = (x - self.mean) / self.std
        y = int(self.labels[idx])
        return torch.as_tensor(x, dtype=self.dtype), torch.as_tensor(y, dtype=torch.long)
    
class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss (Khosla et al., 2020)
    Operates on normalized embeddings; no augmentations needed.
    All samples sharing the same label in a batch are considered positives.
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.tau = temperature

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        z: [B, d] normalized projections
        labels: [B] int labels
        """
        B = z.size(0)
        sim = z @ z.t() / self.tau  # cosine sims since z normalized

        # masks
        eye = torch.eye(B, dtype=torch.bool, device=z.device)
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.t()) & (~eye)   # same-class pairs
        all_mask = ~eye                              # all except self

        # log prob over all others
        logits = sim
        denom = torch.logsumexp(logits.masked_fill(~all_mask, -1e9), dim=1, keepdim=True)
        log_prob = logits - denom

        # average over positives per anchor
        pos_log_prob = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-9)
        loss = -pos_log_prob.mean()
        return loss


def make_weighted_sampler(labels_np: np.ndarray):
    """
    Balances batches by inverse class frequency
    """
    counts = np.array([np.sum(labels_np == 0), np.sum(labels_np == 1)], dtype=np.float64)
    w_per_class = 1.0 / (counts + 1e-12)
    weights = np.array([w_per_class[y] for y in labels_np], dtype=np.float64)
    return WeightedRandomSampler(weights=torch.from_numpy(weights).double(),
                                 num_samples=len(labels_np), replacement=True)

def seed_all(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

@torch.no_grad()
def evaluate_val(val_loader, encoder, device):
    encoder.eval()
    all_g, all_y = [], []
    
    for xb, yb in val_loader:
        xb = xb.to(device, non_blocking=True)
        out = encoder(xb)
        g = out[0] if encoder.use_projection else out  # [B, embed_dim]
        all_g.append(g.cpu())
        all_y.append(yb)

    X = torch.cat(all_g, dim=0).numpy()
    y = torch.cat(all_y, dim=0).numpy()

    # simple linear probe
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, y)
    y_pred = clf.predict_proba(X)[:,1]

    auprc = average_precision_score(y, y_pred)
    return auprc

def train(args):
    seed_all(args.seed)
    # Distributed setup (torchrun-friendly)
    is_ddp = False
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    if world_size > 1 and not args.cpu:
        is_ddp = True
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else "cpu")

    base_train_ds = NPZFingerprints(args.train_npz, dtype=torch.float32, normalize=False)
    # Optionally cap training set for quick runs
    train_ds = base_train_ds
    train_indices = None
    if getattr(args, 'train_limit', None):
        limit = int(args.train_limit)
        if limit > 0 and len(base_train_ds) > limit:
            train_indices = np.arange(limit)
            train_ds = Subset(base_train_ds, train_indices)
            if (not is_ddp) or (rank == 0):
                print(f"Capped training set to first {limit} samples (was {len(base_train_ds)})")
    val_loader = None
    if args.val_npz and os.path.exists(args.val_npz):
        # Use train statistics for normalization to avoid leakage
        if args.normalize:
            # Compute mean/std from the (possibly capped) training set
            if isinstance(train_ds, Subset):
                idx = np.array(train_ds.indices)
                arr = np.asarray(base_train_ds.fps[idx], dtype=np.float32)
                train_mean = arr.mean(axis=0)
                train_std = arr.std(axis=0) + 1e-8
            else:
                # train_ds is NPZFingerprints with normalize=False; compute here
                arr = np.asarray(train_ds.fps, dtype=np.float32)
                train_mean = arr.mean(axis=0)
                train_std = arr.std(axis=0) + 1e-8
            val_ds = NPZFingerprints(args.val_npz, dtype=torch.float32, normalize=True, mean=train_mean, std=train_std)
        else:
            val_ds = NPZFingerprints(args.val_npz, dtype=torch.float32, normalize=False)
        if is_ddp:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler,
                                    num_workers=args.num_workers, pin_memory=(device.type=='cuda'))
        else:
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=(device.type=='cuda'))

    # weighted sample to ensure PKSs appear regularly in batches
    # Labels for balancing (pull from base dataset, respecting any cap)
    if isinstance(train_ds, Subset):
        labels_np = base_train_ds.labels[np.array(train_ds.indices)].astype(np.int64)
    else:
        labels_np = train_ds.labels.astype(np.int64)
    if is_ddp:
        # Use DistributedSampler; disable custom balancing in this simple setup
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True)
        train_loader = DataLoader(train_ds,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  pin_memory=(device.type=='cuda'),
                                  drop_last=True)
    else:
        sampler = make_weighted_sampler(labels_np) if args.balance else None
        train_loader = DataLoader(train_ds,
                                  batch_size=args.batch_size,
                                  sampler=sampler,
                                  shuffle=(sampler is None),
                                  num_workers=args.num_workers,
                                  pin_memory=(device.type=='cuda'),
                                  drop_last=True)

    # model & loss
    base_encoder = fp_CNN_Encoder(fp_dim = args.fp_dim,
                              hidden_channels = (args.c1, args.c2),
                              embed_dim = args.embed_dim,
                              proj_dim = args.proj_dim,
                              use_projection = args.use_projection,
                              batchnorm_safe = args.batchnorm_safe).to(device)
    encoder = base_encoder
    if is_ddp:
        encoder = DDP(base_encoder, device_ids=[local_rank], output_device=local_rank)

    criterion = SupConLoss(temperature=args.temperature).to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    os.makedirs(args.out_dir, exist_ok=True)
    best_monitor = -1.0
    patience = 0

    # training loop
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        epoch_loss, steps = 0.0, 0
        t0 = time.time()

        if is_ddp:
            # Ensure different shuffling per epoch
            train_loader.sampler.set_epoch(epoch)
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                # no augmentations
                _, z = encoder(xb) 
                loss = criterion(z, yb)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(encoder.parameters(), args.grad_clip)
            scaler.step(optimizer); scaler.update()

            epoch_loss += loss.item(); steps += 1

        epoch_loss /= max(1, steps)
        msg = f"[Epoch {epoch:03d}] train_supcon={epoch_loss:.4f} time={time.time()-t0:.1f}s"

        if val_loader is not None:
            # Distributed-safe evaluation: gather embeddings across ranks
            encoder.eval()
            all_g_local, all_y_local = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    out = encoder(xb)
                    g = out[0] if base_encoder.use_projection else out
                    all_g_local.append(g.cpu())
                    all_y_local.append(yb)
            X_local = torch.cat(all_g_local, dim=0).numpy() if all_g_local else np.zeros((0, args.embed_dim), dtype=np.float32)
            y_local = torch.cat(all_y_local, dim=0).numpy() if all_y_local else np.zeros((0,), dtype=np.int64)

            if is_ddp:
                # gather objects across ranks
                obj = (X_local, y_local)
                gathered = [None for _ in range(world_size)]
                dist.all_gather_object(gathered, obj)
                if rank == 0:
                    X = np.concatenate([g[0] for g in gathered if g is not None and len(g[0])>0], axis=0)
                    y = np.concatenate([g[1] for g in gathered if g is not None and len(g[1])>0], axis=0)
                else:
                    X = None; y = None
                # Broadcast a small flag to synchronize
                flag = torch.tensor(1, device=device)
                dist.all_reduce(flag)
            else:
                X, y = X_local, y_local

            if (not is_ddp) or (is_ddp and rank == 0):
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import average_precision_score
                if X is not None and len(X) > 1 and len(np.unique(y)) > 1:
                    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
                    clf.fit(X, y)
                    y_pred = clf.predict_proba(X)[:,1]
                    auprc = average_precision_score(y, y_pred)
                    msg += f" val_AUPRC={auprc:.4f}"
                else:
                    msg += " val_AUPRC=NA"
        if (not is_ddp) or (rank == 0):
            print(msg)

    if (not is_ddp) or (rank == 0):
        state = encoder.module.state_dict() if is_ddp else encoder.state_dict()
        torch.save({"encoder": state, "args": vars(args)},
                   os.path.join(args.out_dir, "last_encoder.pt"))
        print(f"Saved to {args.out_dir}")
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    # Configure arguments here instead of argparse
    args = SimpleNamespace(
        seed=42,
        cpu=False,
        # Data
        train_npz="../data/train/baseline_train_ecfp4.npz",
        val_npz="../data/val/baseline_val_ecfp4.npz",
        normalize=False,
        train_limit=15000,
        balance=True,
        batch_size=1024,
        num_workers=2,
        # Model
        fp_dim=2048,
        c1=64,
        c2=128,
        embed_dim=256,
        proj_dim=120,
        use_projection=True,
        batchnorm_safe=True,
        # Loss / Optim
        temperature=0.1,
        lr=3e-4,
        weight_decay=1e-4,
        grad_clip=0.0,
        # Train loop
        epochs=10,
        out_dir="../models/supcon_cnn",
    )

    os.makedirs(args.out_dir, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    train(args)
