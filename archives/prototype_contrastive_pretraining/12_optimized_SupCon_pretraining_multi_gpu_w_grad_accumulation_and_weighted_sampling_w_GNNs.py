import math
import os
import datetime
import contextlib
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdchem
from torch import amp
from torch import distributed as dist
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

# Disable noisy RDKit logs.
RDLogger.DisableLog('rdApp.*')


# ---- Hyperparameters ----

TRAIN_PARQUET = '../data/train/baseline_train.parquet'
VAL_PARQUET = '../data/val/baseline_val.parquet'

BATCH_SIZE = 128
ACCUMULATION_STEPS = 16
EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.07
SEED = 42
SUBSET_SIZE = 2_000_000
WARMUP_EPOCHS = 3
FP_CACHE_SIZE = 25_000

GNN_HIDDEN_DIM = 256
GNN_HEADS = (4, 4, 4)
GNN_DROPOUT = 0.1
EMBED_DIM = 512
PROJ_DIM = 256

NUM_WORKERS = 4


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
        # Running in single-process mode (e.g., notebook or plain python)
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not dist.is_initialized():
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend, init_method='env://')

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device('cuda', local_rank)
    return torch.device('cpu')


def barrier():
    if is_dist():
        dist.barrier()


torch.manual_seed(SEED)
np.random.seed(SEED)


# ---- Graph feature helpers ----

ATOM_TYPES = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
DEGREES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
NUM_HS = [0, 1, 2, 3, 4]
HYBRIDIZATIONS = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]

BOND_TYPES = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]


def _build_mapping(values: Iterable[int]) -> Dict[int, int]:
    return {value: idx for idx, value in enumerate(values)}


ATOM_MAP = _build_mapping(ATOM_TYPES)
DEGREE_MAP = _build_mapping(DEGREES)
CHARGE_MAP = _build_mapping(FORMAL_CHARGES)
NUM_H_MAP = _build_mapping(NUM_HS)
HYB_MAP = {hyb: idx for idx, hyb in enumerate(HYBRIDIZATIONS)}
BOND_MAP = {bond: idx for idx, bond in enumerate(BOND_TYPES)}


def _one_hot_with_unknown(value, mapping: Dict) -> np.ndarray:
    size = len(mapping) + 1
    vec = np.zeros(size, dtype=np.float32)
    idx = mapping.get(value, len(mapping))
    vec[idx] = 1.0
    return vec


def atom_to_feature(atom: rdchem.Atom) -> np.ndarray:
    features = [
        _one_hot_with_unknown(atom.GetAtomicNum(), ATOM_MAP),
        _one_hot_with_unknown(atom.GetTotalDegree(), DEGREE_MAP),
        _one_hot_with_unknown(atom.GetFormalCharge(), CHARGE_MAP),
        _one_hot_with_unknown(atom.GetTotalNumHs(includeNeighbors=True), NUM_H_MAP),
        _one_hot_with_unknown(atom.GetHybridization(), HYB_MAP),
        np.array([atom.GetIsAromatic()], dtype=np.float32),
        np.array([atom.IsInRing()], dtype=np.float32),
    ]
    return np.concatenate(features, axis=0)


EDGE_FEATURE_DIM = len(BOND_TYPES) + 1  # last index reserved for "other/self"


def bond_to_feature(bond: rdchem.Bond | None) -> np.ndarray:
    vec = np.zeros(EDGE_FEATURE_DIM, dtype=np.float32)
    if bond is None:
        vec[-1] = 1.0
        return vec
    bond_type = bond.GetBondType()
    idx = BOND_MAP.get(bond_type, EDGE_FEATURE_DIM - 1)
    vec[idx] = 1.0
    return vec


def smiles_to_graph(smiles: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        raise ValueError(f"SMILES with no atoms encountered: {smiles}")

    node_feats = np.vstack([atom_to_feature(atom) for atom in mol.GetAtoms()]).astype(np.float32)

    edges: List[Tuple[int, int]] = []
    edge_attrs: List[np.ndarray] = []

    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        feat = bond_to_feature(bond)
        edges.append((start, end))
        edges.append((end, start))
        edge_attrs.append(feat)
        edge_attrs.append(feat)

    # Self-loops ensure every node has at least one incoming edge.
    loop_feat = bond_to_feature(None)
    for idx in range(num_atoms):
        edges.append((idx, idx))
        edge_attrs.append(loop_feat)

    edge_index = np.array(edges, dtype=np.int64).T  # shape [2, E]
    edge_attr = np.vstack(edge_attrs).astype(np.float32)
    return node_feats, edge_index, edge_attr


@dataclass
class GraphSample:
    node_feat: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray


class MolecularGraphDataset(Dataset):
    def __init__(self, parquet_path: str, cache_size: int = 0):
        super().__init__()
        df = pd.read_parquet(parquet_path)
        self.smiles: List[str] = df['smiles'].astype(str).tolist()
        self.labels = (df['source'].astype(str) == 'PKS').astype(np.int64).to_numpy()
        self.cache_size = cache_size
        self.cache: Dict[str, GraphSample] = {}
        self.cache_order: deque[str] = deque()

        # Inspect a single valid molecule to determine feature dimensions.
        for smi in self.smiles:
            try:
                sample = smiles_to_graph(smi)
            except ValueError:
                continue
            self.node_feat_dim = sample[0].shape[1]
            self.edge_feat_dim = sample[2].shape[1]
            break
        else:
            raise RuntimeError("No valid SMILES found in dataset.")

    def __len__(self) -> int:
        return len(self.smiles)

    def _maybe_cache(self, smiles: str, sample: GraphSample) -> GraphSample:
        if self.cache_size <= 0:
            return sample
        if smiles in self.cache:
            return self.cache[smiles]
        if len(self.cache_order) >= self.cache_size:
            evicted = self.cache_order.popleft()
            self.cache.pop(evicted, None)
        self.cache_order.append(smiles)
        self.cache[smiles] = sample
        return sample

    def __getitem__(self, idx: int) -> Tuple[GraphSample, int]:
        smiles = self.smiles[idx]
        if smiles in self.cache:
            sample = self.cache[smiles]
        else:
            node_feat, edge_index, edge_attr = smiles_to_graph(smiles)
            sample = GraphSample(node_feat=node_feat, edge_index=edge_index, edge_attr=edge_attr)
            sample = self._maybe_cache(smiles, sample)
        label = int(self.labels[idx])
        return sample, label


def collate_graph_batch(batch: List[Tuple[GraphSample, int]]) -> Dict[str, torch.Tensor]:
    node_feats: List[torch.Tensor] = []
    edge_indices: List[torch.Tensor] = []
    edge_attrs: List[torch.Tensor] = []
    batch_vec: List[torch.Tensor] = []
    labels: List[int] = []

    node_offset = 0
    for graph, label in batch:
        x = torch.from_numpy(graph.node_feat)
        edge_index = torch.from_numpy(graph.edge_index) + node_offset
        edge_attr = torch.from_numpy(graph.edge_attr)
        num_nodes = x.size(0)

        node_feats.append(x)
        edge_indices.append(edge_index)
        edge_attrs.append(edge_attr)
        batch_vec.append(torch.full((num_nodes,), len(labels), dtype=torch.long))
        labels.append(label)
        node_offset += num_nodes

    node_feat = torch.cat(node_feats, dim=0)
    edge_index = torch.cat(edge_indices, dim=1)
    edge_attr = torch.cat(edge_attrs, dim=0)
    batch_indices = torch.cat(batch_vec, dim=0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return {
        "node_feat": node_feat,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "batch": batch_indices,
        "labels": labels_tensor,
    }


# ---- GNN building blocks ----

class GraphAttentionLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        edge_feat_dim: int | None = None,
        concat: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope

        self.lin = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(heads, out_dim))
        self.att_dst = nn.Parameter(torch.Tensor(heads, out_dim))
        self.bias = nn.Parameter(torch.Tensor(out_dim * heads if concat else out_dim))

        self.edge_proj = nn.Linear(edge_feat_dim, heads, bias=False) if edge_feat_dim is not None else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)
        if self.edge_proj is not None:
            nn.init.xavier_uniform_(self.edge_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.lin(x)
        num_nodes = h.size(0)
        h = h.view(num_nodes, self.heads, self.out_dim)
        att_src = (h * self.att_src).sum(dim=-1)
        att_dst = (h * self.att_dst).sum(dim=-1)

        src, dst = edge_index
        alpha = att_src[src] + att_dst[dst]
        if edge_attr is not None and self.edge_proj is not None:
            alpha = alpha + self.edge_proj(edge_attr)

        alpha = F.leaky_relu(alpha, negative_slope=self.negative_slope)
        alpha = self._edge_softmax(dst, alpha, num_nodes)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = h[src] * alpha.unsqueeze(-1)
        agg = torch.zeros(num_nodes, self.heads, self.out_dim, device=x.device)
        agg.index_add_(0, dst, out)

        if self.concat:
            agg = agg.reshape(num_nodes, self.heads * self.out_dim)
        else:
            agg = agg.mean(dim=1)
        agg = agg + self.bias
        return agg

    def _edge_softmax(self, dst: torch.Tensor, scores: torch.Tensor, num_nodes: int) -> torch.Tensor:
        # scores shape: [E, heads]
        heads = scores.size(1)
        alphas = []
        for head in range(heads):
            s = scores[:, head]
            max_vals = torch.full((num_nodes,), -float('inf'), device=s.device)
            max_vals.scatter_reduce_(0, dst, s, reduce='amax')
            s = s - max_vals[dst]
            exp_s = torch.exp(s)
            denom = torch.zeros(num_nodes, device=s.device).scatter_add_(0, dst, exp_s)
            alphas.append(exp_s / (denom[dst] + 1e-16))
        return torch.stack(alphas, dim=1)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, src.size(-1), device=src.device)
    out.scatter_add_(0, index.unsqueeze(-1).expand(-1, src.size(-1)), src)
    counts = torch.zeros(dim_size, device=src.device)
    counts.scatter_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    counts = counts.clamp_min(1.0)
    out = out / counts.unsqueeze(-1)
    return out


class GraphAttentionEncoder(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int = GNN_HIDDEN_DIM,
        heads: Tuple[int, ...] = GNN_HEADS,
        dropout: float = GNN_DROPOUT,
        embed_dim: int = EMBED_DIM,
        proj_dim: int = PROJ_DIM,
        use_projection: bool = True,
    ):
        super().__init__()
        self.dropout = dropout
        self.use_projection = use_projection

        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        current_dim = hidden_dim
        for head_count in heads:
            layer = GraphAttentionLayer(
                in_dim=current_dim,
                out_dim=hidden_dim,
                heads=head_count,
                dropout=dropout,
                edge_feat_dim=edge_feat_dim,
                concat=False,
            )
            self.layers.append(layer)
            self.norms.append(nn.LayerNorm(hidden_dim))
            current_dim = hidden_dim

        self.graph_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

        if use_projection:
            self.proj_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(embed_dim, proj_dim),
            )
        else:
            self.proj_head = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        for block in self.graph_head:
            if isinstance(block, nn.Linear):
                nn.init.xavier_uniform_(block.weight)
                nn.init.zeros_(block.bias)
        if self.proj_head is not None:
            for block in self.proj_head:
                if isinstance(block, nn.Linear):
                    nn.init.xavier_uniform_(block.weight)
                    nn.init.zeros_(block.bias)

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        batch_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(node_feat)
        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = layer(x, edge_index, edge_attr)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual

        num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 0
        graph_embed = scatter_mean(x, batch_index, num_graphs)
        g = F.normalize(self.graph_head(graph_embed), dim=-1, eps=1e-8)

        if self.use_projection and self.proj_head is not None:
            z = F.normalize(self.proj_head(g), dim=-1, eps=1e-8)
            return g, z
        return g, g


# ---- Loss ----

class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.tau = temperature
        self.eps = eps

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if z.size(0) <= 1:
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        z = F.normalize(z.float(), dim=-1, eps=self.eps)
        sim = (z @ z.t()) / self.tau
        eye = torch.eye(z.size(0), dtype=torch.bool, device=z.device)

        labels = labels.view(-1).long()
        pos_mask = (labels.view(-1, 1) == labels.view(1, -1)) & (~eye)
        valid_mask = pos_mask.sum(1) > 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        sim = sim[valid_mask]
        pos_mask = pos_mask[valid_mask]
        labels = labels[valid_mask]

        sim = sim - sim.max(dim=1, keepdim=True).values
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

        pos_counts = pos_mask.sum(1).clamp_min(1)
        pos_log_prob = (pos_mask * log_prob).sum(1) / pos_counts

        # ✅ dynamic per-class weighting (inverse frequency)
        class_counts = torch.bincount(labels)
        weights = class_counts.max() / class_counts[labels]
        weights = weights / weights.mean()   # normalize so avg weight ~ 1
        weights = weights.to(z.device)

        loss = -(weights * pos_log_prob).mean()
        return loss



# ---- Training Loop ----

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


def select_balanced_indices(labels: np.ndarray, subset_size: int, rng: np.random.Generator) -> np.ndarray:
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    n_pos = min(subset_size // 2, len(pos_idx))
    n_neg = min(subset_size - n_pos, len(neg_idx))
    pos_sample = rng.choice(pos_idx, size=n_pos, replace=False)
    neg_sample = rng.choice(neg_idx, size=n_neg, replace=False)
    combined = np.concatenate([pos_sample, neg_sample]).astype(np.int64)
    rng.shuffle(combined)
    return combined


def main():
    device = setup_distributed()
    rank = get_rank()
    world = get_world_size()
    print0(f"DDP initialized: rank={rank}, world={world}, device={device}")
    print0(f"Using {world} GPU{'s' if world != 1 else ''} with grad accumulation steps = {ACCUMULATION_STEPS}")

    torch.set_num_threads(1)

    base_train = MolecularGraphDataset(TRAIN_PARQUET, cache_size=FP_CACHE_SIZE)
    labels_all = base_train.labels.copy()

    if rank == 0:
        rng = np.random.default_rng(SEED)
        selected_indices = select_balanced_indices(labels_all, SUBSET_SIZE, rng)
        lengths = np.array([len(selected_indices)], dtype=np.int64)
    else:
        selected_indices = None
        lengths = np.zeros((1,), dtype=np.int64)

    if is_dist():
        length_tensor = torch.from_numpy(lengths).to(device)
        dist.broadcast(length_tensor, src=0)
        lengths = length_tensor.cpu().numpy()

    subset_length = int(lengths[0])
    if subset_length == 0:
        raise RuntimeError("Subset selection failed to produce any samples.")

    if rank != 0:
        selected_indices = np.empty(subset_length, dtype=np.int64)

    if is_dist():
        index_tensor = torch.from_numpy(selected_indices).to(device)
        dist.broadcast(index_tensor, src=0)
        selected_indices = index_tensor.cpu().numpy()

    train_ds = Subset(base_train, selected_indices.tolist())

    train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_graph_batch,
        drop_last=False,
    )

    model = GraphAttentionEncoder(
        node_feat_dim=base_train.node_feat_dim,
        edge_feat_dim=base_train.edge_feat_dim,
        hidden_dim=GNN_HIDDEN_DIM,
        heads=GNN_HEADS,
        dropout=GNN_DROPOUT,
        embed_dim=EMBED_DIM,
        proj_dim=PROJ_DIM,
        use_projection=True,
    ).to(device)

    if world > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == 'cuda' else None,
            find_unused_parameters=False,
        )

    criterion = SupConLoss(temperature=TEMPERATURE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_warmup_lambda)
    scaler = amp.GradScaler(device.type if torch.cuda.is_available() else 'cpu')

    print0("Training samples:", len(train_ds))

    for epoch in range(1, EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        steps = 0
        accum_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for batch in train_loader:
            node_feat = batch['node_feat'].to(device, non_blocking=True)
            edge_index = batch['edge_index'].to(device, non_blocking=True)
            edge_attr = batch['edge_attr'].to(device, non_blocking=True)
            batch_index = batch['batch'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            is_sync_step = ((accum_steps + 1) % ACCUMULATION_STEPS == 0) or (world == 1)
            sync_ctx = contextlib.nullcontext() if is_sync_step else model.no_sync()
            with sync_ctx:
                with amp.autocast(device.type, dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16):
                    g, z = model(node_feat, edge_index, batch_index, edge_attr)
                    loss_full = criterion(z, labels)
                    loss = loss_full / ACCUMULATION_STEPS

                if not torch.isfinite(loss_full):
                    loss_full = z.sum() * 0.0
                    loss = loss_full / ACCUMULATION_STEPS

                scaler.scale(loss).backward()

            accum_steps += 1
            steps += 1
            epoch_loss += loss_full.item()

            if (accum_steps % ACCUMULATION_STEPS == 0) or (world == 1 and is_sync_step):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        if accum_steps % ACCUMULATION_STEPS != 0 and accum_steps > 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        loss_tensor = torch.tensor([epoch_loss, steps], dtype=torch.float32, device=device)
        if is_dist():
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

        total_steps = int(loss_tensor[1].item())
        if total_steps == 0:
            print0(f"[Epoch {epoch:03d}] No steps executed.")
        else:
            mean_loss = loss_tensor[0].item() / total_steps
            print0(f"[Epoch {epoch:03d}] train_supcon={mean_loss:.4f}, lr={current_lr:.3e}")

    if get_rank() == 0:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_to_save = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
        checkpoint = {
            "model_state_dict": model_to_save.state_dict(),
            "timestamp": timestamp,
            "hyperparameters": {
                "temperature": TEMPERATURE,
                "embed_dim": EMBED_DIM,
                "proj_dim": PROJ_DIM,
                "hidden_dim": GNN_HIDDEN_DIM,
                "heads": GNN_HEADS,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LR,
                "weight_decay": WEIGHT_DECAY,
                "subset_size": SUBSET_SIZE,
            },
            "feature_dims": {
                "node_feat_dim": base_train.node_feat_dim,
                "edge_feat_dim": base_train.edge_feat_dim,
            },
        }
        model_filename = f"supcon_gnn_{timestamp}.pt"
        model_path = f"../models/{model_filename}"
        torch.save(checkpoint, model_path)
        print0(f"[Checkpoint] Saved model to {model_path}")

    barrier()
    if is_dist():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
