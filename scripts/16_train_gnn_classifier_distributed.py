"""Distributed supervised training of a graph attention network classifier on SMILES.

Loads training/validation parquet splits with `smiles` and `source` columns,
builds molecular graphs on the fly, trains a GAT-like encoder with a linear
classification head using BCEWithLogitsLoss (with class-weighting), and reports
metrics on the validation set each epoch. Designed to be launched with
`torchrun`.
"""

import os
import time
import contextlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdchem
from torch import distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

RDLogger.DisableLog("rdApp.*")


# ---- Configuration ----
TRAIN_PARQUET = "../data/train/baseline_train.parquet"
VAL_PARQUET = "../data/val/baseline_val.parquet"

EPOCHS = 100
BATCH_SIZE = 128
GRAD_ACCUM_STEPS = 1
LR = 2e-4
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 256
HEADS = (4, 4, 4)
DROPOUT = 0.1
MESSAGE_PASSES = 3
NUM_WORKERS = 4
SEED = 42


# ---- Distributed helpers ----

def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)


def setup_device() -> torch.device:
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


def cleanup_dist():
    if is_dist():
        dist.destroy_process_group()


# ---- Graph featurisation ----

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


def _one_hot(value, mapping: Dict) -> np.ndarray:
    size = len(mapping) + 1
    vec = np.zeros(size, dtype=np.float32)
    vec[mapping.get(value, len(mapping))] = 1.0
    return vec


def atom_to_feature(atom: rdchem.Atom) -> np.ndarray:
    feats = [
        _one_hot(atom.GetAtomicNum(), ATOM_MAP),
        _one_hot(atom.GetTotalDegree(), DEGREE_MAP),
        _one_hot(atom.GetFormalCharge(), CHARGE_MAP),
        _one_hot(atom.GetTotalNumHs(includeNeighbors=True), NUM_H_MAP),
        _one_hot(atom.GetHybridization(), HYB_MAP),
        np.array([atom.GetIsAromatic()], dtype=np.float32),
        np.array([atom.IsInRing()], dtype=np.float32),
    ]
    return np.concatenate(feats, axis=0)


EDGE_FEAT_DIM = len(BOND_TYPES) + 1


def bond_to_feature(bond: rdchem.Bond | None) -> np.ndarray:
    vec = np.zeros(EDGE_FEAT_DIM, dtype=np.float32)
    if bond is None:
        vec[-1] = 1.0
    else:
        vec[BOND_MAP.get(bond.GetBondType(), EDGE_FEAT_DIM - 1)] = 1.0
    return vec


def smiles_to_graph(smiles: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    n = mol.GetNumAtoms()
    if n == 0:
        raise ValueError(f"SMILES with no atoms: {smiles}")

    node_feat = np.vstack([atom_to_feature(atom) for atom in mol.GetAtoms()]).astype(np.float32)
    edges: List[Tuple[int, int]] = []
    edge_feat: List[np.ndarray] = []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_to_feature(bond)
        edges.append((u, v))
        edges.append((v, u))
        edge_feat.append(feat)
        edge_feat.append(feat)
    loop = bond_to_feature(None)
    for i in range(n):
        edges.append((i, i))
        edge_feat.append(loop)
    edge_index = np.array(edges, dtype=np.int64).T
    edge_attr = np.vstack(edge_feat).astype(np.float32)
    return node_feat, edge_index, edge_attr


@dataclass
class GraphSample:
    node_feat: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    label: int


class MolecularGraphDataset(Dataset):
    def __init__(self, parquet_path: str):
        df = pd.read_parquet(parquet_path)
        self.smiles = df["smiles"].astype(str).tolist()
        self.labels = (df["source"].astype(str) == "PKS").astype(np.int64).to_numpy()
        for smi in self.smiles:
            try:
                sample = smiles_to_graph(smi)
            except ValueError:
                continue
            self.node_feat_dim = sample[0].shape[1]
            self.edge_feat_dim = sample[2].shape[1]
            break
        else:
            raise RuntimeError("No valid molecules found")

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> GraphSample:
        nf, ei, ea = smiles_to_graph(self.smiles[idx])
        return GraphSample(nf, ei, ea, int(self.labels[idx]))


def collate_graphs(batch: List[GraphSample]) -> Dict[str, torch.Tensor]:
    node_feats = []
    edge_indices = []
    edge_attrs = []
    batch_index = []
    labels = []
    offset = 0
    for graph in batch:
        x = torch.from_numpy(graph.node_feat)
        ei = torch.from_numpy(graph.edge_index) + offset
        ea = torch.from_numpy(graph.edge_attr)
        n = x.size(0)
        node_feats.append(x)
        edge_indices.append(ei)
        edge_attrs.append(ea)
        batch_index.append(torch.full((n,), len(labels), dtype=torch.long))
        labels.append(graph.label)
        offset += n
    return {
        "node_feat": torch.cat(node_feats, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_attr": torch.cat(edge_attrs, dim=0),
        "batch": torch.cat(batch_index, dim=0),
        "labels": torch.tensor(labels, dtype=torch.float32).view(-1, 1),
    }


# ---- Model ----

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int, edge_dim: int, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.lin = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(heads, out_dim))
        self.att_dst = nn.Parameter(torch.Tensor(heads, out_dim))
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        self.edge_proj = nn.Linear(edge_dim, heads, bias=False)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)
        nn.init.xavier_uniform_(self.edge_proj.weight)

    def forward(self, x, edge_index, edge_attr):
        h = self.lin(x)
        num_nodes = h.size(0)
        h = h.view(num_nodes, self.heads, self.out_dim)
        att_src = (h * self.att_src).sum(dim=-1)
        att_dst = (h * self.att_dst).sum(dim=-1)
        src, dst = edge_index
        edge_logits = att_src[src] + att_dst[dst] + self.edge_proj(edge_attr)
        edge_logits = F.leaky_relu(edge_logits, 0.2)
        edge_alpha = edge_softmax(dst, edge_logits, num_nodes)
        edge_alpha = F.dropout(edge_alpha, p=self.dropout, training=self.training)
        out = h[src] * edge_alpha.unsqueeze(-1)
        agg = torch.zeros(num_nodes, self.heads, self.out_dim, device=x.device)
        agg.index_add_(0, dst, out)
        return agg.mean(dim=1) + self.bias


def edge_softmax(dst, scores, num_nodes):
    heads = scores.size(1)
    out = []
    for h in range(heads):
        s = scores[:, h]
        max_vals = torch.full((num_nodes,), -float("inf"), device=s.device)
        max_vals.scatter_reduce_(0, dst, s, reduce="amax")
        s = s - max_vals[dst]
        exp_s = torch.exp(s)
        denom = torch.zeros(num_nodes, device=s.device).scatter_add_(0, dst, exp_s)
        out.append(exp_s / (denom[dst] + 1e-16))
    return torch.stack(out, dim=1)


def scatter_mean(src, index, dim_size):
    out = torch.zeros(dim_size, src.size(-1), device=src.device)
    out.scatter_add_(0, index.unsqueeze(-1).expand(-1, src.size(-1)), src)
    counts = torch.zeros(dim_size, device=src.device)
    counts.scatter_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    counts = counts.clamp_min(1.0)
    return out / counts.unsqueeze(-1)


class MessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = DROPOUT):
        super().__init__()
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        deg = torch.zeros(x.size(0), device=x.device)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=deg.dtype))
        agg = agg / deg.clamp_min(1.0).unsqueeze(-1)
        out = self.lin(agg)
        out = self.dropout(F.relu(out))
        return self.norm(out + x)


class GraphAttentionClassifier(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=HIDDEN_DIM, heads=HEADS, dropout=DROPOUT, msg_passes=MESSAGE_PASSES):
        super().__init__()
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for h in heads:
            self.layers.append(GraphAttentionLayer(hidden_dim, hidden_dim, h, edge_dim, dropout))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.message_layers = nn.ModuleList([MessagePassingLayer(hidden_dim, dropout) for _ in range(msg_passes)])
        jk_dim = hidden_dim * (len(heads) + msg_passes)
        self.graph_head = nn.Sequential(
            nn.Linear(jk_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, node_feat, edge_index, batch_index, edge_attr):
        x = self.input_proj(node_feat)
        jk_feats = []
        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = layer(x, edge_index, edge_attr)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=DROPOUT, training=self.training)
            x = x + residual
            jk_feats.append(x)
        for mp_layer in self.message_layers:
            x = mp_layer(x, edge_index)
            jk_feats.append(x)
        concat_feats = torch.cat(jk_feats, dim=-1) if len(jk_feats) > 1 else jk_feats[0]
        num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 0
        graph_embed = scatter_mean(concat_feats, batch_index, num_graphs)
        graph_embed = self.graph_head(graph_embed)
        logits = self.classifier(graph_embed)
        return logits


# ---- Metrics ----

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    logits_list = []
    labels_list = []
    for batch in loader:
        node_feat = batch["node_feat"].to(device, non_blocking=True)
        edge_index = batch["edge_index"].to(device, non_blocking=True)
        edge_attr = batch["edge_attr"].to(device, non_blocking=True)
        batch_index = batch["batch"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        logits = model(node_feat, edge_index, batch_index, edge_attr)
        logits_list.append(logits.detach().cpu())
        labels_list.append(labels.detach().cpu())
    logits = torch.cat(logits_list, dim=0).numpy().ravel()
    logits = np.clip(logits, -50.0, 50.0)
    labels = torch.cat(labels_list, dim=0).numpy().ravel()
    probs = 1.0 / (1.0 + np.exp(-logits))
    from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score

    auprc = average_precision_score(labels, probs)
    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError:
        auroc = float("nan")
    preds = (probs >= 0.5).astype(np.float32)
    acc = accuracy_score(labels, preds)
    return auprc, auroc, acc


# ---- Training loop ----

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = setup_device()
    rank = get_rank()
    world = get_world_size()
    print0(f"Using device {device} | rank={rank} | world={world}")

    train_ds = MolecularGraphDataset(TRAIN_PARQUET)
    val_ds = MolecularGraphDataset(VAL_PARQUET)

    train_labels_np = np.array(train_ds.labels, dtype=np.int64)
    pos = max(int(train_labels_np.sum()), 1)
    neg = max(int(len(train_labels_np) - pos), 1)
    pos_weight_value = neg / pos
    if get_rank() == 0:
        print0(f"Class counts (train): pos={pos}, neg={neg}, pos_weight={pos_weight_value:.3f}")

    train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_graphs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_graphs,
    )

    model = GraphAttentionClassifier(
        node_dim=train_ds.node_feat_dim,
        edge_dim=train_ds.edge_feat_dim,
        hidden_dim=HIDDEN_DIM,
        heads=HEADS,
        dropout=DROPOUT,
    ).to(device)
    if world > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    pos_weight = torch.tensor([pos_weight_value], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        step_count = 0
        accum_steps = 0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        for batch in train_loader:
            node_feat = batch["node_feat"].to(device, non_blocking=True)
            edge_index = batch["edge_index"].to(device, non_blocking=True)
            edge_attr = batch["edge_attr"].to(device, non_blocking=True)
            batch_index = batch["batch"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            is_sync = ((accum_steps + 1) % GRAD_ACCUM_STEPS == 0) or (world == 1)
            ctx = model.no_sync() if world > 1 and not is_sync else contextlib.nullcontext()
            with ctx:
                with torch.amp.autocast(device_type=device.type if device.type in {"cuda", "cpu"} else "cuda", enabled=(device.type == "cuda")):
                    logits = model(node_feat, edge_index, batch_index, edge_attr)
                    loss_full = criterion(logits, labels)
                    loss = loss_full / GRAD_ACCUM_STEPS
                scaler.scale(loss).backward()

            accum_steps += 1
            epoch_loss += loss_full.item()
            step_count += 1

            if is_sync:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        if accum_steps % GRAD_ACCUM_STEPS != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        loss_tensor = torch.tensor([epoch_loss, step_count], device=device, dtype=torch.float32)
        if is_dist():
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        steps = int(loss_tensor[1].item())
        mean_loss = loss_tensor[0].item() / max(steps, 1)
        if get_rank() == 0:
            print0(f"Epoch {epoch:02d} | train_loss={mean_loss:.4f} | time={time.time()-t0:.1f}s")

        # Validation on rank 0 only
        if get_rank() == 0:
            val_auprc, val_auroc, val_acc = evaluate(model if world == 1 else model.module, val_loader, device)
            print0(
                f"Epoch {epoch:02d} | val_auprc={val_auprc:.4f}"
            )
            if val_auprc > best_val:
                best_val = val_auprc
                best_state = {
                    "state_dict": (model if world == 1 else model.module).state_dict(),
                    "val_auprc": val_auprc,
                    "val_auroc": val_auroc,
                    "val_acc": val_acc,
                    "epoch": epoch,
                }

        if is_dist():
            dist.barrier()

    if get_rank() == 0 and best_state is not None:
        out_dir = Path("../models")
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / f"GNN_classifier_w_attn_and_MPNN_{EPOCHS}ep_JK.pt"
        torch.save(best_state, save_path)
        print0(f"Saved best model to {save_path}")

    cleanup_dist()


if __name__ == "__main__":
    main()
