#!/usr/bin/env python3
"""
Distributed SupCon GNN Pretraining Script

Trains a GNN encoder with Supervised Contrastive Loss on PKS vs non-PKS molecules.
Supports single-node multi-GPU (primary) and multi-node multi-GPU (via SLURM).

Usage:
    # Single-node multi-GPU (primary use case)
    torchrun --nproc_per_node=4 scripts/07_train_supcon_gnn_distributed.py

    # Single GPU (no distributed)
    python scripts/07_train_supcon_gnn_distributed.py

    # Multi-node multi-GPU (SLURM)
    srun torchrun --nnodes=$SLURM_NNODES --nproc-per-node=4 \
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        scripts/07_train_supcon_gnn_distributed.py
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdchem

RDLogger.DisableLog("rdApp.*")


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distributed SupCon GNN Pretraining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data paths
    parser.add_argument(
        "--data_dir", type=str, default="data/",
        help="Path to data directory containing train/val subdirs"
    )
    parser.add_argument(
        "--output_dir", type=str, default="models/supcon_gnn/",
        help="Path for saving checkpoints"
    )

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=64, help="Per-GPU batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--temperature", type=float, default=0.07, help="SupCon temperature")

    # Model architecture
    parser.add_argument("--hidden_dim", type=int, default=256, help="GAT hidden dimension")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--proj_dim", type=int, default=64, help="Projection dimension")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of GAT layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Checkpointing
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    return parser.parse_args()


# =============================================================================
# Distributed Setup
# =============================================================================

def setup_distributed() -> Tuple[int, int, bool]:
    """Initialize distributed training.

    Returns:
        local_rank: GPU index on this node
        world_size: Total number of processes
        is_distributed: Whether running in distributed mode
    """
    # Check if running with torchrun/distributed
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        return local_rank, world_size, True
    else:
        # Single GPU mode
        return 0, 1, False


def cleanup_distributed(is_distributed: bool):
    """Clean up distributed training."""
    if is_distributed:
        dist.destroy_process_group()


def get_rank(is_distributed: bool) -> int:
    """Get current process rank."""
    return dist.get_rank() if is_distributed else 0


def log_rank0(msg: str, is_distributed: bool):
    """Print message only on rank 0."""
    if get_rank(is_distributed) == 0:
        print(msg, flush=True)


# =============================================================================
# Graph Featurization
# =============================================================================

# Atom feature vocabularies
ATOM_TYPES = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]  # H, B, C, N, O, F, Si, P, S, Cl, Br, I
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


def _build_mapping(values: Iterable) -> Dict:
    return {value: idx for idx, value in enumerate(values)}


ATOM_MAP = _build_mapping(ATOM_TYPES)
DEGREE_MAP = _build_mapping(DEGREES)
CHARGE_MAP = _build_mapping(FORMAL_CHARGES)
NUM_H_MAP = _build_mapping(NUM_HS)
HYB_MAP = {hyb: idx for idx, hyb in enumerate(HYBRIDIZATIONS)}
BOND_MAP = {bond: idx for idx, bond in enumerate(BOND_TYPES)}

EDGE_FEAT_DIM = len(BOND_TYPES) + 1  # +1 for self-loop


def _one_hot(value, mapping: Dict) -> np.ndarray:
    """Create one-hot vector with unknown bucket."""
    size = len(mapping) + 1
    vec = np.zeros(size, dtype=np.float32)
    vec[mapping.get(value, len(mapping))] = 1.0
    return vec


def atom_to_feature(atom: rdchem.Atom) -> np.ndarray:
    """Convert RDKit atom to feature vector (40 dim)."""
    feats = [
        _one_hot(atom.GetAtomicNum(), ATOM_MAP),        # 13 dim
        _one_hot(atom.GetTotalDegree(), DEGREE_MAP),    # 7 dim
        _one_hot(atom.GetFormalCharge(), CHARGE_MAP),   # 6 dim
        _one_hot(atom.GetTotalNumHs(includeNeighbors=True), NUM_H_MAP),  # 6 dim
        _one_hot(atom.GetHybridization(), HYB_MAP),     # 6 dim
        np.array([atom.GetIsAromatic()], dtype=np.float32),  # 1 dim
        np.array([atom.IsInRing()], dtype=np.float32),       # 1 dim
    ]
    return np.concatenate(feats, axis=0)


def bond_to_feature(bond) -> np.ndarray:
    """Convert RDKit bond to feature vector (5 dim)."""
    vec = np.zeros(EDGE_FEAT_DIM, dtype=np.float32)
    if bond is None:
        vec[-1] = 1.0  # self-loop marker
    else:
        vec[BOND_MAP.get(bond.GetBondType(), EDGE_FEAT_DIM - 1)] = 1.0
    return vec


def smiles_to_graph(smiles: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert SMILES to graph tensors.

    Returns:
        node_feat: [num_atoms, node_feat_dim]
        edge_index: [2, num_edges] (includes self-loops)
        edge_attr: [num_edges, edge_feat_dim]
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    n = mol.GetNumAtoms()
    if n == 0:
        raise ValueError(f"SMILES with no atoms: {smiles}")

    # Node features
    node_feat = np.vstack([atom_to_feature(atom) for atom in mol.GetAtoms()]).astype(np.float32)

    # Edge features (bidirectional + self-loops)
    edges: List[Tuple[int, int]] = []
    edge_feat: List[np.ndarray] = []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_to_feature(bond)
        edges.append((u, v))
        edges.append((v, u))
        edge_feat.append(feat)
        edge_feat.append(feat)

    # Self-loops
    loop = bond_to_feature(None)
    for i in range(n):
        edges.append((i, i))
        edge_feat.append(loop)

    edge_index = np.array(edges, dtype=np.int64).T
    edge_attr = np.vstack(edge_feat).astype(np.float32)
    return node_feat, edge_index, edge_attr


# Compute node feature dimension from test molecule
_test_nf, _, _ = smiles_to_graph("C")
NODE_FEAT_DIM = _test_nf.shape[1]


# =============================================================================
# Dataset and DataLoader
# =============================================================================

@dataclass
class GraphSample:
    """Container for a single molecular graph."""
    node_feat: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    label: int


class MolecularGraphDataset(Dataset):
    """Dataset that loads molecules from parquet and converts to graphs on-the-fly."""

    def __init__(self, parquet_path: str):
        df = pd.read_parquet(parquet_path)
        self.smiles = df["smiles"].astype(str).tolist()
        self.labels = df["label"].to_numpy().astype(np.int64)

        # Get feature dimensions from first valid molecule
        for smi in self.smiles:
            try:
                nf, _, ea = smiles_to_graph(smi)
                self.node_feat_dim = nf.shape[1]
                self.edge_feat_dim = ea.shape[1]
                break
            except ValueError:
                continue
        else:
            raise RuntimeError("No valid molecules found")

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> GraphSample:
        nf, ei, ea = smiles_to_graph(self.smiles[idx])
        return GraphSample(nf, ei, ea, int(self.labels[idx]))


def collate_graphs(batch: List[GraphSample]) -> Dict[str, torch.Tensor]:
    """Collate graphs into a batched format."""
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
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# =============================================================================
# Model Architecture
# =============================================================================

def edge_softmax(dst: torch.Tensor, scores: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Compute softmax over edges grouped by destination node.

    Args:
        dst: [num_edges] destination node indices
        scores: [num_edges, heads] attention scores
        num_nodes: total number of nodes

    Returns:
        [num_edges, heads] attention weights (softmaxed per dst node)
    """
    heads = scores.size(1)
    out = []
    for h in range(heads):
        s = scores[:, h]
        # Compute max per destination for numerical stability
        max_vals = torch.full((num_nodes,), -float("inf"), device=s.device)
        max_vals.scatter_reduce_(0, dst, s, reduce="amax")
        s = s - max_vals[dst]
        exp_s = torch.exp(s)
        # Sum per destination
        denom = torch.zeros(num_nodes, device=s.device).scatter_add_(0, dst, exp_s)
        out.append(exp_s / (denom[dst] + 1e-16))
    return torch.stack(out, dim=1)


class GraphAttentionLayer(nn.Module):
    """Single GAT layer with edge features."""

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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [num_nodes, in_dim] node features
            edge_index: [2, num_edges] edge indices
            edge_attr: [num_edges, edge_dim] edge features

        Returns:
            [num_nodes, out_dim] updated node features
        """
        h = self.lin(x)  # [N, heads * out_dim]
        num_nodes = h.size(0)
        h = h.view(num_nodes, self.heads, self.out_dim)  # [N, heads, out_dim]

        # Attention scores
        att_src = (h * self.att_src).sum(dim=-1)  # [N, heads]
        att_dst = (h * self.att_dst).sum(dim=-1)  # [N, heads]

        src, dst = edge_index
        edge_logits = att_src[src] + att_dst[dst] + self.edge_proj(edge_attr)  # [E, heads]
        edge_logits = F.leaky_relu(edge_logits, 0.2)

        # Softmax attention
        edge_alpha = edge_softmax(dst, edge_logits, num_nodes)  # [E, heads]
        edge_alpha = F.dropout(edge_alpha, p=self.dropout, training=self.training)

        # Aggregate
        out = h[src] * edge_alpha.unsqueeze(-1)  # [E, heads, out_dim]
        agg = torch.zeros(num_nodes, self.heads, self.out_dim, device=x.device)
        agg.index_add_(0, dst, out)

        return agg.mean(dim=1) + self.bias  # [N, out_dim]


class SupConGNNEncoder(nn.Module):
    """GNN encoder for Supervised Contrastive Learning.

    Architecture:
    - Input projection: node_dim -> hidden_dim
    - GAT layers with residual + LayerNorm
    - Mean pooling over nodes
    - Embedding head: hidden_dim -> embed_dim (produces g)
    - Projection head: embed_dim -> proj_dim (produces z)
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 256,
        heads: int = 4,
        num_layers: int = 3,
        embed_dim: int = 128,
        proj_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(node_dim, hidden_dim)

        # GAT layers with LayerNorm
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(
                GraphAttentionLayer(hidden_dim, hidden_dim, heads, edge_dim, dropout)
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = dropout

        # Embedding head (produces g)
        self.embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Projection head (produces z, discarded after pretraining)
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            node_feat: [total_nodes, node_dim] batched node features
            edge_index: [2, total_edges] batched edge indices
            batch: [total_nodes] node-to-graph assignment
            edge_attr: [total_edges, edge_dim] batched edge features

        Returns:
            g: [batch_size, embed_dim] graph embeddings (for downstream tasks)
            z: [batch_size, proj_dim] projections (for contrastive loss)
        """
        # Input projection
        x = self.input_proj(node_feat)

        # GAT layers with residual + LayerNorm
        for gat, norm in zip(self.gat_layers, self.layer_norms):
            residual = x
            x = gat(x, edge_index, edge_attr)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual  # Residual connection

        # Mean pooling over nodes
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        graph_embed = torch.zeros(num_graphs, self.hidden_dim, device=x.device)
        counts = torch.zeros(num_graphs, device=x.device)
        graph_embed.index_add_(0, batch, x)
        counts.index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype))
        graph_embed = graph_embed / counts.clamp_min(1.0).unsqueeze(-1)

        # Embedding and projection heads
        g = self.embed_head(graph_embed)  # [B, embed_dim]
        g = F.normalize(g, dim=-1, eps=1e-8)  # L2 normalize embeddings

        z = self.proj_head(g)  # [B, proj_dim]
        z = F.normalize(z, dim=-1, eps=1e-8)  # L2 normalize projections

        return g, z


# =============================================================================
# SupCon Loss with Distributed All-Gather
# =============================================================================

class DistributedSupConLoss(nn.Module):
    """Supervised Contrastive Loss with distributed all-gather support.

    For each anchor, pull together samples with the same label
    and push apart samples with different labels.

    In distributed mode, gathers projections from all GPUs before computing
    loss to maintain large effective batch size for contrastive learning.
    """

    def __init__(self, temperature: float = 0.07, eps: float = 1e-8):
        super().__init__()
        self.tau = temperature
        self.eps = eps

    def forward(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        is_distributed: bool = False,
    ) -> torch.Tensor:
        """Compute SupCon loss.

        Args:
            z: [B, proj_dim] L2-normalized projections
            labels: [B] class labels
            is_distributed: Whether to all-gather across GPUs

        Returns:
            Scalar loss
        """
        # All-gather projections and labels in distributed mode
        if is_distributed and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()

            # Gather projections
            z_gathered = [torch.zeros_like(z) for _ in range(world_size)]
            dist.all_gather(z_gathered, z.contiguous())
            # Replace local rank's copy with original grad-attached tensor
            z_gathered[rank] = z
            z = torch.cat(z_gathered, dim=0)

            # Gather labels
            labels_gathered = [torch.zeros_like(labels) for _ in range(world_size)]
            dist.all_gather(labels_gathered, labels.contiguous())
            labels = torch.cat(labels_gathered, dim=0)

        B = z.size(0)
        z = F.normalize(z.float(), dim=-1, eps=self.eps)  # Ensure normalized

        # Similarity matrix scaled by temperature
        sim = (z @ z.t()) / self.tau  # [B, B]

        # Masks
        eye = torch.eye(B, dtype=torch.bool, device=z.device)
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.t()) & (~eye)  # Same class, not self

        # Skip anchors with no positives
        valid_mask = pos_mask.sum(1) > 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        # Restrict to valid anchors
        sim = sim[valid_mask]
        pos_mask = pos_mask[valid_mask]
        eye_valid = eye[valid_mask]

        # Numerical stability: subtract max
        sim = sim - sim.max(dim=1, keepdim=True).values

        # Mask out self-similarities for denominator
        sim_masked = sim.masked_fill(eye_valid, float("-inf"))

        # Log-softmax denominator (excluding self)
        denom = torch.logsumexp(sim_masked, dim=1, keepdim=True)
        log_prob = sim - denom

        # Average log-prob over positives
        pos_counts = pos_mask.sum(1).clamp_min(1)
        pos_log_prob = (pos_mask * log_prob).sum(1) / pos_counts

        loss = -pos_log_prob.mean()

        if not torch.isfinite(loss):
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        return loss


# =============================================================================
# Training Utilities
# =============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: DistributedSupConLoss,
    device: torch.device,
    is_distributed: bool,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        node_feat = batch["node_feat"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_attr = batch["edge_attr"].to(device)
        batch_idx = batch["batch"].to(device)
        labels = batch["labels"].to(device)

        # Skip batches with only one class (no positives for the other)
        if labels.unique().numel() < 2:
            continue

        optimizer.zero_grad()

        _, z = model(node_feat, edge_index, batch_idx, edge_attr)
        loss = criterion(z, labels, is_distributed=is_distributed)

        if not torch.isfinite(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: DistributedSupConLoss,
    device: torch.device,
) -> float:
    """Evaluate loss on a dataset (single GPU, no all-gather)."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        node_feat = batch["node_feat"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_attr = batch["edge_attr"].to(device)
        batch_idx = batch["batch"].to(device)
        labels = batch["labels"].to(device)

        if labels.unique().numel() < 2:
            continue

        _, z = model(node_feat, edge_index, batch_idx, edge_attr)
        # No all-gather during validation (evaluate on local batch)
        loss = criterion(z, labels, is_distributed=False)

        if torch.isfinite(loss):
            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    path: Path,
    is_distributed: bool,
):
    """Save training checkpoint (rank 0 only)."""
    if get_rank(is_distributed) != 0:
        return

    # Get model state (unwrap DDP if needed)
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, float]:
    """Load training checkpoint.

    Returns:
        start_epoch: Epoch to resume from
        best_val_loss: Best validation loss so far
    """
    checkpoint = torch.load(path, map_location=device)

    # Load model state (handle DDP wrapper)
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["epoch"], checkpoint.get("best_val_loss", float("inf"))


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # Setup distributed
    local_rank, world_size, is_distributed = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Set seeds
    torch.manual_seed(args.seed + get_rank(is_distributed))
    np.random.seed(args.seed + get_rank(is_distributed))

    log_rank0(f"Starting SupCon GNN training", is_distributed)
    log_rank0(f"  Device: {device}", is_distributed)
    log_rank0(f"  World size: {world_size}", is_distributed)
    log_rank0(f"  Distributed: {is_distributed}", is_distributed)

    # Create output directory
    output_dir = Path(args.output_dir)
    if get_rank(is_distributed) == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    data_dir = Path(args.data_dir)
    train_path = data_dir / "train" / "supcon_train.parquet"
    val_path = data_dir / "val" / "supcon_val.parquet"

    log_rank0(f"Loading training data from {train_path}...", is_distributed)
    train_ds = MolecularGraphDataset(str(train_path))
    log_rank0(f"Loading validation data from {val_path}...", is_distributed)
    val_ds = MolecularGraphDataset(str(val_path))

    log_rank0(f"  Train samples: {len(train_ds)}", is_distributed)
    log_rank0(f"  Val samples: {len(val_ds)}", is_distributed)

    # Create samplers and loaders
    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_graphs,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_graphs,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Initialize model
    model = SupConGNNEncoder(
        node_dim=train_ds.node_feat_dim,
        edge_dim=train_ds.edge_feat_dim,
        hidden_dim=args.hidden_dim,
        heads=args.num_heads,
        num_layers=args.num_layers,
        embed_dim=args.embed_dim,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log_rank0(f"\nModel parameters: {total_params:,}", is_distributed)
    log_rank0(f"  Hidden dim: {args.hidden_dim}", is_distributed)
    log_rank0(f"  Embed dim: {args.embed_dim}", is_distributed)
    log_rank0(f"  Proj dim: {args.proj_dim}", is_distributed)
    log_rank0(f"  GAT layers: {args.num_layers}", is_distributed)
    log_rank0(f"  GAT heads: {args.num_heads}", is_distributed)

    # Wrap with DDP
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Loss and optimizer
    criterion = DistributedSupConLoss(temperature=args.temperature)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Resume from checkpoint if provided
    start_epoch = 1
    best_val_loss = float("inf")
    if args.checkpoint:
        log_rank0(f"Resuming from checkpoint: {args.checkpoint}", is_distributed)
        start_epoch, best_val_loss = load_checkpoint(
            Path(args.checkpoint), model, optimizer, device
        )
        start_epoch += 1  # Start from next epoch
        log_rank0(f"  Resuming from epoch {start_epoch}", is_distributed)
        log_rank0(f"  Best val loss so far: {best_val_loss:.4f}", is_distributed)

    # Training loop
    log_rank0(f"\nStarting training...", is_distributed)
    log_rank0(f"  Batch size per GPU: {args.batch_size}", is_distributed)
    log_rank0(f"  Effective batch size: {args.batch_size * world_size}", is_distributed)
    log_rank0(f"  Learning rate: {args.lr}", is_distributed)
    log_rank0(f"  Temperature: {args.temperature}", is_distributed)
    log_rank0(f"  Epochs: {args.epochs}", is_distributed)
    log_rank0("", is_distributed)
    log_rank0(f"{'Epoch':>5} {'Train Loss':>12} {'Val Loss':>12} {'Best':>6}", is_distributed)
    log_rank0("-" * 38, is_distributed)

    for epoch in range(start_epoch, args.epochs + 1):
        # Set epoch for DistributedSampler (ensures different shuffling each epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, is_distributed
        )

        # Validate (on rank 0 only to avoid duplicate computation)
        if get_rank(is_distributed) == 0:
            val_loss = eval_epoch(model, val_loader, criterion, device)
        else:
            val_loss = 0.0

        # Broadcast val_loss to all ranks (for consistent logging/checkpointing)
        if is_distributed:
            val_loss_tensor = torch.tensor(val_loss, device=device)
            dist.broadcast(val_loss_tensor, src=0)
            val_loss = val_loss_tensor.item()

        # Track best
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        log_rank0(
            f"{epoch:>5} {train_loss:>12.4f} {val_loss:>12.4f} {'*' if is_best else '':>6}",
            is_distributed,
        )

        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss, best_val_loss,
                ckpt_path, is_distributed,
            )
            log_rank0(f"  Saved checkpoint: {ckpt_path}", is_distributed)

        # Save best model
        if is_best:
            best_path = output_dir / "best_model.pt"
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss, best_val_loss,
                best_path, is_distributed,
            )

    log_rank0("\nTraining complete!", is_distributed)
    log_rank0(f"Best validation loss: {best_val_loss:.4f}", is_distributed)
    log_rank0(f"Checkpoints saved to: {output_dir}", is_distributed)

    # Cleanup
    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
