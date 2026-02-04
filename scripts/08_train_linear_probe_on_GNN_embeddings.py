#!/usr/bin/env python3
"""
Linear Probe Training on GNN Embeddings

Loads a trained SupCon GNN encoder checkpoint, extracts embeddings for
train/val/test sets, trains a logistic regression linear probe, and
evaluates performance.

Usage:
    python scripts/08_train_linear_probe_on_GNN_embeddings.py \
        --checkpoint models/supcon_gnn/best_model.pt

    # With custom settings
    python scripts/08_train_linear_probe_on_GNN_embeddings.py \
        --checkpoint models/supcon_gnn/best_model.pt \
        --batch_size 512 \
        --num_workers 8 \
        --save_embeddings
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdchem

RDLogger.DisableLog("rdApp.*")


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train linear probe on GNN embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to GNN encoder checkpoint"
    )

    # Data paths
    parser.add_argument(
        "--data_dir", type=str, default="data/",
        help="Path to data directory containing train/val/test subdirs"
    )
    parser.add_argument(
        "--output_dir", type=str, default="models/supcon_gnn/",
        help="Path for saving linear probe and metrics"
    )

    # Embedding extraction
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="Batch size for embedding extraction"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8,
        help="Number of DataLoader workers"
    )

    # Linear probe
    parser.add_argument(
        "--max_iter", type=int, default=1000,
        help="LogisticRegression max iterations"
    )

    # Output options
    parser.add_argument(
        "--save_embeddings", action="store_true",
        help="Save extracted embeddings as .npz files"
    )

    # Data file suffix (for testing with subsets)
    parser.add_argument(
        "--data_suffix", type=str, default="",
        help="Suffix for data files (e.g., '_tiny' for supcon_train_tiny.parquet)"
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


# =============================================================================
# Graph Featurization (copied from 07_train_supcon_gnn_distributed.py)
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
# Dataset and DataLoader (copied from 07_train_supcon_gnn_distributed.py)
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
# Model Architecture (copied from 07_train_supcon_gnn_distributed.py)
# =============================================================================

def edge_softmax(dst: torch.Tensor, scores: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Compute softmax over edges grouped by destination node."""
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
        """Forward pass."""
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


class SupConGNNEncoder(nn.Module):
    """GNN encoder for Supervised Contrastive Learning."""

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

        self.input_proj = nn.Linear(node_dim, hidden_dim)

        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(
                GraphAttentionLayer(hidden_dim, hidden_dim, heads, edge_dim, dropout)
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = dropout

        self.embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )

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

        Returns:
            g: [batch_size, embed_dim] graph embeddings (for downstream tasks)
            z: [batch_size, proj_dim] projections (for contrastive loss)
        """
        x = self.input_proj(node_feat)

        for gat, norm in zip(self.gat_layers, self.layer_norms):
            residual = x
            x = gat(x, edge_index, edge_attr)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual

        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        graph_embed = torch.zeros(num_graphs, self.hidden_dim, device=x.device)
        counts = torch.zeros(num_graphs, device=x.device)
        graph_embed.index_add_(0, batch, x)
        counts.index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype))
        graph_embed = graph_embed / counts.clamp_min(1.0).unsqueeze(-1)

        g = self.embed_head(graph_embed)
        g = F.normalize(g, dim=-1, eps=1e-8)

        z = self.proj_head(g)
        z = F.normalize(z, dim=-1, eps=1e-8)

        return g, z


# =============================================================================
# Embedding Extraction
# =============================================================================

@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    parquet_path: str,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract embeddings with parallelized data loading.

    Args:
        model: Trained GNN encoder
        parquet_path: Path to parquet file with SMILES and labels
        device: Device to run inference on
        batch_size: Batch size for inference
        num_workers: Number of DataLoader workers for parallelization

    Returns:
        embeddings: [N, embed_dim] L2-normalized embeddings
        labels: [N] class labels
    """
    dataset = MolecularGraphDataset(parquet_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_graphs,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    model.eval()
    embeddings_list = []
    labels_list = []

    for batch in tqdm(loader, desc=f"Extracting from {Path(parquet_path).name}"):
        g, _ = model(
            batch["node_feat"].to(device),
            batch["edge_index"].to(device),
            batch["batch"].to(device),
            batch["edge_attr"].to(device),
        )
        embeddings_list.append(g.cpu().numpy())
        labels_list.append(batch["labels"].numpy())

    return np.vstack(embeddings_list), np.concatenate(labels_list)


# =============================================================================
# Linear Probe Training
# =============================================================================

def train_linear_probe(
    train_embeds: np.ndarray,
    train_labels: np.ndarray,
    max_iter: int = 1000,
    seed: int = 42,
) -> LogisticRegression:
    """Train logistic regression with parallel solver.

    Args:
        train_embeds: [N, embed_dim] training embeddings
        train_labels: [N] training labels
        max_iter: Maximum iterations for L-BFGS solver
        seed: Random seed

    Returns:
        Trained LogisticRegression classifier
    """
    clf = LogisticRegression(
        max_iter=max_iter,
        class_weight='balanced',
        n_jobs=-1,  # Use all CPU cores
        random_state=seed,
        solver='lbfgs',
    )
    clf.fit(train_embeds, train_labels)
    return clf


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_probe(
    clf: LogisticRegression,
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Evaluate linear probe on a dataset.

    Args:
        clf: Trained classifier
        embeddings: [N, embed_dim] embeddings
        labels: [N] ground truth labels

    Returns:
        Dictionary with accuracy, AUPRC, and F1 metrics
    """
    preds = clf.predict(embeddings)
    probs = clf.predict_proba(embeddings)[:, 1]

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "auprc": float(average_precision_score(labels, probs)),
        "f1": float(f1_score(labels, preds)),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = args.data_suffix
    train_path = data_dir / "train" / f"supcon_train{suffix}.parquet"
    val_path = data_dir / "val" / f"supcon_val{suffix}.parquet"
    test_path = data_dir / "test" / f"supcon_test{suffix}.parquet"

    # Load checkpoint and infer model architecture
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_state = checkpoint["model_state_dict"]

    # Infer architecture from checkpoint weights
    # input_proj.weight has shape [hidden_dim, node_dim]
    hidden_dim = model_state["input_proj.weight"].shape[0]
    node_dim = model_state["input_proj.weight"].shape[1]

    # embed_head.2.weight has shape [embed_dim, hidden_dim]
    embed_dim = model_state["embed_head.2.weight"].shape[0]

    # proj_head.2.weight has shape [proj_dim, embed_dim]
    proj_dim = model_state["proj_head.2.weight"].shape[0]

    # Count GAT layers
    num_layers = sum(1 for k in model_state.keys() if k.startswith("gat_layers.") and k.endswith(".lin.weight"))

    # gat_layers.0.att_src has shape [heads, out_dim]
    heads = model_state["gat_layers.0.att_src"].shape[0]

    # edge_proj.weight has shape [heads, edge_dim]
    edge_dim = model_state["gat_layers.0.edge_proj.weight"].shape[1]

    print(f"  Inferred architecture:")
    print(f"    node_dim={node_dim}, edge_dim={edge_dim}")
    print(f"    hidden_dim={hidden_dim}, embed_dim={embed_dim}, proj_dim={proj_dim}")
    print(f"    num_layers={num_layers}, heads={heads}")

    # Initialize model with inferred architecture
    model = SupConGNNEncoder(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        heads=heads,
        num_layers=num_layers,
        embed_dim=embed_dim,
        proj_dim=proj_dim,
        dropout=0.0,  # No dropout during inference
    ).to(device)

    model.load_state_dict(model_state)
    model.eval()
    print(f"  Model loaded successfully")

    # Extract embeddings
    print(f"\nExtracting embeddings (batch_size={args.batch_size}, workers={args.num_workers})...")

    print(f"\nProcessing training set...")
    train_embeds, train_labels = extract_embeddings(
        model, str(train_path), device, args.batch_size, args.num_workers
    )
    print(f"  Shape: {train_embeds.shape}, PKS ratio: {train_labels.mean():.3f}")

    print(f"\nProcessing validation set...")
    val_embeds, val_labels = extract_embeddings(
        model, str(val_path), device, args.batch_size, args.num_workers
    )
    print(f"  Shape: {val_embeds.shape}, PKS ratio: {val_labels.mean():.3f}")

    print(f"\nProcessing test set...")
    test_embeds, test_labels = extract_embeddings(
        model, str(test_path), device, args.batch_size, args.num_workers
    )
    print(f"  Shape: {test_embeds.shape}, PKS ratio: {test_labels.mean():.3f}")

    # Train linear probe
    print(f"\nTraining linear probe (max_iter={args.max_iter}, n_jobs=-1)...")
    clf = train_linear_probe(train_embeds, train_labels, args.max_iter, args.seed)
    print(f"  Converged in {clf.n_iter_[0]} iterations")

    # Evaluate
    print("\nEvaluating linear probe...")
    metrics = {
        "train": evaluate_probe(clf, train_embeds, train_labels),
        "val": evaluate_probe(clf, val_embeds, val_labels),
        "test": evaluate_probe(clf, test_embeds, test_labels),
    }

    # Print results
    print("\n" + "=" * 60)
    print("Linear Probe Results")
    print("=" * 60)
    print(f"{'Split':<10} {'Accuracy':>12} {'AUPRC':>12} {'F1':>12}")
    print("-" * 60)
    for split, m in metrics.items():
        print(f"{split:<10} {m['accuracy']:>12.4f} {m['auprc']:>12.4f} {m['f1']:>12.4f}")
    print("=" * 60)

    # Save outputs
    probe_path = output_dir / "linear_probe.joblib"
    metrics_path = output_dir / "linear_probe_metrics.json"

    print(f"\nSaving linear probe to {probe_path}...")
    joblib.dump(clf, probe_path)

    print(f"Saving metrics to {metrics_path}...")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Optionally save embeddings
    if args.save_embeddings:
        for split, (embeds, labels) in [
            ("train", (train_embeds, train_labels)),
            ("val", (val_embeds, val_labels)),
            ("test", (test_embeds, test_labels)),
        ]:
            embed_path = output_dir / f"embeddings_{split}.npz"
            print(f"Saving {split} embeddings to {embed_path}...")
            np.savez(embed_path, embeddings=embeds, labels=labels)

    print("\nDone!")


if __name__ == "__main__":
    main()
