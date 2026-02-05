#!/usr/bin/env python3
"""
Single-Molecule Inference with SupCon GNN + Linear Probe

Accepts a SMILES string via CLI, runs it through the trained SupCon GNN
encoder to get an embedding, then through the trained linear probe to
output a PKS probability and label.

Usage:
    python scripts/10_run_inference_with_GNN_embeddings.py --smiles "CCO"
"""

import argparse
import sys
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdchem

RDLogger.DisableLog("rdApp.*")


# =============================================================================
# Configuration
# =============================================================================

CHECKPOINT = "models/supcon_gnn/best_model.pt"
PROBE_PATH = "models/supcon_gnn/linear_probe.joblib"


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
# Inference Functions
# =============================================================================

def load_model(checkpoint_path: str, device: torch.device) -> SupConGNNEncoder:
    """Load a trained SupConGNNEncoder from checkpoint.

    Infers architecture from weight shapes so no config is needed.

    Args:
        checkpoint_path: Path to the checkpoint .pt file
        device: Device to load the model onto

    Returns:
        Model in eval mode with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint["model_state_dict"]

    # Infer architecture from checkpoint weights
    hidden_dim = model_state["input_proj.weight"].shape[0]
    node_dim = model_state["input_proj.weight"].shape[1]
    embed_dim = model_state["embed_head.2.weight"].shape[0]
    proj_dim = model_state["proj_head.2.weight"].shape[0]
    num_layers = sum(1 for k in model_state if k.startswith("gat_layers.") and k.endswith(".lin.weight"))
    heads = model_state["gat_layers.0.att_src"].shape[0]
    edge_dim = model_state["gat_layers.0.edge_proj.weight"].shape[1]

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
    return model


def load_probe(probe_path: str):
    """Load a trained LogisticRegression linear probe.

    Args:
        probe_path: Path to the .joblib file

    Returns:
        Trained LogisticRegression classifier

    Raises:
        FileNotFoundError: If probe file is missing
    """
    from pathlib import Path

    if not Path(probe_path).exists():
        raise FileNotFoundError(
            f"Linear probe not found at {probe_path}. "
            "Run scripts/08_train_linear_probe_on_GNN_embeddings.py first."
        )
    return joblib.load(probe_path)


def predict_single(
    model: SupConGNNEncoder,
    probe,
    smiles: str,
    device: torch.device,
) -> Tuple[float, str, np.ndarray]:
    """Run inference on a single SMILES string.

    Args:
        model: Trained SupConGNNEncoder in eval mode
        probe: Trained LogisticRegression linear probe
        smiles: Input SMILES string
        device: Device for model inference

    Returns:
        probability: PKS probability (float)
        label: "PKS" or "non-PKS"
        embedding: 1-D numpy array of the graph embedding

    Raises:
        ValueError: If the SMILES is invalid or has no atoms
    """
    node_feat, edge_index, edge_attr = smiles_to_graph(smiles)

    node_feat_t = torch.from_numpy(node_feat).to(device)
    edge_index_t = torch.from_numpy(edge_index).to(device)
    edge_attr_t = torch.from_numpy(edge_attr).to(device)
    batch_t = torch.zeros(node_feat.shape[0], dtype=torch.long, device=device)

    with torch.no_grad():
        g, _ = model(node_feat_t, edge_index_t, batch_t, edge_attr_t)

    embedding = g.cpu().numpy().squeeze(0)  # [embed_dim]
    probability = float(probe.predict_proba(embedding.reshape(1, -1))[:, 1][0])
    label = "PKS" if probability >= 0.5 else "non-PKS"

    return probability, label, embedding


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a single molecule using the SupCon GNN + linear probe."
    )
    parser.add_argument(
        "--smiles", type=str, required=True,
        help="SMILES string of the molecule to classify"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    from pathlib import Path

    if not Path(CHECKPOINT).exists():
        print(f"Error: Checkpoint not found at {CHECKPOINT}")
        print("Run scripts/07_train_supcon_gnn_distributed.py first.")
        sys.exit(1)

    model = load_model(CHECKPOINT, device)

    # Load probe
    try:
        probe = load_probe(PROBE_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Run inference
    try:
        probability, label, embedding = predict_single(model, probe, args.smiles, device)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Print results
    print(f"SMILES:       {args.smiles}")
    print(f"Prediction:   {label}")
    print(f"Probability:  {probability:.4f}")
    print(f"Embedding:    [{embedding.shape[0]}-dim vector] (first 5 values: {embedding[:5]})")


if __name__ == "__main__":
    main()
