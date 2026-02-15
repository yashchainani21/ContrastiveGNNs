#!/usr/bin/env python3
"""
Single-Molecule Inference with Supervised GNN Classifier

Accepts a SMILES string via CLI, runs it through the trained supervised GNN
classifier to output a PKS probability and label. Unlike the SupCon pipeline
(script 10), this does not need a separate linear probe — the model outputs
a logit directly, and we apply sigmoid to get the probability.

Usage:
    python scripts/19_run_inference_with_supervised_gnn.py --smiles "CCO"
    python scripts/19_run_inference_with_supervised_gnn.py --smiles "CC(O)CC(=O)O" --checkpoint models/supervised_gnn/checkpoint_epoch_010.pt
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem, RDLogger
from rdkit.Chem import rdchem

RDLogger.DisableLog("rdApp.*")


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CHECKPOINT = "models/supervised_gnn/best_model.pt"


# =============================================================================
# Graph Featurization (copied from 12_evaluate_probes_on_ood_eval_set.py)
# =============================================================================

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


def _build_mapping(values: Iterable) -> Dict:
    return {value: idx for idx, value in enumerate(values)}


ATOM_MAP = _build_mapping(ATOM_TYPES)
DEGREE_MAP = _build_mapping(DEGREES)
CHARGE_MAP = _build_mapping(FORMAL_CHARGES)
NUM_H_MAP = _build_mapping(NUM_HS)
HYB_MAP = {hyb: idx for idx, hyb in enumerate(HYBRIDIZATIONS)}
BOND_MAP = {bond: idx for idx, bond in enumerate(BOND_TYPES)}

EDGE_FEAT_DIM = len(BOND_TYPES) + 1


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


def bond_to_feature(bond) -> np.ndarray:
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


# =============================================================================
# Model Architecture (copied from 12_evaluate_probes_on_ood_eval_set.py)
# =============================================================================

def edge_softmax(dst: torch.Tensor, scores: torch.Tensor, num_nodes: int) -> torch.Tensor:
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


class SupervisedGNNClassifier(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 256,
        heads: int = 4,
        num_layers: int = 3,
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
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        logit = self.cls_head(graph_embed)
        return logit, graph_embed


# =============================================================================
# Checkpoint Loading (copied from 12_evaluate_probes_on_ood_eval_set.py)
# =============================================================================

def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> SupervisedGNNClassifier:
    """Load SupervisedGNNClassifier from a training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint["model_state_dict"]

    if "args" in checkpoint:
        saved_args = checkpoint["args"]
        node_dim = model_state["input_proj.weight"].shape[1]
        edge_dim = model_state["gat_layers.0.edge_proj.weight"].shape[1]
        model = SupervisedGNNClassifier(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=saved_args["hidden_dim"],
            heads=saved_args["num_heads"],
            num_layers=saved_args["num_layers"],
            dropout=saved_args["dropout"],
        )
    else:
        hidden_dim = model_state["input_proj.weight"].shape[0]
        node_dim = model_state["input_proj.weight"].shape[1]
        edge_dim = model_state["gat_layers.0.edge_proj.weight"].shape[1]
        num_layers = sum(1 for k in model_state if k.startswith("gat_layers.") and k.endswith(".lin.weight"))
        heads = model_state["gat_layers.0.att_src"].shape[0]
        model = SupervisedGNNClassifier(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            num_layers=num_layers,
        )

    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("val_loss", float("nan"))
    print(f"  Loaded checkpoint from epoch {epoch} (val loss: {val_loss:.4f})")

    return model


# =============================================================================
# Inference
# =============================================================================

@torch.no_grad()
def predict_single(
    model: SupervisedGNNClassifier,
    smiles: str,
    device: torch.device,
) -> Tuple[float, str]:
    """Run inference on a single SMILES string.

    Returns:
        probability: PKS probability (float)
        label: "PKS" or "non-PKS"

    Raises:
        ValueError: If the SMILES is invalid or has no atoms
    """
    node_feat, edge_index, edge_attr = smiles_to_graph(smiles)

    node_feat_t = torch.from_numpy(node_feat).to(device)
    edge_index_t = torch.from_numpy(edge_index).to(device)
    edge_attr_t = torch.from_numpy(edge_attr).to(device)
    batch_t = torch.zeros(node_feat.shape[0], dtype=torch.long, device=device)

    logit, _ = model(node_feat_t, edge_index_t, batch_t, edge_attr_t)
    probability = torch.sigmoid(logit).cpu().numpy().item()
    label = "PKS" if probability >= 0.5 else "non-PKS"

    return probability, label


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a single molecule using the supervised GNN classifier."
    )
    parser.add_argument(
        "--smiles", type=str, required=True,
        help="SMILES string of the molecule to classify"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
        help="Path to supervised GNN checkpoint (default: %(default)s)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Run the supervised GNN training script first.")
        sys.exit(1)

    print(f"Loading model from {args.checkpoint}...")
    model = load_model_from_checkpoint(args.checkpoint, device)

    # Run inference
    try:
        probability, label = predict_single(model, args.smiles, device)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Print results
    print(f"SMILES:       {args.smiles}")
    print(f"Prediction:   {label}")
    print(f"Probability:  {probability:.4f}")


if __name__ == "__main__":
    main()
