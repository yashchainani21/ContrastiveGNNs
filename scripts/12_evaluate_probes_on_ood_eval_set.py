#!/usr/bin/env python3
"""
Evaluate Supervised GNN Classifier on OOD PKS Eval Set

Loads held-out PKS molecules (from extender codes not seen during training) and
measures recall for both the supervised GNN (direct inference) and an ECFP4
logistic regression baseline.

All eval molecules are true PKS (label=1), so the key metric is recall: what
fraction does each method correctly identify as PKS?

Prerequisites:
    - Run scripts/18_train_supervised_gnn_distributed.py to produce:
        models/supervised_gnn/best_model.pt
    - Run scripts/08_train_linear_probe_on_GNN_embeddings.py to produce:
        models/supcon_gnn/linear_probe_ecfp4.joblib
    - Run scripts/11_synthetically_generate_eval_set.py to produce:
        data/processed/eval_pks_products_*_SMILES.txt

Usage:
    python scripts/12_evaluate_probes_on_ood_eval_set.py
    python scripts/12_evaluate_probes_on_ood_eval_set.py --checkpoint models/supervised_gnn/checkpoint_epoch_020.pt
    python scripts/12_evaluate_probes_on_ood_eval_set.py --ecfp4_probe models/supcon_gnn/linear_probe_ecfp4.joblib
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdchem

RDLogger.DisableLog("rdApp.*")


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Supervised GNN on OOD PKS Eval Set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, default="models/supervised_gnn/best_model.pt",
        help="Path to supervised GNN checkpoint"
    )
    parser.add_argument(
        "--eval_smiles", type=str,
        default="data/processed/eval_pks_products_1_ext_no_stereo_butmal_hexmal_isobutmal_d-isobutmal_dcp_SMILES.txt",
        help="Path to OOD eval SMILES file (all PKS, label=1)"
    )
    parser.add_argument(
        "--ecfp4_probe", type=str, default="models/supcon_gnn/linear_probe_ecfp4.joblib",
        help="Path to pre-trained ECFP4 logistic regression probe"
    )
    parser.add_argument(
        "--output_json", type=str, default="models/supervised_gnn/ood_eval_comparison.json",
        help="Path for saving comparison JSON"
    )
    return parser.parse_args()


# =============================================================================
# Graph Featurization
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


def smiles_to_ecfp4(smiles: str, nbits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nbits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nbits)
    return np.array(fp, dtype=np.float32)


# =============================================================================
# Model Architecture
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
# Checkpoint Loading
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
# GNN Direct Inference on OOD SMILES
# =============================================================================

@torch.no_grad()
def get_gnn_predictions(
    model: SupervisedGNNClassifier,
    smiles_list: List[str],
    device: torch.device,
) -> Tuple[np.ndarray, List[str]]:
    """Run direct GNN inference on a list of SMILES.

    Returns:
        probs: [N_valid] array of P(PKS) from sigmoid(logit)
        failed: list of SMILES that could not be parsed
    """
    model.eval()
    probs = []
    failed = []

    for smi in smiles_list:
        try:
            node_feat, edge_index, edge_attr = smiles_to_graph(smi)
        except ValueError:
            failed.append(smi)
            continue

        node_feat_t = torch.from_numpy(node_feat).to(device)
        edge_index_t = torch.from_numpy(edge_index).to(device)
        edge_attr_t = torch.from_numpy(edge_attr).to(device)
        batch_t = torch.zeros(node_feat.shape[0], dtype=torch.long, device=device)

        logit, _ = model(node_feat_t, edge_index_t, batch_t, edge_attr_t)
        prob = torch.sigmoid(logit).cpu().numpy().item()
        probs.append(prob)

    return np.array(probs), failed


# =============================================================================
# ECFP4 Fingerprint Extraction
# =============================================================================

def get_ecfp4_fingerprints(smiles_list: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Compute ECFP4 fingerprints for a list of SMILES."""
    fingerprints = []
    failed = []

    for smi in smiles_list:
        fp = smiles_to_ecfp4(smi)
        if fp.sum() == 0 and Chem.MolFromSmiles(smi) is None:
            failed.append(smi)
        else:
            fingerprints.append(fp)

    return np.vstack(fingerprints), failed


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    print("OOD Eval: Supervised GNN (direct) vs ECFP4 Probe")
    print("=" * 72)

    # --- Load eval SMILES ---
    eval_path = Path(args.eval_smiles)
    if not eval_path.exists():
        print(f"Error: Eval SMILES file not found at {args.eval_smiles}")
        print("Run scripts/11_synthetically_generate_eval_set.py first.")
        sys.exit(1)

    smiles_list = [line.strip() for line in eval_path.read_text().splitlines() if line.strip()]
    print(f"\nLoaded {len(smiles_list)} OOD PKS molecules (all label=1)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- GNN direct inference ---
    print(f"\n--- Supervised GNN (direct inference) ---")
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)

    print(f"Loading checkpoint from {args.checkpoint}...")
    model = load_model_from_checkpoint(args.checkpoint, device)

    print("Running inference on OOD molecules...")
    gnn_probs, gnn_failed = get_gnn_predictions(model, smiles_list, device)
    if gnn_failed:
        print(f"  WARNING: {len(gnn_failed)} SMILES failed graph conversion")
    print(f"  Predictions: {len(gnn_probs)}")

    # --- ECFP4 baseline (load pre-trained probe) ---
    print("\n--- ECFP4 Fingerprint Probe ---")
    if not Path(args.ecfp4_probe).exists():
        print(f"Error: ECFP4 probe not found at {args.ecfp4_probe}")
        print("Run scripts/08_train_linear_probe_on_GNN_embeddings.py first.")
        sys.exit(1)

    print(f"Loading pre-trained ECFP4 probe from {args.ecfp4_probe}...")
    clf = joblib.load(args.ecfp4_probe)

    print("Computing ECFP4 fingerprints for OOD molecules...")
    ecfp4_fps, ecfp4_failed = get_ecfp4_fingerprints(smiles_list)
    if ecfp4_failed:
        print(f"  WARNING: {len(ecfp4_failed)} SMILES failed fingerprinting")
    print(f"  Fingerprints: {len(ecfp4_fps)}")

    ecfp4_probs = clf.predict_proba(ecfp4_fps)[:, 1]

    # --- Compute metrics ---
    gnn_recall = float(np.mean(gnn_probs >= 0.5))
    gnn_mean_prob = float(np.mean(gnn_probs))
    gnn_median_prob = float(np.median(gnn_probs))

    ecfp4_recall = float(np.mean(ecfp4_probs >= 0.5))
    ecfp4_mean_prob = float(np.mean(ecfp4_probs))
    ecfp4_median_prob = float(np.median(ecfp4_probs))

    # --- Print comparison table ---
    print("\n" + "=" * 72)
    print("OOD Eval: Supervised GNN (direct) vs ECFP4 Probe")
    print(f"  ({len(gnn_probs)} GNN molecules, {len(ecfp4_probs)} ECFP4 molecules, all true PKS)")
    print("=" * 72)
    print(f"{'Metric':<20} {'GNN':>12} {'ECFP4':>12} {'Delta':>12}")
    print("-" * 72)

    rows = [
        ("Recall (>=0.5)", gnn_recall, ecfp4_recall),
        ("Mean PKS prob", gnn_mean_prob, ecfp4_mean_prob),
        ("Median PKS prob", gnn_median_prob, ecfp4_median_prob),
    ]
    for label, gnn_val, ecfp4_val in rows:
        delta = gnn_val - ecfp4_val
        sign = "+" if delta >= 0 else ""
        print(f"{label:<20} {gnn_val:>12.4f} {ecfp4_val:>12.4f} {sign}{delta:>11.4f}")

    print("=" * 72)

    # --- Save JSON ---
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "checkpoint": args.checkpoint,
        "eval_file": args.eval_smiles,
        "n_molecules": len(smiles_list),
        "supervised_gnn_direct": {
            "n_evaluated": len(gnn_probs),
            "n_failed": len(gnn_failed),
            "recall": gnn_recall,
            "mean_prob": gnn_mean_prob,
            "median_prob": gnn_median_prob,
        },
        "ecfp4_probe": {
            "n_evaluated": len(ecfp4_probs),
            "n_failed": len(ecfp4_failed),
            "recall": ecfp4_recall,
            "mean_prob": ecfp4_mean_prob,
            "median_prob": ecfp4_median_prob,
        },
        "delta": {
            "recall": gnn_recall - ecfp4_recall,
            "mean_prob": gnn_mean_prob - ecfp4_mean_prob,
            "median_prob": gnn_median_prob - ecfp4_median_prob,
        },
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
