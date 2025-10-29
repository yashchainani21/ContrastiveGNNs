"""Test-time evaluation for the supervised GNN classifier.

Loads the saved checkpoint, builds the same graph featurisation pipeline used
for training, runs batched inference on the held-out test parquet, reports
metrics (loss, AUPRC, precision, recall, F1, AUROC, accuracy), and shows a few
example SMILES predictions. Designed to mirror the architecture and settings
from `16_train_gnn_classifier_distributed.py`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdchem
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

RDLogger.DisableLog("rdApp.*")


# ---- Paths & evaluation settings ----

DEFAULT_CHECKPOINT = "../models/GNN_classifier_w_attn_and_MPNN_20ep_JK.pt"
TEST_PARQUET = "../data/test/baseline_test.parquet"
BATCH_SIZE = 256
NUM_WORKERS = 4
SMILES_EXAMPLES = [
    "CC(C)OC(=O)C1=CC=CC=C1C(=O)O",  # aspirin
    "CCCCCCCCC(=O)OCC",              # laurate ester
    "C1=CC=C(C=C1)C=O",              # benzaldehyde
]


# ---- Graph featurisation (matches training script) ----

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
EDGE_FEAT_DIM = len(BOND_TYPES) + 1


def _one_hot(value, mapping: Dict[int, int]) -> np.ndarray:
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
    smiles: str


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
            raise RuntimeError("No valid molecules found in dataset")

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> GraphSample:
        nf, ei, ea = smiles_to_graph(self.smiles[idx])
        return GraphSample(nf, ei, ea, int(self.labels[idx]), self.smiles[idx])


def collate_graphs(batch: List[GraphSample]) -> Dict[str, torch.Tensor]:
    node_feats = []
    edge_indices = []
    edge_attrs = []
    batch_vec = []
    labels = []
    smiles = []
    offset = 0
    for sample in batch:
        x = torch.from_numpy(sample.node_feat)
        ei = torch.from_numpy(sample.edge_index) + offset
        ea = torch.from_numpy(sample.edge_attr)
        n = x.size(0)
        node_feats.append(x)
        edge_indices.append(ei)
        edge_attrs.append(ea)
        batch_vec.append(torch.full((n,), len(labels), dtype=torch.long))
        labels.append(sample.label)
        smiles.append(sample.smiles)
        offset += n
    return {
        "node_feat": torch.cat(node_feats, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_attr": torch.cat(edge_attrs, dim=0),
        "batch": torch.cat(batch_vec, dim=0),
        "labels": torch.tensor(labels, dtype=torch.float32).view(-1, 1),
        "smiles": smiles,
    }


# ---- Model (mirrors training architecture) ----


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int, edge_dim: int, dropout: float):
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
    def __init__(self, hidden_dim: int, dropout: float):
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
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        heads: Tuple[int, ...],
        dropout: float,
        msg_passes: int,
    ):
        super().__init__()
        self.dropout = dropout
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
            x = F.dropout(x, p=self.dropout, training=self.training)
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


# ---- Evaluation helpers ----


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    logits = []
    labels = []
    for batch in loader:
        node_feat = batch["node_feat"].to(device, non_blocking=True)
        edge_index = batch["edge_index"].to(device, non_blocking=True)
        edge_attr = batch["edge_attr"].to(device, non_blocking=True)
        batch_index = batch["batch"].to(device, non_blocking=True)
        batch_labels = batch["labels"].to(device, non_blocking=True)
        preds = model(node_feat, edge_index, batch_index, edge_attr)
        logits.append(preds.cpu().numpy())
        labels.append(batch_labels.cpu().numpy())
    logits = np.concatenate(logits, axis=0).ravel()
    labels = np.concatenate(labels, axis=0).ravel()
    logits = np.clip(logits, -50.0, 50.0)
    return logits, labels


def compute_metrics(labels: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    auprc = average_precision_score(labels, probs)
    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError:
        auroc = float("nan")
    preds = (probs >= 0.5).astype(np.float32)
    metrics = {
        "auprc": float(auprc),
        "auroc": float(auroc),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }
    return metrics


def compute_loss(labels: np.ndarray, logits: np.ndarray) -> float:
    labels_t = torch.from_numpy(labels).float().view(-1, 1)
    logits_t = torch.from_numpy(logits).float().view(-1, 1)
    pos = max(int(labels.sum()), 1)
    neg = max(int(len(labels) - pos), 1)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss = criterion(logits_t, labels_t)
    return float(loss.item())


def load_checkpoint(path: Path, device: torch.device) -> Tuple[Dict[str, torch.Tensor], Dict[str, Optional[float]]]:
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt["state_dict"]
    meta = {
        "epoch": ckpt.get("epoch"),
        "val_auprc": ckpt.get("val_auprc"),
        "val_auroc": ckpt.get("val_auroc"),
        "val_acc": ckpt.get("val_acc"),
        "val_precision": ckpt.get("val_precision"),
        "val_recall": ckpt.get("val_recall"),
        "val_f1": ckpt.get("val_f1"),
        "val_loss": ckpt.get("val_loss"),
    }
    return state_dict, meta


def predict_single_smiles(
    model: nn.Module,
    smiles: str,
    device: torch.device,
) -> Tuple[str, float, float]:
    node_feat, edge_index, edge_attr = smiles_to_graph(smiles)
    node_feat_t = torch.from_numpy(node_feat).to(device)
    edge_index_t = torch.from_numpy(edge_index).to(device)
    edge_attr_t = torch.from_numpy(edge_attr).to(device)
    batch_index = torch.zeros(node_feat_t.size(0), dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        logit = model(node_feat_t, edge_index_t, batch_index, edge_attr_t)
    logit_val = float(torch.clamp(logit, -50.0, 50.0).item())
    prob = 1.0 / (1.0 + np.exp(-logit_val))
    return smiles, logit_val, prob


# ---- Main entrypoint ----


def build_model(dataset: MolecularGraphDataset, device: torch.device) -> nn.Module:
    hidden_dim = 512
    heads = (8, 8, 8, 8, 8, 8, 8, 8)
    dropout = 0.2
    msg_passes = 3
    model = GraphAttentionClassifier(
        node_dim=dataset.node_feat_dim,
        edge_dim=dataset.edge_feat_dim,
        hidden_dim=hidden_dim,
        heads=heads,
        dropout=dropout,
        msg_passes=msg_passes,
    ).to(device)
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained GNN classifier on the test split")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-parquet", type=str, default=TEST_PARQUET)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument(
        "--example-smiles",
        type=str,
        nargs="*",
        default=SMILES_EXAMPLES,
        help="SMILES strings to score after evaluation",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = MolecularGraphDataset(args.test_parquet)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_graphs,
    )

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state_dict, meta = load_checkpoint(checkpoint_path, device)

    model = build_model(dataset, device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch | missing={missing} | unexpected={unexpected}")
    model.eval()

    logits, labels = run_inference(model, loader, device)
    metrics = compute_metrics(labels, logits)
    loss = compute_loss(labels, logits)

    print(f"Checkpoint: {checkpoint_path}")
    if meta.get("epoch") is not None:
        print(
            "Validation snapshot:",
            {k: v for k, v in meta.items() if k.startswith("val_") and v is not None},
            "| epoch:",
            meta["epoch"],
        )
    print(f"Test loss: {loss:.4f}")
    print(f"Test AUPRC: {metrics['auprc']:.4f}")
    print(f"Test AUROC: {metrics['auroc']:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")
    print(f"Test F1: {metrics['f1']:.4f}")

    print("\nExample SMILES predictions:")
    example_smiles = args.example_smiles or dataset.smiles[:3]
    for smi in example_smiles[:5]:
        try:
            _, logit_val, prob = predict_single_smiles(model, smi, device)
            print(f"  SMILES: {smi} | logit={logit_val:.3f} | prob_PK={prob:.3f}")
        except ValueError as err:
            print(f"  SMILES: {smi} | error: {err}")


if __name__ == "__main__":
    main()
