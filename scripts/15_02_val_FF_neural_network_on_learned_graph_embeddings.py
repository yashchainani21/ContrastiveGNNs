from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdchem
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import average_precision_score


RDLogger.DisableLog("rdApp.*")


# ---- Configuration ----
VAL_PARQUET = "../data/val/baseline_val.parquet"
GNN_CHECKPOINT = "../models/supcon_gnn_latest.pt"
FFNN_CHECKPOINT = "../models/supcon_gnn_20251008_161101.pt"
FFNN_META = "../models/learned_graph_embedding_ffnn_pks_classifier.meta.json"
EMBED_SOURCE = "preproj"  # "preproj" or "proj"
BATCH_SIZE = 512
NUM_WORKERS = 4


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


def _build_mapping(values):
    return {value: idx for idx, value in enumerate(values)}


ATOM_MAP = _build_mapping(ATOM_TYPES)
DEGREE_MAP = _build_mapping(DEGREES)
CHARGE_MAP = _build_mapping(FORMAL_CHARGES)
NUM_H_MAP = _build_mapping(NUM_HS)
HYB_MAP = {hyb: idx for idx, hyb in enumerate(HYBRIDIZATIONS)}
BOND_MAP = {bond: idx for idx, bond in enumerate(BOND_TYPES)}


def _one_hot_with_unknown(value, mapping):
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


EDGE_FEATURE_DIM = len(BOND_TYPES) + 1


def bond_to_feature(bond: rdchem.Bond | None) -> np.ndarray:
    vec = np.zeros(EDGE_FEATURE_DIM, dtype=np.float32)
    if bond is None:
        vec[-1] = 1.0
        return vec
    idx = BOND_MAP.get(bond.GetBondType(), EDGE_FEATURE_DIM - 1)
    vec[idx] = 1.0
    return vec


def smiles_to_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        raise ValueError(f"SMILES with no atoms: {smiles}")

    node_feats = np.vstack([atom_to_feature(atom) for atom in mol.GetAtoms()]).astype(np.float32)
    edges = []
    edge_attrs = []

    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        feat = bond_to_feature(bond)
        edges.append((start, end))
        edges.append((end, start))
        edge_attrs.append(feat)
        edge_attrs.append(feat)

    loop_feat = bond_to_feature(None)
    for idx in range(num_atoms):
        edges.append((idx, idx))
        edge_attrs.append(loop_feat)

    edge_index = np.array(edges, dtype=np.int64).T
    edge_attr = np.vstack(edge_attrs).astype(np.float32)
    return node_feats, edge_index, edge_attr


class GraphSample:
    __slots__ = ("node_feat", "edge_index", "edge_attr", "label")

    def __init__(self, node_feat, edge_index, edge_attr, label):
        self.node_feat = node_feat
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.label = label


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
            raise RuntimeError("No valid molecules found.")

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx: int):
        nf, ei, ea = smiles_to_graph(self.smiles[idx])
        return GraphSample(nf, ei, ea, int(self.labels[idx]))


def collate_graphs(batch):
    node_feats = []
    edge_indices = []
    edge_attrs = []
    batch_vec = []
    labels = []
    offset = 0
    for sample in batch:
        x = torch.from_numpy(sample.node_feat)
        edge_index = torch.from_numpy(sample.edge_index) + offset
        edge_attr = torch.from_numpy(sample.edge_attr)
        num_nodes = x.size(0)

        node_feats.append(x)
        edge_indices.append(edge_index)
        edge_attrs.append(edge_attr)
        batch_vec.append(torch.full((num_nodes,), len(labels), dtype=torch.long))
        labels.append(sample.label)
        offset += num_nodes

    return {
        "node_feat": torch.cat(node_feats, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_attr": torch.cat(edge_attrs, dim=0),
        "batch": torch.cat(batch_vec, dim=0),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.0, edge_feat_dim=None, negative_slope=0.2):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.dropout = dropout
        self.lin = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(heads, out_dim))
        self.att_dst = nn.Parameter(torch.Tensor(heads, out_dim))
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        self.edge_proj = nn.Linear(edge_feat_dim, heads, bias=False) if edge_feat_dim is not None else None
        self.negative_slope = negative_slope
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)
        if self.edge_proj is not None:
            nn.init.xavier_uniform_(self.edge_proj.weight)

    def forward(self, x, edge_index, edge_attr):
        h = self.lin(x)
        num_nodes = h.size(0)
        h = h.view(num_nodes, self.heads, self.out_dim)
        att_src = (h * self.att_src).sum(dim=-1)
        att_dst = (h * self.att_dst).sum(dim=-1)

        src, dst = edge_index
        alpha = att_src[src] + att_dst[dst]
        if self.edge_proj is not None:
            alpha = alpha + self.edge_proj(edge_attr)
        alpha = F.leaky_relu(alpha, negative_slope=self.negative_slope)
        alpha = edge_softmax(dst, alpha, num_nodes)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = h[src] * alpha.unsqueeze(-1)
        agg = torch.zeros(num_nodes, self.heads, self.out_dim, device=x.device)
        agg.index_add_(0, dst, out)
        agg = agg.mean(dim=1) + self.bias
        return agg


def edge_softmax(dst, scores, num_nodes):
    heads = scores.size(1)
    outputs = []
    for head in range(heads):
        s = scores[:, head]
        max_vals = torch.full((num_nodes,), -float("inf"), device=s.device)
        max_vals.scatter_reduce_(0, dst, s, reduce="amax")
        s = s - max_vals[dst]
        exp_s = torch.exp(s)
        denom = torch.zeros(num_nodes, device=s.device).scatter_add_(0, dst, exp_s)
        outputs.append(exp_s / (denom[dst] + 1e-16))
    return torch.stack(outputs, dim=1)


def scatter_mean(src, index, dim_size):
    out = torch.zeros(dim_size, src.size(-1), device=src.device)
    out.scatter_add_(0, index.unsqueeze(-1).expand(-1, src.size(-1)), src)
    counts = torch.zeros(dim_size, device=src.device)
    counts.scatter_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    counts = counts.clamp_min(1.0)
    return out / counts.unsqueeze(-1)


class GraphAttentionEncoder(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, heads, dropout, embed_dim, proj_dim, use_projection=True):
        super().__init__()
        self.dropout = dropout
        self.use_projection = use_projection
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for head_count in heads:
            self.layers.append(GraphAttentionLayer(hidden_dim, hidden_dim, head_count, dropout, edge_feat_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
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

    def forward(self, node_feat, edge_index, batch_index, edge_attr):
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


class FFClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden1: int = 512, hidden2: int = 256, p_drop: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x)


def build_encoder(node_dim, edge_dim, hparams):
    return GraphAttentionEncoder(
        node_feat_dim=node_dim,
        edge_feat_dim=edge_dim,
        hidden_dim=hparams.get("hidden_dim", 256),
        heads=tuple(hparams.get("heads", (4, 4, 4))),
        dropout=hparams.get("dropout", 0.1),
        embed_dim=hparams.get("embed_dim", 512),
        proj_dim=hparams.get("proj_dim", 256),
        use_projection=True,
    )


def load_gnn(checkpoint_path, node_dim, edge_dim, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    hparams = ckpt.get("hyperparameters", {})
    model = build_encoder(node_dim, edge_dim, hparams)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print("[Warning] Missing keys when loading GNN checkpoint:", missing)
    if unexpected:
        print("[Warning] Unexpected keys when loading GNN checkpoint:", unexpected)
    model.to(device)
    model.eval()
    return model, hparams


def load_classifier(checkpoint_path, meta_path, device):
    meta = json.loads(Path(meta_path).read_text())
    input_dim = meta["input_dim"]
    model = FFClassifier(input_dim=input_dim)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, meta


def compute_embeddings(model, dataset, device):
    loader = DataLoader(
        list(range(len(dataset))),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=lambda idxs: collate_graphs([dataset[i] for i in idxs]),
    )
    use_proj = EMBED_SOURCE.lower() in {"proj", "z"}
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            node_feat = batch["node_feat"].to(device, non_blocking=True)
            edge_index = batch["edge_index"].to(device, non_blocking=True)
            edge_attr = batch["edge_attr"].to(device, non_blocking=True)
            batch_index = batch["batch"].to(device, non_blocking=True)
            g, z = model(node_feat, edge_index, batch_index, edge_attr)
            feats = z if use_proj else g
            embeddings.append(feats.cpu().numpy())
            labels.append(batch["labels"].numpy())
    return np.vstack(embeddings), np.concatenate(labels).astype(np.int64)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading validation molecules...")
    val_dataset = MolecularGraphDataset(VAL_PARQUET)
    print(f"Validation graphs: {len(val_dataset)}")

    print(f"Restoring GNN encoder from {GNN_CHECKPOINT} ...")
    encoder, hparams = load_gnn(GNN_CHECKPOINT, val_dataset.node_feat_dim, val_dataset.edge_feat_dim, device)
    print(
        "Encoder hyperparameters:",
        {k: hparams.get(k, "n/a") for k in ("hidden_dim", "embed_dim", "proj_dim", "heads")},
        "source=", EMBED_SOURCE,
    )

    print("Computing graph embeddings for validation set...")
    embeddings, labels = compute_embeddings(encoder, val_dataset, device)
    print("Embedding shape:", embeddings.shape)

    print(f"Loading FFNN classifier from {FFNN_CHECKPOINT} ...")
    classifier, meta = load_classifier(FFNN_CHECKPOINT, FFNN_META, device)
    print("Classifier input_dim:", meta["input_dim"])

    loader = DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(embeddings.astype(np.float32)),
            torch.from_numpy(labels.astype(np.float32)).view(-1, 1),
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    classifier.eval()
    probs_list = []
    labels_list = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = classifier(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_list.append(probs)
            labels_list.append(yb.cpu().numpy())

    probs = np.concatenate(probs_list).ravel()
    labels = np.concatenate(labels_list).ravel()
    val_auprc = average_precision_score(labels, probs)
    print(f"Validation AUPRC: {val_auprc:.4f}")


if __name__ == "__main__":
    main()
