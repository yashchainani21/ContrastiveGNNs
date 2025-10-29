import pathlib
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdchem
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# Silence RDKit warnings to keep logs clean.
RDLogger.DisableLog('rdApp.*')


# ---- Configuration ----
TRAIN_NPZ = '../data/train/baseline_train_ecfp4.npz'
TRAIN_PARQUET = '../data/train/baseline_train.parquet'
MODEL_CHECKPOINT = '../models/supcon_gnn_latest.pt'  # update to actual filename
OUTPUT_DIR = '../figures'
RAW_TSNE_FILENAME = 'tsne_ecfp4_gnn.png'
LEARNED_TSNE_FILENAME = 'tsne_learned_gnn.png'
NEGATIVE_SAMPLE_SIZE = 50_000
SEED = 42
BATCH_SIZE = 1024
NUM_WORKERS = 4
TSNE_PERPLEXITY = 30.0
KNN_N_NEIGHBORS = 5
KNN_WEIGHTS = 'distance'
KNN_METRIC = 'euclidean'
KNN_TEST_FRACTION = 0.2
USE_PROJECTION_HEAD = False  # True => use projection (z); False => encoder output (g)


# ---- Molecular graph utilities ----
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


EDGE_FEATURE_DIM = len(BOND_TYPES) + 1


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
        raise ValueError(f"SMILES with no atoms: {smiles}")

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

    loop_feat = bond_to_feature(None)
    for idx in range(num_atoms):
        edges.append((idx, idx))
        edge_attrs.append(loop_feat)

    edge_index = np.array(edges, dtype=np.int64).T
    edge_attr = np.vstack(edge_attrs).astype(np.float32)
    return node_feats, edge_index, edge_attr


# ---- Dataset helpers ----
class GraphSample:
    __slots__ = ("node_feat", "edge_index", "edge_attr")

    def __init__(self, node_feat: np.ndarray, edge_index: np.ndarray, edge_attr: np.ndarray):
        self.node_feat = node_feat
        self.edge_index = edge_index
        self.edge_attr = edge_attr


class MolecularGraphSubset:
    def __init__(self, parquet_path: str, indices: np.ndarray):
        df = pd.read_parquet(parquet_path)
        self.smiles = df['smiles'].astype(str).iloc[indices].reset_index(drop=True)
        self.labels = (df['source'].astype(str) == 'PKS').astype(np.int64).iloc[indices].to_numpy()
        sample_graph = smiles_to_graph(self.smiles.iloc[0])
        self.node_feat_dim = sample_graph[0].shape[1]
        self.edge_feat_dim = sample_graph[2].shape[1]

    def __len__(self) -> int:
        return len(self.smiles)

    def get_graph(self, idx: int) -> Tuple[GraphSample, int]:
        node_feat, edge_index, edge_attr = smiles_to_graph(self.smiles.iloc[idx])
        graph = GraphSample(node_feat=node_feat, edge_index=edge_index, edge_attr=edge_attr)
        label = int(self.labels[idx])
        return graph, label


def collate_graphs(batch: List[Tuple[GraphSample, int]]) -> Dict[str, torch.Tensor]:
    node_feats = []
    edge_indices = []
    edge_attrs = []
    batch_vec = []
    labels = []

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
    batch_tensor = torch.cat(batch_vec, dim=0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return {
        "node_feat": node_feat,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "batch": batch_tensor,
        "labels": labels_tensor,
    }


# ---- Graph encoder (mirrors training script) ----
class GraphAttentionLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.0,
        edge_feat_dim: int | None = None,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        x_proj = self.lin(x)
        num_nodes = x_proj.size(0)
        x_proj = x_proj.view(num_nodes, self.heads, self.out_dim)

        att_src = (x_proj * self.att_src).sum(dim=-1)
        att_dst = (x_proj * self.att_dst).sum(dim=-1)

        src, dst = edge_index
        alpha = att_src[src] + att_dst[dst]
        if self.edge_proj is not None:
            alpha = alpha + self.edge_proj(edge_attr)

        alpha = F.leaky_relu(alpha, negative_slope=self.negative_slope)
        alpha = softmax_edges(dst, alpha, num_nodes)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = x_proj[src] * alpha.unsqueeze(-1)
        agg = torch.zeros(num_nodes, self.heads, self.out_dim, device=x.device)
        agg.index_add_(0, dst, out)
        agg = agg.mean(dim=1)
        agg = agg + self.bias
        return agg


def softmax_edges(dst: torch.Tensor, scores: torch.Tensor, num_nodes: int) -> torch.Tensor:
    heads = scores.size(1)
    out = []
    for head in range(heads):
        s = scores[:, head]
        max_vals = torch.full((num_nodes,), -float('inf'), device=s.device)
        max_vals.scatter_reduce_(0, dst, s, reduce='amax')
        s = s - max_vals[dst]
        exp_s = torch.exp(s)
        denom = torch.zeros(num_nodes, device=s.device).scatter_add_(0, dst, exp_s)
        out.append(exp_s / (denom[dst] + 1e-16))
    return torch.stack(out, dim=1)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, src.size(-1), device=src.device)
    out.scatter_add_(0, index.unsqueeze(-1).expand(-1, src.size(-1)), src)
    counts = torch.zeros(dim_size, device=src.device)
    counts.scatter_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    counts = counts.clamp_min(1.0)
    return out / counts.unsqueeze(-1)


class GraphAttentionEncoder(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int,
        heads: Tuple[int, ...],
        dropout: float,
        embed_dim: int,
        proj_dim: int,
        use_projection: bool,
    ):
        super().__init__()
        self.dropout = dropout
        self.use_projection = use_projection

        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for head_count in heads:
            layer = GraphAttentionLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                heads=head_count,
                dropout=dropout,
                edge_feat_dim=edge_feat_dim,
            )
            self.layers.append(layer)
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


# ---- Utility functions ----
def ensure_output_dir(path: str) -> pathlib.Path:
    out_path = pathlib.Path(path)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def select_subset(fps: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    n_pos = len(pos_idx)
    n_neg = min(NEGATIVE_SAMPLE_SIZE, len(neg_idx))

    pos_sample = pos_idx  # keep all positives
    neg_sample = rng.choice(neg_idx, size=n_neg, replace=False)
    selected = np.concatenate([pos_sample, neg_sample]).astype(np.int64)
    rng.shuffle(selected)
    return fps[selected], labels[selected], selected


def compute_tsne(features: np.ndarray, random_state: int) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        init='pca',
        random_state=random_state,
        learning_rate='auto',
    )
    return tsne.fit_transform(features)


def plot_tsne(points: np.ndarray, labels: np.ndarray, title: str, output_path: pathlib.Path) -> None:
    plt.figure(figsize=(8, 6))
    mask = labels == 1
    plt.scatter(
        points[~mask, 0],
        points[~mask, 1],
        c='#1f77b4',
        alpha=0.35,
        s=10,
        label='Non-polyketide (label=0)',
    )
    plt.scatter(
        points[mask, 0],
        points[mask, 1],
        c='#d62728',
        alpha=0.8,
        s=12,
        label='Polyketide (label=1)',
    )
    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved t-SNE figure to {output_path}")


def evaluate_knn(features: np.ndarray, labels: np.ndarray, description: str, random_state: int) -> float:
    X_train, X_val, y_train, y_val = train_test_split(
        features,
        labels,
        test_size=KNN_TEST_FRACTION,
        stratify=labels,
        random_state=random_state,
    )
    clf = KNeighborsClassifier(
        n_neighbors=KNN_N_NEIGHBORS,
        weights=KNN_WEIGHTS,
        metric=KNN_METRIC,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_val)[:, 1]
    score = average_precision_score(y_val, probs)
    print(f"[k-NN] {description} AUPRC={score:.4f}")
    return score


def load_gnn_checkpoint(model_path: str, feat_dims: Dict[str, int]) -> Tuple[nn.Module, Dict]:
    ckpt = torch.load(model_path, map_location='cpu')
    hparams = ckpt.get('hyperparameters', {})
    node_dim = feat_dims['node_feat_dim']
    edge_dim = feat_dims['edge_feat_dim']

    model = GraphAttentionEncoder(
        node_feat_dim=node_dim,
        edge_feat_dim=edge_dim,
        hidden_dim=hparams.get('hidden_dim', 256),
        heads=tuple(hparams.get('heads', (4, 4, 4))),
        dropout=hparams.get('dropout', 0.1),
        embed_dim=hparams.get('embed_dim', 512),
        proj_dim=hparams.get('proj_dim', 256),
        use_projection=True,
    )
    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if missing:
        print("[Warning] Missing keys when loading checkpoint:", missing)
    if unexpected:
        print("[Warning] Unexpected keys when loading checkpoint:", unexpected)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, hparams


def compute_gnn_embeddings(
    model: nn.Module,
    dataset: MolecularGraphSubset,
) -> np.ndarray:
    device = next(model.parameters()).device
    loader = torch.utils.data.DataLoader(
        list(range(len(dataset))),
        batch_size=64,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=lambda indices: collate_graphs([dataset.get_graph(i) for i in indices]),
    )

    embeddings = []
    use_proj = USE_PROJECTION_HEAD

    with torch.no_grad():
        for batch in loader:
            node_feat = batch['node_feat'].to(device, non_blocking=True)
            edge_index = batch['edge_index'].to(device, non_blocking=True)
            edge_attr = batch['edge_attr'].to(device, non_blocking=True)
            batch_index = batch['batch'].to(device, non_blocking=True)
            g, z = model(node_feat, edge_index, batch_index, edge_attr)
            features = z if use_proj else g
            embeddings.append(features.cpu().numpy())

    return np.vstack(embeddings)


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    ensure_output_dir(OUTPUT_DIR)
    raw_out = pathlib.Path(OUTPUT_DIR) / RAW_TSNE_FILENAME
    learned_out = pathlib.Path(OUTPUT_DIR) / LEARNED_TSNE_FILENAME

    data = np.load(TRAIN_NPZ, allow_pickle=False)
    fps = data['fps'].astype(np.float32)
    labels = data['labels'].astype(np.int64)
    print("Loaded fingerprint dataset:", fps.shape)

    fps_subset, labels_subset, selected_indices = select_subset(fps, labels)
    print("Subset size:", fps_subset.shape[0])
    print("Positives:", (labels_subset == 1).sum(), "Negatives:", (labels_subset == 0).sum())

    print("Evaluating k-NN on raw fingerprints...")
    evaluate_knn(fps_subset, labels_subset, "Raw ECFP4", random_state=SEED)

    print("Running t-SNE on raw fingerprints...")
    tsne_raw = compute_tsne(fps_subset, random_state=SEED)
    plot_tsne(tsne_raw, labels_subset, "t-SNE on baseline ECFP4 fingerprints", raw_out)

    graph_dataset = MolecularGraphSubset(TRAIN_PARQUET, selected_indices)
    print("Graph subset loaded:", len(graph_dataset))

    model, ckpt_meta = load_gnn_checkpoint(
        MODEL_CHECKPOINT,
        {"node_feat_dim": graph_dataset.node_feat_dim, "edge_feat_dim": graph_dataset.edge_feat_dim},
    )
    print("Loaded GNN checkpoint with hyperparameters:", ckpt_meta)

    print("Computing graph-based embeddings...")
    gnn_embeddings = compute_gnn_embeddings(model, graph_dataset)
    print("GNN embeddings shape:", gnn_embeddings.shape)

    print("Evaluating k-NN on GNN embeddings...")
    evaluate_knn(gnn_embeddings, labels_subset, "GNN embeddings", random_state=SEED + 1)

    print("Running t-SNE on GNN embeddings...")
    tsne_learned = compute_tsne(gnn_embeddings, random_state=SEED + 1)
    plot_tsne(tsne_learned, labels_subset, "t-SNE on learned graph embeddings", learned_out)


if __name__ == '__main__':
    main()
