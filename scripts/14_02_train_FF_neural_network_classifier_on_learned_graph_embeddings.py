from pathlib import Path
import json
import os
import time
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
from torch import distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score

RDLogger.DisableLog("rdApp.*")

# ---- Configuration ----
TRAIN_PARQUET = "../data/train/baseline_train.parquet"
GNN_CHECKPOINT = "../models/supcon_gnn_20251008_161101.pt"  # update to actual filename
EMBED_SOURCE = "preproj"  # "preproj" (encoder g) or "proj" (projection z)
EMBED_BATCH_SIZE = 256
EMBED_NUM_WORKERS = 4
epochs = 100

# ---- Graph utilities ----
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
    idx = BOND_MAP.get(bond.GetBondType(), EDGE_FEATURE_DIM - 1)
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


@dataclass
class GraphSample:
    node_feat: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray


class MolecularGraphDataset(Dataset):
    def __init__(self, parquet_path: str, cache_size: int = 25000):
        df = pd.read_parquet(parquet_path)
        self.smiles = df["smiles"].astype(str).tolist()
        self.labels = (df["source"].astype(str) == "PKS").astype(np.int64).to_numpy()
        self.cache_size = cache_size
        self.cache: Dict[str, GraphSample] = {}
        self.cache_order: deque[str] = deque()

        for smi in self.smiles:
            try:
                sample = smiles_to_graph(smi)
            except ValueError:
                continue
            self.node_feat_dim = sample[0].shape[1]
            self.edge_feat_dim = sample[2].shape[1]
            break
        else:
            raise RuntimeError("No valid molecules found in dataset.")

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


def collate_graphs(batch: List[Tuple[GraphSample, int]]) -> Dict[str, torch.Tensor]:
    node_feats = []
    edge_indices = []
    edge_attrs = []
    batch_vec = []
    labels = []
    offset = 0
    for graph, label in batch:
        x = torch.from_numpy(graph.node_feat)
        edge_index = torch.from_numpy(graph.edge_index) + offset
        edge_attr = torch.from_numpy(graph.edge_attr)
        num_nodes = x.size(0)

        node_feats.append(x)
        edge_indices.append(edge_index)
        edge_attrs.append(edge_attr)
        batch_vec.append(torch.full((num_nodes,), len(labels), dtype=torch.long))
        labels.append(label)
        offset += num_nodes

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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
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


def edge_softmax(dst: torch.Tensor, scores: torch.Tensor, num_nodes: int) -> torch.Tensor:
    heads = scores.size(1)
    result = []
    for head in range(heads):
        s = scores[:, head]
        max_vals = torch.full((num_nodes,), -float("inf"), device=s.device)
        max_vals.scatter_reduce_(0, dst, s, reduce="amax")
        s = s - max_vals[dst]
        exp_s = torch.exp(s)
        denom = torch.zeros(num_nodes, device=s.device).scatter_add_(0, dst, exp_s)
        result.append(exp_s / (denom[dst] + 1e-16))
    return torch.stack(result, dim=1)


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
            self.layers.append(
                GraphAttentionLayer(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    heads=head_count,
                    dropout=dropout,
                    edge_feat_dim=edge_feat_dim,
                )
            )
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


# ---- Encoder helpers ----
def build_model(node_dim: int, edge_dim: int, hparams: Dict) -> nn.Module:
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


def load_gnn(checkpoint_path: str, node_dim: int, edge_dim: int, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    hparams = ckpt.get("hyperparameters", {})
    model = build_model(node_dim, edge_dim, hparams)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print("[Warning] Missing keys when loading GNN checkpoint:", missing)
    if unexpected:
        print("[Warning] Unexpected keys when loading GNN checkpoint:", unexpected)
    model.to(device)
    model.eval()
    return model, hparams


def compute_graph_embeddings(
    model: nn.Module,
    dataset: MolecularGraphDataset,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        list(range(len(dataset))),
        batch_size=EMBED_BATCH_SIZE,
        shuffle=False,
        num_workers=EMBED_NUM_WORKERS,
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


# ---- FF classifier ----
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


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.X = torch.from_numpy(embeddings.astype(np.float32)).clone()
        self.y = torch.from_numpy(labels.astype(np.float32)).view(-1, 1).clone()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


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


def cleanup_distributed():
    if is_dist():
        dist.destroy_process_group()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    logits_list = []
    labels_list = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        logits_list.append(logits.detach().cpu().numpy())
        labels_list.append(yb.detach().cpu().numpy())

    logits_all = np.concatenate(logits_list, axis=0)
    labels_all = np.concatenate(labels_list, axis=0).ravel()
    probs = 1.0 / (1.0 + np.exp(-logits_all.ravel()))
    auprc = average_precision_score(labels_all, probs)
    try:
        auroc = roc_auc_score(labels_all, probs)
    except ValueError:
        auroc = float("nan")
    preds = (probs >= 0.5).astype(np.float32)
    acc = accuracy_score(labels_all, preds)
    return auprc, auroc, acc


def main():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = setup_distributed()
    rank = get_rank()
    world = get_world_size()
    print0(f"Using device: {device} | rank={rank} | world_size={world}")

    print0("Loading training parquet...")
    train_graphs = MolecularGraphDataset(TRAIN_PARQUET)
    print0(f"Loaded {len(train_graphs)} molecular graphs.")

    print0(f"Restoring GNN encoder from {GNN_CHECKPOINT} ...")
    encoder, hparams = load_gnn(GNN_CHECKPOINT, train_graphs.node_feat_dim, train_graphs.edge_feat_dim, device)
    print0(
        f"Encoder hidden_dim={hparams.get('hidden_dim', 'n/a')}, embed_dim={hparams.get('embed_dim', 'n/a')}, "
        f"proj_dim={hparams.get('proj_dim', 'n/a')}, heads={hparams.get('heads', 'n/a')}, source={EMBED_SOURCE}"
    )

    print0("Computing graph embeddings for training set...")
    embeddings, labels = compute_graph_embeddings(encoder, train_graphs, device)
    print0("Finished computing graph embeddings.")
    print0(f"Embedding shape: {embeddings.shape}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_ds = EmbeddingDataset(embeddings, labels)
    input_dim = train_ds.X.shape[1]
    del embeddings

    def _suggest_workers(default: int) -> int:
        try:
            sct = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
            if sct > 0:
                return max(1, min(default, sct // 2))
        except Exception:
            pass
        return 1

    batch_size = 2048 if device.type == "cuda" else 512
    nw_train = _suggest_workers(4)
    pin = device.type == "cuda"
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world,
        rank=rank,
        shuffle=True,
        drop_last=False,
    ) if world > 1 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=nw_train,
        pin_memory=pin,
    )

    print0("Initialising feed-forward classifier...")
    model = FFClassifier(input_dim=input_dim).to(device)
    if world > 1:
        print0("Wrapping classifier with DistributedDataParallel...")
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    pos = max(int(labels.sum()), 1)
    neg = max(int(len(labels) - pos), 1)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    print0(f"Class counts (train): pos={pos}, neg={neg}, pos_weight={pos_weight.item():.3f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    print0("Starting training loop...")
    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        loss_tensor = torch.tensor([epoch_loss, n_batches], dtype=torch.float32, device=device)
        if is_dist():
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_steps = int(loss_tensor[1].item())
        avg_loss = loss_tensor[0].item() / max(total_steps, 1)

        if get_rank() == 0:
            dt = time.time() - t0
            print0(f"Epoch {epoch:02d} | loss={avg_loss:.4f} | {dt:.1f}s")

        if is_dist():
            dist.barrier()

    if get_rank() == 0:
        print0("Training complete. Saving classifier checkpoint...")
        target_model = model if world == 1 else model.module

        out_dir = Path("../models")
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "learned_graph_embedding_ffnn_pks_classifier.pt"
        meta_path = out_dir / "learned_graph_embedding_ffnn_pks_classifier.meta.json"
        torch.save(
            {
                "state_dict": target_model.state_dict(),
                "input_dim": input_dim,
                "hidden": [512, 256],
                "pos_weight": pos_weight.item(),
            },
            model_path,
        )

        with open(meta_path, "w") as f:
            json.dump(
                {
                    "model_path": str(model_path),
                    "input_dim": input_dim,
                    "hidden": [512, 256],
                    "batch_size_per_rank": batch_size,
                    "epochs": epochs,
                    "world_size": world,
                    "device": str(device),
                    "gnn_checkpoint": GNN_CHECKPOINT,
                    "encoder_hparams": hparams,
                    "embedding_source": EMBED_SOURCE,
                    "class_counts": {"pos": int(pos), "neg": int(neg)},
                    "pos_weight": float(pos_weight.item()),
                },
                f,
                indent=2,
            )

        print0(f"Saved model to {model_path} and metadata to {meta_path}")

    if is_dist():
        dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
