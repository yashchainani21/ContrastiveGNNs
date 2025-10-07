import os
import pathlib
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE


# ---- Configuration ----
TRAIN_NPZ = '../data/train/baseline_train_ecfp4.npz'
MODEL_CHECKPOINT = '../models/Molecular_ResNet_1024_512_50_epochs.pt'  # update to actual filename
OUTPUT_DIR = '../figures'
RAW_TSNE_FILENAME = 'tsne_ecfp4.png'
LEARNED_TSNE_FILENAME = 'tsne_learned.png'
NEGATIVE_SAMPLE_SIZE = 50_000
EMBED_SOURCE = 'preproj'  # 'preproj' (g) or 'proj' (z)
SEED = 42
BATCH_SIZE = 4096
NUM_WORKERS = 4
TSNE_PERPLEXITY = 30.0

# ---- Model Definitions (must mirror training script) ----
class fp_CNN_Encoder(nn.Module):
    def __init__(self, fp_dim=2048, hidden_channels=(64, 128), embed_dim=256,
                 proj_dim=120, use_projection=True, batchnorm_safe=True,
                 dropout_p=0.3):
        super().__init__()
        c1, c2 = hidden_channels
        self.conv = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=5, padding=2),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(1),
        )
        self.fc = nn.Linear(c2, 256)
        self.use_projection = use_projection
        if use_projection:
            self.proj = nn.Linear(256, proj_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.conv(x).squeeze(-1)
        g = F.normalize(self.fc(h), dim=-1)
        if self.use_projection:
            z = F.normalize(self.proj(g), dim=-1)
            return g, z
        return g, g


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, stride=1, padding=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.shortcut = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))


class fp_CNN_ResNetEncoder(nn.Module):
    def __init__(self, fp_dim=2048, hidden_channels=(128, 256), embed_dim=128,
                 proj_dim=64, use_projection=True, batchnorm_safe=True,
                 dropout_p=0.2, use_proj_skip=True):
        super().__init__()
        c1, c2 = hidden_channels
        self.stem = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualBlock(c1, c1)
        self.layer2 = ResidualBlock(c1, c2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c2, embed_dim)
        self.use_projection = use_projection
        self.use_proj_skip = use_proj_skip
        if self.use_projection:
            self.proj_fc1 = nn.Linear(embed_dim, embed_dim)
            self.proj_fc2 = nn.Linear(embed_dim, proj_dim)
            self.relu = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.pool(h).squeeze(-1)
        g = F.normalize(self.fc(h), dim=-1)
        if self.use_projection:
            h_proj = self.relu(self.proj_fc1(g))
            h_proj = self.dropout(h_proj)
            if self.use_proj_skip:
                h_proj = h_proj + g
            z = F.normalize(self.proj_fc2(h_proj), dim=-1)
            return g, z
        return g, g


class TinyMLP(nn.Module):
    def __init__(self, fp_dim, embed_dim=512, proj_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(fp_dim, 256)
        self.fc2 = nn.Linear(256, embed_dim)
        self.proj = nn.Linear(embed_dim, proj_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        g = F.normalize(self.fc2(h), dim=-1)
        z = F.normalize(self.proj(g), dim=-1)
        return g, z


def build_model(model_type="resnet", fp_dim=2048, embed_dim=512, proj_dim=256):
    if model_type.lower() == "mlp":
        return TinyMLP(fp_dim=fp_dim, embed_dim=embed_dim, proj_dim=proj_dim)
    if model_type.lower() == "cnn":
        return fp_CNN_Encoder(fp_dim=fp_dim, embed_dim=embed_dim, proj_dim=proj_dim)
    if model_type.lower() == "resnet":
        return fp_CNN_ResNetEncoder(fp_dim=fp_dim, embed_dim=embed_dim, proj_dim=proj_dim)
    raise ValueError(f"Unknown model_type={model_type}")


# ---- Helpers ----
def ensure_output_dir(path: str) -> pathlib.Path:
    output_path = pathlib.Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def select_subset(fps: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_fps = fps[pos_mask]
    pos_labels = labels[pos_mask]

    neg_indices = np.where(neg_mask)[0]
    if NEGATIVE_SAMPLE_SIZE >= len(neg_indices):
        sampled_neg_indices = neg_indices
    else:
        sampled_neg_indices = rng.choice(neg_indices, size=NEGATIVE_SAMPLE_SIZE, replace=False)
    neg_fps = fps[sampled_neg_indices]
    neg_labels = labels[sampled_neg_indices]

    subset_fps = np.concatenate([pos_fps, neg_fps], axis=0)
    subset_labels = np.concatenate([pos_labels, neg_labels], axis=0)

    perm = rng.permutation(len(subset_labels))
    return subset_fps[perm], subset_labels[perm]


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
    label_mask = labels == 1
    plt.scatter(
        points[~label_mask, 0],
        points[~label_mask, 1],
        c='#1f77b4',
        alpha=0.4,
        s=10,
        label='Non-polyketide (label=0)',
    )
    plt.scatter(
        points[label_mask, 0],
        points[label_mask, 1],
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


def load_model() -> Tuple[nn.Module, str, int, int]:
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location='cpu')
    config = checkpoint.get('hyperparameters', {})
    model_type = checkpoint.get('model_type', 'resnet')
    embed_dim = config.get('embed_dim', 512)
    proj_dim = config.get('proj_dim', 256)
    model = build_model(model_type=model_type, fp_dim=2048,
                        embed_dim=embed_dim, proj_dim=proj_dim)
    missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if missing:
        print("[Warning] Missing keys when loading checkpoint:", missing)
    if unexpected:
        print("[Warning] Unexpected keys when loading checkpoint:", unexpected)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return model, model_type, embed_dim, proj_dim


def compute_learned_embeddings(model: nn.Module, fps_subset: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    tensor_dataset = torch.utils.data.TensorDataset(torch.from_numpy(fps_subset))
    loader = torch.utils.data.DataLoader(
        tensor_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )
    use_preproj = EMBED_SOURCE.lower() in {'preproj', 'encoder', 'g'}
    features = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            outputs = model(xb)
            if isinstance(outputs, tuple):
                g, z = outputs
            else:
                g, z = outputs, outputs
            batch_feats = g if use_preproj else z
            features.append(batch_feats.cpu().numpy())
    return np.vstack(features)


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    ensure_output_dir(OUTPUT_DIR)
    raw_output_path = pathlib.Path(OUTPUT_DIR) / RAW_TSNE_FILENAME
    learned_output_path = pathlib.Path(OUTPUT_DIR) / LEARNED_TSNE_FILENAME

    data = np.load(TRAIN_NPZ, allow_pickle=False)
    fps = data['fps'].astype(np.float32)
    labels = data['labels'].astype(np.int64)
    print("Loaded training fingerprints:", fps.shape)

    fps_subset, labels_subset = select_subset(fps, labels)
    print("Subset size:", fps_subset.shape[0])
    print("Class distribution - positives:", (labels_subset == 1).sum(),
          "negatives:", (labels_subset == 0).sum())

    print("Running t-SNE on raw ECFP4 fingerprints...")
    tsne_raw = compute_tsne(fps_subset, random_state=SEED)
    plot_tsne(tsne_raw, labels_subset, "t-SNE on baseline ECFP4 fingerprints", raw_output_path)

    print("Loading encoder checkpoint...")
    model, model_type, embed_dim, proj_dim = load_model()
    print(f"Loaded model type: {model_type}, embed_dim: {embed_dim}, proj_dim: {proj_dim}")

    print("Computing learned embeddings...")
    learned_features = compute_learned_embeddings(model, fps_subset)
    print("Embeddings shape:", learned_features.shape)

    print("Running t-SNE on learned embeddings...")
    tsne_learned = compute_tsne(learned_features, random_state=SEED + 1)
    plot_tsne(tsne_learned, labels_subset, "t-SNE on learned molecular embeddings", learned_output_path)


if __name__ == '__main__':
    main()
