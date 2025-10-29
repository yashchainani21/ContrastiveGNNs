from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score


# ---- Configuration ----
VAL_NPZ = "../data/val/baseline_val_ecfp4.npz"
ENCODER_CHECKPOINT = "../models/Molecular_ResNet_1024_512_100_epochs_w_weighted_loss.pt"  # match training script
FFNN_CHECKPOINT = "../models/learned_ecfp_resnet_embedding_ffnn_pks_classifier.pt"
FFNN_META = "../models/learned_ecfp_resnet_embedding_ffnn_pks_classifier.meta.json"
EMBED_SOURCE = "preproj"  # "preproj" or "proj"
BATCH_SIZE = 1024
NUM_WORKERS = 4


def load_split_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    fps = data["fps"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    return fps, labels


class fp_CNN_Encoder(nn.Module):
    def __init__(self, fp_dim=2048, hidden_channels=(64, 128),
                 embed_dim=256, proj_dim=120, use_projection=True,
                 batchnorm_safe=True, dropout_p=0.3):
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
        self.fc = nn.Linear(c2, embed_dim)
        self.use_projection = use_projection
        if use_projection:
            self.proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p),
                nn.Linear(embed_dim, proj_dim),
            )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.conv(x).squeeze(-1)
        g = F.normalize(self.fc(h), dim=-1, eps=1e-8)
        if self.use_projection:
            z = F.normalize(self.proj(g), dim=-1, eps=1e-8)
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
        self.shortcut = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

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
        g = F.normalize(self.fc(h), dim=-1, eps=1e-8)
        if self.use_projection:
            h_proj = self.relu(self.proj_fc1(g))
            h_proj = self.dropout(h_proj)
            if self.use_proj_skip:
                h_proj = h_proj + g
            z = F.normalize(self.proj_fc2(h_proj), dim=-1, eps=1e-8)
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
        g = F.normalize(self.fc2(h), dim=-1, eps=1e-8)
        z = F.normalize(self.proj(g), dim=-1, eps=1e-8)
        return g, z


def build_encoder(model_type="resnet", fp_dim=2048, embed_dim=512, proj_dim=256):
    if model_type.lower() == "mlp":
        return TinyMLP(fp_dim=fp_dim, embed_dim=embed_dim, proj_dim=proj_dim)
    if model_type.lower() == "cnn":
        return fp_CNN_Encoder(fp_dim=fp_dim, embed_dim=embed_dim, proj_dim=proj_dim)
    if model_type.lower() == "resnet":
        return fp_CNN_ResNetEncoder(fp_dim=fp_dim, embed_dim=embed_dim, proj_dim=proj_dim)
    raise ValueError(f"Unknown model_type={model_type}")


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


def load_encoder(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt.get("hyperparameters", {})
    model_type = ckpt.get("model_type", "resnet")
    embed_dim = config.get("embed_dim", 512)
    proj_dim = config.get("proj_dim", 256)
    encoder = build_encoder(model_type=model_type, fp_dim=2048, embed_dim=embed_dim, proj_dim=proj_dim)
    missing, unexpected = encoder.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print("[Warning] Missing keys when loading encoder:", missing)
    if unexpected:
        print("[Warning] Unexpected keys when loading encoder:", unexpected)
    encoder.to(device)
    encoder.eval()
    return encoder, model_type, embed_dim, proj_dim


def load_classifier(checkpoint_path: str, meta_path: str, device: torch.device):
    meta = json.loads(Path(meta_path).read_text())
    input_dim = meta["input_dim"]
    model = FFClassifier(input_dim=input_dim)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, meta


def compute_embeddings(encoder: nn.Module, fps: np.ndarray, device: torch.device) -> np.ndarray:
    dataset = TensorDataset(torch.from_numpy(fps))
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    use_proj = EMBED_SOURCE.lower() in {"proj", "z"}
    features = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            outputs = encoder(xb)
            if isinstance(outputs, tuple):
                g, z = outputs
            else:
                g, z = outputs, outputs
            feats = z if use_proj else g
            features.append(feats.cpu().numpy())
    return np.vstack(features)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading validation split...")
    val_fps, val_labels = load_split_npz(VAL_NPZ)
    print("Validation fingerprints:", val_fps.shape)

    print(f"Restoring encoder from {ENCODER_CHECKPOINT} ...")
    encoder, model_type, embed_dim, proj_dim = load_encoder(ENCODER_CHECKPOINT, device)
    print(f"Encoder type={model_type}, embed_dim={embed_dim}, proj_dim={proj_dim}, source={EMBED_SOURCE}")

    print("Computing embeddings for validation set...")
    val_embeddings = compute_embeddings(encoder, val_fps, device)
    print("Embedding shape:", val_embeddings.shape)

    print(f"Loading FFNN classifier from {FFNN_CHECKPOINT} ...")
    classifier, meta = load_classifier(FFNN_CHECKPOINT, FFNN_META, device)
    print("Classifier input_dim:", meta["input_dim"])

    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_embeddings.astype(np.float32)), torch.from_numpy(val_labels.astype(np.float32)).view(-1, 1)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    classifier.eval()
    probs_list = []
    labels_list = []
    with torch.no_grad():
        for xb, yb in val_loader:
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
