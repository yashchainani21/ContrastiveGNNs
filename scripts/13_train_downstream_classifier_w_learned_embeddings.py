import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from joblib import dump


# ---- Hyperparameters ----
MODEL_TYPE = None        # 'resnet' | 'cnn' | 'mlp'; None = load from checkpoint
EMBED_DIM = None         # override encoder embed dim; None = load from checkpoint metadata
PROJ_DIM = None          # override projection dim; None = load from checkpoint metadata
EMBED_SOURCE = 'preproj'  # 'preproj' (g) or 'proj' (z) for downstream features
BATCH_SIZE = 8192
NUM_WORKERS = 4

# ---- Paths ----
MODEL_CHECKPOINT = '../models/supcon_ddp_resnet_20251006_100411.pt'  # update to actual filename
TRAIN_NPZ = '../data/train/baseline_train_ecfp4.npz'
OUTPUT_EMBEDS = '../models/train_embeddings.npy'
OUTPUT_LABELS = '../models/train_labels.npy'
OUTPUT_CLF = '../models/downstream_logreg.pkl'


# ---- Model Definitions (must match training script) ----
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


# ---- Load checkpoint ----

checkpoint = torch.load(MODEL_CHECKPOINT, map_location='cpu')
config = checkpoint.get('hyperparameters', {})
model_type = MODEL_TYPE or checkpoint.get('model_type', 'resnet')
embed_dim = EMBED_DIM or config.get('embed_dim', 512)
proj_dim = PROJ_DIM or config.get('proj_dim', 256)
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


# ---- Load data ----

data = np.load(TRAIN_NPZ, allow_pickle=False)
fps = data['fps'].astype(np.float32)
labels = data['labels'].astype(np.int64)
use_preproj = EMBED_SOURCE.lower() in {'preproj', 'encoder', 'g'}
print("Loaded training set:", fps.shape)
print(f"Model type: {model_type}, embed_dim: {embed_dim}, proj_dim: {proj_dim}")
print(f"Using {'pre-projection' if use_preproj else 'projection'} embeddings for downstream training")


# ---- Compute embeddings ----
tensor_dataset = TensorDataset(torch.from_numpy(fps))
loader = DataLoader(
    tensor_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=NUM_WORKERS > 0,
)

embeddings = []
with torch.no_grad():
    for (xb,) in loader:
        xb = xb.to(device, non_blocking=True)
        outputs = model(xb)
        if isinstance(outputs, tuple):
            g, z = outputs
        else:
            g, z = outputs, outputs
        features = g if use_preproj else z
        embeddings.append(features.cpu().numpy())

embeddings = np.vstack(embeddings)
np.save(OUTPUT_EMBEDS, embeddings)
np.save(OUTPUT_LABELS, labels)
print("Saved training embeddings to", OUTPUT_EMBEDS)


# ---- Train downstream classifier ----

clf = LogisticRegression(max_iter=10_000, class_weight='balanced')
clf.fit(embeddings, labels)
preds = clf.predict_proba(embeddings)[:, 1]
auprc = average_precision_score(labels, preds)
print(f"LogReg (train) AUPRC={auprc:.4f}")

os.makedirs(os.path.dirname(OUTPUT_CLF), exist_ok=True)
dump(clf, OUTPUT_CLF)
print(f"Downstream classifier saved to {OUTPUT_CLF}")
