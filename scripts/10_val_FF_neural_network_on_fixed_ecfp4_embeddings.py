from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score


# ---- Configuration ----
VAL_PARQUET = "../data/val/baseline_val_ecfp4.parquet"  # falls back to .csv if not found
MODEL_CHECKPOINT = "../models/baseline_ffnn_pks_classifier.pt"
BATCH_SIZE = 8192
NUM_WORKERS = 4


def find_val_file() -> Path:
    base = Path("../data/val")
    for name in ("baseline_val_ecfp4.parquet", "baseline_val_ecfp4.csv", "baseline_val.parquet", "baseline_val.csv"):
        candidate = base / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate validation split under {base}")


def load_split(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "source" not in df.columns:
        raise ValueError(f"'source' column not found in {path}")
    return df


def get_fp_columns(df: pd.DataFrame) -> list[str]:
    fp_cols = [c for c in df.columns if str(c).startswith("fp_")]
    if not fp_cols:
        raise ValueError("No fingerprint columns (fp_*) found in validation dataframe")
    return sorted(fp_cols, key=lambda s: int(str(s).split("_")[1]))


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


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    logits_list = []
    labels_list = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        logits_list.append(logits.detach().cpu().numpy())
        labels_list.append(yb.detach().cpu().numpy())

    logits = np.concatenate(logits_list, axis=0).ravel()
    labels = np.concatenate(labels_list, axis=0).ravel()
    probs = 1.0 / (1.0 + np.exp(-logits))
    auprc = average_precision_score(labels, probs)
    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError:
        auroc = float("nan")
    preds = (probs >= 0.5).astype(np.float32)
    acc = accuracy_score(labels, preds)
    return auprc, auroc, acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    val_path = find_val_file()
    print(f"Loading validation split from {val_path} ...")
    df_val = load_split(val_path)
    fp_cols = get_fp_columns(df_val)

    fps = df_val[fp_cols].to_numpy(dtype=np.float32)
    labels = (df_val["source"].astype(str) == "PKS").astype(np.float32).to_numpy()
    print("Validation data:", fps.shape)

    print(f"Loading classifier checkpoint from {MODEL_CHECKPOINT} ...")
    ckpt = torch.load(MODEL_CHECKPOINT, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    first_weight_key = next(k for k in state_dict if k.endswith("0.weight"))
    input_dim = state_dict[first_weight_key].shape[1]
    if fps.shape[1] != input_dim:
        if fps.shape[1] > input_dim:
            print(
                f"Warning: validation fingerprints have {fps.shape[1]} dims but model expects {input_dim}. "
                f"Truncating to the first {input_dim} columns to match training configuration."
            )
            fps = fps[:, :input_dim]
        else:
            raise ValueError(
                f"Validation fingerprints have {fps.shape[1]} dims which is fewer than model input {input_dim}."
            )
    model = FFClassifier(input_dim=input_dim)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(fps), torch.from_numpy(labels).view(-1, 1)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    auprc, auroc, acc = evaluate(model, loader, device)
    print(f"Validation AUPRC={auprc:.4f} | AUROC={auroc:.4f} | ACC={acc:.4f}")


if __name__ == "__main__":
    main()
