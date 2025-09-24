from pathlib import Path
import json
import math
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score


def find_split_file(split: str) -> Path:
    base = Path("../data") / split
    for name in (
        f"baseline_{split}_ecfp4.parquet",
        f"baseline_{split}_ecfp4.csv",
        f"baseline_{split}.parquet",
        f"baseline_{split}.csv",
    ):
        p = base / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find split file for '{split}' under {base}")


def load_split(split: str) -> pd.DataFrame:
    p = find_split_file(split)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    if "smiles" not in df.columns or "source" not in df.columns:
        raise ValueError(f"Expected 'smiles' and 'source' in {p}")
    return df


def get_fp_columns(df: pd.DataFrame) -> List[str]:
    fp_cols = [c for c in df.columns if str(c).startswith("fp_")]
    if not fp_cols:
        raise ValueError("No fingerprint columns found (expected fp_0..fp_2047)")
    return sorted(fp_cols, key=lambda s: int(str(s).split("_")[1]))


class FingerprintDataset(Dataset):
    def __init__(self, df: pd.DataFrame, fp_cols: List[str]):
        X = df[fp_cols].to_numpy(dtype=np.float32)
        y = (df["source"].astype(str) == "PKS").astype(np.float32).to_numpy()
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).view(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FFClassifier(nn.Module):
    def __init__(self, input_dim: int = 2048, hidden1: int = 512, hidden2: int = 256, p_drop: float = 0.2):
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
            nn.Linear(hidden2, 1),  # output logit
        )

    def forward(self, x):
        return self.net(x)


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

    # Metrics
    auprc = average_precision_score(labels_all, probs)
    try:
        auroc = roc_auc_score(labels_all, probs)
    except ValueError:
        auroc = float("nan")  # if only one class present
    preds = (probs >= 0.5).astype(np.float32)
    acc = accuracy_score(labels_all, preds)
    return auprc, auroc, acc


def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load splits
    df_train = load_split("train")
    df_val = load_split("val")
    df_test = load_split("test")
    fp_cols = get_fp_columns(df_train)

    # Datasets and loaders
    train_ds = FingerprintDataset(df_train, fp_cols)
    val_ds = FingerprintDataset(df_val, fp_cols)
    test_ds = FingerprintDataset(df_test, fp_cols)

    batch_size = 4096 if device.type == "cuda" else 1024
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=(device.type=="cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device.type=="cuda"))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device.type=="cuda"))

    # Model
    input_dim = len(fp_cols)
    model = FFClassifier(input_dim=input_dim).to(device)

    # Class imbalance handling via pos_weight in BCEWithLogitsLoss
    y_train = (df_train["source"].astype(str) == "PKS").astype(np.int64)
    pos = max(int(y_train.sum()), 1)
    neg = max(int((1 - y_train).sum()), 1)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    print(f"Class counts (train): pos={pos}, neg={neg}, pos_weight={pos_weight.item():.3f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    best_val_auprc = -1.0
    best_state = None
    epochs = 25
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        # Validation
        val_auprc, val_auroc, val_acc = evaluate(model, val_loader, device)
        avg_loss = epoch_loss / max(n_batches, 1)
        dt = time.time() - t0
        print(f"Epoch {epoch:02d} | loss={avg_loss:.4f} | val_auprc={val_auprc:.4f} | val_auroc={val_auroc:.4f} | val_acc={val_acc:.4f} | {dt:.1f}s")

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_state = {"model": model.state_dict(), "epoch": epoch, "val_auprc": val_auprc}

    # Load best and evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state["model"]) 

    val_auprc, val_auroc, val_acc = evaluate(model, val_loader, device)
    test_auprc, test_auroc, test_acc = evaluate(model, test_loader, device)
    print(f"Best Val AUPRC={val_auprc:.4f} | AUROC={val_auroc:.4f} | ACC={val_acc:.4f}")
    print(f"Test  AUPRC={test_auprc:.4f} | AUROC={test_auroc:.4f} | ACC={test_acc:.4f}")

    # Save model and metadata
    out_dir = Path("../models")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "baseline_ffnn_pks_classifier.pt"
    meta_path = out_dir / "baseline_ffnn_pks_classifier.meta.json"
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden": [512, 256],
        "pos_weight": pos_weight.item(),
        "best_val_auprc": float(val_auprc),
        "best_val_auroc": float(val_auroc),
        "best_val_acc": float(val_acc),
        "test_auprc": float(test_auprc),
        "test_auroc": float(test_auroc),
        "test_acc": float(test_acc),
    }, model_path)

    with open(meta_path, "w") as f:
        json.dump({
            "model_path": str(model_path),
            "input_dim": input_dim,
            "hidden": [512, 256],
            "batch_size": batch_size,
            "epochs": epochs,
            "device": str(device),
            "class_counts": {"pos": int(pos), "neg": int(neg)},
            "pos_weight": float(pos_weight.item()),
            "val_metrics": {"auprc": float(val_auprc), "auroc": float(val_auroc), "acc": float(val_acc)},
            "test_metrics": {"auprc": float(test_auprc), "auroc": float(test_auroc), "acc": float(test_acc)},
        }, f, indent=2)

    print(f"Saved model to {model_path} and metadata to {meta_path}")


if __name__ == "__main__":
    main()

