from pathlib import Path
import os
import json
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
import multiprocessing as mp


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


# Globals for worker processes (to avoid pickling large arrays for each task)
_X_train = None
_train_counts = None
_train_labels = None
_train_smiles = None


def _init_worker(X_train: np.ndarray, train_counts: np.ndarray, train_labels: np.ndarray, train_smiles: np.ndarray):
    global _X_train, _train_counts, _train_labels, _train_smiles
    _X_train = X_train
    _train_counts = train_counts
    _train_labels = train_labels
    _train_smiles = train_smiles


def _predict_nn_for_indices(args: Tuple[np.ndarray, np.ndarray]):
    idxs, X_test_chunk = args
    preds = np.zeros(len(idxs), dtype=np.int8)
    sims = np.zeros(len(idxs), dtype=np.float32)
    nn_smiles = []

    # Compute similarity to all train entries using dot products
    # X_train: (N_train, d) uint8; test: (d,) uint8
    for i, x in enumerate(X_test_chunk):
        test_bits = x.astype(np.uint8)
        inter = _X_train.dot(test_bits.astype(np.int32))  # (N_train,)
        test_count = int(test_bits.sum())
        union = _train_counts + test_count - inter
        # Avoid division by zero
        union = np.maximum(union, 1)
        tanimoto = inter / union
        j = int(np.argmax(tanimoto))
        preds[i] = _train_labels[j]
        sims[i] = float(tanimoto[j])
        nn_smiles.append(_train_smiles[j])

    return idxs, preds, sims, nn_smiles


def main():
    # Load splits
    df_train = load_split("train")
    df_test = load_split("test")
    fp_cols = get_fp_columns(df_train)

    # Optional caps for speed (set env SIM_TRAIN_LIMIT / SIM_TEST_LIMIT)
    # Default: cap BOTH train and test to 10,000 rows for quick checks
    train_limit = int(os.environ.get("SIM_TRAIN_LIMIT", "10000")) or None
    test_limit = int(os.environ.get("SIM_TEST_LIMIT", "15000")) or None

    if train_limit and len(df_train) > train_limit:
        df_train = df_train.iloc[:train_limit].copy()
        print(f"Capped train to {train_limit:,} rows")
    if test_limit and len(df_test) > test_limit:
        df_test = df_test.iloc[:test_limit].copy()
        print(f"Capped test to {test_limit:,} rows")

    # Prepare arrays
    X_train = df_train[fp_cols].to_numpy(dtype=np.uint8)
    y_train = (df_train["source"].astype(str) == "PKS").astype(np.int8).to_numpy()
    smiles_train = df_train["smiles"].astype(str).to_numpy()
    train_counts = X_train.sum(axis=1).astype(np.int32)

    X_test = df_test[fp_cols].to_numpy(dtype=np.uint8)
    y_test = (df_test["source"].astype(str) == "PKS").astype(np.int8).to_numpy()
    smiles_test = df_test["smiles"].astype(str).to_numpy()

    print(f"Train: {len(X_train):,} | Test: {len(X_test):,} | Dim: {X_train.shape[1]}")

    # Multiprocessing over test indices
    processes = int(os.environ.get("SIM_PROCESSES", str(max(1, mp.cpu_count() - 1))))
    chunksize = int(os.environ.get("SIM_CHUNKSIZE", "256"))

    # Build chunks of indices
    N = len(X_test)
    indices = np.arange(N, dtype=np.int64)
    chunks = [indices[i:i + chunksize] for i in range(0, N, chunksize)]

    preds = np.zeros(N, dtype=np.int8)
    sims = np.zeros(N, dtype=np.float32)
    nn_smiles_all = [None] * N

    # Prefer 'fork' to avoid copying large arrays; fallback to 'spawn' with fewer processes
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context("spawn")

    # Set globals before forking so children inherit without pickling
    global _X_train, _train_counts, _train_labels, _train_smiles
    _X_train = X_train
    _train_counts = train_counts
    _train_labels = y_train
    _train_smiles = smiles_train

    with ctx.Pool(processes=processes) as pool:
        tasks = []
        for idx_chunk in chunks:
            X_chunk = X_test[idx_chunk]
            tasks.append((idx_chunk, X_chunk))

        for idxs, preds_chunk, sims_chunk, nn_smiles in pool.imap_unordered(_predict_nn_for_indices, tasks):
            preds[idxs] = preds_chunk
            sims[idxs] = sims_chunk
            for k, s in zip(idxs, nn_smiles):
                nn_smiles_all[int(k)] = s

    # Metrics
    try:
        auprc = average_precision_score(y_test, preds)
    except Exception:
        auprc = float("nan")
    try:
        auroc = roc_auc_score(y_test, preds)
    except Exception:
        auroc = float("nan")
    acc = accuracy_score(y_test, preds)
    print(f"Chemical NN baseline — ACC={acc:.4f} | AUROC={auroc:.4f} | AUPRC={auprc:.4f}")

    # Save results
    out_dir = Path("../data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chem_similarity_nn_baseline.parquet"
    out_df = pd.DataFrame({
        "smiles": df_test["smiles"].astype(str).to_numpy(),
        "true_label": np.where(y_test == 1, "PKS", "non-PKS"),
        "pred_label": np.where(preds == 1, "PKS", "non-PKS"),
        "nn_similarity": sims,
        "nn_train_smiles": nn_smiles_all,
    })
    out_df.to_parquet(out_path, index=False)
    print(f"Saved per-molecule results to {out_path}")

    # Also save summary metrics to models folder (e.g., for tracking baselines)
    models_dir = Path("../models")
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = models_dir / "chem_similarity_nn_baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "acc": float(acc),
            "auroc": float(auroc) if not (isinstance(auroc, float) and (auroc != auroc)) else None,
            "auprc": float(auprc) if not (isinstance(auprc, float) and (auprc != auprc)) else None,
            "n_test": int(len(y_test)),
            "n_train": int(len(y_train)),
        }, f, indent=2)
    print(f"Saved summary metrics to {metrics_path}")


if __name__ == "__main__":
    main()
