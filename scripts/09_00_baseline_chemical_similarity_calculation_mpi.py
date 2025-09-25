from pathlib import Path
import os
from typing import List

import numpy as np
import pandas as pd
from mpi4py import MPI
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score


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


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # All ranks load the same splits from shared FS (avoids broadcasting huge arrays)
    df_train = load_split("train")
    df_test = load_split("test")
    fp_cols = get_fp_columns(df_train)

    # Optional caps via env to control workload
    # Default: cap BOTH train and test to 10,000 rows for quick checks
    train_limit = int(os.environ.get("SIM_TRAIN_LIMIT", "10000")) or None
    test_limit = int(os.environ.get("SIM_TEST_LIMIT", "10000")) or None
    if train_limit and len(df_train) > train_limit:
        df_train = df_train.iloc[:train_limit].copy()
        if rank == 0:
            print(f"Capped train to {train_limit:,} rows")
    if test_limit and len(df_test) > test_limit:
        df_test = df_test.iloc[:test_limit].copy()
        if rank == 0:
            print(f"Capped test to {test_limit:,} rows")

    # Prepare arrays
    X_train = df_train[fp_cols].to_numpy(dtype=np.uint8)
    y_train = (df_train["source"].astype(str) == "PKS").astype(np.int8).to_numpy()
    smiles_train = df_train["smiles"].astype(str).to_numpy()
    train_counts = X_train.sum(axis=1).astype(np.int32)

    X_test = df_test[fp_cols].to_numpy(dtype=np.uint8)
    y_test = (df_test["source"].astype(str) == "PKS").astype(np.int8).to_numpy()
    smiles_test = df_test["smiles"].astype(str).to_numpy()

    N = len(X_test)
    if rank == 0:
        print(f"MPI ranks={size} | Train: {len(X_train):,} | Test: {N:,} | Dim: {X_train.shape[1]}")

    # Split test indices across ranks
    indices = np.array_split(np.arange(N, dtype=np.int64), size)[rank]
    my_n = len(indices)
    print(f"[Rank {rank}] processing {my_n} test molecules", flush=True)

    # Compute nearest neighbor prediction per assigned test index
    preds = np.zeros(my_n, dtype=np.int8)
    sims = np.zeros(my_n, dtype=np.float32)
    nn_smiles = []

    for i, k in enumerate(indices):
        test_bits = X_test[k].astype(np.uint8)
        inter = X_train.dot(test_bits.astype(np.int32))  # (N_train,)
        tcount = int(test_bits.sum())
        union = train_counts + tcount - inter
        union = np.maximum(union, 1)
        tanimoto = inter / union
        j = int(np.argmax(tanimoto))
        preds[i] = y_train[j]
        sims[i] = float(tanimoto[j])
        nn_smiles.append(smiles_train[j])

    # Write per-rank shard to avoid large gathers
    out_dir = Path("../data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    shard = out_dir / f"chem_similarity_nn_baseline_mpi.rank{rank}.parquet"
    out_df = pd.DataFrame({
        "index": indices,
        "smiles": smiles_test[indices],
        "true_label": np.where(y_test[indices] == 1, "PKS", "non-PKS"),
        "pred_label": np.where(preds == 1, "PKS", "non-PKS"),
        "nn_similarity": sims,
        "nn_train_smiles": nn_smiles,
    })
    out_df.to_parquet(shard, index=False)
    print(f"[Rank {rank}] wrote {len(out_df)} rows to {shard}")

    comm.Barrier()

    if rank == 0:
        # Merge shards
        shards = sorted(out_dir.glob("chem_similarity_nn_baseline_mpi.rank*.parquet"))
        parts = [pd.read_parquet(p) for p in shards]
        merged = pd.concat(parts, ignore_index=True).sort_values("index")

        # Compute metrics on merged
        y_true = (merged["true_label"] == "PKS").astype(np.int8).to_numpy()
        y_pred = (merged["pred_label"] == "PKS").astype(np.int8).to_numpy()
        acc = accuracy_score(y_true, y_pred)
        try:
            auroc = roc_auc_score(y_true, y_pred)
        except Exception:
            auroc = float("nan")
        try:
            auprc = average_precision_score(y_true, y_pred)
        except Exception:
            auprc = float("nan")
        print(f"Chemical NN baseline (MPI) — ACC={acc:.4f} | AUROC={auroc:.4f} | AUPRC={auprc:.4f}")

        final_path = out_dir / "chem_similarity_nn_baseline_mpi.parquet"
        merged.drop(columns=["index"], inplace=True)
        merged.to_parquet(final_path, index=False)
        print(f"Saved merged results to {final_path}")
