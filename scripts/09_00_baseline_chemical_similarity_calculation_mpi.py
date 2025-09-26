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

    # Prepare arrays (bit-packed for memory-efficient popcount)
    X_train_bits = df_train[fp_cols].to_numpy(dtype=np.uint8)
    y_train = (df_train["source"].astype(str) == "PKS").astype(np.int8).to_numpy()
    smiles_train = df_train["smiles"].astype(str).to_numpy()

    X_test_bits = df_test[fp_cols].to_numpy(dtype=np.uint8)
    y_test = (df_test["source"].astype(str) == "PKS").astype(np.int8).to_numpy()
    smiles_test = df_test["smiles"].astype(str).to_numpy()

    # Lookup table for popcount(0..255)
    LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    # Pack bits along feature axis to reduce memory 8x
    X_train_packed = np.packbits(X_train_bits, axis=1)
    # Precompute bit counts per train row
    train_counts = LUT[X_train_packed].sum(axis=1).astype(np.int32)

    # Use the correct test array (bit-packed input names changed above)
    N = len(X_test_bits)
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

    # Blocked nearest-neighbor search using packed bits and LUT popcount
    BLOCK = int(os.environ.get("SIM_TRAIN_BLOCK", "100000"))  # rows per train block
    n_train = X_train_packed.shape[0]
    for i, k in enumerate(indices):
        test_bits = X_test_bits[k].astype(np.uint8)
        test_packed = np.packbits(test_bits)
        tcount = int(LUT[test_packed].sum())

        best_sim = -1.0
        best_j = -1
        # Iterate over train in blocks to bound memory
        for start in range(0, n_train, BLOCK):
            end = min(start + BLOCK, n_train)
            block_and = X_train_packed[start:end] & test_packed  # (B, packed_dim)
            inter_block = LUT[block_and].sum(axis=1).astype(np.int32)  # (B,)
            union_block = train_counts[start:end] + tcount - inter_block
            union_block = np.maximum(union_block, 1)
            tanimoto_block = inter_block / union_block
            j_local = int(np.argmax(tanimoto_block))
            if tanimoto_block[j_local] > best_sim:
                best_sim = float(tanimoto_block[j_local])
                best_j = start + j_local

        preds[i] = y_train[best_j]
        sims[i] = best_sim
        nn_smiles.append(smiles_train[best_j])

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

    # NOTE: Do not use collectives (e.g., Barrier or Gather). If any rank fails,
    # the communicator often becomes unusable and the job aborts. By writing
    # per-rank shards eagerly, partial progress is preserved even on node failure.
    # Use the standalone merge script to combine available shards afterwards.
    if rank == 0:
        print("Rank 0: Skipping in-job merge to be resilient to rank failures.")
        print("Use scripts/09_00_merge_chem_similarity_shards.py to merge existing shards later.")
