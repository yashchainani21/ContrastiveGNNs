from pathlib import Path
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from mpi4py import MPI
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit import RDLogger
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

RDLogger.DisableLog('rdApp.*')


def find_split_file(split: str) -> Path:
    """Return the non-fingerprint baseline split file (parquet preferred)."""
    base = Path("../data") / split
    for name in (
        f"baseline_{split}.parquet",
        f"baseline_{split}.csv",
    ):
        p = base / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find non-ECFP split for '{split}' under {base}")


def load_split(split: str) -> pd.DataFrame:
    p = find_split_file(split)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    if "smiles" not in df.columns or "source" not in df.columns:
        raise ValueError(f"Expected 'smiles' and 'source' in {p}")
    return df


def smiles_to_mol(smi: str) -> Optional[Chem.Mol]:
    try:
        mol = Chem.MolFromSmiles(smi)
        return mol
    except Exception:
        return None


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    df_train = load_split("train")
    df_test = load_split("test")

    # Default caps for safety; override with env vars
    train_limit = int(os.environ.get("SIM_TRAIN_LIMIT", "1000")) or None
    test_limit = int(os.environ.get("SIM_TEST_LIMIT", "1000")) or None
    if train_limit and len(df_train) > train_limit:
        df_train = df_train.iloc[:train_limit].copy()
        if rank == 0:
            print(f"Capped train to {train_limit:,} rows")
    if test_limit and len(df_test) > test_limit:
        df_test = df_test.iloc[:test_limit].copy()
        if rank == 0:
            print(f"Capped test to {test_limit:,} rows")

    # Build molecules; drop invalids
    train_smiles = df_train["smiles"].astype(str).to_numpy()
    y_train = (df_train["source"].astype(str) == "PKS").astype(np.int8).to_numpy()
    train_mols = []
    keep_idx = []
    for i, smi in enumerate(train_smiles):
        m = smiles_to_mol(smi)
        if m is not None:
            train_mols.append(m)
            keep_idx.append(i)
    train_smiles = train_smiles[keep_idx]
    y_train = y_train[keep_idx]

    test_smiles = df_test["smiles"].astype(str).to_numpy()
    y_test = (df_test["source"].astype(str) == "PKS").astype(np.int8).to_numpy()
    test_mols = []
    keep_t = []
    for i, smi in enumerate(test_smiles):
        m = smiles_to_mol(smi)
        if m is not None:
            test_mols.append(m)
            keep_t.append(i)
    test_smiles = test_smiles[keep_t]
    y_test = y_test[keep_t]

    # Split test indices across ranks
    N = len(test_mols)
    if rank == 0:
        print(f"MPI ranks={size} | Train: {len(train_mols):,} | Test: {N:,}")
    indices = np.array_split(np.arange(N, dtype=np.int64), size)[rank]
    print(f"[Rank {rank}] processing {len(indices)} test molecules", flush=True)

    # MCS parameters
    timeout = float(os.environ.get("MCS_TIMEOUT", "1.0"))
    bond_cmp = Chem.rdFMCS.BondCompare.CompareOrderExact

    # For each assigned test mol, find most similar train by MCS score
    preds = np.zeros(len(indices), dtype=np.int8)
    scores = np.zeros(len(indices), dtype=np.float32)
    nn_smiles = []

    for i, k in enumerate(indices):
        tm = test_mols[int(k)]
        na = tm.GetNumAtoms()
        best_s = -1.0
        best_j = -1
        for j, trm in enumerate(train_mols):
            nb = trm.GetNumAtoms()
            try:
                res = rdFMCS.FindMCS([trm, tm], timeout=timeout, matchValences=True,
                                      matchChiralTag=False, bondCompare=bond_cmp)
                if res.canceled:
                    s = 0.0
                else:
                    common = float(res.numAtoms)
                    denom = float(na + nb - res.numAtoms)
                    s = common / denom if denom > 0 else 0.0
            except Exception:
                s = 0.0
            if s > best_s:
                best_s = s
                best_j = j

        preds[i] = y_train[best_j]
        scores[i] = best_s
        nn_smiles.append(train_smiles[best_j])

    # Write per-rank shard; avoid collectives for resilience
    out_dir = Path("../data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    shard = out_dir / f"mcs_similarity_nn_baseline_mpi.rank{rank}.parquet"
    out_df = pd.DataFrame({
        "index": indices,
        "smiles": test_smiles[indices],
        "true_label": np.where(y_test[indices] == 1, "PKS", "non-PKS"),
        "pred_label": np.where(preds == 1, "PKS", "non-PKS"),
        "mcs_score": scores,
        "nn_train_smiles": nn_smiles,
    })
    out_df.to_parquet(shard, index=False)
    print(f"[Rank {rank}] wrote {len(out_df)} rows to {shard}")

    if rank == 0:
        print("Rank 0: Skipping in-job merge for resilience. Use a merge script to combine shards later.")
