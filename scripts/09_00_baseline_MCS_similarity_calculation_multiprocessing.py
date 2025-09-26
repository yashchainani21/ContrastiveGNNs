from pathlib import Path
import os
import json
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import multiprocessing as mp
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
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


# Globals for worker processes
_train_mols: List[Chem.Mol] | None = None
_train_labels: np.ndarray | None = None
_train_smiles: np.ndarray | None = None
_timeout: float = 1.0
_bond_cmp = Chem.rdFMCS.BondCompare.CompareOrderExact


def _init_globals(train_mols, train_labels, train_smiles, timeout: float):
    global _train_mols, _train_labels, _train_smiles, _timeout
    _train_mols = train_mols
    _train_labels = train_labels
    _train_smiles = train_smiles
    _timeout = timeout


def _predict_nn_for_indices(args: Tuple[np.ndarray, List[Chem.Mol]]):
    idxs, test_mols_chunk = args
    preds = np.zeros(len(idxs), dtype=np.int8)
    scores = np.zeros(len(idxs), dtype=np.float32)
    nn_smiles: List[str] = []

    for i, tm in enumerate(test_mols_chunk):
        if tm is None:
            preds[i] = 0
            scores[i] = 0.0
            nn_smiles.append("")
            continue
        na = tm.GetNumAtoms()
        best_s = -1.0
        best_j = -1
        for j, trm in enumerate(_train_mols or []):
            if trm is None:
                continue
            nb = trm.GetNumAtoms()
            try:
                res = rdFMCS.FindMCS([trm, tm], timeout=_timeout, matchValences=True,
                                      matchChiralTag=False, bondCompare=_bond_cmp)
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

        preds[i] = int((_train_labels or np.array([0]))[best_j]) if best_j >= 0 else 0
        scores[i] = best_s if best_j >= 0 else 0.0
        nn_smiles.append(((_train_smiles or np.array([""]))[best_j]) if best_j >= 0 else "")

    return idxs, preds, scores, nn_smiles


def main():
    # Load splits
    df_train = load_split("train")
    df_test = load_split("test")

    # Default caps; override with env
    train_limit = int(os.environ.get("SIM_TRAIN_LIMIT", "1000")) or None
    test_limit = int(os.environ.get("SIM_TEST_LIMIT", "1000")) or None
    if train_limit and len(df_train) > train_limit:
        df_train = df_train.iloc[:train_limit].copy()
        print(f"Capped train to {train_limit:,} rows")
    if test_limit and len(df_test) > test_limit:
        df_test = df_test.iloc[:test_limit].copy()
        print(f"Capped test to {test_limit:,} rows")

    # Convert to mols
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

    N = len(test_mols)
    print(f"Multiprocessing MCS baseline | Train: {len(train_mols):,} | Test: {N:,}")

    # Tasks
    chunksize = int(os.environ.get("SIM_CHUNKSIZE", "128"))
    processes = int(os.environ.get("SIM_PROCESSES", str(max(1, mp.cpu_count() - 1))))
    timeout = float(os.environ.get("MCS_TIMEOUT", "1.0"))

    indices = np.arange(N, dtype=np.int64)
    chunks = [indices[i:i + chunksize] for i in range(0, N, chunksize)]

    preds = np.zeros(N, dtype=np.int8)
    scores = np.zeros(N, dtype=np.float32)
    nn_smiles_all: List[str] = [""] * N

    # Prefer fork when available to share large objects; fallback to spawn otherwise
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context("spawn")

    # Set globals before forking so children inherit without pickling
    global _train_mols, _train_labels, _train_smiles, _timeout
    _train_mols = train_mols
    _train_labels = y_train
    _train_smiles = train_smiles
    _timeout = timeout

    with ctx.Pool(processes=processes) as pool:
        tasks = []
        for idx_chunk in chunks:
            test_chunk = [test_mols[int(k)] for k in idx_chunk]
            tasks.append((idx_chunk, test_chunk))

        for idxs, preds_chunk, scores_chunk, nn_smiles in pool.imap_unordered(_predict_nn_for_indices, tasks):
            preds[idxs] = preds_chunk
            scores[idxs] = scores_chunk
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
    print(f"MCS NN baseline — ACC={acc:.4f} | AUROC={auroc:.4f} | AUPRC={auprc:.4f}")

    # Save per-molecule results
    out_dir = Path("../data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mcs_similarity_nn_baseline.parquet"
    out_df = pd.DataFrame({
        "smiles": test_smiles,
        "true_label": np.where(y_test == 1, "PKS", "non-PKS"),
        "pred_label": np.where(preds == 1, "PKS", "non-PKS"),
        "mcs_score": scores,
        "nn_train_smiles": nn_smiles_all,
    })
    out_df.to_parquet(out_path, index=False)
    print(f"Saved per-molecule results to {out_path}")

    # Save summary metrics in models folder
    models_dir = Path("../models")
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = models_dir / "mcs_similarity_nn_baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "acc": float(acc),
            "auroc": None if (isinstance(auroc, float) and (auroc != auroc)) else float(auroc),
            "auprc": None if (isinstance(auprc, float) and (auprc != auprc)) else float(auprc),
            "n_test": int(len(y_test)),
            "n_train": int(len(y_train)),
            "timeout": timeout,
        }, f, indent=2)
    print(f"Saved summary metrics to {metrics_path}")


if __name__ == "__main__":
    main()

