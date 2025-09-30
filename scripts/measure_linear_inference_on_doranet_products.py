import os
import sys
import time
import pickle
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit import DataStructs

RDLogger.DisableLog('rdApp.*')


RADIUS = 2
N_BITS = 2048


def smiles_to_bits(smi: str) -> Tuple[bool, np.ndarray]:
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False, np.empty((0,), dtype=np.uint8)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS)
        arr = np.zeros((N_BITS,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return True, arr
    except Exception:
        return False, np.empty((0,), dtype=np.uint8)


def load_model(model_path: str):
    # Try joblib if available, else fall back to pickle
    try:
        import joblib  # type: ignore
        return joblib.load(model_path)
    except Exception:
        with open(model_path, "rb") as f:
            return pickle.load(f)


def read_smiles_list(path: str) -> List[str]:
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f]
    # Deduplicate and keep non-empty
    return [s for s in dict.fromkeys(lines) if s]


def main():
    parser = argparse.ArgumentParser(
        description="Measure inference latency of a simple linear classifier on DORAnet products"
    )
    default_txt = os.path.join(
        os.path.dirname(__file__), "doranet_retro_cryptofolione_products_all.txt"
    )
    default_model = os.path.join(
        os.path.dirname(__file__), "..", "models", "baseline_linear_clf.pkl"
    )
    parser.add_argument("--smiles-txt", default=default_txt, help="Path to .txt of SMILES (one per line)")
    parser.add_argument("--model-pkl", default=default_model, help="Path to pickled sklearn linear classifier")
    parser.add_argument("--out-csv", default=None, help="Where to save per-molecule results CSV")
    parser.add_argument("--timing-txt", default=None, help="Where to save aggregate timing summary")
    args = parser.parse_args()

    smiles_path = os.path.abspath(args.smiles_txt)
    model_path = os.path.abspath(args.model_pkl)

    if args.out_csv is None:
        out_csv = os.path.join(os.path.dirname(__file__), "linear_inference_on_doranet_products.csv")
    else:
        out_csv = os.path.abspath(args.out_csv)

    if args.timing_txt is None:
        timing_txt = os.path.join(os.path.dirname(__file__), "linear_inference_on_doranet_products_timing.txt")
    else:
        timing_txt = os.path.abspath(args.timing_txt)

    # Load resources
    smiles = read_smiles_list(smiles_path)
    clf = load_model(model_path)

    # Prepare outputs
    rows = []
    total_infer = 0.0
    total_feat = 0.0
    ok_count = 0

    for smi in smiles:
        # Featurization timing
        t0 = time.perf_counter()
        ok, bits = smiles_to_bits(smi)
        t1 = time.perf_counter()
        feat_s = t1 - t0
        total_feat += feat_s

        if not ok:
            rows.append({
                "smiles": smi,
                "valid": 0,
                "featurize_seconds": feat_s,
                "infer_seconds": np.nan,
                "score": np.nan,
                "error": "invalid_smiles",
            })
            continue

        # Inference timing (per-sample)
        x = bits.astype(np.float32)[None, :]
        t2 = time.perf_counter()
        try:
            if hasattr(clf, "predict_proba"):
                score = float(clf.predict_proba(x)[:, 1][0])
            elif hasattr(clf, "decision_function"):
                score = float(clf.decision_function(x))
            else:
                # Fallback to predict
                score = float(clf.predict(x)[0])
            err = ""
        except Exception as e:
            score = np.nan
            err = str(e)
        t3 = time.perf_counter()
        infer_s = t3 - t2
        total_infer += infer_s
        ok_count += 1 if not np.isnan(score) else 0

        rows.append({
            "smiles": smi,
            "valid": 1,
            "featurize_seconds": feat_s,
            "infer_seconds": infer_s,
            "score": score,
            "error": err,
        })

    # Save per-molecule CSV
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # Save timing summary
    with open(timing_txt, "w") as f:
        f.write(f"n_molecules={len(smiles)}\n")
        f.write(f"n_valid={int(df['valid'].sum())}\n")
        f.write(f"total_featurize_seconds={total_feat:.6f}\n")
        f.write(f"total_infer_seconds={total_infer:.6f}\n")
        f.write(f"mean_infer_seconds={df['infer_seconds'].dropna().mean():.8f}\n")
        f.write(f"median_infer_seconds={df['infer_seconds'].dropna().median():.8f}\n")

    print(f"Saved per-molecule results: {out_csv}")
    print(f"Saved timing summary: {timing_txt}")


if __name__ == "__main__":
    main()

