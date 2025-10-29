import os
import sys
import time
import glob
import argparse
from typing import List, Set

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


def sanitize_smiles_no_stereo(smi: str):
    if smi is None:
        return None
    smi = smi.strip()
    if not smi:
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    except Exception:
        return None


def read_doranet_smiles(txt_path: str) -> List[str]:
    with open(txt_path, 'r') as f:
        raw = [ln.strip() for ln in f if ln.strip()]
    out: List[str] = []
    for s in raw:
        c = sanitize_smiles_no_stereo(s)
        if c:
            out.append(c)
    # Deduplicate but preserve order roughly
    return list(dict.fromkeys(out))


def load_pks_smiles_from_processed(processed_dir: str) -> Set[str]:
    # Find the combined processed file created by 04_compile_all_molecules.py
    pattern = os.path.join(processed_dir, "all_PKS_and_non_PKS_molecules_*_no_stereo.parquet")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No processed parquet matching {pattern}")
    path = matches[0]
    df = pd.read_parquet(path)
    if not {"smiles", "source"}.issubset(df.columns):
        raise ValueError(f"Expected columns 'smiles' and 'source' in {path}")
    pks = df[df["source"] == "PKS"]["smiles"].astype(str).tolist()
    # Ensure canonical no-stereo (file should already be), but sanitize defensively
    pks_clean = []
    for s in pks:
        c = sanitize_smiles_no_stereo(s)
        if c:
            pks_clean.append(c)
    return set(pks_clean)


def main():
    parser = argparse.ArgumentParser(description="Measure time to check DORAnet product membership in PKS list")
    default_doranet_txt = os.path.join(os.path.dirname(__file__), "doranet_retro_cryptofolione_products_all.txt")
    default_processed = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    parser.add_argument("--doranet-txt", default=default_doranet_txt, help="Path to DORAnet products .txt")
    parser.add_argument("--processed-dir", default=default_processed, help="Path to processed data directory")
    parser.add_argument("--out-csv", default=None, help="Where to save membership results CSV")
    parser.add_argument("--timing-txt", default=None, help="Where to save timing summary")
    args = parser.parse_args()

    doranet_txt = os.path.abspath(args.doranet_txt)
    processed_dir = os.path.abspath(args.processed_dir)
    out_csv = args.out_csv or os.path.join(os.path.dirname(__file__), "doranet_vs_pks_membership.csv")
    timing_txt = args.timing_txt or os.path.join(os.path.dirname(__file__), "doranet_vs_pks_membership_timing.txt")

    # Load inputs
    t0 = time.perf_counter()
    doranet_list = read_doranet_smiles(doranet_txt)
    t1 = time.perf_counter()
    pks_set = load_pks_smiles_from_processed(processed_dir)
    t2 = time.perf_counter()

    # Membership checks one-by-one
    in_flags: List[int] = []
    t_start = time.perf_counter()
    for smi in doranet_list:
        in_flags.append(1 if smi in pks_set else 0)
    t_end = time.perf_counter()

    total_checks = len(doranet_list)
    duration = t_end - t_start
    load_doranet_s = t1 - t0
    load_pks_s = t2 - t1

    # Save results
    pd.DataFrame({
        "smiles": doranet_list,
        "in_pks": in_flags,
    }).to_csv(out_csv, index=False)

    with open(timing_txt, 'w') as f:
        f.write(f"n_doranet={total_checks}\n")
        f.write(f"n_pks={len(pks_set)}\n")
        f.write(f"load_doranet_seconds={load_doranet_s:.6f}\n")
        f.write(f"load_pks_seconds={load_pks_s:.6f}\n")
        f.write(f"membership_total_seconds={duration:.6f}\n")
        f.write(f"membership_mean_seconds={duration/total_checks if total_checks else 0.0:.9f}\n")

    print(f"Checked {total_checks} DORAnet molecules against {len(pks_set)} PKS SMILES")
    print(f"Membership time: {duration:.6f} s (avg {duration/total_checks if total_checks else 0.0:.9f} s/molecule)")
    print(f"Saved results: {out_csv}")
    print(f"Saved timing: {timing_txt}")


if __name__ == "__main__":
    main()

