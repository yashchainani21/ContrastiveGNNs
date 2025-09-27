#!/usr/bin/env python3
"""
Convert a fingerprint split to .npz for training, given only the split name.

Usage:
  - Set SPLIT at the top of this file to one of: 'train', 'val', 'test'.
  - Run the script; it auto-discovers the input and writes the output next to it.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# -------- Configure only this --------
SPLIT = "train"   # choose from: 'train', 'val', 'test'
# ------------------------------------

LABEL_COL = "source"
FP_PREFIX = "fp_"
POSITIVE_LABEL = "PKS"


def _resolve_input(split: str) -> Path:
    base = Path("../data") / split
    candidates = [
        base / f"baseline_{split}_ecfp4.parquet",
        base / f"baseline_{split}_ecfp4.csv",
        base / f"baseline_{split}.parquet",
        base / f"baseline_{split}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No input found for split '{split}' in {base}")


def _resolve_output(split: str) -> Path:
    base = Path("../data") / split
    base.mkdir(parents=True, exist_ok=True)
    return base / f"baseline_{split}_ecfp4.npz"


def convert_split_to_npz(split: str):
    in_path = _resolve_input(split)
    out_path = _resolve_output(split)

    print(f"Split: {split}")
    print(f"Input:  {in_path}")
    print(f"Output: {out_path}")

    if in_path.suffix == ".parquet":
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path)

    # Identify fingerprint columns
    fp_cols = [c for c in df.columns if c.startswith(FP_PREFIX)]
    if not fp_cols:
        raise ValueError("No fingerprint columns found; expected columns starting with 'fp_'")
    fp_cols = sorted(fp_cols, key=lambda s: int(s.split("_")[1]))
    print(f"Found {len(fp_cols)} fingerprint columns")

    # Extract fingerprints and labels
    fps = df[fp_cols].to_numpy(dtype=np.float32)
    raw_labels = df[LABEL_COL].astype(str).to_numpy()
    labels = np.array([1 if x == POSITIVE_LABEL else 0 for x in raw_labels], dtype=np.int64)

    print(f"Shapes: fps={fps.shape}, labels={labels.shape}")
    print(f"Class counts: PKS={labels.sum()}, non-PKS={len(labels)-labels.sum()}")

    # Save compressed npz
    np.savez_compressed(out_path, fps=fps, labels=labels)
    print(f"Saved NPZ file: {out_path}")


if __name__ == "__main__":
    convert_split_to_npz(SPLIT)
