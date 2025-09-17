from pathlib import Path
from typing import Optional, Set
import pandas as pd
import pytest


def _find_split_path(base: Path, split: str) -> Optional[Path]:
    # Prefer baseline split files
    p_parquet = base / split / f"baseline_{split}.parquet"
    p_csv = base / split / f"baseline_{split}.csv"
    if p_parquet.exists():
        return p_parquet
    if p_csv.exists():
        return p_csv
    # Fallback to fingerprinted outputs
    p_fp_parquet = base / split / f"baseline_{split}_ecfp4.parquet"
    p_fp_csv = base / split / f"baseline_{split}_ecfp4.csv"
    if p_fp_parquet.exists():
        return p_fp_parquet
    if p_fp_csv.exists():
        return p_fp_csv
    return None


def _load_smiles_set(path: Path) -> Set[str]:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "smiles" not in df.columns:
        raise AssertionError(f"'smiles' column not found in {path}")
    return set(str(s).strip() for s in df["smiles"].dropna().astype(str))


def test_no_smiles_leakage_across_splits():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    train_path = _find_split_path(data_dir, "train")
    val_path = _find_split_path(data_dir, "val")
    test_path = _find_split_path(data_dir, "test")

    missing = [name for name, p in ("train", train_path), ("val", val_path), ("test", test_path) if p is None]
    if missing:
        pytest.skip(f"Missing split files for: {', '.join(missing)}")

    train_smiles = _load_smiles_set(train_path)
    val_smiles = _load_smiles_set(val_path)
    test_smiles = _load_smiles_set(test_path)

    # Ensure no overlap between any pair of splits
    inter_train_val = train_smiles & val_smiles
    inter_train_test = train_smiles & test_smiles
    inter_val_test = val_smiles & test_smiles

    assert not inter_train_val, f"SMILES leakage between train and val: {list(sorted(inter_train_val))[:10]} (and more)"
    assert not inter_train_test, f"SMILES leakage between train and test: {list(sorted(inter_train_test))[:10]} (and more)"
    assert not inter_val_test, f"SMILES leakage between val and test: {list(sorted(inter_val_test))[:10]} (and more)"
