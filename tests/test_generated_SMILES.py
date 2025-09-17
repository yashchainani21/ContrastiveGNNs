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

    pairs = [("train", train_path), ("val", val_path), ("test", test_path)]
    missing = [name for (name, p) in pairs if p is None]
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


def test_no_stereo_characters_in_smiles():
    """
    Ensure no stereochemical markers remain in SMILES across all splits.
    Forbidden characters: '@', '@@', '/', '\\'. Checking '@' covers '@@'.
    """
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    splits = ["train", "val", "test"]
    checked_any = False
    errors = []

    forbidden = ["@", "/", "\\"]  # '@@' is subsumed by '@'

    for split in splits:
        path = _find_split_path(data_dir, split)
        if path is None:
            continue
        checked_any = True
        # Load as list to report specific offending entries
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        if "smiles" not in df.columns:
            errors.append(f"Split {split}: missing 'smiles' column in {path}")
            continue
        smiles_list = [str(s).strip() for s in df["smiles"].dropna().astype(str)]
        bad = [s for s in smiles_list if any(ch in s for ch in forbidden)]
        if bad:
            preview = bad[:10]
            errors.append(
                f"Split {split}: found {len(bad)} SMILES with stereochemical markers. Examples: {preview}"
            )

    if not checked_any:
        pytest.skip("No split files found to check stereo characters")

    assert not errors, "\n".join(errors)

