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


def _find_split_with_source(base: Path, split: str) -> Optional[Path]:
    # Prefer non-fingerprint first, then fingerprinted
    for name in [f"baseline_{split}.parquet", f"baseline_{split}.csv",
                 f"baseline_{split}_ecfp4.parquet", f"baseline_{split}_ecfp4.csv"]:
        p = base / split / name
        if p.exists():
            return p
    return None


def _pks_ratio(path: Path) -> float:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "source" not in df.columns:
        raise AssertionError(f"'source' column not found in {path}")
    total = len(df)
    if total == 0:
        return 0.0
    pks = (df["source"].astype(str) == "PKS").sum()
    return float(pks) / float(total)


def _pks_counts(path: Path):
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "source" not in df.columns:
        raise AssertionError(f"'source' column not found in {path}")
    total = len(df)
    pks = int((df["source"].astype(str) == "PKS").sum())
    nonpks = int(total - pks)
    ratio = (pks / total) if total else 0.0
    return pks, nonpks, total, ratio


def test_pks_pkl_has_no_duplicate_values():
    """
    Ensure PKS pkl files don't lose data due to key collisions.

    This catches the bug where using pks_design as dict key caused
    thiolysis products to be overwritten by cyclization products.
    """
    import pickle

    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"
    pkl_files = list(processed_dir.glob("pks_products_*.pkl"))

    if not pkl_files:
        pytest.skip("No PKS pkl files found")

    for pkl_path in pkl_files:
        if "_SMILES" in pkl_path.name:
            continue  # Skip txt files

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            values = list(data.values())
            assert len(values) == len(set(values)), (
                f"{pkl_path.name}: has {len(values)} values but only "
                f"{len(set(values))} unique - indicates data loss"
            )


def test_pks_ratio_similarity_across_splits():
    """
    Check that the PKS fraction is similar across train/val/test.
    Uses an absolute tolerance on the proportion difference.
    """
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    paths = {
        split: _find_split_with_source(data_dir, split)
        for split in ("train", "val", "test")
    }
    missing = [s for s, p in paths.items() if p is None]
    if missing:
        pytest.skip(f"Missing split files for: {', '.join(missing)}")

    # Compute and print ratios and counts for visibility
    counts = {s: _pks_counts(p) for s, p in paths.items()}
    ratios = {s: c[3] for s, c in counts.items()}

    # Print a concise summary (use `-s` with pytest to always see prints)
    for split, (pks, nonpks, total, ratio) in counts.items():
        print(f"{split}: PKS ratio={ratio:.3f} (PKS={pks}, non-PKS={nonpks}, total={total})")

    # Tolerance in absolute proportion (e.g., 0.05 = 5 percentage points)
    tol = 0.05

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    errors = []
    for a, b in pairs:
        diff = abs(ratios[a] - ratios[b])
        if diff > tol:
            errors.append(f"PKS ratio differs more than {tol:.2f} between {a} ({ratios[a]:.3f}) and {b} ({ratios[b]:.3f})")

    assert not errors, "\n".join(errors)


# ============================================================================
# SupCon Split Tests
# ============================================================================

def _find_supcon_split_path(base: Path, split: str) -> Optional[Path]:
    """Find SupCon split file for a given split."""
    # Prefer _V1 files first
    p_v1 = base / split / f"supcon_{split}_V1.parquet"
    if p_v1.exists():
        return p_v1
    p = base / split / f"supcon_{split}.parquet"
    if p.exists():
        return p
    return None


def _load_supcon_df(path: Path) -> pd.DataFrame:
    """Load a SupCon split parquet file."""
    return pd.read_parquet(path)


def test_supcon_no_smiles_leakage():
    """Verify no SMILES overlap across SupCon train/val/test splits."""
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    train_path = _find_supcon_split_path(data_dir, "train")
    val_path = _find_supcon_split_path(data_dir, "val")
    test_path = _find_supcon_split_path(data_dir, "test")

    pairs = [("train", train_path), ("val", val_path), ("test", test_path)]
    missing = [name for (name, p) in pairs if p is None]
    if missing:
        pytest.skip(f"Missing SupCon split files for: {', '.join(missing)}")

    train_smiles = set(_load_supcon_df(train_path)['smiles'])
    val_smiles = set(_load_supcon_df(val_path)['smiles'])
    test_smiles = set(_load_supcon_df(test_path)['smiles'])

    inter_train_val = train_smiles & val_smiles
    inter_train_test = train_smiles & test_smiles
    inter_val_test = val_smiles & test_smiles

    assert not inter_train_val, (
        f"SMILES leakage between train and val: {len(inter_train_val)} molecules. "
        f"Examples: {list(sorted(inter_train_val))[:5]}"
    )
    assert not inter_train_test, (
        f"SMILES leakage between train and test: {len(inter_train_test)} molecules. "
        f"Examples: {list(sorted(inter_train_test))[:5]}"
    )
    assert not inter_val_test, (
        f"SMILES leakage between val and test: {len(inter_val_test)} molecules. "
        f"Examples: {list(sorted(inter_val_test))[:5]}"
    )


def test_supcon_class_ratios():
    """Verify ~33% PKS ratio in each SupCon split (inherent to triplet structure)."""
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    paths = {
        split: _find_supcon_split_path(data_dir, split)
        for split in ("train", "val", "test")
    }
    missing = [s for s, p in paths.items() if p is None]
    if missing:
        pytest.skip(f"Missing SupCon split files for: {', '.join(missing)}")

    # Expected ratio is 1/3 (one PKS per triplet, two augmentations)
    expected_ratio = 1 / 3
    tol = 0.02  # 2 percentage points tolerance

    errors = []
    for split, path in paths.items():
        df = _load_supcon_df(path)
        pks_ratio = (df['label'] == 1).mean()
        print(f"SupCon {split}: PKS ratio = {pks_ratio:.3f} (expected ~{expected_ratio:.3f})")

        if abs(pks_ratio - expected_ratio) > tol:
            errors.append(
                f"SupCon {split} PKS ratio {pks_ratio:.3f} differs from expected "
                f"{expected_ratio:.3f} by more than {tol:.2f}"
            )

    assert not errors, "\n".join(errors)


def test_supcon_split_sizes():
    """Verify ~80/10/10 split ratios for SupCon data."""
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    paths = {
        split: _find_supcon_split_path(data_dir, split)
        for split in ("train", "val", "test")
    }
    missing = [s for s, p in paths.items() if p is None]
    if missing:
        pytest.skip(f"Missing SupCon split files for: {', '.join(missing)}")

    sizes = {split: len(_load_supcon_df(path)) for split, path in paths.items()}
    total = sum(sizes.values())

    ratios = {split: size / total for split, size in sizes.items()}
    expected = {'train': 0.8, 'val': 0.1, 'test': 0.1}
    tol = 0.02  # 2 percentage points tolerance

    print(f"SupCon split sizes: {sizes}")
    print(f"SupCon split ratios: train={ratios['train']:.3f}, val={ratios['val']:.3f}, test={ratios['test']:.3f}")

    errors = []
    for split in ['train', 'val', 'test']:
        if abs(ratios[split] - expected[split]) > tol:
            errors.append(
                f"SupCon {split} ratio {ratios[split]:.3f} differs from expected "
                f"{expected[split]:.2f} by more than {tol:.2f}"
            )

    assert not errors, "\n".join(errors)


def test_supcon_triplet_integrity():
    """Verify each triplet_id has exactly 3 rows (1 PKS + 2 augmentations)."""
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    paths = {
        split: _find_supcon_split_path(data_dir, split)
        for split in ("train", "val", "test")
    }
    missing = [s for s, p in paths.items() if p is None]
    if missing:
        pytest.skip(f"Missing SupCon split files for: {', '.join(missing)}")

    errors = []
    for split, path in paths.items():
        df = _load_supcon_df(path)

        # Check each triplet has exactly 3 rows
        triplet_counts = df.groupby('triplet_id').size()
        invalid_triplets = triplet_counts[triplet_counts != 3]
        if len(invalid_triplets) > 0:
            errors.append(
                f"SupCon {split}: {len(invalid_triplets)} triplets don't have exactly 3 rows. "
                f"Examples: {invalid_triplets.head(5).to_dict()}"
            )

        # Check each triplet has exactly 1 PKS (label=1) and 2 augmentations (label=0)
        triplet_pks_counts = df.groupby('triplet_id')['label'].sum()
        invalid_pks = triplet_pks_counts[triplet_pks_counts != 1]
        if len(invalid_pks) > 0:
            errors.append(
                f"SupCon {split}: {len(invalid_pks)} triplets don't have exactly 1 PKS molecule. "
                f"Examples: {invalid_pks.head(5).to_dict()}"
            )

        # Check source distribution per triplet
        for triplet_id in df['triplet_id'].unique()[:100]:  # Sample first 100 triplets
            triplet_df = df[df['triplet_id'] == triplet_id]
            sources = set(triplet_df['source'])
            expected_sources = {'pks', 'enzymatic_aug', 'synthetic_aug'}
            if sources != expected_sources:
                errors.append(
                    f"SupCon {split}: triplet {triplet_id} has sources {sources}, "
                    f"expected {expected_sources}"
                )
                break  # Only report first error

    assert not errors, "\n".join(errors)
