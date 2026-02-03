"""
Local multiprocessing-based fingerprinting script for SupCon splits.

Supports both ECFP4 (Morgan) and Atom Pair fingerprints. Processes all three splits
(train/val/test) in a single run using Python's multiprocessing.

Usage:
    python scripts/04_local_fingerprint_molecules.py
"""

from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

# =============================================================================
# Configuration
# =============================================================================

FP_TYPE: str = "atompair"
"""Fingerprint type: 'ecfp4' or 'atompair'"""

N_BITS: int = 2048
"""Number of bits in the fingerprint"""

N_WORKERS: int = None  # None = use all available CPUs
"""Number of parallel workers. Set to None to use all CPUs."""

# ECFP4-specific (Morgan fingerprint)
MORGAN_RADIUS: int = 2
"""Radius for Morgan fingerprint (ECFP4 uses radius=2)"""

# Atom Pair-specific
AP_MIN_LENGTH: int = 1
"""Minimum path length for atom pair fingerprint"""

AP_MAX_LENGTH: int = 30
"""Maximum path length for atom pair fingerprint"""

# =============================================================================
# Fingerprint Functions
# =============================================================================


def compute_ecfp4(mol) -> np.ndarray:
    """Compute Morgan fingerprint (ECFP4) with configured radius and bits."""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=N_BITS)
    arr = np.zeros((N_BITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_atompair(mol) -> np.ndarray:
    """Compute hashed atom pair fingerprint with configured bits and path lengths."""
    fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(
        mol, nBits=N_BITS, minLength=AP_MIN_LENGTH, maxLength=AP_MAX_LENGTH
    )
    arr = np.zeros((N_BITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smiles_to_fingerprint(smi: str) -> Tuple[bool, np.ndarray]:
    """
    Convert SMILES to fingerprint based on FP_TYPE config.

    Returns:
        Tuple of (success, fingerprint_array). If success is False,
        the array is empty.
    """
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False, np.empty((0,), dtype=np.uint8)
        if FP_TYPE == "ecfp4":
            return True, compute_ecfp4(mol)
        elif FP_TYPE == "atompair":
            return True, compute_atompair(mol)
        else:
            raise ValueError(f"Unknown fingerprint type: {FP_TYPE}")
    except Exception:
        return False, np.empty((0,), dtype=np.uint8)


def process_row(row: Tuple[str, int, str, int]) -> Tuple[bool, str, int, str, int, np.ndarray]:
    """
    Process a single row: compute fingerprint for SMILES.

    Args:
        row: Tuple of (smiles, label, source, triplet_id)

    Returns:
        Tuple of (success, smiles, label, source, triplet_id, fingerprint)
    """
    smi, label, source, triplet_id = row
    ok, bits = smiles_to_fingerprint(smi)
    return ok, smi, label, source, triplet_id, bits


# =============================================================================
# Helper Functions
# =============================================================================


def find_supcon_input(split: str) -> Path:
    """Find supcon_{split}_V1.parquet file."""
    base_dir = Path(__file__).parent.parent / "data" / split
    # Prefer _V1 files first
    parquet_v1 = base_dir / f"supcon_{split}_V1.parquet"
    if parquet_v1.exists():
        return parquet_v1
    parquet = base_dir / f"supcon_{split}.parquet"
    if parquet.exists():
        return parquet
    raise FileNotFoundError(f"No input found for split '{split}': {parquet_v1} or {parquet}")


def load_supcon_df(split: str) -> pd.DataFrame:
    """Load SupCon split and validate required columns."""
    in_path = find_supcon_input(split)
    df = pd.read_parquet(in_path)
    required_cols = {"smiles", "label", "source", "triplet_id"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Input dataframe missing columns: {missing}")
    return df


# =============================================================================
# Process Single Split
# =============================================================================


def process_split(split: str, n_workers: int) -> None:
    """
    Process one split with multiprocessing parallelization.
    """
    print(f"\n{'='*50}")
    print(f"Processing {split} split with {FP_TYPE} fingerprints")
    print(f"{'='*50}")

    # Load data
    df_in = load_supcon_df(split)
    rows = list(
        df_in[["smiles", "label", "source", "triplet_id"]].itertuples(
            index=False, name=None
        )
    )
    print(f"Loaded {len(rows)} rows")

    # Process with multiprocessing
    print(f"Processing with {n_workers} workers...")
    with Pool(n_workers) as pool:
        results = pool.map(process_row, rows)

    # Collect successful results
    ok_smiles: List[str] = []
    ok_labels: List[int] = []
    ok_sources: List[str] = []
    ok_triplet_ids: List[int] = []
    ok_bits: List[np.ndarray] = []
    dropped = 0

    for ok, smi, label, source, triplet_id, bits in results:
        if ok:
            ok_smiles.append(smi)
            ok_labels.append(label)
            ok_sources.append(source)
            ok_triplet_ids.append(triplet_id)
            ok_bits.append(bits)
        else:
            dropped += 1

    print(f"Completed. Dropped {dropped} invalid rows.")

    # Build output dataframe
    out_dir = Path(__file__).parent.parent / "data" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    if ok_smiles:
        bit_array = np.vstack(ok_bits).astype(np.uint8)
        cols = [f"fp_{i}" for i in range(N_BITS)]
        df_bits = pd.DataFrame(bit_array, columns=cols)
        df_out = pd.DataFrame(
            {
                "smiles": ok_smiles,
                "label": ok_labels,
                "source": ok_sources,
                "triplet_id": ok_triplet_ids,
            }
        )
        df_out = pd.concat([df_out, df_bits], axis=1)

        final_parquet = out_dir / f"supcon_{split}_V1_{FP_TYPE}.parquet"
        df_out.to_parquet(final_parquet, index=False)
        print(f"Wrote {final_parquet.name} ({len(df_out)} rows)")
    else:
        print(f"WARNING: No valid molecules in {split} split!")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    n_workers = N_WORKERS if N_WORKERS is not None else cpu_count()

    print(f"Starting fingerprinting with FP_TYPE={FP_TYPE}, N_BITS={N_BITS}")
    print(f"Using {n_workers} parallel workers")

    for split in ["train", "val", "test"]:
        process_split(split, n_workers)

    print("\nAll splits fingerprinted successfully!")
