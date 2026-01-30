"""
MPI-based fingerprinting script for SupCon splits.

Supports ECFP4 (Morgan) and Atom Pair fingerprints. Processes all three splits
(train/val/test) in a single run.

Usage:
    # Single process
    python scripts/04_fingerprint_molecules.py

    # Multi-process with MPI
    mpirun -np 4 python scripts/04_fingerprint_molecules.py
"""

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from mpi4py import MPI
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

# =============================================================================
# Configuration
# =============================================================================

FP_TYPE: str = "ecfp4"
"""Fingerprint type: 'ecfp4' or 'atompair'"""

N_BITS: int = 2048
"""Number of bits in the fingerprint"""

CLEANUP_SHARDS: bool = True
"""If True, remove per-rank shard files after merge on rank 0."""

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


# =============================================================================
# Helper Functions
# =============================================================================


def find_supcon_input(split: str) -> Path:
    """Find supcon_{split}.parquet file."""
    base_dir = Path(__file__).parent.parent / "data" / split
    parquet = base_dir / f"supcon_{split}.parquet"
    if parquet.exists():
        return parquet
    raise FileNotFoundError(f"No input found for split '{split}': {parquet}")


def load_supcon_df(split: str) -> pd.DataFrame:
    """Load SupCon split and validate required columns."""
    in_path = find_supcon_input(split)
    df = pd.read_parquet(in_path)
    required_cols = {"smiles", "label", "source", "triplet_id"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Input dataframe missing columns: {missing}")
    return df


def chunkify(lst: List, n: int) -> List[List]:
    """Divide list into n roughly equal chunks."""
    n = max(1, min(n, len(lst)))
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


# =============================================================================
# Process Single Split
# =============================================================================


def process_split(split: str, comm, rank: int, size: int) -> None:
    """
    Process one split with MPI parallelization.

    Steps:
    1. Rank 0 loads df, extracts rows, chunks by MPI size
    2. Scatter chunks to all ranks
    3. Each rank computes fingerprints for its chunk
    4. Each rank writes shard file
    5. Barrier sync
    6. Rank 0 merges shards, writes final output, cleans up
    """
    # Rank 0 loads data and distributes
    if rank == 0:
        df_in = load_supcon_df(split)
        # Extract all columns we need to preserve
        rows = list(
            df_in[["smiles", "label", "source", "triplet_id"]].itertuples(
                index=False, name=None
            )
        )
        chunks = chunkify(rows, size)
        # Pad with empty chunks so length equals size (required by Scatter)
        if len(chunks) < size:
            chunks.extend([[] for _ in range(size - len(chunks))])
    else:
        chunks = None

    # All ranks participate in Scatter
    my_rows = comm.scatter(chunks, root=0)

    print(f"[Rank {rank}] Processing {len(my_rows)} rows for {split}", flush=True)

    # Compute fingerprints for assigned rows
    ok_smiles: List[str] = []
    ok_labels: List[int] = []
    ok_sources: List[str] = []
    ok_triplet_ids: List[int] = []
    ok_bits: List[np.ndarray] = []
    dropped = 0

    for smi, label, source, triplet_id in my_rows:
        ok, bits = smiles_to_fingerprint(smi)
        if ok:
            ok_smiles.append(smi)
            ok_labels.append(label)
            ok_sources.append(source)
            ok_triplet_ids.append(triplet_id)
            ok_bits.append(bits)
        else:
            dropped += 1

    print(f"[Rank {rank}] Completed {split}. Dropped {dropped} invalid rows.", flush=True)

    # Write shard per rank to avoid large gather messages
    out_dir = Path(__file__).parent.parent / "data" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_parquet = out_dir / f"supcon_{split}_{FP_TYPE}.rank{rank}.parquet"

    if ok_smiles:
        bit_array = np.vstack(ok_bits).astype(np.uint8)
        cols = [f"fp_{i}" for i in range(N_BITS)]
        df_bits = pd.DataFrame(bit_array, columns=cols)
        df_shard = pd.DataFrame(
            {
                "smiles": ok_smiles,
                "label": ok_labels,
                "source": ok_sources,
                "triplet_id": ok_triplet_ids,
            }
        )
        df_shard = pd.concat([df_shard, df_bits], axis=1)
        df_shard.to_parquet(shard_parquet, index=False)
        print(f"[Rank {rank}] Wrote shard {shard_parquet.name} ({len(df_shard)} rows)")
    else:
        # Clean any stale shard from prior runs
        if shard_parquet.exists():
            shard_parquet.unlink()

    # Synchronize ranks before merge
    comm.Barrier()

    # Rank 0 merges shards
    if rank == 0:
        shard_paths = sorted(out_dir.glob(f"supcon_{split}_{FP_TYPE}.rank*.parquet"))
        if not shard_paths:
            raise RuntimeError(
                f"No shard files found for {split}. Check worker logs."
            )

        parts = []
        for p in shard_paths:
            try:
                parts.append(pd.read_parquet(p))
            except Exception as e:
                print(f"Skipping shard {p} due to read error: {e}")
        if not parts:
            raise RuntimeError(
                f"All shard reads failed for {split}; cannot assemble final dataframe."
            )

        df_out = pd.concat(parts, ignore_index=True)
        final_parquet = out_dir / f"supcon_{split}_{FP_TYPE}.parquet"
        df_out.to_parquet(final_parquet, index=False)
        print(
            f"Merged {len(parts)} shards -> {final_parquet.name} ({len(df_out)} rows)"
        )

        # Cleanup shard files if requested
        if CLEANUP_SHARDS:
            removed = 0
            for p in shard_paths:
                try:
                    p.unlink()
                    removed += 1
                except Exception as e:
                    print(f"Warning: failed to remove shard {p}: {e}")
            if removed:
                print(f"Cleanup: removed {removed} shard file(s)")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"Starting fingerprinting with FP_TYPE={FP_TYPE}, N_BITS={N_BITS}")
        print(f"MPI size: {size} processes")

    for split in ["train", "val", "test"]:
        if rank == 0:
            print(f"\n{'='*50}")
            print(f"Processing {split} split with {FP_TYPE} fingerprints")
            print(f"{'='*50}")
        process_split(split, comm, rank, size)
        comm.Barrier()  # Sync between splits

    if rank == 0:
        print("\nAll splits fingerprinted successfully!")
