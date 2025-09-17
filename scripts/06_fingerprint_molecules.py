import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from mpi4py import MPI
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit import DataStructs

RDLogger.DisableLog('rdApp.*')

# Choose which split to fingerprint: 'train', 'val', or 'test'
SPLIT = 'train'

RADIUS = 2
N_BITS = 2048


def find_input_file(split: str) -> Path:
    base_dir = Path("../data") / split
    parquet = base_dir / f"baseline_{split}.parquet"
    csv = base_dir / f"baseline_{split}.csv"
    if parquet.exists():
        return parquet
    if csv.exists():
        return csv
    raise FileNotFoundError(f"No input found for split '{split}': {parquet} or {csv}")


def load_split_df(split: str) -> pd.DataFrame:
    in_path = find_input_file(split)
    if in_path.suffix == ".parquet":
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path)
    if not {"smiles", "source"}.issubset(df.columns):
        raise ValueError("Input dataframe must contain 'smiles' and 'source' columns")
    return df


def smiles_to_bits_row(smi: str) -> Tuple[bool, np.ndarray]:
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


def chunkify(lst: List, n: int) -> List[List]:
    n = max(1, min(n, len(lst)))
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        df_in = load_split_df(SPLIT)
        rows = list(df_in[["smiles", "source"]].itertuples(index=False, name=None))
        chunks = chunkify(rows, size)
        # Pad with empty chunks so length equals size (required by Scatter)
        if len(chunks) < size:
            chunks.extend([[] for _ in range(size - len(chunks))])
    else:
        df_in = None
        chunks = None

    # All ranks participate in Scatter with a list of length == size
    my_rows = comm.scatter(chunks, root=0)

    print(f"[Rank {rank}] processing {len(my_rows)} rows", flush=True)

    # Compute fingerprints for assigned rows
    ok_smiles: List[str] = []
    ok_sources: List[str] = []
    ok_bits: List[np.ndarray] = []
    dropped = 0
    for smi, src in my_rows:
        ok, bits = smiles_to_bits_row(smi)
        if ok:
            ok_smiles.append(smi)
            ok_sources.append(src)
            ok_bits.append(bits)
        else:
            dropped += 1

    print(f"[Rank {rank}] completed. Dropped {dropped} invalid rows.", flush=True)

    # Write shard per rank to avoid large gather messages
    out_dir = Path("../data") / SPLIT
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_parquet = out_dir / f"baseline_{SPLIT}_ecfp4.rank{rank}.parquet"
    shard_csv = out_dir / f"baseline_{SPLIT}_ecfp4.rank{rank}.csv"

    if ok_smiles:
        bit_array = np.vstack(ok_bits).astype(np.uint8)
        cols = [f"fp_{i}" for i in range(N_BITS)]
        df_bits = pd.DataFrame(bit_array, columns=cols)
        df_shard = pd.DataFrame({
            "smiles": ok_smiles,
            "source": ok_sources,
        })
        df_shard = pd.concat([df_shard, df_bits], axis=1)
        try:
            df_shard.to_parquet(shard_parquet, index=False)
            print(f"[Rank {rank}] wrote shard {shard_parquet} ({len(df_shard)} rows)")
        except Exception as e:
            df_shard.to_csv(shard_csv, index=False)
            print(f"[Rank {rank}] parquet failed ({e}); wrote CSV shard {shard_csv}")
    else:
        # Clean any stale shard from prior runs
        try:
            if shard_parquet.exists(): shard_parquet.unlink()
            if shard_csv.exists(): shard_csv.unlink()
        except Exception:
            pass

    # Synchronize ranks before merge
    comm.Barrier()

    if rank == 0:
        # Try parquet shards first, else CSV shards
        shard_paths = sorted(out_dir.glob(f"baseline_{SPLIT}_ecfp4.rank*.parquet"))
        use_csv = False
        if not shard_paths:
            shard_paths = sorted(out_dir.glob(f"baseline_{SPLIT}_ecfp4.rank*.csv"))
            use_csv = True
        if not shard_paths:
            raise RuntimeError("No shard files found to merge. Check worker logs.")

        parts = []
        for p in shard_paths:
            try:
                parts.append(pd.read_parquet(p) if not use_csv else pd.read_csv(p))
            except Exception as e:
                print(f"Skipping shard {p} due to read error: {e}")
        if not parts:
            raise RuntimeError("All shard reads failed; cannot assemble final dataframe.")

        df_out = pd.concat(parts, ignore_index=True)
        final_parquet = out_dir / f"baseline_{SPLIT}_ecfp4.parquet"
        try:
            df_out.to_parquet(final_parquet, index=False)
            print(f"Merged {len(parts)} shards → {final_parquet} ({len(df_out)} rows)")
        except Exception as e:
            final_csv = out_dir / f"baseline_{SPLIT}_ecfp4.csv"
            df_out.to_csv(final_csv, index=False)
            print(f"Parquet merge failed ({e}); wrote CSV {final_csv}")
