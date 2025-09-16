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
        num_active = len(chunks)
    else:
        df_in = None
        chunks = None
        num_active = None

    num_active = comm.bcast(num_active if rank == 0 else None, root=0)
    if rank < num_active:
        my_rows = comm.scatter(chunks, root=0)
    else:
        my_rows = []

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

    # Prepare serializable payload
    payload = (ok_smiles, ok_sources, [arr.tolist() for arr in ok_bits])

    gathered = comm.gather(payload, root=0)

    if rank == 0:
        all_smiles: List[str] = []
        all_sources: List[str] = []
        all_bits_lists: List[List[int]] = []
        for smi_list, src_list, bits_list in gathered:
            all_smiles.extend(smi_list)
            all_sources.extend(src_list)
            all_bits_lists.extend(bits_list)

        if not all_smiles:
            raise RuntimeError("No valid fingerprints were produced by any rank.")

        # Build the final DataFrame with explicit bit columns
        bit_array = np.array(all_bits_lists, dtype=np.uint8)
        assert bit_array.shape[1] == N_BITS, f"Bit array has wrong width: {bit_array.shape}"
        cols = [f"fp_{i}" for i in range(N_BITS)]
        df_bits = pd.DataFrame(bit_array, columns=cols)
        df_out = pd.DataFrame({
            "smiles": all_smiles,
            "source": all_sources,
        })
        df_out = pd.concat([df_out, df_bits], axis=1)

        # Save alongside the split
        out_dir = Path("../data") / SPLIT
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"baseline_{SPLIT}_ecfp4.parquet"
        try:
            df_out.to_parquet(out_path, index=False)
            print(f"Saved fingerprints to {out_path} ({len(df_out)} rows, {N_BITS} bits)")
        except Exception as e:
            # Fallback CSV if parquet fails
            out_csv = out_dir / f"baseline_{SPLIT}_ecfp4.csv"
            df_out.to_csv(out_csv, index=False)
            print(f"Parquet save failed ({e}); saved CSV to {out_csv}")
