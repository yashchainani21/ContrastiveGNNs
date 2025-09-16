from rdkit import Chem
from rdkit import RDLogger
import pandas as pd
from typing import List, Dict, Optional
import os

RDLogger.DisableLog('rdApp.*')

max_extension_modules = 3
num_bio_steps = 1
num_chem_steps = 1

PKS_products_filepath = f'../data/interim/unbound_PKS_products_{max_extension_modules}_ext_mods_no_stereo.pkl'
DORAnet_bio_products_filepath = f'../data/interim/DORAnet_BIO{num_bio_steps}_from_PKS_products_{max_extension_modules}_ext_mods_no_stereo.txt'
DORAnet_chem_products_filepath = f'../data/interim/DORAnet_CHEM{num_chem_steps}_from_PKS_products_{max_extension_modules}_ext_mods_no_stereo.txt'

all_molecules_output_filepath = f'../data/processed/all_PKS_and_non_PKS_molecules_{max_extension_modules}_BIO{num_bio_steps}_CHEM{num_chem_steps}_no_stereo.parquet'

def sanitize_smiles_no_stereo(smi: str) -> Optional[str]:
    """Return a sanitized, canonical SMILES without stereochemistry, or None if invalid."""
    if smi is None:
        return None
    smi = smi.strip()
    if not smi:
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        # remove stereo info from atoms and bonds, then sanitize and canonicalize
        Chem.RemoveStereochemistry(mol)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    except Exception:
        return None

def load_txt_smiles(filepath: str) -> List[str]:
    """Read a .txt containing one SMILES per line, sanitize and drop invalids."""
    if not os.path.exists(filepath):
        print(f"Warning: file not found: {filepath}")
        return []
    with open(filepath, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    out = []
    for smi in lines:
        s = sanitize_smiles_no_stereo(smi)
        if s:
            out.append(s)
    return out

def load_pks_smiles_from_pickle(filepath: str) -> List[str]:
    """Load a pickled dict of PKS products (values are SMILES), sanitize and drop invalids."""
    if not os.path.exists(filepath):
        print(f"Warning: file not found: {filepath}")
        return []
    try:
        data = pd.read_pickle(filepath)
    except Exception as e:
        print(f"Error reading PKS pickle {filepath}: {e}")
        return []
    # Expecting dict-like with SMILES in values
    if hasattr(data, 'values') and not isinstance(data, dict):
        # If it's a pandas Series or similar, pull values
        try:
            values = list(data.values())
        except Exception:
            values = list(data)
    elif isinstance(data, dict):
        values = list(data.values())
    else:
        # Fallback: try to iterate
        try:
            values = list(data)
        except Exception:
            print("Unrecognized PKS pickle format; expected dict-like")
            return []

    out = []
    for smi in values:
        s = sanitize_smiles_no_stereo(str(smi))
        if s:
            out.append(s)
    return out

if __name__ == "__main__":
    # Load and sanitize non-PKS products (bio and chem)
    bio_smiles = load_txt_smiles(DORAnet_bio_products_filepath)
    chem_smiles = load_txt_smiles(DORAnet_chem_products_filepath)

    # Build DataFrame for bio/chem and remove duplicates across them (keep arbitrary first)
    df_bio = pd.DataFrame({"smiles": bio_smiles, "source": ["bio"] * len(bio_smiles)})
    df_chem = pd.DataFrame({"smiles": chem_smiles, "source": ["chem"] * len(chem_smiles)})
    df_non_pks = pd.concat([df_bio, df_chem], ignore_index=True)
    df_non_pks = df_non_pks.drop_duplicates(subset=["smiles"], keep="first")

    # Load PKS products; sanitize but DO NOT remove duplicates from PKS per instructions
    pks_smiles = load_pks_smiles_from_pickle(PKS_products_filepath)
    df_pks = pd.DataFrame({"smiles": pks_smiles, "source": ["PKS"] * len(pks_smiles)})

    # Combine, keeping all PKS entries
    df_all = pd.concat([df_pks, df_non_pks], ignore_index=True)

    # Final dedup pass: keep all PKS rows; only remove duplicates among bio/chem
    mask_non_pks = df_all["source"] != "PKS"
    df_non_pks_final = df_all[mask_non_pks].drop_duplicates(subset=["smiles"], keep="first")
    df_pks_final = df_all[~mask_non_pks]
    df_final = pd.concat([df_pks_final, df_non_pks_final], ignore_index=True)

    print(
        f"Loaded: PKS={len(df_pks)}, bio={len(df_bio)}, chem={len(df_chem)} | "
        f"After dedup non-PKS={len(df_non_pks)}, Final total={len(df_final)}"
    )

    # Save to parquet if available; else fallback to CSV
    try:
        df_final.to_parquet(all_molecules_output_filepath, index=False)
        print(f"Saved dataframe to {all_molecules_output_filepath}")
    except Exception as e:
        alt_path = all_molecules_output_filepath.rsplit('.', 1)[0] + '.csv'
        df_final.to_csv(alt_path, index=False)
        print(f"Parquet write failed ({e}); saved CSV to {alt_path}")
