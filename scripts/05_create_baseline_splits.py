import os
from pathlib import Path
import pandas as pd
from rdkit import RDLogger
from sklearn.model_selection import train_test_split

RDLogger.DisableLog('rdApp.*')

# Match the defaults used when generating the combined molecules file
max_extension_modules = 3
num_bio_steps = 1
num_chem_steps = 1

input_base = f"../data/processed/all_PKS_and_non_PKS_molecules_{max_extension_modules}_BIO{num_bio_steps}_CHEM{num_chem_steps}_no_stereo"
input_parquet = input_base + ".parquet"
input_csv = input_base + ".csv"

out_train_dir = Path("../data/train")
out_val_dir = Path("../data/val")
out_test_dir = Path("../data/test")

def read_input_df() -> pd.DataFrame:
    if os.path.exists(input_parquet):
        df = pd.read_parquet(input_parquet)
        print(f"Loaded dataframe from {input_parquet} with shape {df.shape}")
        return df
    if os.path.exists(input_csv):
        df = pd.read_csv(input_csv)
        print(f"Loaded dataframe from {input_csv} with shape {df.shape}")
        return df
    raise FileNotFoundError(f"Neither {input_parquet} nor {input_csv} was found.")

def ensure_columns(df: pd.DataFrame):
    required = {"smiles", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input dataframe missing required columns: {missing}")

def save_df(df: pd.DataFrame, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"baseline_{name}.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"Saved {name} to {parquet_path} ({len(df)} rows)")
    except Exception as e:
        csv_path = out_dir / f"baseline_{name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Parquet save failed ({e}); saved {name} CSV to {csv_path} ({len(df)} rows)")

if __name__ == "__main__":
    df = read_input_df()
    ensure_columns(df)

    # Build unique-SMILES view and label mapping: PKS if any row for that SMILES is PKS, else non-PKS
    # This ensures all duplicates of a SMILES are assigned to the same split to avoid leakage.
    smiles_group = df.groupby("smiles")["source"].apply(lambda s: "PKS" if (s == "PKS").any() else "non-PKS").reset_index(name="label")

    # Determine if we can use stratification safely on unique SMILES.
    class_counts = smiles_group["label"].value_counts(dropna=False)
    min_per_class = class_counts.min() if not class_counts.empty else 0
    use_stratify = min_per_class >= 10

    if not use_stratify:
        print(
            "Warning: some classes have fewer than 10 unique SMILES; "
            "falling back to non-stratified split. Unique counts: " + str(class_counts.to_dict())
        )

    # First split on unique SMILES: Train (80%) vs Temp (20%)
    uniq_train, uniq_temp = train_test_split(
        smiles_group,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=smiles_group["label"] if use_stratify else None,
    )

    # Second split on unique SMILES: Temp -> Val/Test (50/50)
    uniq_val, uniq_test = train_test_split(
        uniq_temp,
        test_size=0.5,
        random_state=42,
        shuffle=True,
        stratify=uniq_temp["label"] if use_stratify else None,
    )

    # Map back to full dataframe: include all rows whose SMILES is in the respective set
    train_smiles_set = set(uniq_train["smiles"]) 
    val_smiles_set = set(uniq_val["smiles"]) 
    test_smiles_set = set(uniq_test["smiles"]) 

    df_train = df[df["smiles"].isin(train_smiles_set)].copy()
    df_val = df[df["smiles"].isin(val_smiles_set)].copy()
    df_test = df[df["smiles"].isin(test_smiles_set)].copy()

    print(
        "Split sizes (rows) — train: {}, val: {}, test: {}".format(
            len(df_train), len(df_val), len(df_test)
        )
    )
    print(
        "Unique SMILES per split — train: {}, val: {}, test: {}".format(
            len(train_smiles_set), len(val_smiles_set), len(test_smiles_set)
        )
    )

    # Save outputs
    save_df(df_train, out_train_dir, "train")
    save_df(df_val, out_val_dir, "val")
    save_df(df_test, out_test_dir, "test")
