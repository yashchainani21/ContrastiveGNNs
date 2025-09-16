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

    # Determine if we can use stratification safely.
    # For two-stage split (80/20 then 50/50), require at least 10 examples per class
    # so temp (20%) has at least 2 per class for the second stratified split.
    class_counts = df["source"].value_counts(dropna=False)
    min_per_class = class_counts.min() if not class_counts.empty else 0
    use_stratify = min_per_class >= 10

    if not use_stratify:
        print(
            "Warning: some classes have fewer than 10 samples; "
            "falling back to non-stratified split. Counts: " + str(class_counts.to_dict())
        )

    # First split: Train (80%) vs Temp (20%)
    df_train, df_temp = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=df["source"] if use_stratify else None,
    )

    # Second split: split Temp into Val (10%) and Test (10%) of the total -> 50/50 of temp
    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.5,
        random_state=42,
        shuffle=True,
        stratify=df_temp["source"] if use_stratify else None,
    )

    print(
        "Split sizes — train: {}, val: {}, test: {}".format(
            len(df_train), len(df_val), len(df_test)
        )
    )

    # Save outputs
    save_df(df_train, out_train_dir, "train")
    save_df(df_val, out_val_dir, "val")
    save_df(df_test, out_test_dir, "test")

