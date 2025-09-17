from pathlib import Path
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from collections import Counter


def find_input_file() -> Path:
    base_dir = Path("../data/train")
    parquet = base_dir / "baseline_train_ecfp4.parquet"
    csv = base_dir / "baseline_train_ecfp4.csv"
    if parquet.exists():
        return parquet
    if csv.exists():
        return csv
    raise FileNotFoundError(f"Could not find train set at {parquet} or {csv}")


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def encode_labels(df: pd.DataFrame) -> np.ndarray:
    if "source" not in df.columns:
        raise ValueError("Expected 'source' column in dataset")
    # PKS -> 1, bio/chem -> 0
    return (df["source"].astype(str) == "PKS").astype(np.int64).to_numpy()


def extract_features(df: pd.DataFrame) -> np.ndarray:
    fp_cols = [c for c in df.columns if c.startswith("fp_")]
    if not fp_cols:
        raise ValueError("No fingerprint columns found (expected fp_0..fp_2047)")
    # sort numerically by index
    fp_cols_sorted = sorted(fp_cols, key=lambda s: int(s.split("_")[1]))
    X = df[fp_cols_sorted].to_numpy(dtype=np.float32)
    return X, fp_cols_sorted


if __name__ == "__main__":
    in_path = find_input_file()
    df = load_dataset(in_path)

    y = encode_labels(df)
    X, fp_cols = extract_features(df)

    # Inspect class balance and set class_weight accordingly
    counts = Counter(y)
    print(f"Class counts (0=non-PKS, 1=PKS): {dict(counts)}")
    class_weight = "balanced"

    # Simple linear classifier; saga supports n_jobs for multinomial but also fine here.
    # Bits are 0/1 so no scaling required.
    clf = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=1.0,
        max_iter=2000,
        n_jobs=-1,
        class_weight=class_weight,
        random_state=42,
    )
    clf.fit(X, y)

    out_dir = Path("../models")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "baseline_linear_clf.pkl"
    meta_path = out_dir / "baseline_linear_clf.meta.json"

    joblib.dump({
        "model": clf,
        "feature_columns": fp_cols,
        "label_mapping": {"PKS": 1, "bio/chem": 0},
        "train_rows": int(X.shape[0]),
        "train_path": str(in_path),
        "class_weight": class_weight,
        "class_counts": dict(counts),
    }, model_path)

    with open(meta_path, "w") as f:
        json.dump({
            "feature_columns": fp_cols,
            "label_mapping": {"PKS": 1, "bio_or_chem": 0},
            "train_rows": int(X.shape[0]),
            "train_path": str(in_path),
            "model_path": str(model_path),
            "classifier": "LogisticRegression(saga, l2, C=1.0, class_weight=balanced)",
            "class_weight": class_weight,
            "class_counts": dict(counts),
        }, f, indent=2)

    print(f"Saved model to {model_path} and metadata to {meta_path}")
