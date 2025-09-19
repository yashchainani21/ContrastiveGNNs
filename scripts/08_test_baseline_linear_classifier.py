from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    average_precision_score,
)


def find_input_file() -> Path:
    base_dir = Path("../data/test")
    parquet = base_dir / "baseline_test_ecfp4.parquet"
    csv = base_dir / "baseline_test_ecfp4.csv"
    if parquet.exists():
        return parquet
    if csv.exists():
        return csv
    raise FileNotFoundError(f"Could not find test set at {parquet} or {csv}")


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def encode_labels(df: pd.DataFrame) -> np.ndarray:
    if "source" not in df.columns:
        raise ValueError("Expected 'source' column in dataset")
    return (df["source"].astype(str) == "PKS").astype(np.int64).to_numpy()


def extract_features(df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    # Ensure all required columns exist
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing[:5]}{'...' if len(missing)>5 else ''}")
    return df[feature_columns].to_numpy(dtype=np.float32)


if __name__ == "__main__":
    # Load model and metadata
    model_path = Path("../models/baseline_linear_clf.pkl")
    payload = joblib.load(model_path)
    clf = payload["model"]
    feature_columns = payload["feature_columns"]

    in_path = find_input_file()
    df = load_dataset(in_path)

    y_true = encode_labels(df)
    X = extract_features(df, feature_columns)

    y_prob = None
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X)[:, 1]
    elif hasattr(clf, "decision_function"):
        # map decision function to [0,1] via sigmoid for AUC
        scores = clf.decision_function(X)
        y_prob = 1.0 / (1.0 + np.exp(-scores))

    y_pred = clf.predict(X)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None
    auprc = average_precision_score(y_true, y_prob) if y_prob is not None else None

    report = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "auprc": auprc,
        "confusion_matrix": cm,
        "n_samples": int(len(y_true)),
        "test_path": str(in_path),
        "model_path": str(model_path),
    }

    print(json.dumps(report, indent=2))

    out_dir = Path("../models")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "baseline_linear_clf_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved test report to {out_dir / 'baseline_linear_clf_test_report.json'}")
