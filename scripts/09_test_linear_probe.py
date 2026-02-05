#!/usr/bin/env python3
"""
Test Linear Probes: GNN Embeddings vs ECFP4 Fingerprints

Lightweight evaluation script (no torch/RDKit needed) that loads pre-saved
probes and test data, evaluates both on the test set, and prints a side-by-side
comparison table.

Prerequisites:
    - Run scripts/08_train_linear_probe_on_GNN_embeddings.py to produce:
        models/supcon_gnn/linear_probe.joblib
        models/supcon_gnn/linear_probe_ecfp4.joblib
        models/supcon_gnn/embeddings_test.npz
    - Run scripts/04_local_fingerprint_molecules.py to produce:
        data/test/supcon_test_ecfp4.parquet

Usage:
    python scripts/09_test_linear_probe.py
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


# =============================================================================
# Configuration
# =============================================================================

GNN_PROBE_PATH = "models/supcon_gnn/linear_probe.joblib"
GNN_EMBEDDINGS_PATH = "models/supcon_gnn/embeddings_test.npz"
ECFP4_PROBE_PATH = "models/supcon_gnn/linear_probe_ecfp4.joblib"
ECFP4_DATA_PATH = "data/test/supcon_test_ecfp4.parquet"
OUTPUT_PATH = "models/supcon_gnn/linear_probe_comparison.json"


# =============================================================================
# Data Loading
# =============================================================================

def load_gnn_test_embeddings(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load GNN test embeddings from .npz file.

    Args:
        npz_path: Path to embeddings_test.npz (saved by script 08)

    Returns:
        embeddings: [N, embed_dim] array
        labels: [N] array
    """
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(
            f"GNN test embeddings not found: {npz_path}\n"
            f"Run scripts/08_train_linear_probe_on_GNN_embeddings.py first."
        )
    data = np.load(npz_path)
    return data["embeddings"], data["labels"]


def load_ecfp4_test_data(parquet_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load ECFP4 fingerprints from parquet.

    Args:
        parquet_path: Path to supcon_test_ecfp4.parquet

    Returns:
        X: [N, n_bits] float32 fingerprint array
        y: [N] int labels
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(
            f"ECFP4 test data not found: {parquet_path}\n"
            f"Run scripts/04_local_fingerprint_molecules.py first."
        )
    df = pd.read_parquet(parquet_path)
    fp_cols = [c for c in df.columns if c.startswith("fp_")]
    X = df[fp_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int64)
    return X, y


def load_probe(probe_path: str, name: str) -> LogisticRegression:
    """Load a saved linear probe.

    Args:
        probe_path: Path to .joblib file
        name: Human-readable name for error messages

    Returns:
        Trained LogisticRegression classifier
    """
    path = Path(probe_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{name} probe not found: {probe_path}\n"
            f"Run scripts/08_train_linear_probe_on_GNN_embeddings.py first."
        )
    return joblib.load(probe_path)


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_probe(
    clf: LogisticRegression,
    features: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Evaluate linear probe on a dataset.

    Args:
        clf: Trained classifier
        features: [N, dim] feature array
        labels: [N] ground truth labels

    Returns:
        Dictionary with accuracy, AUPRC, AUROC, and F1 metrics
    """
    preds = clf.predict(features)
    probs = clf.predict_proba(features)[:, 1]

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "auprc": float(average_precision_score(labels, probs)),
        "auroc": float(roc_auc_score(labels, probs)),
        "f1": float(f1_score(labels, preds)),
    }


# =============================================================================
# Comparison Display
# =============================================================================

def print_comparison_table(gnn_metrics: Dict[str, float], ecfp4_metrics: Dict[str, float]) -> None:
    """Print side-by-side comparison table with delta column."""
    metric_names = ["accuracy", "auprc", "auroc", "f1"]
    labels = ["Accuracy", "AUPRC", "AUROC", "F1"]

    print("\n" + "=" * 72)
    print("Test Set Comparison: GNN Embeddings vs ECFP4 Fingerprints")
    print("=" * 72)
    print(f"{'Metric':<12} {'GNN':>12} {'ECFP4':>12} {'Delta':>12}")
    print("-" * 72)

    for label, key in zip(labels, metric_names):
        gnn_val = gnn_metrics[key]
        ecfp4_val = ecfp4_metrics[key]
        delta = gnn_val - ecfp4_val
        sign = "+" if delta >= 0 else ""
        print(f"{label:<12} {gnn_val:>12.4f} {ecfp4_val:>12.4f} {sign}{delta:>11.4f}")

    print("=" * 72)


# =============================================================================
# Main
# =============================================================================

def main():
    print("Linear Probe Test Set Evaluation")
    print("=" * 72)

    # Load GNN probe and test embeddings
    print("\nLoading GNN probe and test embeddings...")
    gnn_clf = load_probe(GNN_PROBE_PATH, "GNN")
    gnn_features, gnn_labels = load_gnn_test_embeddings(GNN_EMBEDDINGS_PATH)
    print(f"  GNN embeddings: {gnn_features.shape}, PKS ratio: {gnn_labels.mean():.3f}")

    # Load ECFP4 probe and test data
    print("\nLoading ECFP4 probe and test data...")
    ecfp4_clf = load_probe(ECFP4_PROBE_PATH, "ECFP4")
    ecfp4_features, ecfp4_labels = load_ecfp4_test_data(ECFP4_DATA_PATH)
    print(f"  ECFP4 features: {ecfp4_features.shape}, PKS ratio: {ecfp4_labels.mean():.3f}")

    # Evaluate both probes
    print("\nEvaluating probes on test set...")
    gnn_metrics = evaluate_probe(gnn_clf, gnn_features, gnn_labels)
    ecfp4_metrics = evaluate_probe(ecfp4_clf, ecfp4_features, ecfp4_labels)

    # Print comparison
    print_comparison_table(gnn_metrics, ecfp4_metrics)

    # Save comparison JSON
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comparison = {
        "gnn_embeddings": gnn_metrics,
        "ecfp4_fingerprints": ecfp4_metrics,
        "delta": {
            key: gnn_metrics[key] - ecfp4_metrics[key]
            for key in gnn_metrics
        },
    }

    print(f"\nSaving comparison to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
