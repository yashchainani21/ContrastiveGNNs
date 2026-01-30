#!/usr/bin/env python3
"""
Evaluate trained XGBoost classifier on the held-out test set.

This script is intentionally separate from training to prevent any possibility
of data leakage - test data is never loaded alongside training data.

Computes all metrics with bootstrap 95% confidence intervals.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

# =============================================================================
# Configuration
# =============================================================================

# Paths
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
TEST_PATH = DATA_DIR / "test" / "supcon_test_V1_ecfp4.parquet"

# Model files (use portable JSON format)
MODEL_NAME = "xgb_classifier_supcon_V1"
MODEL_PATH = MODELS_DIR / f"{MODEL_NAME}.json"
METADATA_PATH = MODELS_DIR / f"{MODEL_NAME}_metadata.json"
TEST_REPORT_PATH = MODELS_DIR / f"{MODEL_NAME}_test_report.json"

# Bootstrap settings
BOOTSTRAP_ITERATIONS = 1000
CI_LEVEL = 0.95
RANDOM_STATE = 42


# =============================================================================
# Data Loading
# =============================================================================


def load_fingerprint_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load parquet file and extract fingerprint features and labels."""
    df = pd.read_parquet(path)
    fp_cols = [c for c in df.columns if c.startswith("fp_")]
    X = df[fp_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)
    return X, y


def print_class_distribution(y: np.ndarray, split_name: str) -> None:
    """Print class distribution statistics."""
    n_total = len(y)
    n_pks = y.sum()
    n_non_pks = n_total - n_pks
    pks_ratio = n_pks / n_total * 100
    print(f"{split_name}: {n_total:,} samples | PKS: {n_pks:,} ({pks_ratio:.1f}%) | Non-PKS: {n_non_pks:,}")


# =============================================================================
# Model Loading
# =============================================================================


def load_model(model_path: Path, metadata_path: Path):
    """Load trained model and metadata using portable JSON format."""
    model = XGBClassifier()
    model.load_model(model_path)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return model, metadata


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================


def bootstrap_metric(
    y_true: np.ndarray, y_score: np.ndarray, metric_fn, n_iterations: int = BOOTSTRAP_ITERATIONS
) -> dict:
    """Compute metric with bootstrap 95% CI."""
    rng = np.random.default_rng(RANDOM_STATE)
    n_samples = len(y_true)
    bootstrap_scores = []

    for _ in range(n_iterations):
        indices = rng.integers(0, n_samples, size=n_samples)
        y_true_boot = y_true[indices]
        y_score_boot = y_score[indices]

        # Skip if only one class present in bootstrap sample
        if len(np.unique(y_true_boot)) < 2:
            continue

        try:
            score = metric_fn(y_true_boot, y_score_boot)
            bootstrap_scores.append(score)
        except Exception:
            continue

    bootstrap_scores = np.array(bootstrap_scores)
    alpha = 1 - CI_LEVEL
    lower = np.percentile(bootstrap_scores, alpha / 2 * 100)
    upper = np.percentile(bootstrap_scores, (1 - alpha / 2) * 100)
    mean = np.mean(bootstrap_scores)

    return {"mean": float(mean), "ci_lower": float(lower), "ci_upper": float(upper)}


def compute_all_metrics_with_ci(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute all metrics with bootstrap 95% CIs."""
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {}

    # AUPRC (primary metric)
    metrics["auprc"] = bootstrap_metric(y_true, y_prob, average_precision_score)

    # AUROC
    metrics["auroc"] = bootstrap_metric(y_true, y_prob, roc_auc_score)

    # Accuracy
    metrics["accuracy"] = bootstrap_metric(y_true, y_pred, accuracy_score)

    # Precision
    metrics["precision"] = bootstrap_metric(y_true, y_pred, precision_score)

    # Recall
    metrics["recall"] = bootstrap_metric(y_true, y_pred, recall_score)

    # F1
    metrics["f1"] = bootstrap_metric(y_true, y_pred, f1_score)

    return metrics


# =============================================================================
# Save Functions
# =============================================================================


def save_test_report(metrics: dict, model_metadata: dict) -> None:
    """Save test report with metrics and model info."""
    report = {
        "model_info": {
            "trained_on": model_metadata.get("trained_on", "unknown"),
            "n_train_samples": model_metadata.get("n_train_samples", "unknown"),
            "hyperparameters": {
                k: int(v) if k in ["max_depth", "n_estimators", "min_child_weight"] else float(v)
                for k, v in model_metadata.get("hyperparameters", {}).items()
            },
        },
        "test_metrics": metrics,
        "bootstrap_settings": {
            "iterations": BOOTSTRAP_ITERATIONS,
            "ci_level": CI_LEVEL,
        },
    }
    with open(TEST_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved test report to {TEST_REPORT_PATH}")


# =============================================================================
# Main Pipeline
# =============================================================================


def main():
    print("=" * 70)
    print("XGBoost Classifier - Test Set Evaluation")
    print("=" * 70)

    # Step 1: Load test data ONLY
    print("\n[1/4] Loading TEST data only (no train/val to prevent leakage)...")
    X_test, y_test = load_fingerprint_data(TEST_PATH)
    print(f"Loaded test: {X_test.shape}")
    print_class_distribution(y_test, "Test")

    # Step 2: Load trained model
    print("\n[2/4] Loading trained model...")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run 05_train_baseline_XGB_models.py and 05_export_xgb_model.py first."
        )
    model, model_metadata = load_model(MODEL_PATH, METADATA_PATH)
    print(f"Loaded model trained on: {model_metadata.get('trained_on', 'unknown')}")
    print(f"Training samples: {model_metadata.get('n_train_samples', 'unknown'):,}")

    # Step 3: Evaluate on test set with bootstrap CIs
    print(f"\n[3/4] Evaluating on test set with bootstrap CIs ({BOOTSTRAP_ITERATIONS} iterations)...")
    y_test_prob = model.predict_proba(X_test)[:, 1]
    test_metrics = compute_all_metrics_with_ci(y_test, y_test_prob)

    # Step 4: Save test report
    print("\n[4/4] Saving test report...")
    save_test_report(test_metrics, model_metadata)

    # Print results
    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)
    print(f"\nMetrics (with 95% CI from {BOOTSTRAP_ITERATIONS} bootstrap iterations):")
    print("-" * 55)
    print(f"{'Metric':<12} {'Mean':>10} {'95% CI':>30}")
    print("-" * 55)
    for metric in ["auprc", "auroc", "accuracy", "precision", "recall", "f1"]:
        m = test_metrics[metric]
        ci_str = f"[{m['ci_lower']:.4f}, {m['ci_upper']:.4f}]"
        print(f"{metric.upper():<12} {m['mean']:>10.4f} {ci_str:>30}")
    print("-" * 55)

    print("\nDone!")


if __name__ == "__main__":
    main()
