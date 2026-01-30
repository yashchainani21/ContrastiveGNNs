#!/usr/bin/env python3
"""
Train XGBoost classifiers on ECFP4 fingerprints with Bayesian hyperparameter optimization.

Uses validation AUPRC as the optimization target. After finding optimal hyperparameters,
retrains on combined train+val data for maximum training samples.

Test set evaluation is handled separately by 06_test_baseline_XGB_models.py to prevent
any possibility of data leakage.
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

# =============================================================================
# Configuration
# =============================================================================

# Paths
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
TRAIN_PATH = DATA_DIR / "train" / "supcon_train_V1_ecfp4.parquet"
VAL_PATH = DATA_DIR / "val" / "supcon_val_V1_ecfp4.parquet"

# Output files
MODEL_NAME = "xgb_classifier_supcon_V1"
MODEL_PATH_PKL = MODELS_DIR / f"{MODEL_NAME}.pkl"  # Pickle format (legacy)
MODEL_PATH_JSON = MODELS_DIR / f"{MODEL_NAME}.json"  # Portable JSON format
METADATA_PATH = MODELS_DIR / f"{MODEL_NAME}_metadata.json"  # Model metadata
HYPERPARAMS_PATH = MODELS_DIR / f"{MODEL_NAME}_hyperparams.json"
SEARCH_HISTORY_PATH = MODELS_DIR / f"{MODEL_NAME}_search_history.json"

# Bayesian optimization settings
INIT_POINTS = 5  # Random exploration
N_ITER = 25  # Optimization iterations

# Hyperparameter search bounds
PARAM_BOUNDS = {
    "learning_rate": (0.01, 0.5),
    "max_depth": (3, 12),
    "n_estimators": (50, 500),
    "min_child_weight": (1, 10),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "colsample_bylevel": (0.5, 1.0),
    "colsample_bynode": (0.5, 1.0),
    "reg_alpha": (0.0, 1.0),
    "reg_lambda": (0.0, 1.0),
    "scale_pos_weight": (1.0, 5.0),
}

# XGBoost fixed settings
RANDOM_STATE = 42
N_JOBS = os.cpu_count()
TREE_METHOD = "hist"
EVAL_METRIC = "logloss"

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
# XGBoost Model Creation
# =============================================================================


def create_xgb_classifier(params: dict) -> XGBClassifier:
    """Create XGBClassifier with given parameters."""
    # Convert integer parameters
    int_params = ["max_depth", "n_estimators", "min_child_weight"]
    params_copy = params.copy()
    for p in int_params:
        if p in params_copy:
            params_copy[p] = int(params_copy[p])

    return XGBClassifier(
        **params_copy,
        n_jobs=N_JOBS,
        tree_method=TREE_METHOD,
        eval_metric=EVAL_METRIC,
        random_state=RANDOM_STATE,
        verbosity=0,
    )


# =============================================================================
# Bayesian Optimization
# =============================================================================


def create_objective_function(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
    """Create objective function for Bayesian optimization (returns validation AUPRC)."""

    def objective(**params):
        model = create_xgb_classifier(params)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        auprc = average_precision_score(y_val, y_prob)
        return auprc

    return objective


def run_bayesian_optimization(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> tuple[dict, list[dict]]:
    """Run Bayesian optimization and return best params and search history."""
    objective = create_objective_function(X_train, y_train, X_val, y_val)

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=PARAM_BOUNDS,
        random_state=RANDOM_STATE,
        verbose=2,
    )

    optimizer.maximize(init_points=INIT_POINTS, n_iter=N_ITER)

    # Extract search history
    search_history = []
    for i, res in enumerate(optimizer.res):
        search_history.append(
            {
                "iteration": i + 1,
                "params": res["params"],
                "val_auprc": res["target"],
            }
        )

    best_params = optimizer.max["params"]
    return best_params, search_history


def compute_metrics_no_ci(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute metrics without bootstrap (for quick validation checks)."""
    return {
        "auprc": float(average_precision_score(y_true, y_prob)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
    }


# =============================================================================
# Save Functions
# =============================================================================


def save_model(model: XGBClassifier, best_params: dict, n_train_samples: int) -> None:
    """Save model in both pickle (legacy) and portable JSON formats."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Convert integer parameters for JSON
    params_json = best_params.copy()
    for p in ["max_depth", "n_estimators", "min_child_weight"]:
        if p in params_json:
            params_json[p] = int(params_json[p])

    # Save pickle format (legacy, may have version compatibility issues)
    data = {
        "model": model,
        "hyperparameters": best_params,
        "random_state": RANDOM_STATE,
        "trained_on": "train+val combined",
        "n_train_samples": n_train_samples,
    }
    with open(MODEL_PATH_PKL, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved model (pickle) to {MODEL_PATH_PKL}")

    # Save portable JSON format (works across XGBoost versions)
    model.save_model(MODEL_PATH_JSON)
    print(f"Saved model (JSON) to {MODEL_PATH_JSON}")

    # Save metadata separately for JSON format
    metadata = {
        "hyperparameters": params_json,
        "random_state": RANDOM_STATE,
        "trained_on": "train+val combined",
        "n_train_samples": n_train_samples,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {METADATA_PATH}")


def save_hyperparams(best_params: dict, val_metrics: dict) -> None:
    """Save hyperparameters and validation metrics."""
    # Convert integer parameters for JSON
    params_json = best_params.copy()
    for p in ["max_depth", "n_estimators", "min_child_weight"]:
        if p in params_json:
            params_json[p] = int(params_json[p])

    data = {
        "hyperparameters": params_json,
        "validation_metrics": val_metrics,
        "optimization_settings": {
            "init_points": INIT_POINTS,
            "n_iter": N_ITER,
            "objective": "validation_auprc",
        },
    }
    with open(HYPERPARAMS_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved hyperparameters to {HYPERPARAMS_PATH}")


def save_search_history(search_history: list[dict]) -> None:
    """Save Bayesian optimization search history."""
    with open(SEARCH_HISTORY_PATH, "w") as f:
        json.dump(search_history, f, indent=2)
    print(f"Saved search history to {SEARCH_HISTORY_PATH}")


# =============================================================================
# Main Pipeline
# =============================================================================


def main():
    print("=" * 70)
    print("XGBoost Classifier with Bayesian Hyperparameter Optimization")
    print("=" * 70)

    # Step 1: Load train and validation data (NO test data - handled separately)
    print("\n[1/7] Loading fingerprint data (train + val only)...")
    X_train, y_train = load_fingerprint_data(TRAIN_PATH)
    X_val, y_val = load_fingerprint_data(VAL_PATH)
    print(f"Loaded train: {X_train.shape}, val: {X_val.shape}")

    # Step 2: Log class distributions
    print("\n[2/7] Class distributions:")
    print_class_distribution(y_train, "Train")
    print_class_distribution(y_val, "Val")

    # Step 3: Train default model as baseline
    print("\n[3/7] Training DEFAULT XGBoost model (no tuning)...")
    default_model = XGBClassifier(
        n_jobs=N_JOBS,
        tree_method=TREE_METHOD,
        eval_metric=EVAL_METRIC,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    default_model.fit(X_train, y_train)

    # Evaluate default model on val
    y_val_prob_default = default_model.predict_proba(X_val)[:, 1]
    default_val_metrics = compute_metrics_no_ci(y_val, y_val_prob_default)
    print(f"Default model val AUPRC: {default_val_metrics['auprc']:.4f}")

    # Step 4: Run Bayesian optimization
    print(f"\n[4/7] Running Bayesian optimization ({INIT_POINTS} init + {N_ITER} iter)...")
    best_params, search_history = run_bayesian_optimization(X_train, y_train, X_val, y_val)

    # Find the best AUPRC from search history
    best_auprc = max(h["val_auprc"] for h in search_history)
    print(f"\nBest validation AUPRC: {best_auprc:.4f}")

    # Step 5: Save search history
    print("\n[5/7] Saving search history...")
    save_search_history(search_history)

    # Step 6: Combine train+val and train final model
    print("\n[6/7] Training final model on COMBINED train+val data...")
    X_trainval = np.concatenate([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    print(f"Combined train+val: {X_trainval.shape}")
    print_class_distribution(y_trainval, "Train+Val")

    final_model = create_xgb_classifier(best_params)
    final_model.fit(X_trainval, y_trainval)

    # Step 7: Save model and hyperparameters
    print("\n[7/7] Saving model and hyperparameters...")
    # Save validation metrics from hyperparameter search (before combining)
    val_metrics_for_saving = {"auprc": best_auprc}
    save_model(final_model, best_params, n_train_samples=len(y_trainval))
    save_hyperparams(best_params, val_metrics_for_saving)

    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"\nDefault model val AUPRC: {default_val_metrics['auprc']:.4f}")
    print(f"Optimized model val AUPRC: {best_auprc:.4f}")
    print(f"Improvement: {(best_auprc - default_val_metrics['auprc']) * 100:.2f} percentage points")

    print(f"\nFinal model trained on {len(y_trainval):,} samples (train+val combined)")

    print("\nBest hyperparameters:")
    for k, v in best_params.items():
        if k in ["max_depth", "n_estimators", "min_child_weight"]:
            print(f"  {k}: {int(v)}")
        else:
            print(f"  {k}: {v:.4f}")

    print("\nNext step: Run 05_test_baseline_XGB_models.py to evaluate on test set")
    print("\nDone!")


if __name__ == "__main__":
    main()
