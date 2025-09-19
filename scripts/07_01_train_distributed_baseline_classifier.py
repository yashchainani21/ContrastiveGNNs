"""
Distributed baseline classifier training using Dask-ML Incremental wrapper.

Trains a binary logistic model to distinguish PKS (1) vs non-PKS (0) on ECFP4 bits.

Cluster connection:
- If env DASK_SCHEDULER is set, connects to that address (e.g., tcp://scheduler:8786)
- Else if env DASK_SCHEDULER_FILE is set, uses that scheduler file path
- Else starts a LocalCluster (single node, multi-core)

Run examples (single node):
  python scripts/07_01_train_distributed_baseline_classifier.py

Run on multi-node (one option):
  # Launch a dask scheduler + workers via your cluster manager or dask-mpi
  mpiexec -n <N> dask-mpi --nthreads <T>
  # Or use dask-jobqueue SLURMCluster and then run this script pointing to scheduler
  DASK_SCHEDULER=tcp://<scheduler-host>:8786 \\
    python scripts/07_01_train_distributed_baseline_classifier.py
"""

# Bypass RAPIDS dask import shim when GPUs aren't used
import os
os.environ.setdefault("RAPIDS_NO_INITIALIZE", "1")

from pathlib import Path
import json
import joblib
import numpy as np
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from dask_ml.wrappers import Incremental
from sklearn.linear_model import SGDClassifier


def connect_client() -> Client:
    addr = os.getenv("DASK_SCHEDULER")
    sched_file = os.getenv("DASK_SCHEDULER_FILE")
    if addr:
        print(f"Connecting to Dask scheduler at {addr}")
        return Client(addr)
    if sched_file and Path(sched_file).exists():
        print(f"Connecting via scheduler file {sched_file}")
        return Client(scheduler_file=sched_file)
    # Fallback: local cluster (multi-core)
    n_workers = max(1, (os.cpu_count() or 2) - 1)
    threads_per_worker = 1
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
    print(f"Started LocalCluster at {cluster.dashboard_link} with {n_workers} workers")
    return Client(cluster)


def find_train_file() -> Path:
    base = Path("../data/train")
    for name in ("baseline_train_ecfp4.parquet", "baseline_train_ecfp4.csv",
                 "baseline_train.parquet", "baseline_train.csv"):
        p = base / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find train set under {base}")


def get_feature_columns_any(path: Path) -> list[str]:
    if path.suffix == ".parquet":
        # Read only schema by loading a small part
        df_head = dd.read_parquet(path, columns=None).head(1)
    else:
        df_head = dd.read_csv(path).head(1)
    cols = [c for c in df_head.columns if str(c).startswith("fp_")]
    if not cols:
        raise ValueError("No fingerprint columns found (expected fp_0..fp_2047)")
    cols_sorted = sorted(cols, key=lambda s: int(str(s).split("_")[1]))
    return cols_sorted


if __name__ == "__main__":
    client = connect_client()
    print("Dask client connected.")

    train_path = find_train_file()
    print(f"Loading training data from {train_path}")

    fp_cols = get_feature_columns_any(train_path)

    if train_path.suffix == ".parquet":
        ddf = dd.read_parquet(train_path, columns=["smiles", "source", *fp_cols])
    else:
        ddf = dd.read_csv(train_path, usecols=["smiles", "source", *fp_cols])

    # Binary labels: PKS -> 1 else 0
    ddf = ddf.assign(label=(ddf["source"].astype(str) == "PKS").astype("int8"))

    # Convert to dask arrays for Incremental fit
    X = ddf[fp_cols].to_dask_array(lengths=True).astype(np.float32)
    y = ddf["label"].to_dask_array(lengths=True)

    # Set up SGD with logistic loss; suitable for partial_fit
    base_est = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=1,  # one pass per partial_fit chunk
        tol=None,
        random_state=42,
        class_weight="balanced",
    )
    inc = Incremental(base_est)

    print("Starting distributed incremental training …")
    inc.fit(X, y, classes=[0, 1])
    est = inc.estimator_ if hasattr(inc, "estimator_") else base_est
    print("Training complete.")

    out_dir = Path("../models")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "baseline_linear_clf_distributed.pkl"
    meta_path = out_dir / "baseline_linear_clf_distributed.meta.json"

    payload = {
        "model": est,
        "feature_columns": fp_cols,
        "label_mapping": {"PKS": 1, "bio/chem": 0},
        "train_path": str(train_path),
        "trainer": "dask-ml Incremental(SGDClassifier, log_loss)",
    }
    joblib.dump(payload, model_path)

    try:
        info = client.scheduler_info()
    except Exception:
        info = None
    with open(meta_path, "w") as f:
        json.dump({
            "feature_columns": fp_cols,
            "label_mapping": {"PKS": 1, "bio_or_chem": 0},
            "train_path": str(train_path),
            "model_path": str(model_path),
            "trainer": "dask-ml Incremental(SGDClassifier, log_loss)",
            "scheduler_info": info,
        }, f, indent=2)

    print(f"Saved distributed model to {model_path} and metadata to {meta_path}")
