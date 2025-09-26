from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score


def main():
    parser = argparse.ArgumentParser(description="Merge per-rank similarity shards and compute metrics.")
    parser.add_argument(
        "--glob",
        default="chem_similarity_nn_baseline_mpi.rank*.parquet",
        help="Glob pattern of shard files to merge (e.g., 'chem_similarity_nn_baseline_mpi.rank*.parquet' or 'mcs_similarity_nn_baseline_mpi.rank*.parquet')",
    )
    args = parser.parse_args()

    out_dir = Path("../data/processed")
    shards = sorted(out_dir.glob(args.glob))
    if not shards:
        raise SystemExit(f"No shard files found for glob: {args.glob}")

    parts = []
    for p in shards:
        try:
            parts.append(pd.read_parquet(p))
        except Exception as e:
            print(f"Warning: skipping shard {p} due to read error: {e}")

    if not parts:
        raise SystemExit("All shard reads failed; nothing to merge.")

    merged = pd.concat(parts, ignore_index=True)
    if "index" in merged.columns:
        merged = merged.sort_values("index").drop(columns=["index"])  # clean index if present

    # Metrics (optional; robust to missing classes)
    y_true = (merged["true_label"] == "PKS").astype(np.int8).to_numpy()
    y_pred = (merged["pred_label"] == "PKS").astype(np.int8).to_numpy()
    acc = accuracy_score(y_true, y_pred)
    try:
        auroc = roc_auc_score(y_true, y_pred)
    except Exception:
        auroc = float("nan")
    try:
        auprc = average_precision_score(y_true, y_pred)
    except Exception:
        auprc = float("nan")
    print(f"Merged shards — ACC={acc:.4f} | AUROC={auroc:.4f} | AUPRC={auprc:.4f}")

    # Derive final path based on shard prefix
    final_name = "chem_similarity_nn_baseline_mpi.parquet"
    if "mcs_" in args.glob:
        final_name = "mcs_similarity_nn_baseline_mpi.parquet"
    final_path = out_dir / final_name
    merged.to_parquet(final_path, index=False)
    print(f"Saved merged results to {final_path}")

    # Also write metrics JSON to models folder for tracking
    models_dir = Path("../models")
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_name = (
        "mcs_similarity_nn_baseline_metrics.json"
        if "mcs_" in args.glob
        else "chem_similarity_nn_baseline_metrics.json"
    )
    metrics_path = models_dir / metrics_name
    def _clean(x):
        try:
            return None if (isinstance(x, float) and (x != x)) else float(x)
        except Exception:
            return None
    import json
    with open(metrics_path, "w") as f:
        json.dump({
            "acc": _clean(acc),
            "auroc": _clean(auroc),
            "auprc": _clean(auprc),
            "n_rows": int(len(merged)),
            "n_pos": int(int((merged["true_label"] == "PKS").sum())),
            "source": metrics_name.replace("_metrics.json", ""),
        }, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Cleanup: remove all per-rank shard files for both chem and MCS baselines
    removed = 0
    for pat in ("chem_similarity_nn_baseline*", "mcs_similarity_nn_baseline*"):
        for p in out_dir.glob(pat):
            # keep the final merged files
            if p.name in {"chem_similarity_nn_baseline_mpi.parquet", "mcs_similarity_nn_baseline_mpi.parquet"}:
                continue
            try:
                p.unlink()
                removed += 1
            except Exception as e:
                print(f"Warning: failed to remove {p}: {e}")
    print(f"Cleanup: removed {removed} shard file(s)")

if __name__ == "__main__":
    main()
