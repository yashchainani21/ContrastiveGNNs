from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score


def main():
    out_dir = Path("../data/processed")
    shards = sorted(out_dir.glob("chem_similarity_nn_baseline_mpi.rank*.parquet"))
    if not shards:
        raise SystemExit("No shard files found. Expected chem_similarity_nn_baseline_mpi.rank*.parquet")

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

    final_path = out_dir / "chem_similarity_nn_baseline_mpi.parquet"
    merged.to_parquet(final_path, index=False)
    print(f"Saved merged results to {final_path}")


if __name__ == "__main__":
    main()

