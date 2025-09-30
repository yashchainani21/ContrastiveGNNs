import os
import argparse
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def parse_timing_txt(path: str) -> Dict[str, float]:
    data: Dict[str, float] = {}
    if not os.path.exists(path):
        print(f"Warning: timing file not found: {path}")
        return data
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            try:
                data[k] = float(v)
            except ValueError:
                # non-float entries (e.g., counts) are ignored here
                continue
    return data


def main():
    parser = argparse.ArgumentParser(description="Plot time comparison across three routes as a bar chart")
    default_dir = os.path.dirname(__file__)
    parser.add_argument(
        "--retrotide-timing",
        default=os.path.join(default_dir, "doranet_retro_cryptofolione_retrotide_timing.txt"),
        help="Timing file for RetroTide (expects key 'retrotide_total_seconds')",
    )
    parser.add_argument(
        "--linear-timing",
        default=os.path.join(default_dir, "linear_inference_on_doranet_products_timing.txt"),
        help="Timing file for linear inference (expects key 'total_infer_seconds')",
    )
    parser.add_argument(
        "--membership-timing",
        default=os.path.join(default_dir, "doranet_vs_pks_membership_timing.txt"),
        help="Timing file for membership checks (expects key 'membership_total_seconds')",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(default_dir, "time_comparison_bar.png"),
        help="Output image filepath (PNG)",
    )
    parser.add_argument(
        "--title",
        default="Runtime Comparison",
        help="Plot title",
    )
    args = parser.parse_args()

    # Load timings
    t_ret = parse_timing_txt(args.retrotide_timing)
    t_lin = parse_timing_txt(args.linear_timing)
    t_mem = parse_timing_txt(args.membership_timing)

    bars: List[Tuple[str, float]] = []

    if "retrotide_total_seconds" in t_ret:
        bars.append(("RetroTide (total)", t_ret["retrotide_total_seconds"]))
    else:
        print("Warning: key 'retrotide_total_seconds' not found; skipping RetroTide bar")

    if "total_infer_seconds" in t_lin:
        bars.append(("Linear Inference (total)", t_lin["total_infer_seconds"]))
    else:
        print("Warning: key 'total_infer_seconds' not found; skipping Linear Inference bar")

    if "membership_total_seconds" in t_mem:
        bars.append(("PKS Membership (total)", t_mem["membership_total_seconds"]))
    else:
        print("Warning: key 'membership_total_seconds' not found; skipping Membership bar")

    if not bars:
        raise SystemExit("No valid timing entries found. Ensure timing files exist and have expected keys.")

    labels, values = zip(*bars)

    plt.figure(figsize=(8, 4))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    plt.bar(range(len(values)), values, color=colors[: len(values)])
    plt.xticks(range(len(values)), labels, rotation=15, ha="right")
    plt.ylabel("Seconds (total)")
    plt.title(args.title)
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.2f}s", ha="center", va="bottom")
    plt.tight_layout()
    out_path = os.path.abspath(args.out)
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()

