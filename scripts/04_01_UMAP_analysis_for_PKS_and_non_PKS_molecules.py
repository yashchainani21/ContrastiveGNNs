from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def find_split(base: Path, split: str) -> Path | None:
    candidates = [
        base / split / f"baseline_{split}_ecfp4.parquet",
        base / split / f"baseline_{split}_ecfp4.csv",
        base / split / f"baseline_{split}.parquet",
        base / split / f"baseline_{split}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


if __name__ == "__main__":
    # Prefer RAPIDS cuML UMAP; fallback to umap-learn on CPU if unavailable
    try:
        from cuml.manifold import UMAP as UMAP_GPU
        UMAP_IMPL = "cuml"
    except Exception:
        try:
            from umap import UMAP as UMAP_CPU
            UMAP_IMPL = "umap-learn"
        except Exception as e:
            raise RuntimeError(
                "Neither RAPIDS cuML UMAP nor umap-learn is available. Install one of them."
            ) from e

    data_dir = Path("../data")
    splits = ["train", "val", "test"]
    paths = {s: find_split(data_dir, s) for s in splits}

    dfs = []
    for s, p in paths.items():
        if p is None:
            print(f"Warning: missing split {s}; skipping")
            continue
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
        df["split"] = s
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No split files found to load")

    df_all = pd.concat(dfs, ignore_index=True)

    fp_cols = [c for c in df_all.columns if str(c).startswith("fp_")]
    if not fp_cols:
        raise ValueError("No fingerprint columns found (expected fp_0..fp_2047)")
    fp_cols = sorted(fp_cols, key=lambda s: int(str(s).split("_")[1]))

    X = df_all[fp_cols].to_numpy(dtype=np.float32)
    y = (df_all["source"].astype(str) == "PKS").astype(int).to_numpy()

    n = X.shape[0]
    print(f"Loaded {n:,} molecules with {X.shape[1]}-dim ECFP4 bits")

    # Configure UMAP parameters
    n_neighbors = int(min(50, max(10, np.sqrt(n) // 10)))  # heuristic
    min_dist = 0.1
    n_components = 2

    if UMAP_IMPL == "cuml":
        # Use NN-descent for large datasets; allow host data to avoid GPU memory spikes
        build_algo = "nn_descent" if n >= 50000 else "auto"
        build_kwds = {"nnd_graph_degree": 32}
        if n >= 200000:
            build_kwds.update({"nnd_do_batch": True, "nnd_n_clusters": 4})

        try:
            umap = UMAP_GPU(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=min_dist,
                build_algo=build_algo,
                build_kwds=build_kwds,
                random_state=42,
                verbose=True,
            )
            # Feed host data explicitly for large datasets
            data_on_host = n >= 50000
            emb = umap.fit_transform(X, data_on_host=data_on_host)
        except Exception as e:
            print("cuML UMAP failed (likely CUDA/CuPy toolchain issue):", repr(e))
            print("Falling back to CPU umap-learn…")
            from umap import UMAP as UMAP_CPU
            umap = UMAP_CPU(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=min_dist,
                random_state=42,
                verbose=True,
            )
            emb = umap.fit_transform(X)
    else:
        # CPU umap-learn fallback
        umap = UMAP_CPU(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            random_state=42,
            verbose=True,
        )
        emb = umap.fit_transform(X)

    df_all["umap_1"] = emb[:, 0]
    df_all["umap_2"] = emb[:, 1]
    df_all["is_pks"] = np.where(df_all["source"].astype(str) == "PKS", "PKS", "non-PKS")

    out_parquet = data_dir / "processed" / "pks_nonpks_umap_embeddings.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_all[["smiles", "source", "split", "umap_1", "umap_2", "is_pks"]].to_parquet(
        out_parquet, index=False
    )
    print(f"Saved embeddings to {out_parquet}")

    # Plot
    # High-visibility palette and styling
    sns.set_style("white")
    plt.figure(figsize=(9, 7), facecolor="white")
    palette = {"PKS": "#0C7BDC", "non-PKS": "#D62728"}  # bright blue vs red
    sns.scatterplot(
        data=df_all.sample(min(200000, len(df_all)), random_state=42),
        x="umap_1",
        y="umap_2",
        hue="is_pks",
        palette=palette,
        alpha=0.7,
        s=12,
        linewidth=0,
        rasterized=True,
    )
    plt.title("UMAP of ECFP4 fingerprints: PKS vs non-PKS", fontsize=13)
    plt.legend(title="label", markerscale=2)
    plt.tight_layout()
    out_png = Path("../plots") / "pks_nonpks_umap.png"
    plt.savefig(out_png, dpi=300)
    print(f"Saved plot to {out_png}")
