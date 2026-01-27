"""
Create leak-free train/val/test splits for Supervised Contrastive Learning.

Uses connected-component splitting to prevent SMILES leakage: the same SMILES
can appear across multiple triplets, so we group all connected triplets together
and split at the component level.

Input: data/processed/pks_augmentation_pairs.parquet (triplet format)
Output: Flat (smiles, label, source, triplet_id) files for SupCon training

Usage:
    python scripts/05_create_supcon_splits.py
"""

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


class UnionFind:
    """Union-Find data structure for connected-component detection."""

    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> None:
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


def build_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build connected components from triplet data.

    Edges connect PKS -> enzymatic_aug and PKS -> synthetic_aug.
    All SMILES in a connected component will be assigned to the same split.

    Returns dataframe with added 'component_id' column.
    """
    uf = UnionFind()

    for _, row in df.iterrows():
        uf.union(row['pks_smiles'], row['enzymatic_aug_smiles'])
        uf.union(row['pks_smiles'], row['synthetic_aug_smiles'])

    # Map triplets to components via the PKS SMILES root
    df = df.copy()
    df['component_id'] = df['pks_smiles'].apply(uf.find)
    return df


def split_components_greedy(
    df: pd.DataFrame,
    target_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> Dict[str, str]:
    """
    Assign components to splits using greedy bin packing.

    Sorts components by size (descending) and assigns each to the split
    that is furthest below its target size. This approximates the target
    ratios while keeping all triplets in a component together.

    Returns mapping of component_id -> split name ('train', 'val', 'test').
    """
    component_sizes = df.groupby('component_id').size().to_dict()
    total = sum(component_sizes.values())

    targets = {
        'train': total * target_ratios[0],
        'val': total * target_ratios[1],
        'test': total * target_ratios[2]
    }
    current = {'train': 0, 'val': 0, 'test': 0}
    assignments: Dict[str, str] = {}

    # Sort by size descending for better bin packing
    for comp_id, size in sorted(component_sizes.items(), key=lambda x: x[1], reverse=True):
        gaps = {k: targets[k] - current[k] for k in targets}
        best_split = max(gaps, key=gaps.get)
        assignments[comp_id] = best_split
        current[best_split] += size

    return assignments


def melt_triplets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert triplet rows to flat (smiles, label, source, triplet_id, split) format.

    Each triplet row becomes 3 rows in the output:
    - PKS molecule (label=1, source='pks')
    - Enzymatic augmentation (label=0, source='enzymatic_aug')
    - Synthetic augmentation (label=0, source='synthetic_aug')
    """
    rows = []
    for idx, row in df.iterrows():
        triplet_id = idx
        split = row['split']

        rows.append({
            'smiles': row['pks_smiles'],
            'label': 1,
            'source': 'pks',
            'triplet_id': triplet_id,
            'split': split
        })
        rows.append({
            'smiles': row['enzymatic_aug_smiles'],
            'label': 0,
            'source': 'enzymatic_aug',
            'triplet_id': triplet_id,
            'split': split
        })
        rows.append({
            'smiles': row['synthetic_aug_smiles'],
            'label': 0,
            'source': 'synthetic_aug',
            'triplet_id': triplet_id,
            'split': split
        })

    return pd.DataFrame(rows)


def compute_component_stats(df: pd.DataFrame) -> Dict:
    """Compute statistics about connected components."""
    component_sizes = df.groupby('component_id').size()

    return {
        'total_components': len(component_sizes),
        'largest_component': int(component_sizes.max()),
        'median_component_size': int(component_sizes.median()),
        'single_triplet_components': int((component_sizes == 1).sum()),
        'multi_triplet_components': int((component_sizes > 1).sum()),
    }


def verify_no_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    """Verify no SMILES overlap across splits."""
    train_smiles = set(train_df['smiles'])
    val_smiles = set(val_df['smiles'])
    test_smiles = set(test_df['smiles'])

    train_val = train_smiles & val_smiles
    train_test = train_smiles & test_smiles
    val_test = val_smiles & test_smiles

    if train_val or train_test or val_test:
        print(f"LEAKAGE DETECTED:")
        print(f"  Train & Val overlap: {len(train_val)}")
        print(f"  Train & Test overlap: {len(train_test)}")
        print(f"  Val & Test overlap: {len(val_test)}")
        return False
    return True


def write_stats_file(
    stats_path: Path,
    input_path: str,
    total_triplets: int,
    total_unique_smiles: int,
    component_stats: Dict,
    triplet_counts: Dict[str, int],
    flat_counts: Dict[str, int],
    pks_ratios: Dict[str, float],
    no_leakage: bool
) -> None:
    """Write statistics file."""
    total_triplets_sum = sum(triplet_counts.values())
    total_flat_sum = sum(flat_counts.values())

    with open(stats_path, 'w') as f:
        f.write("=== SupCon Split Statistics ===\n\n")

        f.write(f"Input: {input_path}\n")
        f.write(f"Total triplets: {total_triplets:,}\n")
        f.write(f"Total unique SMILES: {total_unique_smiles:,}\n\n")

        f.write("Connected Components:\n")
        f.write(f"  Total components: {component_stats['total_components']:,}\n")
        f.write(f"  Largest component: {component_stats['largest_component']} SMILES\n")
        f.write(f"  Median component size: {component_stats['median_component_size']} SMILES\n")
        single_pct = 100 * component_stats['single_triplet_components'] / component_stats['total_components']
        multi_pct = 100 * component_stats['multi_triplet_components'] / component_stats['total_components']
        f.write(f"  Components with 1 triplet: {component_stats['single_triplet_components']:,} ({single_pct:.1f}%)\n")
        f.write(f"  Components with 2+ triplets: {component_stats['multi_triplet_components']:,} ({multi_pct:.1f}%)\n\n")

        f.write("Split Sizes (triplets):\n")
        for split in ['train', 'val', 'test']:
            pct = 100 * triplet_counts[split] / total_triplets_sum if total_triplets_sum else 0
            f.write(f"  {split.capitalize()}: {triplet_counts[split]:,} ({pct:.1f}%)\n")
        f.write("\n")

        f.write("Split Sizes (flat rows):\n")
        for split in ['train', 'val', 'test']:
            pct = 100 * flat_counts[split] / total_flat_sum if total_flat_sum else 0
            f.write(f"  {split.capitalize()}: {flat_counts[split]:,} ({pct:.1f}%)\n")
        f.write("\n")

        f.write("Class Ratios:\n")
        for split in ['train', 'val', 'test']:
            f.write(f"  {split.capitalize()} PKS ratio: {100 * pks_ratios[split]:.1f}%\n")
        f.write("\n")

        if no_leakage:
            f.write("No SMILES leakage detected: Yes\n")
        else:
            f.write("No SMILES leakage detected: FAILED\n")


def main():
    # Paths
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "processed" / "pks_augmentation_pairs.parquet"
    train_dir = project_root / "data" / "train"
    val_dir = project_root / "data" / "val"
    test_dir = project_root / "data" / "test"
    stats_path = project_root / "data" / "processed" / "supcon_split_stats.txt"

    # Load triplet data
    print(f"Loading triplets from {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} triplets")

    # Count unique SMILES across all columns
    all_smiles = set(df['pks_smiles']) | set(df['enzymatic_aug_smiles']) | set(df['synthetic_aug_smiles'])
    total_unique_smiles = len(all_smiles)
    print(f"Total unique SMILES: {total_unique_smiles:,}")

    # Build connected components
    print("Building connected components...")
    df = build_components(df)
    component_stats = compute_component_stats(df)
    print(f"Found {component_stats['total_components']:,} components")
    print(f"  Largest component: {component_stats['largest_component']} triplets")
    print(f"  Single-triplet components: {component_stats['single_triplet_components']:,}")

    # Split components
    print("Splitting components (80/10/10)...")
    assignments = split_components_greedy(df)
    df['split'] = df['component_id'].map(assignments)

    # Count triplets per split
    triplet_counts = df['split'].value_counts().to_dict()
    for split in ['train', 'val', 'test']:
        if split not in triplet_counts:
            triplet_counts[split] = 0
    total = sum(triplet_counts.values())
    print(f"Triplet split sizes:")
    for split in ['train', 'val', 'test']:
        pct = 100 * triplet_counts[split] / total if total else 0
        print(f"  {split}: {triplet_counts[split]:,} ({pct:.1f}%)")

    # Melt triplets to flat format
    print("Converting triplets to flat format...")
    flat_df = melt_triplets(df)
    print(f"Created {len(flat_df):,} flat rows")

    # Split flat dataframe
    train_df = flat_df[flat_df['split'] == 'train'].drop(columns=['split']).reset_index(drop=True)
    val_df = flat_df[flat_df['split'] == 'val'].drop(columns=['split']).reset_index(drop=True)
    test_df = flat_df[flat_df['split'] == 'test'].drop(columns=['split']).reset_index(drop=True)

    flat_counts = {
        'train': len(train_df),
        'val': len(val_df),
        'test': len(test_df)
    }

    # Compute PKS ratios
    pks_ratios = {
        'train': (train_df['label'] == 1).mean() if len(train_df) > 0 else 0,
        'val': (val_df['label'] == 1).mean() if len(val_df) > 0 else 0,
        'test': (test_df['label'] == 1).mean() if len(test_df) > 0 else 0,
    }

    print(f"Flat split sizes:")
    for split, df_split in [('train', train_df), ('val', val_df), ('test', test_df)]:
        total_flat = len(train_df) + len(val_df) + len(test_df)
        pct = 100 * len(df_split) / total_flat if total_flat else 0
        print(f"  {split}: {len(df_split):,} ({pct:.1f}%), PKS ratio: {100 * pks_ratios[split]:.1f}%")

    # Verify no leakage
    print("Verifying no SMILES leakage...")
    no_leakage = verify_no_leakage(train_df, val_df, test_df)
    if no_leakage:
        print("No SMILES leakage detected")
    else:
        print("WARNING: SMILES leakage detected!")

    # Create output directories
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Save split files
    train_path = train_dir / "supcon_train.parquet"
    val_path = val_dir / "supcon_val.parquet"
    test_path = test_dir / "supcon_test.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"Saved splits:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")

    # Write statistics file
    write_stats_file(
        stats_path=stats_path,
        input_path=str(input_path.relative_to(project_root)),
        total_triplets=len(df),
        total_unique_smiles=total_unique_smiles,
        component_stats=component_stats,
        triplet_counts=triplet_counts,
        flat_counts=flat_counts,
        pks_ratios=pks_ratios,
        no_leakage=no_leakage
    )
    print(f"Saved statistics to {stats_path}")


if __name__ == "__main__":
    main()
