# ContrastiveGNNs

A machine learning research project for distinguishing Polyketide Synthase (PKS) products from non-PKS molecules using Graph Neural Networks and Supervised Contrastive Learning.

## Scientific Motivation

**Polyketides** are a large class of secondary metabolites produced by bacteria, fungi, and plants. They include many pharmaceutically important compounds (antibiotics, immunosuppressants, anticancer agents). PKS products have characteristic structural features arising from their biosynthetic assembly-line mechanism.

**The Classification Problem**: Given a molecule's structure, can we determine whether it was (or could be) produced by a PKS? This is challenging because:
1. PKS products are structurally diverse
2. Many non-PKS molecules share substructural features with PKS products
3. Standard molecular fingerprints (ECFP4) capture local substructures but may miss the global structural patterns characteristic of PKS biosynthesis

**Our Approach**: Learn molecular representations using Supervised Contrastive Learning (SupCon) that capture the structural differences between PKS and non-PKS molecules better than handcrafted fingerprints.

## Project Evolution

### Phase 1: Baseline Classification (Completed)
- Generated PKS products using the `bcs` library (RetroTide forward prediction)
- Generated non-PKS molecules using DORAnet enzymatic and synthetic chemistry
- Trained baseline classifiers (SVM, feedforward NN, GAT-based GNN) on ECFP4 fingerprints
- **Result**: High accuracy but potentially learning spurious correlations in fingerprint space

### Phase 2: Supervised Contrastive Learning (Current)
The key insight: instead of randomly sampling non-PKS molecules, we generate **hard negatives** that are chemically similar to each PKS molecule but structurally distinct due to different biosynthetic origins.

#### The SupCon Data Pipeline

For each PKS molecule, we generate a **triplet**:
1. **Anchor (PKS)**: The original PKS product (label = 1)
2. **Enzymatic augmentation**: Most similar DORAnet enzymatic product (label = 0)
3. **Synthetic augmentation**: Most similar DORAnet synthetic product (label = 0)

This creates hard negatives that force the model to learn subtle structural differences rather than obvious chemical dissimilarities.

#### Data Leakage Prevention

The same augmentation SMILES can appear across multiple triplets (e.g., the same DORAnet product might be the closest match for multiple PKS molecules). Naive splitting would leak SMILES across train/val/test.

**Solution**: Connected-component splitting
1. Build a graph where edges connect PKS → enzymatic_aug and PKS → synthetic_aug
2. Find connected components using Union-Find
3. Split at the **component level** (not triplet or SMILES level)
4. All SMILES in a component stay in the same split

## Current Data Pipeline

```
scripts/00_generate_bound_PKS_products.py      # PKS products (bound to enzyme)
scripts/01_generate_unbound_PKS_products.py    # PKS products (released via thiolysis/cyclization)
scripts/02_generate_PKS_augmentations.py       # Generate triplets (PKS + enzymatic_aug + synthetic_aug)
scripts/03_create_supcon_splits.py             # Connected-component train/val/test splits
```

### Input/Output Files

| Script | Input | Output |
|--------|-------|--------|
| 00 | PKS designs | `data/raw/bound_PKS_products_*.pkl` |
| 01 | Bound PKS | `data/processed/pks_products_*_SMILES.txt` |
| 02 | PKS SMILES | `data/processed/pks_augmentation_pairs.parquet` |
| 03 | Augmentation pairs | `data/{train,val,test}/supcon_*.parquet` |

### SupCon Split Format

Each split file contains flat rows with columns:
- `smiles`: Molecule SMILES string
- `label`: 1 = PKS, 0 = augmentation (non-PKS)
- `source`: "pks", "enzymatic_aug", or "synthetic_aug"
- `triplet_id`: Links back to original triplet for batch construction

The 1:2 class ratio (33% PKS, 67% non-PKS) is inherent to the triplet structure.

## Usage

```bash
# Generate PKS products
python scripts/00_generate_bound_PKS_products.py
python scripts/01_generate_unbound_PKS_products.py

# Generate augmentation triplets (MPI-distributed, slow)
mpirun -np <NUM_PROCESSES> python scripts/02_generate_PKS_augmentations.py

# Create leak-free splits
python scripts/03_create_supcon_splits.py

# Run tests to verify no leakage
pytest tests/test_generated_SMILES.py -v
```

## Testing

The test suite (`tests/test_generated_SMILES.py`) verifies:

**Baseline tests:**
- No SMILES overlap between train/val/test splits
- No stereochemical markers in SMILES
- PKS class ratio consistent across splits

**SupCon tests:**
- `test_supcon_no_smiles_leakage()` - No SMILES overlap across splits
- `test_supcon_class_ratios()` - ~33% PKS ratio in each split
- `test_supcon_split_sizes()` - ~80/10/10 split ratios
- `test_supcon_triplet_integrity()` - Each triplet_id has exactly 3 rows

## Key Dependencies

- `rdkit`: Molecular parsing, fingerprinting, canonicalization
- `torch` / `torch_geometric`: GNN implementation
- `bcs`: PKS product generation (RetroTide)
- `doranet`: Enzymatic and synthetic reaction network generation
- `mpi4py`: Distributed processing for augmentation generation
- `pandas`: Data handling (parquet format)

## Project Structure

```
ContrastiveGNNs/
├── scripts/           # Numbered pipeline scripts
├── data/
│   ├── raw/          # Original PKS products
│   ├── interim/      # Intermediate files
│   ├── processed/    # Final datasets, augmentation pairs
│   ├── train/        # Training split
│   ├── val/          # Validation split
│   └── test/         # Test split
├── models/           # Saved checkpoints and metrics
├── notebooks/        # Jupyter notebooks for exploration
├── tests/            # pytest test suite
├── archives/         # Deprecated code
├── CLAUDE.md         # AI assistant instructions
└── README.md         # This file
```

## Next Steps

1. **SupCon Training Script**: Implement supervised contrastive loss training
   - Use triplet structure for batch construction (anchor + positives + negatives)
   - Train GNN encoder to produce embeddings where PKS molecules cluster together

2. **Evaluation**:
   - Linear probe on frozen SupCon embeddings vs. ECFP4 fingerprints
   - t-SNE/UMAP visualization of learned embedding space
   - Analyze which structural features drive PKS vs. non-PKS separation

3. **Interpretability**:
   - Attention visualization on molecular graphs
   - Identify substructures most predictive of PKS origin

## Session Notes (2025-01-27)

### What We Built Today

1. **`scripts/03_create_supcon_splits.py`**: New splitting script with:
   - `UnionFind` class for connected-component detection
   - `build_components()` to group triplets sharing SMILES
   - `split_components_greedy()` for 80/10/10 bin packing
   - `melt_triplets()` to convert triplet format to flat training format
   - Statistics file generation for debugging

2. **Test suite extensions**: Added 4 new tests for SupCon data integrity

3. **Moved to archives**: Old `05_create_train_test_val_splits.py` (baseline pipeline)

### Key Design Decisions

- **Hard negatives via chemical similarity**: Rather than random non-PKS molecules, we use the most similar DORAnet products as negatives. This forces the model to learn subtle PKS-specific patterns.

- **Connected-component splitting**: Ensures zero SMILES leakage even when the same augmentation appears in multiple triplets.

- **Flat output format**: Each row is a single molecule with metadata. The `triplet_id` column allows reconstructing triplet batches during training if needed.

- **Class labels**: PKS = 1 (positive), augmentations = 0 (negative). The SupCon loss will pull PKS embeddings together and push augmentation embeddings apart.
