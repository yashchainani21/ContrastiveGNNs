"""
Unified MPI script to generate paired augmentations for SupCon training.

For each PKS molecule, generates:
- The most chemically similar enzymatic DORAnet product (negative)
- The most chemically similar synthetic DORAnet product (negative, must differ from enzymatic)

Output is a parquet file with triplets (pks, enzymatic_aug, synthetic_aug) plus similarity scores.

Usage:
    mpirun -np <NUM_PROCESSES> python 02_generate_PKS_augmentations.py [--input INPUT] [--output OUTPUT]

Replaces: 02_generate_DORAnet_bio_products.py and 03_generate_DORAnet_chem_products.py
"""

import argparse
import json
import uuid
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from mpi4py import MPI
from rdkit import Chem, DataStructs
from rdkit import RDLogger
from rdkit.Chem import AllChem

import doranet.modules.enzymatic as enzymatic
import doranet.modules.synthetic as synthetic

RDLogger.DisableLog('rdApp.*')

# Default paths
DEFAULT_INPUT = '../data/processed/pks_products_3_ext_no_stereo_mal_mmal_allylmal_hmal_emal_mxmal_SMILES.txt'
DEFAULT_OUTPUT = '../data/processed/pks_augmentation_pairs.parquet'
DEFAULT_STATS = '../data/processed/pks_augmentation_stats.json'
DEFAULT_SKIPPED = '../data/processed/pks_augmentation_skipped.json'

# Fingerprint parameters
FP_RADIUS = 2
FP_NBITS = 2048

# Helper SMILES to exclude from synthetic products
HELPER_SMILES_SET = frozenset((
    "O", "O=O", "[H][H]", "O=C=O", "C=O", "[C-]#[O+]", "Br", "[Br][Br]",
    "CO", "C=C", "O=S(O)O", "N", "O=S(=O)(O)O", "O=NO", "N#N",
    "O=[N+]([O-])O", "NO", "C#N", "S", "O=S=O"
))


def compute_fingerprint(smiles: str):
    """Compute Morgan fingerprint for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS)


def compute_tanimoto(fp1, fp2) -> float:
    """Compute Tanimoto similarity between two fingerprints."""
    if fp1 is None or fp2 is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def rank_by_similarity(anchor_fp, candidates: List[str], exclude: Set[str]) -> List[Tuple[str, float]]:
    """
    Rank candidate SMILES by Tanimoto similarity to anchor fingerprint.

    Returns list of (smiles, similarity) sorted by similarity descending,
    excluding any SMILES in the exclude set.
    """
    results = []
    for cand in candidates:
        if cand in exclude:
            continue
        cand_fp = compute_fingerprint(cand)
        if cand_fp is None:
            continue
        sim = compute_tanimoto(anchor_fp, cand_fp)
        results.append((cand, sim))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def generate_enzymatic_products(pks_smiles: str) -> List[str]:
    """Generate all enzymatic DORAnet products (gen=1) for a PKS molecule."""
    unique_id = str(uuid.uuid4())
    unique_jobname = f'{pks_smiles}_{unique_id}'

    try:
        forward_network = enzymatic.generate_network(
            job_name=unique_jobname,
            starters={pks_smiles},
            gen=1,
            direction="forward"
        )

        products = []
        for mol in forward_network.mols:
            try:
                product_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(mol.uid))
                if product_smiles:
                    products.append(product_smiles)
            except Exception:
                continue
        return products
    except Exception:
        return []


def generate_synthetic_products(pks_smiles: str) -> List[str]:
    """Generate all synthetic DORAnet products (gen=1) for a PKS molecule."""
    unique_id = str(uuid.uuid4())
    unique_jobname = f'{pks_smiles}_{unique_id}'

    try:
        forward_network = synthetic.generate_network(
            job_name=unique_jobname,
            starters={pks_smiles},
            helpers=tuple(HELPER_SMILES_SET),
            gen=1,
            direction="forward"
        )

        products = []
        for mol in forward_network.mols:
            try:
                smiles = Chem.MolToSmiles(Chem.MolFromSmiles(mol.uid))
                if smiles and smiles not in HELPER_SMILES_SET:
                    products.append(smiles)
            except Exception:
                continue
        return products
    except Exception:
        return []


def process_single_pks(pks_smiles: str) -> dict:
    """
    Process a single PKS molecule to find its paired augmentations.

    Returns a dict with either successful triplet data or skip reason.
    """
    result = {
        'pks_smiles': pks_smiles,
        'success': False,
        'skip_reason': None,
        'enzymatic_aug_smiles': None,
        'synthetic_aug_smiles': None,
        'enzymatic_similarity': None,
        'synthetic_similarity': None,
        'synthetic_rank': None
    }

    # Compute anchor fingerprint
    anchor_fp = compute_fingerprint(pks_smiles)
    if anchor_fp is None:
        result['skip_reason'] = 'rdkit_error'
        return result

    # Generate enzymatic products
    enz_products = generate_enzymatic_products(pks_smiles)
    if not enz_products:
        result['skip_reason'] = 'no_enzymatic'
        return result

    # Rank enzymatic products by similarity (excluding original PKS)
    ranked_enz = rank_by_similarity(anchor_fp, enz_products, exclude={pks_smiles})
    if not ranked_enz:
        result['skip_reason'] = 'no_enzymatic'
        return result

    # Select most similar enzymatic product
    enz_aug, enz_sim = ranked_enz[0]

    # Generate synthetic products
    syn_products = generate_synthetic_products(pks_smiles)
    if not syn_products:
        result['skip_reason'] = 'no_synthetic'
        return result

    # Rank synthetic products by similarity (excluding original PKS)
    ranked_syn = rank_by_similarity(anchor_fp, syn_products, exclude={pks_smiles})
    if not ranked_syn:
        result['skip_reason'] = 'no_synthetic'
        return result

    # Find first synthetic product that differs from enzymatic augmentation
    syn_aug = None
    syn_sim = None
    syn_rank = None
    for rank_idx, (syn_smiles, sim) in enumerate(ranked_syn, start=1):
        if syn_smiles != enz_aug:
            syn_aug = syn_smiles
            syn_sim = sim
            syn_rank = rank_idx
            break

    if syn_aug is None:
        result['skip_reason'] = 'all_synthetic_match_enzymatic'
        return result

    # Success
    result['success'] = True
    result['enzymatic_aug_smiles'] = enz_aug
    result['synthetic_aug_smiles'] = syn_aug
    result['enzymatic_similarity'] = enz_sim
    result['synthetic_similarity'] = syn_sim
    result['synthetic_rank'] = syn_rank

    return result


def chunkify(lst: list, n: int) -> List[list]:
    """Split list into n (roughly) equal-sized chunks. Avoid creating empty chunks."""
    n = max(1, min(n, len(lst)))
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def parse_args():
    parser = argparse.ArgumentParser(description='Generate PKS augmentation pairs for SupCon training')
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT,
                        help='Input file with PKS SMILES (one per line)')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                        help='Output parquet file for augmentation pairs')
    parser.add_argument('--stats', type=str, default=DEFAULT_STATS,
                        help='Output JSON file for generation statistics')
    parser.add_argument('--skipped', type=str, default=DEFAULT_SKIPPED,
                        help='Output JSON file for skipped molecules')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # MPI initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Only rank 0 reads the input file
    if rank == 0:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        with open(input_path, 'r') as f:
            pks_list = [line.strip() for line in f if line.strip()]

        print(f"Loaded {len(pks_list)} PKS molecules from {input_path}", flush=True)

        # Prepare chunks for distribution
        chunks = chunkify(pks_list, size)
        # Pad with empty chunks if needed
        while len(chunks) < size:
            chunks.append([])
    else:
        chunks = None

    # Scatter data to all ranks
    my_pks_list = comm.scatter(chunks, root=0)
    print(f"[Rank {rank}] received {len(my_pks_list)} PKS molecules to process.", flush=True)

    # Process assigned PKS molecules
    my_results = []
    for i, pks_smiles in enumerate(my_pks_list):
        result = process_single_pks(pks_smiles)
        my_results.append(result)

        # Progress logging every 100 molecules
        if (i + 1) % 100 == 0:
            print(f"[Rank {rank}] processed {i + 1}/{len(my_pks_list)} molecules", flush=True)

    print(f"[Rank {rank}] completed processing {len(my_pks_list)} molecules.", flush=True)

    # Gather results from all ranks to rank 0
    all_results = comm.gather(my_results, root=0)

    if rank == 0:
        # Flatten results
        flat_results = [r for sublist in all_results for r in sublist]

        # Separate successful triplets and skipped molecules
        successful = [r for r in flat_results if r['success']]
        skipped = [r for r in flat_results if not r['success']]

        # Build output dataframe
        if successful:
            df_out = pd.DataFrame({
                'pks_smiles': [r['pks_smiles'] for r in successful],
                'enzymatic_aug_smiles': [r['enzymatic_aug_smiles'] for r in successful],
                'synthetic_aug_smiles': [r['synthetic_aug_smiles'] for r in successful],
                'enzymatic_similarity': [r['enzymatic_similarity'] for r in successful],
                'synthetic_similarity': [r['synthetic_similarity'] for r in successful],
            })

            # Write parquet
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_out.to_parquet(output_path, index=False)
            print(f"Wrote {len(df_out)} triplets to {output_path}", flush=True)
        else:
            print("WARNING: No successful triplets generated!", flush=True)
            df_out = pd.DataFrame()

        # Compute statistics
        skip_reasons = {}
        for r in skipped:
            reason = r['skip_reason']
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

        enz_sims = [r['enzymatic_similarity'] for r in successful]
        syn_sims = [r['synthetic_similarity'] for r in successful]
        syn_ranks = [r['synthetic_rank'] for r in successful]

        # Compute rank distribution
        rank_dist = {'rank_1': 0, 'rank_2': 0, 'rank_3_plus': 0}
        for rk in syn_ranks:
            if rk == 1:
                rank_dist['rank_1'] += 1
            elif rk == 2:
                rank_dist['rank_2'] += 1
            else:
                rank_dist['rank_3_plus'] += 1

        stats = {
            'total_pks_input': len(flat_results),
            'successful_triplets': len(successful),
            'skipped_no_enzymatic': skip_reasons.get('no_enzymatic', 0),
            'skipped_no_synthetic': skip_reasons.get('no_synthetic', 0),
            'skipped_all_synthetic_match_enzymatic': skip_reasons.get('all_synthetic_match_enzymatic', 0),
            'skipped_rdkit_error': skip_reasons.get('rdkit_error', 0),
            'enzymatic_similarity': {
                'mean': float(np.mean(enz_sims)) if enz_sims else None,
                'median': float(np.median(enz_sims)) if enz_sims else None,
                'min': float(np.min(enz_sims)) if enz_sims else None,
                'max': float(np.max(enz_sims)) if enz_sims else None,
            },
            'synthetic_similarity': {
                'mean': float(np.mean(syn_sims)) if syn_sims else None,
                'median': float(np.median(syn_sims)) if syn_sims else None,
                'min': float(np.min(syn_sims)) if syn_sims else None,
                'max': float(np.max(syn_sims)) if syn_sims else None,
            },
            'synthetic_rank_distribution': rank_dist
        }

        # Write stats
        stats_path = Path(args.stats)
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Wrote statistics to {stats_path}", flush=True)

        # Write skipped molecules
        skipped_data = [{'pks_smiles': r['pks_smiles'], 'reason': r['skip_reason']} for r in skipped]
        skipped_path = Path(args.skipped)
        with open(skipped_path, 'w') as f:
            json.dump(skipped_data, f, indent=2)
        print(f"Wrote {len(skipped_data)} skipped molecules to {skipped_path}", flush=True)

        # Print summary
        print("\n" + "="*60)
        print("GENERATION SUMMARY")
        print("="*60)
        print(f"Total PKS input:        {stats['total_pks_input']}")
        print(f"Successful triplets:    {stats['successful_triplets']}")
        print(f"Skipped (no enzymatic): {stats['skipped_no_enzymatic']}")
        print(f"Skipped (no synthetic): {stats['skipped_no_synthetic']}")
        print(f"Skipped (syn=enz):      {stats['skipped_all_synthetic_match_enzymatic']}")
        print(f"Skipped (RDKit error):  {stats['skipped_rdkit_error']}")
        if enz_sims:
            print(f"\nEnzymatic similarity:   mean={stats['enzymatic_similarity']['mean']:.4f}, "
                  f"median={stats['enzymatic_similarity']['median']:.4f}")
            print(f"Synthetic similarity:   mean={stats['synthetic_similarity']['mean']:.4f}, "
                  f"median={stats['synthetic_similarity']['median']:.4f}")
            print(f"\nSynthetic rank distribution:")
            print(f"  Rank 1 (most similar): {rank_dist['rank_1']}")
            print(f"  Rank 2:                {rank_dist['rank_2']}")
            print(f"  Rank 3+:               {rank_dist['rank_3_plus']}")
        print("="*60)
