"""
Generate unbound PKS products from bound PKS products via thiolysis and cyclization reactions.

This script uses multiprocessing to parallelize the generation of unbound PKS products.
Each worker processes molecules independently, and results are collected and deduplicated
in the main process to ensure deterministic output (same as sequential version).
"""

import bcs
import pickle
import multiprocessing as mp
from dataclasses import dataclass
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Dict, Tuple, Optional
import time
import os
from tqdm import tqdm

# Configuration
NUM_WORKERS: Optional[int] = None  # None = use all available CPUs
CHUNKSIZE: int = 500  # Items per worker task (tune for performance)

# Script parameters
REMOVE_STEREOCHEMISTRY = True
MAX_EXTENSION_MODULES = 3

# Input/Output filepaths
INPUT_FILEPATH = f"../data/raw/expanded_bound_PKS_products_{MAX_EXTENSION_MODULES}_ext_mods_mal_mmal_v2.pkl"

# Trivial products to filter out
TRIVIAL_PRODUCTS = {'S', 'O=C=O'}


@dataclass
class UnboundProductResult:
    """Container for results from processing a single bound PKS molecule."""
    original_index: int
    pks_design_bytes: bytes  # Pickled bcs.Cluster
    thiolysis_smiles: List[str]  # All valid SMILES from thiolysis
    cyclization_smiles: List[str]  # All valid SMILES from cyclization
    thiolysis_error: Optional[str] = None
    cyclization_error: Optional[str] = None


def run_pks_release_reaction(pks_release_mechanism: str,
                             bound_product_mol: Chem.Mol) -> List[Chem.Mol]:
    """
    Run an offloading reaction to release a bound PKS product.
    Two types of offloading reactions are currently supported: thiolysis and cyclization.
    Always returns a list of product molecules.
    """

    if pks_release_mechanism == 'thiolysis':
        Chem.SanitizeMol(bound_product_mol)
        rxn = AllChem.ReactionFromSmarts(
            '[C:1](=[O:2])[S:3]>>[C:1](=[O:2])[O].[S:3]'
        )
        products = rxn.RunReactants((bound_product_mol,))
        if not products:
            raise ValueError("Unable to perform thiolysis reaction")

        unbound_products = []
        for prod_tuple in products:
            for prod in prod_tuple:
                try:
                    Chem.SanitizeMol(prod)
                    unbound_products.append(prod)
                except:
                    continue
        return unbound_products

    if pks_release_mechanism == 'cyclization':
        Chem.SanitizeMol(bound_product_mol)
        rxn = AllChem.ReactionFromSmarts(
            '([C:1](=[O:2])[S:3].[O,N:4][C:5][C:6])>>[C:1](=[O:2])[*:4][C:5][C:6].[S:3]'
        )
        products = rxn.RunReactants((bound_product_mol,))
        if not products:
            raise ValueError("Unable to perform cyclization reaction")

        unbound_products = []
        for prod_tuple in products:
            for prod in prod_tuple:
                try:
                    Chem.SanitizeMol(prod)
                    unbound_products.append(prod)
                except:
                    continue
        return unbound_products

    raise ValueError(f"Unsupported PKS release mechanism: {pks_release_mechanism}")


def process_single_bound_product(args: Tuple[int, bytes, bool]) -> UnboundProductResult:
    """
    Process one bound PKS molecule through thiolysis and cyclization.

    Args:
        args: (original_index, pickled_cluster_and_mol, remove_stereochemistry)

    Returns:
        UnboundProductResult containing all generated SMILES and any errors.
    """
    original_index, pickled_data, remove_stereochemistry = args

    # Unpickle the input data
    pks_design, bound_pks_mol = pickle.loads(pickled_data)

    # Re-pickle just the design for returning (to avoid serialization issues)
    pks_design_bytes = pickle.dumps(pks_design)

    thiolysis_smiles: List[str] = []
    cyclization_smiles: List[str] = []
    thiolysis_error: Optional[str] = None
    cyclization_error: Optional[str] = None

    # Try thiolysis reaction
    try:
        unbound_products = run_pks_release_reaction("thiolysis", bound_pks_mol)

        for unbound_mol in unbound_products:
            if remove_stereochemistry:
                Chem.RemoveStereochemistry(unbound_mol)

            smiles = Chem.MolToSmiles(unbound_mol)

            # Filter trivial products
            if smiles not in TRIVIAL_PRODUCTS:
                thiolysis_smiles.append(smiles)

    except Exception as e:
        thiolysis_error = str(e)

    # Try cyclization reaction
    try:
        unbound_products = run_pks_release_reaction("cyclization", bound_pks_mol)

        for unbound_mol in unbound_products:
            if remove_stereochemistry:
                Chem.RemoveStereochemistry(unbound_mol)

            smiles = Chem.MolToSmiles(unbound_mol)

            # Filter trivial products
            if smiles not in TRIVIAL_PRODUCTS:
                cyclization_smiles.append(smiles)

    except Exception as e:
        cyclization_error = str(e)

    return UnboundProductResult(
        original_index=original_index,
        pks_design_bytes=pks_design_bytes,
        thiolysis_smiles=thiolysis_smiles,
        cyclization_smiles=cyclization_smiles,
        thiolysis_error=thiolysis_error,
        cyclization_error=cyclization_error,
    )


def prepare_work_items(bound_pks_products: List[Tuple[bcs.Cluster, Chem.Mol]],
                       remove_stereochemistry: bool) -> List[Tuple[int, bytes, bool]]:
    """
    Pre-pickle inputs with their original indices for passing to workers.

    Args:
        bound_pks_products: List of (Cluster, Mol) tuples
        remove_stereochemistry: Whether to remove stereochemistry from products

    Returns:
        List of (original_index, pickled_data, remove_stereochemistry) tuples
    """
    work_items = []
    for i, (pks_design, bound_mol) in enumerate(bound_pks_products):
        pickled_data = pickle.dumps((pks_design, bound_mol))
        work_items.append((i, pickled_data, remove_stereochemistry))
    return work_items


def run_parallel_processing(work_items: List[Tuple[int, bytes, bool]],
                            num_workers: Optional[int],
                            chunksize: int) -> List[UnboundProductResult]:
    """
    Process all bound products in parallel using a multiprocessing Pool.

    Args:
        work_items: Pre-pickled work items with indices
        num_workers: Number of worker processes (None = all CPUs)
        chunksize: Items per worker task

    Returns:
        List of UnboundProductResult (unordered)
    """
    num_workers = num_workers or mp.cpu_count()
    total_items = len(work_items)

    print(f"Starting parallel processing with {num_workers} workers...")
    print(f"Total items to process: {total_items}")
    print(f"Chunksize: {chunksize}")

    results: List[UnboundProductResult] = []

    with mp.Pool(processes=num_workers) as pool:
        # Use imap_unordered for streaming results (better memory usage)
        # Wrap with tqdm for progress bar
        with tqdm(total=total_items, desc="Processing PKS designs", unit="mol") as pbar:
            for result in pool.imap_unordered(process_single_bound_product,
                                              work_items,
                                              chunksize=chunksize):
                results.append(result)
                pbar.update(1)

    return results


def collect_and_deduplicate_results(results: List[UnboundProductResult]) -> Tuple[Dict[bcs.Cluster, str], int, int]:
    """
    Sort results by original index and deduplicate SMILES (first occurrence wins).

    This matches the original sequential behavior exactly:
    - Process molecules in order of their original input index
    - For each molecule, process thiolysis products first, then cyclization
    - First occurrence of a SMILES gets added to the dictionary

    Args:
        results: Unordered list of results from parallel processing

    Returns:
        Tuple of (final_dict, num_thiolysis, num_cyclization)
    """
    # Sort by original index to ensure deterministic ordering
    sorted_results = sorted(results, key=lambda r: r.original_index)

    unique_smiles: set = set()
    final_dict: Dict[bcs.Cluster, str] = {}
    num_thiolysis = 0
    num_cyclization = 0

    for result in sorted_results:
        pks_design = pickle.loads(result.pks_design_bytes)

        # Process thiolysis first (matches original script order)
        for smiles in result.thiolysis_smiles:
            if smiles not in unique_smiles:
                unique_smiles.add(smiles)
                final_dict[pks_design] = smiles
                num_thiolysis += 1

        # Then cyclization
        for smiles in result.cyclization_smiles:
            if smiles not in unique_smiles:
                unique_smiles.add(smiles)
                final_dict[pks_design] = smiles
                num_cyclization += 1

    return final_dict, num_thiolysis, num_cyclization


def main(input_filepath: str):
    """Main entry point for parallel PKS unbound product generation.

    Args:
        input_filepath: Path to the pickle file containing bound PKS products.
    """
    start_time = time.time()

    # Determine output filepath
    if REMOVE_STEREOCHEMISTRY:
        output_filepath = f"../data/interim/expanded_unbound_PKS_products_{MAX_EXTENSION_MODULES}_ext_mods_no_stereo_mal_mmal_allylmal_emal_mxmal_hmal_extenders.pkl"
    else:
        output_filepath = f"../data/interim/expanded_unbound_PKS_products_{MAX_EXTENSION_MODULES}_ext_mods_with_stereo_mal_mmal_allylmal_emal_mxmal_hmal_extenders.pkl"

    # Load bound PKS products
    print(f"Loading bound PKS products from: {input_filepath}")

    with open(input_filepath, "rb") as f:
        bound_pks_products = pickle.load(f)

    print(f"Loaded {len(bound_pks_products)} bound PKS products")

    # Prepare work items (pre-pickle for multiprocessing)
    print("Preparing work items...")
    prep_start = time.time()
    work_items = prepare_work_items(bound_pks_products, REMOVE_STEREOCHEMISTRY)
    prep_time = time.time() - prep_start
    print(f"Work items prepared in {prep_time:.2f} seconds")

    # Run parallel processing
    proc_start = time.time()
    results = run_parallel_processing(work_items, NUM_WORKERS, CHUNKSIZE)
    proc_time = time.time() - proc_start
    print(f"Parallel processing completed in {proc_time:.2f} seconds")

    # Log any errors encountered
    thiolysis_errors = sum(1 for r in results if r.thiolysis_error)
    cyclization_errors = sum(1 for r in results if r.cyclization_error)
    if thiolysis_errors or cyclization_errors:
        print(f"Encountered {thiolysis_errors} thiolysis errors and {cyclization_errors} cyclization errors")

    # Collect and deduplicate results (deterministic)
    print("Deduplicating results...")
    dedup_start = time.time()
    unbound_pks_products_dict, num_thiolysis, num_cyclization = collect_and_deduplicate_results(results)
    dedup_time = time.time() - dedup_start
    print(f"Deduplication completed in {dedup_time:.2f} seconds")

    # Print summary
    total_time = time.time() - start_time
    print('\n--------------------------------------------\n')
    print(f"Generated {len(unbound_pks_products_dict)} unique unbound PKS products.\n")
    print(f"Number of successful thiolysis reactions: {num_thiolysis}\n")
    print(f"Number of successful cyclization reactions: {num_cyclization}\n")
    print(f"Total time: {total_time:.2f} seconds")

    # Save the dictionary of unbound PKS products
    print(f"\nSaving to: {output_filepath}")
    with open(output_filepath, "wb") as f:
        pickle.dump(unbound_pks_products_dict, f)

    # Save the unique SMILES strings as a text file (one per line)
    smiles_txt_filepath = output_filepath.replace(".pkl", "_SMILES.txt")
    print(f"Saving SMILES to: {smiles_txt_filepath}")
    unique_smiles_list = list(unbound_pks_products_dict.values())
    with open(smiles_txt_filepath, "w") as f:
        for smiles in unique_smiles_list:
            f.write(smiles + "\n")

    print("Done!")


if __name__ == "__main__":
    main(INPUT_FILEPATH)
