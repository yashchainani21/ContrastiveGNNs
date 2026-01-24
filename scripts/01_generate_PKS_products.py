"""
Unified PKS Product Generation Script

This script generates PKS (Polyketide Synthase) products by:
1. Generating bound PKS products for 1, 2, ..., N extension modules
2. Processing each bound product through thiolysis and cyclization reactions
3. Deduplicating and saving the final unbound PKS products

This combines the functionality of the former 00_generate_bound_PKS_products.py and
01_generate_unbound_PKS_products.py into a single unified pipeline.
"""

from typing import List, Dict, Tuple, Optional
from collections import OrderedDict
from itertools import product
from dataclasses import dataclass
import multiprocessing as mp
import pickle
import time
import os

from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import bcs


# =============================================================================
# CONFIGURATION
# =============================================================================

# Extension module range (will generate 1, 2, ..., MAX extension modules)
MAX_EXTENSION_MODULES = 3

# Starter units (None = use all available)
STARTER_CODES: Optional[List[str]] = None

# Extender units to allow
EXTENDER_CODES: List[str] = ['Malonyl-CoA', 'Methylmalonyl-CoA', 'allylmal', 'hmal', 'emal', 'mxmal']

# Stereochemistry handling
REMOVE_STEREOCHEMISTRY = True

# Parallelization settings
NUM_WORKERS: Optional[int] = None  # None = use all CPUs
CHUNKSIZE_BOUND = 1000  # For bound product generation
CHUNKSIZE_UNBOUND = 500  # For unbound product processing

# Output settings
OUTPUT_DIR = "../data/processed"
SAVE_BOUND_PRODUCTS = False  # Whether to also save intermediate bound products

# Trivial products to filter out
TRIVIAL_PRODUCTS = {'S', 'O=C=O'}


# =============================================================================
# PHASE 1: BOUND PRODUCT GENERATION
# =============================================================================

def modify_bcs_starters_extenders(starter_codes: Optional[List[str]] = None,
                                  extender_codes: Optional[List[str]] = None):
    """
    Modify the starter and extender acyl-CoA units used to generate PKS products.

    Parameters
    ----------
    starter_codes : Optional[List[str]], optional
        List of starter unit codes to be used. If None, all starter units are used.
    extender_codes : Optional[List[str]], optional
        List of extender unit codes to be used. If None, all extender units are used.
    """
    if starter_codes is not None:
        for key in list(bcs.starters.keys()):
            if key not in starter_codes:
                bcs.starters.pop(key, None)

    if extender_codes is not None:
        for key in list(bcs.extenders.keys()):
            if key not in extender_codes:
                bcs.extenders.pop(key, None)


def build_bcs_cluster_and_product(starter: str, extension_mods_combo):
    """
    Build a bcs PKS cluster and its corresponding PKS product given a starter unit
    and extension module combination.

    Parameters
    ----------
    starter : str
        The starter unit code
    extension_mods_combo : tuple
        Tuple of extension modules

    Returns
    -------
    tuple or (None, None)
        (cluster, product_mol) if successful, (None, None) otherwise
    """
    try:
        # Build loading module
        loading_AT_domain = bcs.AT(active=True, substrate=starter)
        loading_module = bcs.Module(domains=OrderedDict({bcs.AT: loading_AT_domain}), loading=True)
        full_modules = [loading_module] + list(extension_mods_combo)

        # Create bcs cluster
        cluster = bcs.Cluster(modules=full_modules)

        # Generate PKS product
        product_mol = cluster.computeProduct(structureDB)
        return cluster, product_mol

    except Exception as e:
        print(f"Error building loading module with starter {starter} combo {extension_mods_combo}: {e}")
        return None, None


def generate_all_bound_products() -> List[Tuple]:
    """
    Generate all bound PKS products for 1 to MAX_EXTENSION_MODULES extension modules.

    Returns
    -------
    List[Tuple]
        List of (cluster, product_mol) tuples
    """
    all_cluster_product_pairs = []

    with mp.Pool() as pool:
        for i in range(1, MAX_EXTENSION_MODULES + 1):
            print(f"\nGenerating clusters and products with {i} extension module(s)...")

            # Create all possible (starter, extension_combo) pairs
            starter_plus_ext_mods_combos = product(
                bcs.starters.keys(),
                product(extension_modules_list, repeat=i)
            )

            # Build clusters and products in parallel
            results_i = pool.starmap(
                build_bcs_cluster_and_product,
                starter_plus_ext_mods_combos,
                chunksize=CHUNKSIZE_BOUND
            )

            # Filter out failed builds
            results_i = [r for r in results_i if None not in r]

            all_cluster_product_pairs.extend(results_i)
            print(f"  Generated {len(results_i)} products with {i} extension module(s)")

    return all_cluster_product_pairs


# =============================================================================
# PHASE 2: UNBOUND PRODUCT PROCESSING
# =============================================================================

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

    Two types of offloading reactions are supported: thiolysis and cyclization.

    Parameters
    ----------
    pks_release_mechanism : str
        Either 'thiolysis' or 'cyclization'
    bound_product_mol : Chem.Mol
        The bound PKS product molecule

    Returns
    -------
    List[Chem.Mol]
        List of unbound product molecules
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

    Parameters
    ----------
    args : Tuple[int, bytes, bool]
        (original_index, pickled_cluster_and_mol, remove_stereochemistry)

    Returns
    -------
    UnboundProductResult
        Results containing all generated SMILES and any errors
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


def prepare_work_items(bound_pks_products: List[Tuple],
                       remove_stereochemistry: bool) -> List[Tuple[int, bytes, bool]]:
    """
    Pre-pickle inputs with their original indices for passing to workers.

    Parameters
    ----------
    bound_pks_products : List[Tuple]
        List of (Cluster, Mol) tuples
    remove_stereochemistry : bool
        Whether to remove stereochemistry from products

    Returns
    -------
    List[Tuple[int, bytes, bool]]
        List of (original_index, pickled_data, remove_stereochemistry) tuples
    """
    work_items = []
    for i, (pks_design, bound_mol) in enumerate(bound_pks_products):
        pickled_data = pickle.dumps((pks_design, bound_mol))
        work_items.append((i, pickled_data, remove_stereochemistry))
    return work_items


def process_all_unbound_products(bound_pks_products: List[Tuple]) -> List[UnboundProductResult]:
    """
    Process all bound products through thiolysis and cyclization in parallel.

    Parameters
    ----------
    bound_pks_products : List[Tuple]
        List of (cluster, product_mol) tuples

    Returns
    -------
    List[UnboundProductResult]
        Unordered list of results from parallel processing
    """
    # Prepare work items (pre-pickle for multiprocessing)
    print("\nPreparing work items for unbound product generation...")
    prep_start = time.time()
    work_items = prepare_work_items(bound_pks_products, REMOVE_STEREOCHEMISTRY)
    prep_time = time.time() - prep_start
    print(f"Work items prepared in {prep_time:.2f} seconds")

    # Run parallel processing
    num_workers = NUM_WORKERS or mp.cpu_count()
    total_items = len(work_items)

    print(f"\nStarting parallel processing with {num_workers} workers...")
    print(f"Total items to process: {total_items}")
    print(f"Chunksize: {CHUNKSIZE_UNBOUND}")

    results: List[UnboundProductResult] = []

    with mp.Pool(processes=num_workers) as pool:
        with tqdm(total=total_items, desc="Processing PKS designs", unit="mol") as pbar:
            for result in pool.imap_unordered(process_single_bound_product,
                                              work_items,
                                              chunksize=CHUNKSIZE_UNBOUND):
                results.append(result)
                pbar.update(1)

    return results


# =============================================================================
# PHASE 3: DEDUPLICATION AND OUTPUT
# =============================================================================

def collect_and_deduplicate_results(results: List[UnboundProductResult]) -> Tuple[Dict[str, str], int, int]:
    """
    Sort results by original index and deduplicate SMILES (first occurrence wins).

    This matches deterministic behavior:
    - Process molecules in order of their original input index
    - For each molecule, process thiolysis products first, then cyclization
    - First occurrence of a SMILES gets added to the dictionary

    Uses identity mapping (smiles -> smiles) to avoid data loss when a single
    PKS design produces unique SMILES from both thiolysis and cyclization.

    Parameters
    ----------
    results : List[UnboundProductResult]
        Unordered list of results from parallel processing

    Returns
    -------
    Tuple[Dict[str, str], int, int]
        (final_dict, num_thiolysis, num_cyclization) where final_dict maps
        each unique SMILES to itself
    """
    # Sort by original index to ensure deterministic ordering
    sorted_results = sorted(results, key=lambda r: r.original_index)

    unique_smiles: set = set()
    final_dict: Dict[str, str] = {}
    num_thiolysis = 0
    num_cyclization = 0

    for result in sorted_results:
        # Process thiolysis first (matches original script order)
        for smiles in result.thiolysis_smiles:
            if smiles not in unique_smiles:
                unique_smiles.add(smiles)
                final_dict[smiles] = smiles  # Identity mapping - no overwrites
                num_thiolysis += 1

        # Then cyclization
        for smiles in result.cyclization_smiles:
            if smiles not in unique_smiles:
                unique_smiles.add(smiles)
                final_dict[smiles] = smiles  # Identity mapping - no overwrites
                num_cyclization += 1

    return final_dict, num_thiolysis, num_cyclization


def get_extender_summary(extender_codes: List[str]) -> str:
    """
    Generate a summary string from extender codes for use in filenames.

    Parameters
    ----------
    extender_codes : List[str]
        List of extender unit codes

    Returns
    -------
    str
        Summary string like "mal_mmal_allylmal_hmal_emal_mxmal"
    """
    code_map = {
        'Malonyl-CoA': 'mal',
        'Methylmalonyl-CoA': 'mmal',
        'allylmal': 'allylmal',
        'hmal': 'hmal',
        'emal': 'emal',
        'mxmal': 'mxmal',
    }
    return '_'.join(code_map.get(c, c.lower()) for c in extender_codes)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main entry point for unified PKS product generation."""
    start_time = time.time()

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate output filename
    extender_summary = get_extender_summary(EXTENDER_CODES)
    stereo_suffix = "no_stereo" if REMOVE_STEREOCHEMISTRY else "with_stereo"
    base_filename = f"pks_products_{MAX_EXTENSION_MODULES}_ext_{stereo_suffix}_{extender_summary}"

    print("=" * 70)
    print("UNIFIED PKS PRODUCT GENERATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Max extension modules: {MAX_EXTENSION_MODULES}")
    print(f"  Starter codes: {STARTER_CODES or 'All available'}")
    print(f"  Extender codes: {EXTENDER_CODES}")
    print(f"  Remove stereochemistry: {REMOVE_STEREOCHEMISTRY}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Output filename: {base_filename}.pkl")

    # =========================================================================
    # PHASE 1: Generate Bound PKS Products
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Generating Bound PKS Products")
    print("=" * 70)

    phase1_start = time.time()
    bound_pks_products = generate_all_bound_products()
    phase1_time = time.time() - phase1_start

    print(f"\nSuccessfully generated {len(bound_pks_products)} bound PKS products")
    print(f"Phase 1 completed in {phase1_time:.2f} seconds")

    # Optionally save bound products
    if SAVE_BOUND_PRODUCTS:
        bound_filepath = os.path.join(OUTPUT_DIR, f"bound_{base_filename}.pkl")
        print(f"\nSaving bound products to: {bound_filepath}")
        with open(bound_filepath, "wb") as f:
            pickle.dump(bound_pks_products, f)

    # =========================================================================
    # PHASE 2: Process Through Thiolysis/Cyclization
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Processing Through Thiolysis/Cyclization")
    print("=" * 70)

    phase2_start = time.time()
    results = process_all_unbound_products(bound_pks_products)
    phase2_time = time.time() - phase2_start

    # Log any errors encountered
    thiolysis_errors = sum(1 for r in results if r.thiolysis_error)
    cyclization_errors = sum(1 for r in results if r.cyclization_error)
    if thiolysis_errors or cyclization_errors:
        print(f"\nEncountered {thiolysis_errors} thiolysis errors and {cyclization_errors} cyclization errors")

    print(f"Phase 2 completed in {phase2_time:.2f} seconds")

    # =========================================================================
    # PHASE 3: Deduplicate and Save
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Deduplicating and Saving Results")
    print("=" * 70)

    phase3_start = time.time()
    unbound_pks_products_dict, num_thiolysis, num_cyclization = collect_and_deduplicate_results(results)
    phase3_time = time.time() - phase3_start

    print(f"\nDeduplication completed in {phase3_time:.2f} seconds")

    # Save the dictionary of unbound PKS products
    output_filepath = os.path.join(OUTPUT_DIR, f"{base_filename}.pkl")
    print(f"\nSaving to: {output_filepath}")
    with open(output_filepath, "wb") as f:
        pickle.dump(unbound_pks_products_dict, f)

    # Save the unique SMILES strings as a text file
    smiles_filepath = os.path.join(OUTPUT_DIR, f"{base_filename}_SMILES.txt")
    print(f"Saving SMILES to: {smiles_filepath}")
    unique_smiles_list = list(unbound_pks_products_dict.values())
    with open(smiles_filepath, "w") as f:
        for smiles in unique_smiles_list:
            f.write(smiles + "\n")

    # =========================================================================
    # Summary
    # =========================================================================
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nGenerated {len(unbound_pks_products_dict)} unique unbound PKS products")
    print(f"  - From thiolysis: {num_thiolysis}")
    print(f"  - From cyclization: {num_cyclization}")
    print(f"\nTiming:")
    print(f"  - Phase 1 (bound products): {phase1_time:.2f} seconds")
    print(f"  - Phase 2 (unbound products): {phase2_time:.2f} seconds")
    print(f"  - Phase 3 (deduplication): {phase3_time:.2f} seconds")
    print(f"  - Total: {total_time:.2f} seconds")
    print(f"\nOutput files:")
    print(f"  - {output_filepath}")
    print(f"  - {smiles_filepath}")
    print("\nDone!")


# =============================================================================
# SCRIPT INITIALIZATION
# =============================================================================

# Modify bcs starters and extenders before importing retrotide
modify_bcs_starters_extenders(starter_codes=STARTER_CODES, extender_codes=EXTENDER_CODES)
print(f"\nNumber of starter units: {len(bcs.starters)}")
print(f"Number of extender units: {len(bcs.extenders)}\n")

# Import retrotide and structureDB only after modifying bcs.starters and bcs.extenders
from retrotide import retrotide, structureDB
print(f"Number of entries in structureDB: {len(structureDB)}\n")

# Get list of extension modules
extension_modules_list = list(structureDB.keys())


if __name__ == "__main__":
    main()
