"""
In this script, bound PKS products are synthetically generated using the bcs package.
All starter units are allowed, but the extender units are restricted to Malonyl-CoA and Methylmalonyl-CoA.
The number of stereoisomers produced is also restricted by limiting the KR subtypes that are allowed (this can be changed).
"""

from typing import List, Optional
from collections import OrderedDict
from itertools import product
import multiprocessing as mp
import bcs

# set the maximum number of extension modules to be used in generating PKS products
max_extension_modules = 3

# restrict the number of stereoisomers produced by restricting the KR subtypes that are allowed
# this can be changed to allow for more stereoisomers to be generated later on
allowed_KR_subtypes = ['B1', 'B']

# set output filepath for saving generated (cluster, product) pairs
output_filepath = f"../data/raw/bound_PKS_products_{max_extension_modules}_ext_mods.pkl"

def modify_bcs_starters_extenders(starter_codes: Optional[List[str]] = None,
                                  extender_codes: Optional[List[str]] = None):
    """
    Modify the starter and extender acyl-CoA units used to generate PKS products

    Parameters
    ----------
    starter_codes : Optional[List[str]], optional
        List of starter unit codes to be used. If None, all starter units are used.
    extender_codes : Optional[List[str]], optional
        List of extender unit codes to be used. If None, all extender units are used.

    Returns
    -------
    None
    """
    if starter_codes is not None:
        for key in list(bcs.starters.keys()):
            if key not in starter_codes:
                bcs.starters.pop(key, None) # removes key corresponding to starter code that was not specified

    if extender_codes is not None:
        for key in list(bcs.extenders.keys()):
            if key not in extender_codes:
                bcs.extenders.pop(key, None) # removes key corresponding to extender code that was not specified

# allow for all starter units to be used in generatation of PKS products
starter_codes = None 

# for extenders, only allow Malonyl-CoA and Methylmalonyl-CoA
extender_codes = ['Malonyl-CoA', 'Methylmalonyl-CoA']

modify_bcs_starters_extenders(starter_codes = starter_codes, extender_codes = extender_codes)
print((f"\nNumber of starter units: {len(bcs.starters)}"))
print((f"\nNumber of extender units: {len(bcs.extenders)}\n"))

# import retrotide and structureDB only after modifying bcs.starters and bcs.extenders
from retrotide import retrotide, structureDB
print(f"\nNumber of entries in structureDB: {len(structureDB)}\n")

def build_bcs_cluster_and_product(starter: str, 
                                  extension_mods_combo):
    """
    Build a bcs PKS cluster and its corresponding PKS product given a starter unit and extension module
    """
    try:
        # build loading module
        loading_AT_domain = bcs.AT(active = True, substrate = starter)
        loading_module = bcs.Module(domains = OrderedDict({bcs.AT: loading_AT_domain}), loading = True)
        full_modules = [loading_module] + list(extension_mods_combo)

        # create bcs cluster
        cluster = bcs.Cluster(modules = full_modules)

        # generate PKS product
        product_mol = cluster.computeProduct(structureDB)
        return cluster, product_mol

    except Exception as e:
        print(f"Error building loading module with starter {starter} combo {extension_mods_combo}: {e}")
        return None, None
    
extension_modules_list = list(structureDB.keys())

if __name__ == "__main__":

    all_cluster_product_pairs = []

    with mp.Pool() as pool:
        for i in range(1, max_extension_modules + 1):
            print(f"\nGenerating clusters and products with {i} extension module(s)...\n")

        # create all possible (starter, extension_combo) pairs
        starter_plus_ext_mods_combos = product(bcs.starters.keys(), product(extension_modules_list, repeat = i))

        # build clusters and products in parallel
        results_i = pool.starmap(build_bcs_cluster_and_product, starter_plus_ext_mods_combos)

        # filter out failed builds
        results_i = [r for r in results_i if None not in r]

        all_cluster_product_pairs.extend(results_i)

    print(f"Successfully generated {len(all_cluster_product_pairs)} (cluster, product) pairs.\n")

    with open(output_filepath, "wb") as f:
        import pickle
        pickle.dump(all_cluster_product_pairs, f)
    print(f"Saved all (cluster, product) pairs to {output_filepath}\n")
