from typing import List, Optional
from collections import OrderedDict
import bcs

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

# restrict the number of stereoisomers produced by restricting the KR subtypes that are allowed
# this can be changed to allow for more stereoisomers to be generated later on
allowed_KR_subtypes = ['B1', 'B']

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
