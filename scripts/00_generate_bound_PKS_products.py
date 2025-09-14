from typing import List, Optional
import bcs

def modify_bcs_starters_extenders(starter_codes: Optional[List[str]] = None,
                                  extender_codes: Optional[List[str]] = None):
    """
    Modify the starter and extender acyl-CoA units used to generate PKS products
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