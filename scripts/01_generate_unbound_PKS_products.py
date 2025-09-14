import pickle
from typing import List
from rdkit import Chem

with open("../data/raw/bound_PKS_products_3_ext_mods.pkl", "rb") as f:
    bound_PKS_products = pickle.load(f)

bound_PKS_mols: List[Chem.Mol] = [item[1] for item in bound_PKS_products]
