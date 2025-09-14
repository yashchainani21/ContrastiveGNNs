import pickle
from typing import List
from rdkit import Chem

from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List

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


with open("../data/raw/bound_PKS_products_3_ext_mods.pkl", "rb") as f:
    bound_PKS_products = pickle.load(f)

