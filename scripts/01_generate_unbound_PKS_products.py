import bcs
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Dict

remove_stereochemistry = True
max_extension_modules = 3

if remove_stereochemistry:
    output_filepath = f"../data/interim/unbound_PKS_products_{max_extension_modules}_ext_mods_no_stereo.pkl"
else:
    output_filepath = f"../data/interim/unbound_PKS_products_{max_extension_modules}_ext_mods_with_stereo.pkl"

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

with open(f"../data/raw/bound_PKS_products_{max_extension_modules}_ext_mods.pkl", "rb") as f:
    bound_PKS_products = pickle.load(f)

PKS_designs = [bound_PKS_products[i][0] for i in range(len(bound_PKS_products))]
bound_PKS_mols = [bound_PKS_products[i][1] for i in range(len(bound_PKS_products))]

unbound_PKS_products_dict: Dict[bcs.Cluster, Chem.Mol] = {}
unique_PKS_product_smiles = set()

num_successful_thyiolysis_rxns = 0
num_successful_cyclization_rxns = 0

for i, bound_PKS_mol in enumerate(bound_PKS_mols):
    PKS_design = PKS_designs[i]

    # try a thiolysis reaction to offload the PKS product
    try:
        unbound_PKS_products_list: List[Chem.Mol] = run_pks_release_reaction("thiolysis", bound_PKS_mol)
        
        # for each product generated from thiolysis
        for unbound_PKS_product_mol in unbound_PKS_products_list:

            # remove stereochemistry from resulting carboxylic acid product if specified
            if remove_stereochemistry:
                Chem.RemoveStereochemistry(unbound_PKS_product_mol)
            
            unbound_PKS_product_smiles = Chem.MolToSmiles(unbound_PKS_product_mol)

            # if each carboxylic acid product is unique, save the PKS design and its corresponding product SMILES
            if unbound_PKS_product_smiles not in unique_PKS_product_smiles and unbound_PKS_product_smiles != 'S' and unbound_PKS_product_smiles != 'O=C=O':
                unbound_PKS_products_dict[PKS_design] = unbound_PKS_product_smiles
                num_successful_thyiolysis_rxns += 1
                
                # keep track of all unique PKS product SMILES
                unique_PKS_product_smiles.add(unbound_PKS_product_smiles)
                

    except Exception as e:
        print(f"Error in thiolysis for {PKS_design}: {e}")
        continue

    # try a cyclization reaction to offload the PKS product
    try:
        unbound_PKS_products_list: List[Chem.Mol] = run_pks_release_reaction("cyclization", bound_PKS_mol)

        # for each product generated from cyclization
        for unbound_PKS_product_mol in unbound_PKS_products_list:
            unbound_PKS_product_smiles = Chem.MolToSmiles(unbound_PKS_product_mol)
            
            # remove stereochemistry from resulting cyclized product if specified
            if remove_stereochemistry:
                Chem.RemoveStereochemistry(unbound_PKS_product_mol)

            unbound_PKS_product_smiles = Chem.MolToSmiles(unbound_PKS_product_mol)

            # if each cyclized product is unique, save the PKS design and its corresponding product SMILES
            if unbound_PKS_product_smiles not in unique_PKS_product_smiles and unbound_PKS_product_smiles != 'S' and unbound_PKS_product_smiles != 'O=C=O':
                unbound_PKS_products_dict[PKS_design] = unbound_PKS_product_smiles
                num_successful_cyclization_rxns += 1

                # keep track of all unique PKS product SMILES
                unique_PKS_product_smiles.add(unbound_PKS_product_smiles)

    except Exception as e:
        print(f"Error in cyclization for {PKS_design}: {e}")
        continue

print('\n--------------------------------------------\n')
print(f"Generated {len(unbound_PKS_products_dict)} unique unbound PKS products.\n")
print(f"Number of successful thiolysis reactions: {num_successful_thyiolysis_rxns}\n")
print(f"Number of successful cyclization reactions: {num_successful_cyclization_rxns}\n")

# save the dictionary of unbound PKS products
with open(output_filepath, "wb") as f:
    pickle.dump(unbound_PKS_products_dict, f)