import os
import sys
import csv
import time
from typing import List, Set

from rdkit import Chem
from rdkit import RDLogger

# Silence RDKit noise
RDLogger.DisableLog('rdApp.*')


def get_workspace_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def ensure_retrotide_on_path() -> None:
    # Add BioPKS-Pipeline/biopks_pipeline to PYTHONPATH so we can import retrotide
    root = get_workspace_root()
    biopks_path = os.path.join(root, "BioPKS-Pipeline", "biopks_pipeline")
    if biopks_path not in sys.path:
        sys.path.insert(0, biopks_path)


def run_doranet_retro(start_smiles: str, gens: int = 1) -> Set[str]:
    """Run DORAnet synthetic module in retro mode starting from start_smiles.

    Returns a set of SMILES generated in the network (including the starter).
    """
    from doranet.modules import synthetic

    helper_smiles = (
        "O", "O=O", "[H][H]", "O=C=O", "C=O", "[C-]#[O+]", "Br", "[Br][Br]", "CO", "C=C",
        "O=S(O)O", "N", "O=S(=O)(O)O", "O=NO", "N#N", "O=[N+]([O-])O", "NO", "C#N", "S", "O=S=O"
    )

    print("Running DORAnet (synthetic, retro)")
    net = synthetic.generate_network(
        job_name="cryptofolione_retro",
        starters={start_smiles},
        helpers=tuple(helper_smiles),
        gen=gens,
        direction="retro",
    )

    smiles_set: Set[str] = set()
    for mol in net.mols:
        try:
            smi = Chem.MolToSmiles(Chem.MolFromSmiles(mol.uid))
        except Exception:
            continue
        if smi:
            smiles_set.add(smi)

    return smiles_set


def try_retrotide_design(target_smiles: str, max_designs: int = 15, similarity: str = "mcs_without_stereo"):
    """Attempt to design a PKS for the target using RetroTide.

    Returns a tuple: (best_score: float, total_rounds: int, top_smiles: str)
    """
    ensure_retrotide_on_path()
    from retrotide import retrotide

    target_mol = Chem.MolFromSmiles(target_smiles)
    if target_mol is None:
        return None

    try:
        rounds = retrotide.designPKS(
            targetMol=target_mol,
            maxDesignsPerRound=max_designs,
            similarity=similarity,
        )
    except Exception as e:
        print(f"RetroTide error for {target_smiles}: {e}")
        return None

    # rounds is a list; last element contains designs for final round
    final_round = rounds[-1] if rounds else []
    if not final_round:
        return (0.0, len(rounds), "")

    # Each entry in a round is [Cluster, score, structure_mol]
    best_design = max(final_round, key=lambda x: x[1])
    best_score = float(best_design[1])
    best_mol = best_design[2]
    best_smiles = Chem.MolToSmiles(best_mol) if best_mol is not None else ""
    return (best_score, len(rounds), best_smiles)


def main():
    # Parameters
    # cryptofolione
    start_smiles = "C1C=CC(=O)OC1C=CCC(CC(/C=C/C2=CC=CC=C2)O)O"
    doranet_gens = int(os.environ.get("DORANET_GENS", "1"))

    # Run DORAnet retro expansion from benzene
    products_all = run_doranet_retro(start_smiles, gens=doranet_gens)

    # Remove helpers and the starter itself
    helpers = {
        "O", "O=O", "[H][H]", "O=C=O", "C=O", "[C-]#[O+]", "Br", "[Br][Br]", "CO", "C=C",
        "O=S(O)O", "N", "O=S(=O)(O)O", "O=NO", "N#N", "O=[N+]([O-])O", "NO", "C#N", "S", "O=S=O"
    }
    if start_smiles in products_all:
        products_all.remove(start_smiles)
    products = {s for s in products_all if s not in helpers}

    print(f"Total unique DORAnet molecules (all, excl. starter): {len(products_all)}")
    print(f"Total unique DORAnet products (excluding helpers/start): {len(products)}")

    # Save product SMILES for downstream ML inference timing
    out_dir = os.path.dirname(__file__)
    all_txt = os.path.join(out_dir, "doranet_retro_cryptofolione_products_all.txt")
    filtered_txt = os.path.join(out_dir, "doranet_retro_cryptofolione_products_filtered.txt")
    with open(all_txt, "w") as f_all:
        for smi in sorted(products_all):
            f_all.write(f"{smi}\n")
    with open(filtered_txt, "w") as f_f:
        for smi in sorted(products):
            f_f.write(f"{smi}\n")
    print(f"Saved DORAnet products (all): {all_txt}")
    print(f"Saved DORAnet products (filtered): {filtered_txt}")

    # Try RetroTide for each product (in series) and time the total duration
    results = []
    rt_start = time.time()
    total_targets = len(products)
    for idx, smi in enumerate(sorted(products), start=1):
        print(f"\n[{idx}/{total_targets}] RetroTide on: {smi}")
        res = try_retrotide_design(smi)
        if res is None:
            results.append({
                "target_smiles": smi,
                "retrotide_best_score": "ERROR",
                "retrotide_rounds": "-",
                "retrotide_top_product": "",
            })
        else:
            best_score, n_rounds, top_smi = res
            results.append({
                "target_smiles": smi,
                "retrotide_best_score": best_score,
                "retrotide_rounds": n_rounds,
                "retrotide_top_product": top_smi,
            })
    rt_elapsed = time.time() - rt_start
    print(f"\nTotal RetroTide time (all products, series): {rt_elapsed:.2f} s")

    # Save to CSV alongside this script
    out_csv = os.path.join(os.path.dirname(__file__), "doranet_retro_cryptofolione_retrotide_results.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "target_smiles", "retrotide_best_score", "retrotide_rounds", "retrotide_top_product"
        ])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # Save timing info
    timing_path = os.path.join(os.path.dirname(__file__), "doranet_retro_cryptofolione_retrotide_timing.txt")
    with open(timing_path, "w") as tf:
        tf.write(f"retro_products_count={len(results)}\n")
        tf.write(f"retrotide_total_seconds={rt_elapsed:.3f}\n")
    print(f"\nSaved results: {out_csv}")
    print(f"Saved RetroTide timing: {timing_path}")


if __name__ == "__main__":
    main()
