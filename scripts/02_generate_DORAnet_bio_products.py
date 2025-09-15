import uuid
from mpi4py import MPI
from rdkit import Chem
import doranet.modules.enzymatic as enzymatic
import pandas as pd
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

if __name__ == '__main__':

    # define PKS products to modify
    max_extension_modules = 3

    # MPI initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # define filepaths for PKS unbound PKS products
    precursors_filepath = f'../data/interim/unbound_PKS_products_{max_extension_modules}_ext_mods_no_stereo.pkl'
    output_filepath = f'../data/interim/DORAnet_BIO1_from_PKS_products_{max_extension_modules}_ext_mods_no_stereo.txt'

    # only rank 0 reads files and sets up initial data
    if rank == 0:

        # read in unbound PKS product SMILES
        with open(precursors_filepath, 'rb') as precursors_file:
            unbound_PKS_products = pd.read_pickle(precursors_file)
            precursors_list = [smi for smi in unbound_PKS_products.values()]

    # broadcast the precursors list to all processes
    precursors_list = comm.bcast(precursors_list, root = 0)

    # define a helper function to evenly split the precursors list into n chunks

    # define a helper function to evenly split the precursors list into n chunks
    def chunkify(lst, n):
        """Split lst into n (roughly) equal-sized chunks. Avoid creating empty chunks."""
        n = min(n, len(lst))  # avoid creating more chunks than data
        k, m = divmod(len(lst), n)
        return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

    # scatter the data (only rank 0 prepares the chunks)
    if rank == 0:
        chunks = chunkify(precursors_list, size)
        num_active_ranks = len(chunks)
    else:
        chunks = None

    # broadcast num_active_ranks to all processes
    num_active_ranks = comm.bcast(num_active_ranks if rank == 0 else None, root=0)

    # Scatter to all ranks — unused ranks get empty lists
    if rank < num_active_ranks:
        my_precursors = comm.scatter(chunks, root=0)
    else:
        my_precursors = []

    print(f"[Rank {rank}] received {len(my_precursors)} precursors.", flush=True)

    def perform_DORAnet_bio_1step(precursor_smiles: str):
        """
        Generates one-step DORAnet products for a given precursor SMILES string.
        """
        unique_id = str(uuid.uuid4())
        unique_jobname = f'{precursor_smiles}_{unique_id}'

        forward_network = enzymatic.generate_network(
            job_name = unique_jobname,
            starters = {precursor_smiles},
            gen = 1,
            direction = "forward")

        generated_bioproducts_list = []

        for mol in forward_network.mols:
            generated_bioproduct_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(mol.uid))

            # Store only non-cofactor unique products
            if generated_bioproduct_smiles and generated_bioproduct_smiles not in cofactors_list:
                generated_bioproducts_list.append(generated_bioproduct_smiles)

        return generated_bioproducts_list
