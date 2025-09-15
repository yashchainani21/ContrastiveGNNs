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
    num_chem_steps = 1

    # MPI initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # define filepaths for unbound PKS products
    precursors_filepath = f'../data/interim/unbound_PKS_products_{max_extension_modules}_ext_mods_no_stereo.pkl'
    output_filepath = f'../data/interim/DORAnet_CHEM{num_chem_steps}_from_PKS_products_{max_extension_modules}_ext_mods_no_stereo.txt'

    # only rank 0 reads files and sets up initial data
    if rank == 0:
        with open(precursors_filepath, 'rb') as precursors_file:
            unbound_PKS_products = pd.read_pickle(precursors_file)
            precursors_list = [smi for smi in unbound_PKS_products.values()]
    else:
        precursors_list = None  # <-- define it for all non-root ranks

    # broadcast the precursors list to all processes
    precursors_list = comm.bcast(precursors_list, root=0)

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