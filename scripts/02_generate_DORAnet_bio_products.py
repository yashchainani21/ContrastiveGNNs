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
