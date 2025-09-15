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
