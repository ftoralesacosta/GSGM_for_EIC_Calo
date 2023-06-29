import numpy as np
import h5py
from tqdm import tqdm

from binning_utils import *


def get_smears(width, shape):

    smears = np.random.default_rng().uniform(low=-width/2,
                                            high=width/2, 
                                            size=shape)
    return smears


geant4_name = "../improvedMIP_200cells_FPCD.hdf5"

bin_dict = get_bin_dict(geant4_name)

g4 = h5py.File(geant4_name, 'r')
nevents = np.shape(g4['hcal_cells'])[0]
chunk_size=2000
#nevents = 100

ncells = np.shape(g4['hcal_cells'])[1]
nvar = np.shape(g4['hcal_cells'])[2]
ncluster_var = np.shape(g4['cluster'])[1]
chunk_size = 100

with h5py.File(f'../test_smear.h5', 'w') as newfile:
    # create empty data set
    dset = newfile.create_dataset('hcal_cells', 
                                shape=(nevents,ncells,nvar),
                                maxshape=(nevents,ncells,nvar), 
                                chunks=(chunk_size, ncells, nvar),
                                dtype=np.float32)



    cluster_dset = newfile.create_dataset('cluster', data=g4['cluster'])
    
    MASK = g4['hcal_cells'][:nevents,:,-1]
    dset[:nevents,:,-1] = MASK

    for var in tqdm(range(0,4)):
        
        width = bin_dict[f"width{var_str[var]}"]

        g4_data = g4['hcal_cells'][:nevents,:,var]

        smears = get_smears(width, np.shape(g4_data))
        dset[:nevents,:,var] = (g4_data + smears)*MASK


    for var in range(0,4):
        n_test = 4
        print(f"Sample: {g4_data[:n_test, 0]} ===> {dset[:n_test, 0, var]}")
