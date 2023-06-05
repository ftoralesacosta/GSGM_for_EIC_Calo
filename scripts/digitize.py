import h5py
import numpy as np
from tqdm import tqdm


def get_bin_edges(g4_cell_data):

    centers = np.unique(g4_cell_data)

    # Don't want 0 as first bin (important for ZXY)
    if (centers[0] == 0):
        centers = centers[1:]

    width = np.round(centers[1] - centers[0],2)

    edges = centers - width/2
    max_edge = centers[-1] + width/2
    edges = (np.append(edges,max_edge))
    
    return centers, edges


def get_bin_dict(geant4_name, var_str, nevts = 100_000):

    bin_dict = {}
    with h5py.File(geant4_name, 'r') as g4:

        for var in range(1,4):

            g4_data = g4['hcal_cells'][:nevts,:,var]
            centers, edges = get_bin_edges(g4_data)
        
            bin_dict[f"centers{var_str[var]}"] = centers
            bin_dict[f"edges{var_str[var]}"] = edges 

        print("\nL35: Dictionary Keys = ",bin_dict.keys())

    return bin_dict


def get_digits_dict(continuous_file, dset_name, bin_dict):

    digit_dict = {}

    for var in range(1,4):
        continuous_data = continuous_file[dset_name][:,:,var]
        digits = np.digitize(continuous_data,bin_dict[f"edges{var_str[var]}"])
        print("Sample Bin # ",var_str[var],": ",digits[100,:10])
        digit_dict[f"digits{var_str[var]}"] = digits - 1  # -1 for 0th index

    return digit_dict



#  ======= MAIN ======

# Start with a naturally discrete, G4 Dataset
geant4_name = "improvedMIP_200cells_FPCD.hdf5"

# Dataset Names
smeared_G4 = False

if (smeared_G4):

    dset_name = 'hcal_cells' #for smeared G4 data, not for GSGM
    cluster_name = "cluster"
    discrete_name = "G4_Discrete.h5"
    continuous_name = "newMIP_smeared_20keV_200cells_FPCD.hdf5"

else:

    dset_name = 'cell_features'
    cluster_name = 'cluster_features'
    discrete_name = "GSGM_Discrete.h5"
    continuous_name = "GSGM.h5"


var_str = ["E","X","Y","Z"]

# Get Cell-Binning from Discrete G4
bin_dict = get_bin_dict(geant4_name, var_str)

#Load the contituous file you want digitized
continuous_file = h5py.File(continuous_name, "r")

# Get what bin each datum belongs in
digit_dict = get_digits_dict(continuous_file, dset_name, bin_dict)

#Copy the structure of the continuous File
nevents = np.shape(continuous_file[dset_name])[0]
ncells = np.shape(continuous_file[dset_name])[1]
nvar = np.shape(continuous_file[dset_name])[2]
ncluster_var = np.shape(continuous_file[cluster_name])[1]

# nevents = 10_000
# ncells = 200
# nvar = 4
# ncluster_var = 2
chunk_size = 100

# Testing
# discrete_name = "TEST_PY_GSGM_Discrete.h5"

# create empty data set
with h5py.File(discrete_name, 'w') as newfile:

    dset = newfile.create_dataset(dset_name, 
                                shape=(nevents,ncells,nvar),
                                maxshape=(nevents,ncells,nvar), 
                                chunks=(chunk_size, ncells, nvar),
                                dtype=np.float32)

    cluster_dset = newfile.create_dataset(cluster_name, 
                                          data=continuous_file[cluster_name])
    
    dset[:,:,0] = continuous_file[dset_name][:,:,0] # Just copy E

    if (smeared_G4):
        dset[:,:,-1] = continuous_file[dset_name][:,:,-1]
        # Copy MASK in G4

    for var in range(1,4):
        
        g4_centers = bin_dict[f"centers{var_str[var]}"]  #what data is set to
        n_bins = len(bin_dict[f"centers{var_str[var]}"]) 
        var_mask =  digit_dict[f"digits{var_str[var]}"]  #which data to edit


        # Load the data to digitize
        continuous_data = continuous_file[dset_name][:,:,var]

        # Set data to bin center
        for evt in tqdm(range(nevents)):
            for ibin in range(n_bins):

                bin_mask = var_mask[evt] == ibin
                continuous_data[evt][bin_mask] = g4_centers[ibin]
                
        dset[:,:,var] = np.round(continuous_data,2)

        print("\nSample Check", np.round(continuous_data[100,25:35],2))
        print(f"Done with {var_str[var]}")
