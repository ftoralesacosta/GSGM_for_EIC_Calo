# This code voxelizez Geant4 data int 3D images 
# Not to be confused with digitizing point cloud data

import numpy as np
import h5py
from tqdm import tqdm
from binning_utils import *



def voxel_binning(bins, voxel_factor):
    # pass dict by reference :]

    start = int(voxel_factor/2) #first center

    for var in var_str:

        if var == "E":
            continue
        
        bins[f"centers{var}"] = bins[f"centers{var}"][start::voxel_factor]
        bins[f"edges{var}"] = bins[f"edges{var}"][::voxel_factor]

        # print(len(bins[f"centers{var}"]))
        # print(var,bins[f"centers{var}"])
        # print(var,bins[f"edges{var}"])


def XYZ_to_ZYX(data):
# Data must be EZXY

    if len(np.shape(data)) != 3:
        print("data in XYZ to ZXY converter not 3D")

    data[:,:,[1,3]] = data[:,:,[3,1]]#Swap Z and X
    data[:,:,[2,3]] = data[:,:,[3,2]]#Swap Y and X


# ===== MAIN ====

# File I/O
geant4_name = "../improvedMIP_200cells_FPCD.hdf5"
g4_file = h5py.File(geant4_name, 'r')
g4_file.keys()


#Reduction factor of cells ==> voxels
voxel_factor = 5

# Get Binning Dictionary, and reduce it
bin_dict = get_bin_dict( geant4_name )
voxel_binning(bin_dict,voxel_factor)


# Get shape of datasets
nevents = np.shape(g4_file['hcal_cells'])[0]
nXY = len(bin_dict["centersX"])
nZ = len(bin_dict["centersZ"])


binX = bin_dict["edgesX"]
binY = bin_dict["edgesY"]
binZ = bin_dict["edgesZ"]


chunk_size = 2000
image_shape = [nevents, nZ, nXY, nXY]  #ZXY
chunk_shape = (chunk_size, nZ, nXY, nXY)

ntruth = 2 # use [:2] to write just genP and genTheta


with h5py.File(f'epic_hcal_images_{voxel_factor}x{voxel_factor}.h5','w') as newfile:

    dset = newfile.create_dataset('calo_images', 
                                  shape=((image_shape)),
                                  maxshape=(image_shape), 
                                  chunks=(chunk_shape),
                                  dtype=np.float32)

    truth_dset = newfile.create_dataset('cluster', 
                                        data=g4_file['cluster'][:,:ntruth])


    for chunk in tqdm(range(int(nevents/chunk_size))):

        start = chunk*chunk_size

        data = g4_file['hcal_cells'][start:start+chunk_size]
        XYZ_to_ZYX(data)

        images = []
        for evt in range(chunk_size):
            counts, binedges = np.histogramdd(data[evt,:,1:-1],  # omit E and Mask
                                              bins=(binZ, binX, binY), 
                                              weights=data[evt,:,0])  # weight=E
            images.append(counts)

        images=np.asarray(images)

        dset[start:start+chunk_size, :, :,:] = images[:, :, :,:]


