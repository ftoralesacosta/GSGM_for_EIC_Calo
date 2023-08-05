# This code voxelizez Geant4 data int 3D images 
# Not to be confused with digitizing point cloud data

import numpy as np
import h5py
from tqdm import tqdm
from binning_utils import *


def XYZ_to_ZYX(data):
# Data must be EZXY

    if len(np.shape(data)) != 3:
        print("data in XYZ to ZXY converter not 3D")

    data[:,:,[1,3]] = data[:,:,[3,1]]#Swap Z and X
    data[:,:,[2,3]] = data[:,:,[3,2]]#Swap Y and X


# ===== MAIN ====

# File I/O
discrete_name = "../G4_Nominal.hdf5" # vanilla G4 (previously improvedMIP...)

#input_file = "../G4_smeared.h5"
# input_name = "../GSGM.h5"
# input_name = "../GSGM_1cond.h5"
# input_name = "../GSGM_256mlp.h5"
input_name = "../GSGM_128mlp.h5"
# input_name = "../GSGM_largePart.h5"

input_file = h5py.File(input_name, 'r')
input_file.keys()

dset_name = "hcal_cells"
cluster_name = "cluster"
label = "full_gran_"

if ("GSGM" in input_name):
    dset_name = "cell_features"
    # cluster_name = "cluster_features"
    cluster_name = "truth_features"
    # label = "GSGM_full_gran_"
    label = ""


#Reduction factor of cells ==> voxels
voxel_factor = 5 #have been using 5

# Output H5 File
voxel_out_name = f"../{label}epic_hcal_images_{voxel_factor}x{voxel_factor}.h5"
print("Will write out to ", voxel_out_name)


# Get Binning Dictionary, and reduce it
bin_dict = get_bin_dict( discrete_name )
voxel_binning(bin_dict,voxel_factor)

# Get shape of datasets
nevents = np.shape(input_file[dset_name])[0]
nXY = len(bin_dict["centersX"])
nZ = len(bin_dict["centersZ"])

print("N Bins = ",nZ, nXY, nXY)

binX = bin_dict["edgesX"]
binY = bin_dict["edgesY"]
binZ = bin_dict["edgesZ"]


chunk_size = 1000
image_shape = [nevents, nZ, nXY, nXY]  #ZXY
chunk_shape = (chunk_size, nZ, nXY, nXY)

ntruth = 2 # use [:2] to write just genP and genTheta


with h5py.File(voxel_out_name,'w') as newfile:

    dset = newfile.create_dataset('calo_images', 
                                  shape=((image_shape)),
                                  maxshape=(image_shape), 
                                  chunks=(chunk_shape),
                                  dtype=np.float32)

    truth_dset = newfile.create_dataset(cluster_name, 
                                        data=input_file[cluster_name][:,:ntruth])


    for chunk in tqdm(range(int(nevents/chunk_size))):

        start = chunk*chunk_size

        data = input_file[dset_name][start:start+chunk_size]
        XYZ_to_ZYX(data)

        # if ("GSGM" in input_name):
        #     data[:,:,0] = 10**data[:,:,0] 

        images = []
        for evt in range(chunk_size):

            if ("GSGM" in input_name):
                counts, binedges = np.histogramdd(data[evt,:,1:],  #omit E
                                              bins=(binZ, binX, binY), 
                                              weights=10**data[evt,:,0])  # weight=E
            else:
                counts, binedges = np.histogramdd(data[evt,:,1:-1], #omit MASK
                                              bins=(binZ, binX, binY), 
                                              weights=data[evt,:,0])  # weight=E

            images.append(counts)

        images=np.asarray(images)

        dset[start:start+chunk_size, :, :,:] = images[:, :, :,:]


