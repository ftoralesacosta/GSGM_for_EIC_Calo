import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import gridspec
import argparse
import h5py as h5
import os
import utils
import tensorflow as tf
from GSGM import GSGM
from GSGM_distill import GSGM_distill
import time
import gc
import sys
sys.path.append("JetNet")
from jetnet.evaluation import w1p,w1m,w1efp,cov_mmd,fpnd
from scipy.stats import wasserstein_distance
from plot_class import PlottingConfig

def write_list_to_file(path, my_list):
    with open(path, 'w') as file:
        for item in my_list:
            file.write(str(item) + '\n')

import numpy as np

def create_uniform_2d_array(x_min, x_max, x_min2, x_max2, length):

    if length < 1:
        raise ValueError("Length should be at least 1 to create a meaningful array.")

    # Generate a 1D array of evenly spaced values for each column
    column1 = np.linspace(x_min, x_max, length)
    column2 = np.linspace(x_min2, x_max2, length)

    # Combine the columns to form the (length, 2) 2D array
    result_array = np.column_stack((column1, column2))
    return result_array

def W1(
        cluster1,
        cluster2,
        num_batches = 10,
        return_std = True,
        num_eval=50000,
):

    w1s = []
    
    for j in range(num_batches):
        rand1 = np.random.choice(len(cluster1), size=num_eval,replace=True)
        rand2 = np.random.choice(len(cluster2), size=num_eval,replace=True)

        rand_sample1 = cluster1[rand1]
        rand_sample2 = cluster2[rand2]

        w1 = [wasserstein_distance(rand_sample1, rand_sample2)]
        w1s.append(w1)
        
    means = np.mean(w1s, axis=0)
    stds = np.std(w1s, axis=0)
    return means, stds

def plot(cluster1,cluster2,cond1,cond2,nplots,title,plot_folder,is_big):

    print('nplots',nplots)
    
    for ivar in range(nplots):

        print('ivar',ivar)
        print('title',title)
        print('is_big',is_big)

        config = PlottingConfig(title,ivar,is_big)

        name = utils.names[ivar]

        feed_dict = {
            '{}_truth'.format(name): cluster1[:,ivar],
            '{}_gen'.format(name):  cluster2[:,ivar]
        }

        print('feed_dict',feed_dict)

        if ivar == 0:                            
            fig,gs,_ = utils.HistRoutine(feed_dict,xlabel=config.var,
                                         binning=config.binning,
                                         plot_ratio=False,
                                         reference_name='{}_truth'.format(name),
                                         ylabel= 'Normalized entries',logy=config.logy)
        else:
            fig,gs,_ = utils.HistRoutine(feed_dict,xlabel=config.var,
                                         reference_name='{}_truth'.format(name),
                                         plot_ratio=False,
                                         fig=fig,gs=gs,binning=config.binning,
                                         ylabel= 'Normalized entries',logy=config.logy)
        ax0 = plt.subplot(gs[0])     
        ax0.set_ylim(top=config.max_y)
        if config.logy == False:
            yScalarFormatter = utils.ScalarFormatterClass(useMathText=True)
            yScalarFormatter.set_powerlimits((100,0))
            ax0.yaxis.set_major_formatter(yScalarFormatter)

        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        fig.savefig('{}/GSGM_{}_{}.pdf'.format(plot_folder,title,ivar),bbox_inches='tight')


if __name__ == "__main__":
    print( "Running plot_jet.py as script" )
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    utils.SetStyle()


    parser = argparse.ArgumentParser()

    #parser.add_argument('--data_folder', default='./', help='Folder containing data and MC files')
    parser.add_argument('--data_folder', default='/usr/workspace/hip/eic/scratch/', help='Folder containing data and MC files')
    parser.add_argument('--plot_folder', default='./plots/', help='Folder to save results')
    parser.add_argument('--config', default='config_cluster.json', help='Training parameters')

    parser.add_argument('--model', default='GSGM', help='Type of generative model to load')
    parser.add_argument('--distill', action='store_true', default=False,help='Use the distillation model')
    parser.add_argument('--test', action='store_true', default=False,help='Test if inverse transform returns original data')
    parser.add_argument('--big', action='store_true', default=False,help='Use bigger dataset (1000 cells) as opposed to 200 cells')
    parser.add_argument('--sample', action='store_true', default=False,help='Sample from the generative model')
    parser.add_argument('--comp', action='store_true', default=False, help='Compare the results for diffusion models with different diffusion steps')
    parser.add_argument('--factor', type=int,default=1, help='Step reduction for distillation model')


    flags = parser.parse_args()
    config = utils.LoadJson(flags.config)

    if flags.big:
        labels = utils.labels1000
        npart=1000
    else:
        labels=utils.labels200
        npart=200

    cells, clusters, condition = utils.DataLoader(flags.data_folder,
                                                  labels=labels,
                                                  npart=npart,
                                                  num_condition=config['NUM_COND'],
                                                  make_tf_data=False)

    #print(condition.shape) # (10, 2)
    #print(condition) # [[ 1.0066884  17.722715  ] , [ 1.1772832  17.876328  ] , [ 1.2580142  15.626081  ] , [ 0.344686   18.366722  ] , [ 1.8469115  18.092258  ] , [ 0.21613598 18.01947   ] , [ 0.49931717 16.488306  ] , [ 0.24487707 17.234745  ] , [ 1.6350818  17.398014  ] , [ 0.7529935  17.117504  ]]

    if flags.test:
        cells_gen, clusters_gen, condition_gen = utils.SimpleLoader(flags.data_folder,labels=labels)
        sample_name = "test_mode"

    else:
        model_name = config['MODEL_NAME']
        if flags.big:
            model_name+='_big'

        sample_name = model_name
        if flags.distill:
            sample_name += '_d{}'.format(flags.factor)

        if flags.sample:            
            model = GSGM(config=config,factor=flags.factor,npart=npart)
            checkpoint_folder = '../checkpoints_{}/checkpoint'.format(model_name)
            # checkpoint_folder = '../checkpoints_GSGM_128mlp/checkpoint'.format(model_name)
            if flags.distill:
                checkpoint_folder = '../checkpoints_{}_d{}/checkpoint'.format(model_name,flags.factor)
                model = GSGM_distill(model.ema_cluster,model.ema_part,config=config,
                                     factor=flags.factor,npart=npart)
                print("Loading distilled model from: {}".format(checkpoint_folder))
            model.load_weights('{}'.format(checkpoint_folder)).expect_partial()

            cells_gen = []
            clusters_gen = []

            # nsplit = 100 #number of batches, in which to split nevts in utils.py
            nsplit = 4 #number of batches, in which to split nevts in utils.py

            #print(clusters)
            #print(clusters.shape) # (10000, 2) --> 20 arrays of (500,2)

            split_part = np.array_split(clusters,nsplit) # split the clusters

            nvts = len(condition)

            print('split_part',len(split_part)) # 20 , 5
            print('split_part.shape',split_part[0].shape) # split_part.shape (500, 2)
            print('condition', len(condition)) # condition 10000 (nevts) , 40
            print('condition.shape', condition[0].shape) # condition.shape (2,)
            print('condition 0',condition)

            #condition = create_uniform_2d_array(0, 2.1198688, 12, 20, 10) 
            condition = create_uniform_2d_array(1, 1, 15.567589, 18.432405, 20) # nvets/nsplit
            condition = tf.cast(condition, tf.float32)

            print('condition 1',condition)
            #print('condition 1', len(condition)) # condition 1 10
            #print('condition.shape', condition[0].shape) # condition.shape (2,)

            for i, split in enumerate(np.array_split(condition,nsplit)):
                
                print('split',split.shape) # split (500, 2) --> condition , nevts/nplit (5,2)
                print('split_part[i]',split_part[i].shape) # split_part[i] (500, 2) --> cluster
                #,split_part[i]
                # genP as input to model.genearet()
                start = time.time()
                p,j = model.generate(split,split_part[i])

                print(f"Time to sample {np.shape(split_part[i])[0]} events is {time.time() - start} seconds")
                print(p.shape)# (5, 200, 4)
                print(j.shape)# (5, 2)
                #print('j', j)


                cells_gen.append(p)
                clusters_gen.append(j)

            print(cells_gen[0].shape) # (5, 200, 4)
            print(clusters_gen[0].shape) # (5, 2)
            #exit()

            cells_gen = np.concatenate(cells_gen)
            clusters_gen = np.concatenate(clusters_gen)

            print("L 162: ReversePrep Call")
            cells_gen, clusters_gen = utils.ReversePrep(cells_gen,clusters_gen,npart=npart)
            # clusters_gen = np.concatenate([clusters_gen,np.expand_dims(np.argmax(condition,-1),-1)],-1)

            print(cells_gen.shape) # (10, 200, 4)
            print(clusters_gen.shape) # (10, 2)
            print(condition.shape) # (10, 2)
            #print(condition)

            #condition = create_uniform_2d_array(0, 2.1198688, 15.567589, 18.432405, 10)
            #print(condition)

            path_gen = '/usr/workspace/sinha4/GSGM_new/GSGM_for_EIC_Calo/scripts/gen_data'

            path_gen_cells = '/usr/workspace/sinha4/GSGM_new/GSGM_for_EIC_Calo/scripts/gen_data/cells_g_'+str(nvts)+'.txt'
            path_gen_clus = '/usr/workspace/sinha4/GSGM_new/GSGM_for_EIC_Calo/scripts/gen_data/clusters_g_'+str(nvts)+'.txt'
            path_gen_cond = '/usr/workspace/sinha4/GSGM_new/GSGM_for_EIC_Calo/scripts/gen_data/cond_g_'+str(nvts)+'.txt'

            write_list_to_file(path_gen_cells, cells_gen)
            write_list_to_file(path_gen_clus, clusters_gen) 
            write_list_to_file(path_gen_cond, condition)

            with h5.File(os.path.join(path_gen,sample_name+'.h5'),"w") as h5f:
                dset = h5f.create_dataset("cell_features", data=cells_gen)
                dset = h5f.create_dataset("cluster_features", data=clusters_gen)

        else:
            with h5.File(os.path.join(flags.data_folder,sample_name+'.h5'),"r") as h5f:
                cells_gen = h5f['cell_features'][:]
                #print(cells_gen)
                #print(cells_gen.shape)
                clusters_gen = h5f['cluster_features'][:]
                #print(clusters_gen)
                #print(clusters_gen.shape)

        condition_gen = clusters_gen[:,-1]

        # assert np.all(condition_gen == np.argmax(condition,-1)), 'The order between the cells dont match'
        clusters_gen = clusters_gen[:,:-1]

    print("L 179: ReversePrep Call")
    cells, clusters = utils.ReversePrep(cells,clusters,npart=npart)

    print(cells.shape) # (10, 200, 4)
    print(clusters.shape) # (10, 2)

    print(cells_gen.shape)  # (10, 200, 4)
    print(clusters_gen.shape) # (10, 1)

    print(condition_gen.shape) # (10, )


    plot(clusters,clusters_gen,condition,condition,title='cluster',
         nplots=1,plot_folder=flags.plot_folder,is_big=flags.big)

    path_gen = '/usr/workspace/sinha4/GSGM_new/GSGM_for_EIC_Calo/scripts/gen_data/clusters_gen_'+str(nvts)+'.txt'

    write_list_to_file(path_gen, clusters_gen)

    condition = np.tile(np.expand_dims(condition,1),(1,cells_gen.shape[1],1)).reshape((-1,condition.shape[-1]))

    print('*'*30)

    print(cells_gen.shape) # (10, 200, 4) ; (5, 200, 4)

    cells_gen= cells_gen.reshape((-1,4))
    mask_gen = cells_gen[:,2]>0. # masking which column.
    cells_gen=cells_gen[mask_gen]
    cells=cells.reshape((-1,4))
    mask = cells[:,2]>0.
    cells=cells[mask]

    condition_gen = condition[mask_gen]
    condition = condition[mask]

    print('*'*30)

    print(cells_gen.shape) # (313, 4) ; (235, 4)
    print(condition.shape) # (187, 2) ; (73, 2)
    print(condition_gen.shape) # (313, 2) ; (235, 2)

    path_gen = '/usr/workspace/sinha4/GSGM_new/GSGM_for_EIC_Calo/scripts/gen_data/cells_final_'+str(nvts)+'.txt'
    path_gen_2 = '/usr/workspace/sinha4/GSGM_new/GSGM_for_EIC_Calo/scripts/gen_data/condition_gen_final_'+str(nvts)+'.txt'

    write_list_to_file(path_gen, cells_gen)
    write_list_to_file(path_gen_2, condition_gen)

    plot(cells,cells_gen,
         condition,condition_gen,
         title='part',
         nplots=1,
         plot_folder=flags.plot_folder,
         is_big=flags.big)


    # phi = (0-360)
    # Theta = 0 -> Directly pointing to the center
    # Generated events : Donut shaped (10 - 30 degrees), width of the donut.

    # Select events in the range of (10,15) degrees. (x-y plane lot heat map)
    # x-y positions of the cells.
    # should look different from 20-25 degree values.

    # Compare the values from the generated data.

    # Total cluster energy , first model (condition to the 2nd model) -> learn the number of cells and absolute Energy.
    # x,y,z,E -> learns the energy scale.



# /usr/workspace/sinha4/GSGM_new/GSGM_for_EIC_Calo/scripts/plot_jet.py