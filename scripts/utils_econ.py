import json, yaml
import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mtick
from sklearn.utils import shuffle
import tensorflow as tf
import tensorflow.keras.utils as utils
#from keras.utils.np_utils import to_categorical
#import energyflow as ef

np.random.seed(0) #fix the seed to keep track of validation split

line_style = {
    'true':'dotted',
    'gen':'-',
    'Geant':'dotted',
    'GSGM':'-',
    'P_truth':'-',
    'P_gen':'dotted',
    'Theta_truth':'-',
    'Theta_gen':'dotted',
}

colors = {
    'true':'black',
    'gen':'#7570b3',
    'Geant':'black',
    'GSGM':'#7570b3',

    'P_truth':'#7570b3',
    'P_gen':'#7570b3',
    'Theta_truth':'#d95f02',
    'Theta_gen':'#d95f02',
}

name_translate={
    'true':'True distribution',
    'gen':'Generated distribution',
    'Geant':'Geant 4',
    'GSGM':'Graph Diffusion',

    'P_truth':'Sim.: P',
    'P_gen':'FPCD: P',
    'Theta_truth':'Sim.: Theta',
    'Theta_gen':'FPCD: Theta',
    }

names = ['P','Theta']

labels200 = {
    'log10_Uniform_03-23.hdf5':0, # log10_pi+_100_10k_10-30deg_1.hdf5 for new data
    }

labels1000 = {
    'log10_Uniform_03-23.hdf5':0, # log10_pi+_100_10k_10-30deg_1.hdf5
}

nevts = 100

def SetStyle():
    from matplotlib import rc
    rc('text', usetex=True)

    import matplotlib as mpl
    rc('font', family='serif')
    rc('font', size=22)
    rc('xtick', labelsize=15)
    rc('ytick', labelsize=15)
    rc('legend', fontsize=15)

    # #
    mpl.rcParams.update({'font.size': 19})
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams.update({'xtick.labelsize': 18}) 
    mpl.rcParams.update({'ytick.labelsize': 18}) 
    mpl.rcParams.update({'axes.labelsize': 18}) 
    mpl.rcParams.update({'legend.frameon': False}) 
    mpl.rcParams.update({'lines.linewidth': 2})
    
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.style.use("CMS") 


def SetGrid(ratio=True):
    fig = plt.figure(figsize=(9, 9))
    if ratio:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1]) 
        gs.update(wspace=0.025, hspace=0.1)
    else:
        gs = gridspec.GridSpec(1, 1)
    return fig,gs

        
def PlotRoutine(feed_dict,xlabel='',ylabel='',reference_name='gen'):
    assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"
    
    fig,gs = SetGrid() 
    ax0 = plt.subplot(gs[0])
    plt.xticks(fontsize=0)
    ax1 = plt.subplot(gs[1],sharex=ax0)

    for ip,plot in enumerate(feed_dict.keys()):
        if 'steps' in plot or 'r=' in plot:
            ax0.plot(np.mean(feed_dict[plot],0),label=plot,marker=line_style[plot],color=colors[plot],lw=0)
        else:
            ax0.plot(np.mean(feed_dict[plot],0),label=plot,linestyle=line_style[plot],color=colors[plot])
        if reference_name!=plot:
            ratio = 100*np.divide(np.mean(feed_dict[reference_name],0)-np.mean(feed_dict[plot],0),np.mean(feed_dict[reference_name],0))
            #ax1.plot(ratio,color=colors[plot],marker='o',ms=10,lw=0,markerfacecolor='none',markeredgewidth=3)
            if 'steps' in plot or 'r=' in plot:
                ax1.plot(ratio,color=colors[plot],markeredgewidth=1,marker=line_style[plot],lw=0)
            else:
                ax1.plot(ratio,color=colors[plot],linewidth=2,linestyle=line_style[plot])


    FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0)
    ax0.legend(loc='best',fontsize=16,ncol=1)

    plt.ylabel('Difference. (%)')
    plt.xlabel(xlabel)
    plt.axhline(y=0.0, color='r', linestyle='--',linewidth=1)
    plt.axhline(y=10, color='r', linestyle='--',linewidth=1)
    plt.axhline(y=-10, color='r', linestyle='--',linewidth=1)
    plt.ylim([-100,100])

    return fig,ax0

class ScalarFormatterClass(mtick.ScalarFormatter):
    #https://www.tutorialspoint.com/show-decimal-places-and-scientific-notation-on-the-axis-of-a-matplotlib-plot
    def _set_format(self):
        self.format = "%1.1f"


def FormatFig(xlabel,ylabel,ax0):
    #Limit number of digits in ticks
    # y_loc, _ = plt.yticks()
    # y_update = ['%.1f' % y for y in y_loc]
    # plt.yticks(y_loc, y_update) 
    ax0.set_xlabel(xlabel,fontsize=20)
    ax0.set_ylabel(ylabel)


    # xposition = 0.9
    # yposition=1.03
    # text = 'H1'
    # WriteText(xposition,yposition,text,ax0)


def WriteText(xpos,ypos,text,ax0):

    plt.text(xpos, ypos,text,
             horizontalalignment='center',
             verticalalignment='center',
             transform = ax0.transAxes, fontsize=25, fontweight='bold')


def HistRoutine(feed_dict,
                xlabel='',ylabel='',
                reference_name='Geant',
                logy=False,binning=None,
                fig = None, gs = None,
                plot_ratio= True,
                idx = None,
                label_loc='best'):

    assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"

    if fig is None:
        fig,gs = SetGrid(plot_ratio) 
    ax0 = plt.subplot(gs[0])
    if plot_ratio:
        plt.xticks(fontsize=0)
        ax1 = plt.subplot(gs[1],sharex=ax0)


    if binning is None:
        binning = np.linspace(np.quantile(feed_dict[reference_name],0.0),np.quantile(feed_dict[reference_name],1),5)

    xaxis = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]
    reference_hist,_ = np.histogram(feed_dict[reference_name],bins=binning,density=True)
    maxy = np.max(reference_hist) 
    print(reference_hist) # [2.16 0.72 0.72 0.   0.   0.   0.   0.   0.  ]
    print(maxy) # 2.1599999999999997

    for ip,plot in enumerate(feed_dict.keys()):

        # print("Plot",ip,": Shape of feed_dict = ",np.shape(feed_dict[plot]))
        # print("Feed Dict Keys = ",feed_dict.keys())
        # print("Name translate keys = ",name_translate.keys())
        # print("Line_style keys = ",line_style.keys())
        # print("Colors keys = ",colors.keys())

        dist,_,_=ax0.hist(feed_dict[plot], bins=binning, 
                          label=name_translate[plot],
                          linestyle=line_style[plot],
                          color=colors[plot], 
                          density=True,
                          histtype="step")

        if plot_ratio:
            if reference_name!=plot:
                ratio = 100*np.divide(reference_hist-dist,reference_hist) # mark.
                ax1.plot(xaxis,ratio,color=colors[plot],marker='o',ms=10,lw=0,markerfacecolor='none',markeredgewidth=3)

    ax0.legend(loc=label_loc,fontsize=12,ncol=5)

    if logy:
        ax0.set_yscale('log')

    if plot_ratio:
        FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0)
        plt.ylabel('Difference. (%)')
        plt.xlabel(xlabel)
        plt.axhline(y=0.0, color='r', linestyle='-',linewidth=1)
        # plt.axhline(y=10, color='r', linestyle='--',linewidth=1)
        # plt.axhline(y=-10, color='r', linestyle='--',linewidth=1)
        plt.ylim([-100,100])
    else:
        FormatFig(xlabel = xlabel, ylabel = ylabel,ax0=ax0)

    return fig,gs, binning


def LoadJson(file_name):
    import json,yaml
    JSONPATH = os.path.join(file_name)
    return yaml.safe_load(open(JSONPATH))

def SaveJson(save_file,data):
    with open(save_file,'w') as f:
        json.dump(data, f)


def revert_npart(npart,max_npart):

    #Revert the preprocessing to recover the cell multiplicity
    alpha = 1e-6
    data_dict = LoadJson('preprocessing_{}.json'.format(max_npart))
    x = npart*data_dict['std_cluster'][-1] + data_dict['mean_cluster'][-1]
    x = revert_logit(x)
    x = x * (data_dict['max_cluster'][-1]-data_dict['min_cluster'][-1]) + data_dict['min_cluster'][-1]
    #x = np.exp(x)
    return np.round(x).astype(np.int32)

def revert_logit(x):
    alpha = 1e-6
    exp = np.exp(x)
    x = exp/(1+exp)
    return (x-alpha)/(1 - 2*alpha)                

def ReversePrep(cells,clusters,npart):

    alpha = 1e-6
    data_dict = LoadJson('preprocessing_{}.json'.format(npart))
    num_part = cells.shape[1]    
    cells=cells.reshape(-1,cells.shape[-1])
    mask=np.expand_dims(cells[:,3]!=0,-1) #for 4D cell, this is Z

    def _revert(x,name='cluster'):  

        x = x*data_dict['std_{}'.format(name)] + data_dict['mean_{}'.format(name)]
        x = revert_logit(x)
        x = x * (np.array(data_dict['max_{}'.format(name)]) -data_dict['min_{}'.format(name)]) + data_dict['min_{}'.format(name)]
        return x
    
    cells = _revert(cells,'cell')
    cells = (cells*mask).reshape(clusters.shape[0],num_part,-1)
    clusters = _revert(clusters,'cluster')
    clusters[:,-1] = np.round(clusters[:,-1]) #num cells

    return cells,clusters

# only for plot_jet, with flag.test
def SimpleLoader(data_path,
                 labels,
                 ncluster_var = 2,
                 num_condition = 2):

    cells = []
    clusters = []
    cond = []

    for label in labels:
        #if 'w' in label or 'z' in label: continue #no evaluation for w and z
        with h5.File(os.path.join(data_path,label),"r") as h5f:
            # ntotal = int(nevts)
            cell = h5f['hcal_cells'][int(0.7*ntotal):].astype(np.float32)
            cluster = h5f['cluster'][int(0.7*ntotal):].astype(np.float32)
            cluster = np.concatenate([cluster,labels[label]*np.ones(shape=(cluster.shape[0],1),dtype=np.float32)],-1)

            cells.append(cell)
            clusters.append(cluster)

    cells = np.concatenate(cells)
    clusters = np.concatenate(clusters)

    #Split Conditioned Features and Cluster Training Features
    cond = clusters[:,:num_condition] # GenP, GenTheta 
    clusters = clusters[:,ncluster_var:] # ClusterSum, N_Hits
    cells,clusters = shuffle(cells,clusters, random_state=0)
    mask = np.expand_dims(cells[:nevts,:,-1],-1)

    # Divide the particles of each event into 55 cells by iterating over all the events.
    clusters_new = []
    for i in range(clusters.shape[0]): # iterate over the events.

        part_event = cells[i]
        mean_a = np.mean(part_event, axis=0) # [-7.643371    0.05157143  0.06052196  0.47040254  0.095     ]
        min_a = np.min(part_event, axis=0) # [-8.3204718e+00 -5.6295991e-01 -2.0788333e-03 -5.3362942e+00 0.0000000e+00]
        max_a = np.max(part_event, axis=0) # [-0.1320551   0.11035314  0.6781157   0.5742491   1.        ]

        interval_array = np.linspace(min_a[2], max_a[2], 55) # make 55 intervals between start and end of z.

        array_data = part_event # [200,5]
        z_values = array_data[:, 2]
        sums_of_E = np.zeros(interval_array.shape[0]) # 55 dim  

        # Iterate through each interval and calculate the sum of E values within the interval
        for i in range(interval_array.shape[0]):
            lower_bound = interval_array[i]
            upper_bound = interval_array[i + 1] if i < interval_array.shape[0] - 1 else 1.0
            mask = np.logical_and(z_values >= lower_bound, z_values < upper_bound)
            sums_of_E[i] = np.sum(array_data[mask, 3])

        sum_array = (np.sum(array_data,axis=0)) # sanity check to see whether all intervals add to sums_of_E or not
        sums_of_E = np.array(sums_of_E, dtype="float32")
        sums_of_E = sums_of_E.reshape(-1,1)

        #if (sum_array !=)

        cls_i = np.array(clusters[i],dtype="float32") 
        cls_i = cls_i.reshape(-1,1) # (55,1)

        a = np.concatenate((cls_i,sums_of_E)) # append E of 55 intervals to the 2 dim clusters

        clusters_new.append(a.reshape(a.shape[0],))

    return cells[:nevts,:,:-1]*mask,clusters[:nevts],cond[:nevts]


def DataLoader(data_path,labels,
               npart,
               rank=0,size=1,
               ncluster_var=2,
               num_condition=2,#genP,genTheta
               batch_size=64,make_tf_data=True):
    cells = []
    clusters = []

    def _preprocessing(cells,clusters,save_json=False):
        num_part = cells.shape[1]

        print('clusters1',clusters.shape)
        print('cells',cells.shape) # cells (69930, 200, 5)
        print('clusters',clusters.shape) # clusters (69930, 2)
        print('cond',cond.shape) # clusters (69930, 2)

        print('mean', np.mean(clusters, axis=0)) # mean [ 1.0650898 17.038385 ]
        print('min', np.min(clusters, axis=0)) # min [9.4318224e-05 1.5567589e+01]
        print('max', np.max(clusters, axis=0)) # ax [ 2.1198688 18.432405 ]

        # If clusters is of 2 dim then only make it 57 dim.
        # While testing, if the save_dic is already having cluster of 57 dim, then don't enter.
        if(clusters.shape[1]==2):

            print('cluster_s cdim hanging to 57 dim')
            clusters_new = []

            for i in range(clusters.shape[0]): # for each event

                part_event = cells[i]
                mean_a = np.mean(part_event, axis=0) # [-7.643371    0.05157143  0.06052196  0.47040254  0.095     ]
                min_a = np.min(part_event, axis=0) # [-8.3204718e+00 -5.6295991e-01 -2.0788333e-03 -5.3362942e+00 0.0000000e+00]
                max_a = np.max(part_event, axis=0) # [-0.1320551   0.11035314  0.6781157   0.5742491   1.        ]

                interval_array = np.linspace(min_a[2], max_a[2], 55)

                array_data = part_event # (200,5)
                z_values = array_data[:, 2]
                sums_of_E = np.zeros(interval_array.shape[0])

                # Iterate through each interval and calculate the sum of E values within the interval
                for i in range(interval_array.shape[0]): # 55 intervals
                    lower_bound = interval_array[i]
                    upper_bound = interval_array[i + 1] if i < interval_array.shape[0] - 1 else 1.0
                    mask = np.logical_and(z_values >= lower_bound, z_values < upper_bound)
                    sums_of_E[i] = np.sum(array_data[mask, 3]) # i in 55.

                sum_array = (np.sum(array_data,axis=0)) # sum_array from the data (particle) array. (mean across each 200 particles)
                sums_of_E = np.array(sums_of_E, dtype="float32") # sums_of_E
                sum_array_2 = (np.sum(sums_of_E,axis=0)) 
                
                #print('sum_array',sum_array)
                #print('sums_of_E_2',sum_array_2) # This should be equal to sum_array[3]
                #print('clusters[i]',clusters[i])

                sums_of_E = sums_of_E.reshape(-1,1)

                cls_i = np.array(clusters[i],dtype="float32")
                cls_i = cls_i.reshape(-1,1)

                a = np.concatenate((cls_i,sums_of_E))

                clusters_new.append(a.reshape(a.shape[0],))

            #print(np.mean(cells[1]))
            print(len(clusters_new))
            print(clusters_new[2].shape)
            print('clusters_prev',clusters.shape)

            clusters = np.array(clusters_new)
            print('clusters_aft',clusters.shape)  

        cells=cells.reshape(-1,cells.shape[-1]) #flattens D0 and D1

        def _logit(x):                            
            alpha = 1e-6
            x = alpha + (1 - 2*alpha)*x
            return np.ma.log(x/(1-x)).filled(0)

        #Transformations

        if save_json:
            mask = cells[:,-1] == 1 #saves array of BOOLS instead of ints
            print(f"L 357: Masked {np.shape(cells[mask])[0]} / {len(mask)} cells")

            data_dict = {
                'max_cluster':np.max(clusters[:,:],0).tolist(),
                'min_cluster':np.min(clusters[:,:],0).tolist(),

                # With Mask
                'max_cell':np.max(cells[mask][:,:-1],0).tolist(), #-1 avoids mask
                'min_cell':np.min(cells[mask][:,:-1],0).tolist(),
            }                
            
            SaveJson('preprocessing_{}.json'.format(npart),data_dict)
        else:
            data_dict = LoadJson('preprocessing_{}.json'.format(npart))


        #normalize
        print('data_dict',data_dict)
        print(clusters.shape)

        clusters[:,:] = np.ma.divide(clusters[:,:]-data_dict['min_cluster'],np.array(data_dict['max_cluster'])- data_dict['min_cluster']).filled(0)        
        cells[:,:-1]= np.ma.divide(cells[:,:-1]-data_dict['min_cell'],np.array(data_dict['max_cell'])- data_dict['min_cell']).filled(0)

        print('mean', np.mean(clusters, axis=0)) # mean [ 1.0650898 17.038385 ]
        print('min', np.min(clusters, axis=0)) # min [9.4318224e-05 1.5567589e+01]
        print('max', np.max(clusters, axis=0)) # ax [ 2.1198688 18.432405 ]

        # make gaus-like. 
        clusters = _logit(clusters)
        cells[:,:-1] = _logit(cells[:,:-1])
        print('clusters2',clusters.shape)

        # save the mean and std of celss and clusters (55 dim)
        if save_json:

            print('save_json',save_json)
            print('clusters2',clusters.shape)

            mask = cells[:,-1]
            mean_cell = np.average(cells[:,:-1],axis=0,weights=mask)
            data_dict['mean_cluster']=np.mean(clusters,0).tolist()
            data_dict['std_cluster']=np.std(clusters,0).tolist()
            data_dict['mean_cell']=mean_cell.tolist()
            data_dict['std_cell']=np.sqrt(np.average((cells[:,:-1] - mean_cell)**2,axis=0,weights=mask)).tolist()                        

            SaveJson('preprocessing_{}.json'.format(npart),data_dict)


        clusters = np.ma.divide(clusters-data_dict['mean_cluster'],data_dict['std_cluster']).filled(0)
        cells[:,:-1]= np.ma.divide(cells[:,:-1]-data_dict['mean_cell'],data_dict['std_cell']).filled(0)

        cells = cells.reshape(clusters.shape[0],num_part,-1)

        print(f"\nL 380: Shape of Cells in DataLoader = {np.shape(cells)}") # Shape of Cells in DataLoader = (69930, 200, 5)
        print(f"\nL 381: Cells in DataLoader = \n{cells[0,15:20,:]}")

        print('clusters_final',clusters.shape)

        print('mean', np.mean(clusters, axis=0)) # mean [ 1.0650898 17.038385 ]
        print('min', np.min(clusters, axis=0)) # min [9.4318224e-05 1.5567589e+01]
        print('max', np.max(clusters, axis=0)) # ax [ 2.1198688 18.432405 ]

        return cells.astype(np.float32),clusters.astype(np.float32)


    # start getting the data

    for label in labels:

        with h5.File(os.path.join(data_path,label),"r") as h5f:
            ntotal = h5f['cluster'][:].shape[0]
            # ntotal = int(nevts)

            if make_tf_data:
                cell=h5f['hcal_cells'][rank:int(0.7*ntotal):size].astype(np.float32)
                cluster=h5f['cluster'][rank:int(0.7*ntotal):size].astype(np.float32)

            else:
                #load evaluation data
                cell = h5f['hcal_cells'][int(0.7*ntotal):].astype(np.float32)
                cluster = h5f['cluster'][int(0.7*ntotal):].astype(np.float32)

            cells.append(cell)
            clusters.append(cluster)

    cells = np.concatenate(cells)
    clusters = np.concatenate(clusters)

    #Split Cluster Data into Input and Condition
    cond = clusters[:,:num_condition]#GenP, GenTheta 
    clusters = clusters[:,ncluster_var:] #ClusterSum, N_Hits

    #Additional Pre-Processing, Log10 of E
    cells[:,:,0] = np.log10(cells[:,:,0]) #Log10(CellE)
    cond[:,0] = np.log10(cond[:,0]) #Log10 of GenP #Make sure maks is applied in _preprocess
    # clusters = np.log10(clusters[:,0]) # ClusterSumE, after cond split

    # Shuffel and pre-process the data.
    cells,clusters,cond = shuffle(cells, clusters, cond, random_state=0)
    cells,clusters = _preprocessing(cells, clusters, save_json=True) 
    # cells,clusters = _preprocessing(cells,clusters,save_json=False) 

    clusters_new = []

    print(clusters.shape) # (69930, 57)

    # This is not required while training, only required while generating.
    if(clusters.shape[1]==2):

        for i in range(clusters.shape[0]):

            part_event = cells[i]
            mean_a = np.mean(part_event, axis=0) # [-7.643371    0.05157143  0.06052196  0.47040254  0.095     ]
            min_a = np.min(part_event, axis=0) # [-8.3204718e+00 -5.6295991e-01 -2.0788333e-03 -5.3362942e+00 0.0000000e+00]
            max_a = np.max(part_event, axis=0) # [-0.1320551   0.11035314  0.6781157   0.5742491   1.        ]

            interval_array = np.linspace(min_a[2], max_a[2], 55)

            array_data = part_event
            z_values = array_data[:, 2]
            sums_of_E = np.zeros(interval_array.shape[0])

            # Iterate through each interval and calculate the sum of E values
            for i in range(interval_array.shape[0]):
                lower_bound = interval_array[i]
                upper_bound = interval_array[i + 1] if i < interval_array.shape[0] - 1 else 1.0
                mask = np.logical_and(z_values >= lower_bound, z_values < upper_bound)
                sums_of_E[i] = np.sum(array_data[mask, 3])

            sum_array = (np.sum(array_data,axis=0)) # sum_array from the mean array. (mean across each 200 particles)
            sums_of_E = np.array(sums_of_E, dtype="float32")
            sums_of_E = sums_of_E.reshape(-1,1)

            cls_i = np.array(clusters[i],dtype="float32")
            cls_i = cls_i.reshape(-1,1)

            a = np.concatenate((cls_i,sums_of_E))

            clusters_new.append(a.reshape(a.shape[0],))

        clusters = np.array(clusters_new)

    # Do Train/Test Split, or just return data
    data_size = clusters.shape[0]

    if make_tf_data:
 
        train_cells = cells[:int(0.8*data_size)] #This is 80% train (whcih 70% of total)
        train_clusters = clusters[:int(0.8*data_size)]
        train_cond = cond[:int(0.8*data_size)]
        
        test_cells = cells[int(0.8*data_size):]
        test_clusters = clusters[int(0.8*data_size):]
        test_cond = cond[int(0.8*data_size):]
        
        def _prepare_batches(cells,clusters,cond):
            
            nevts = clusters.shape[0]
            tf_cluster = tf.data.Dataset.from_tensor_slices(clusters)

            tf_cond = tf.data.Dataset.from_tensor_slices(cond)
            mask = np.expand_dims(cells[:,:,-1],-1)

            masked = cells[:,:,:-1]*mask
            masked[masked[:,:,:] == -0.0] = 0.0

            # Really good check on mask and data before training
            print(f" First Cells in _prepare_batches = \n",masked[10,:10,:])
            print(f"Last Cells in _prepare_batches = \n",masked[10,-10:,:])

            tf_part = tf.data.Dataset.from_tensor_slices(masked)
            tf_mask = tf.data.Dataset.from_tensor_slices(mask)
            tf_zip = tf.data.Dataset.zip((tf_part, tf_cluster,tf_cond,tf_mask))

            return tf_zip.shuffle(nevts).repeat().batch(batch_size)
    
        train_data = _prepare_batches(train_cells,train_clusters,train_cond)
        test_data  = _prepare_batches(test_cells,test_clusters,test_cond)    
        return data_size, train_data, test_data
    
    else:

        mask = np.expand_dims(cells[:nevts,:,-1],-1)
        return cells[:nevts,:,:-1]*mask,clusters[:nevts], cond[:nevts]
