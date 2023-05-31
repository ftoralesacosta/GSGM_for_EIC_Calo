import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mtick


class PlottingConfig():
    def __init__(self,name,idx,is_big=False,one_class=False):
        self.name=name
        self.idx=idx
        self.big=is_big
        self.one_class=one_class
        self.binning = self.get_binning()
        self.logy=self.get_logy()
        self.var = self.get_name()
        self.max_y = self.get_y()


    def get_name(self):
        if self.name == 'cluster':
            name_translate = [
                r'Cluster $\sum E_\mathrm{cell}$ [GeV]',
                r'Cluster cell multiplicity',
            ]
        else:
            name_translate = [
                r'All cells $E$',
                r'All cells $X$',
                r'All cells $Y$',
                r'All cells $Z$',
            ]

        return name_translate[self.idx]
    
    def get_binning(self):
        if self.name == 'cluster':
            binning_dict = {
                0 : np.linspace(0.0,2.5,10),
                1 : np.linspace(0,200,200),
            }
            if self.big:
                binning_dict[1] = np.linspace(0,1000,1000)
        else:
            binning_dict = {
                0 : np.linspace(0.0,0.05,100),
                1 : np.linspace(-2700,2700,55),
                2 : np.linspace(-2700,2700,55),
                3 : np.linspace(3800,5000,54),                
            }
            
        return binning_dict[self.idx]

    def get_logy(self):
        if self.name == 'cluster':
            binning_dict = {
                0 : True,
                1 : False,
            }

        else:
            binning_dict = {
                0 : False,
                1 : False,
                2 : False,
                3 : False,
            }
            
        return binning_dict[self.idx]



    def get_y(self):
        if self.name == 'cluster':
            binning_dict = {
                0 : 0.1,
                1 : 0.7,
            }
            if self.big:
                binning_dict[3] = 0.5
            if self.one_class:
                binning_dict[2] = 0.03

        else:
            binning_dict = {
                0 : 12,
                1 : 12,
                2 : 120,
            }
            
        return binning_dict[self.idx]
