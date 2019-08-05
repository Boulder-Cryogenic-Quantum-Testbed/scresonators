# -*- coding: utf-8 -*-
"""
Created on Mon May 14 19:32:43 2018

@author: hung93
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #speadsheet commands

from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d
from fit_resonator import *
np.set_printoptions(precision=4,suppress=True)# display numbers with 4 sig. figures (digits)

                         ## Code Starts Here ##

dir = "path to folder with your data here" #path to directory with data, make sure to use \\ instead of just /
filename = 'example.csv'
filepath = dir+'\\'+filename

#############################################
## create Method

fit_type = 'DCM'
MC_iteration = 10
MC_rounds = 1e3
MC_fix = ['w1']
#manual_init = [Qi,Qc,freq,phi]        #make your own initial guess: [Qi, Qc, freq, phi] (instead of phi used Qa for CPZM)
manual_init = None # find initial guess by itself

try:
    Method = Fit_Method(fit_type, MC_iteration, MC_rounds=MC_rounds,\
                 MC_fix=MC_fix, manual_init=manual_init, MC_step_const=0.3) #mcrounds = 100,000 unless otherwise specified
except:
    print("Failed to initialize method, please change parameters")
    quit()

##############################################################

normalize = 10

# Fit Resonator function without background removal
params1,fig1,chi1,init1 = Fit_Resonator(filename,filepath,Method,normalize)

# Fit Resonator function with background removal
#path_to_background = dir+'\\'+'example_background.csv'
#params1,fig1,chi1,init1 = Fit_Resonator(filename,filepath,Method,normalize,path_to_background)

fig1.savefig(dir+'\\'+filename+'_'+fit_type+'_fit.png')
###############################################
