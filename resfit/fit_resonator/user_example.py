import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #speadsheet commands
import sys #update paths
import os #import os in order to find relative path

from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d
pathToParent = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #set a variable that equals the relative path of parent directory
sys.path.append(pathToParent)#path to Fit_Cavity
import fit_resonator.resonator as res
import fit_resonator.fit_S_data as fsd
np.set_printoptions(precision=4,suppress=True)# display numbers with 4 sig. figures (digits)

                         ## Code Starts Here ##

dir = '/Users/coreyraemcrae/Documents/GitHub/data/epi-GaAs_LEres/GaAs_Die1/BlueFors/4p792GHz/2019_08_22_11_36_20_highpowersweep/' #make sure to use / instead of \
filename = 'VNA_3_formatted.csv'

#############################################
## create Method

fit_type = 'DCM'
MC_iteration = 10
MC_rounds = 1e3
MC_fix = ['w1']
#manual_init = [Qi,Qc,freq,phi]        #make your own initial guess: [Qi, Qc, freq, phi] (instead of phi used Qa for CPZM)
manual_init = None # find initial guess by itself

try:
    Method = res.FitMethod(fit_type, MC_iteration, MC_rounds=MC_rounds,\
                MC_fix=MC_fix, manual_init=manual_init, MC_step_const=0.3) #mcrounds = 100,000 unless otherwise specified
except:
    print("Failed to initialize method, please change parameters")
    quit()

##############################################################

normalize = 10

### Fit Resonator function without background removal ###
params,conf_array,fig1,chi1,init1 = fsd.fit_resonator(filename = filename,Method = Method,normalize = normalize,dir = dir)

### Fit Resonator function with background removal ###
#background_file = 'example_background.csv'
#params1,fig1,chi1,init1 = Fit_Resonator(filename = filename,Method = Method,normalize = normalize,dir = dir,background = background_file)
###############################################
