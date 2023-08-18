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
import fit_resonator.fit as fit
np.set_printoptions(precision=4,suppress=True)# display numbers with 4 sig. figures (digits)

                         ## Code Starts Here ##

dir = 'F:/Paper Data/Noisy Sims/linear and BG noise sims/' #make sure to use / instead of \
filename = 'R~10^6.txt'

#############################################
## create Method

fit_type = 'DCM'
MC_iteration = 10
MC_rounds = 1e3
#MC_fix = ['Q','Qi','Qc','Qa','phi','w1']
MC_fix = []
manual_init = [10000,100000,5.797,-0.558561]        #make your own initial guess: [Qi, Qc, freq, phi] (instead of phi used Qa for CPZM)
#manual_init = None # find initial guess by itself

#try:
Method = res.FitMethod(fit_type, MC_iteration, MC_rounds=MC_rounds,\
            MC_fix=MC_fix, manual_init=manual_init, MC_step_const=0.3) #mcrounds = 100,000 unless otherwise specified
#except:
#    print("Failed to initialize method, please change parameters")
#    quit()

##############################################################

normalize = 10

### Fit Resonator function without background removal ###
params,conf_array,fig1,chi1,init1 = fit.fit(filename = filename, Method = Method, normalize = normalize, dir = dir, preprocess_method ="circle")

### Fit Resonator function with background removal ###
#background_file = 'example_background.csv'
#params1,fig1,chi1,init1 = Fit_Resonator(filename = filename,Method = Method,normalize = normalize,dir = dir,background = background_file)
###############################################
