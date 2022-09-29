import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #speadsheet commands
import sys #update paths
import os #import os in order to find relative path

from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d
pathToParent = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #set a variable that equals the relative path of parent directory
sys.path.append(pathToParent)#path to Fit_Cavity

import fit_resonator.resonator as scres

                         ## Code Starts Here ##
dir = '/Users/Lehnert Lab/GitHub/measurement/temperature_control/powersweep_220617_11_56_17'
#make sure to use / instead of \
filename = 'M3D6_03_BARE_220617_7_-15dB_300000000_mKpcsv.csv'
# The object all following code will be called from
my_resonator = scres.Resonator()

# Load the raw data
url = 'https://raw.githubusercontent.com/Boulder-Cryogenic-Quantum-Testbed/scresonators/master/cryores/test_data/AWR/AWR_Data.csv'
raw = np.loadtxt(url, delimiter=',')
# Can also use our file input system of my_resonator.from_file(url)

# Test with file load into class
my_resonator.from_columns(raw)

# Assign your desired fit method variables
fit_type = 'DCM'
MC_iteration = 10
MC_rounds = 1e3
MC_fix = ['w1']
MC_fix = []
#manual_init = [Qi,Qc,freq,phi]        #make your own initial guess: [Qi, Qc, freq, phi] (instead of phi used Qa for CPZM)
manual_init = None # find initial guess by itself

# Pass these to your resonator object
my_resonator.fit_method(fit_type, MC_iteration, MC_rounds=MC_rounds, MC_fix=MC_fix, manual_init=manual_init,
                        MC_step_const=0.3)

# Fit!
my_resonator.fit()
### Fit Resonator function with background removal ###
#background_file = 'path/to/example_background.csv'
#my_resonator.init_background(background_file, (optional)fscale)

