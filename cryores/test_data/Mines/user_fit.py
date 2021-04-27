# -*- encoding: utf-8 -*-
"""
User code to fit S21 data for the Mines Al 6061 3D Cavity

Data collection information:
---------------------------

Collector: Nick Materise
VNA: Keysight PNA-X
Date collected: 210426, 210427
Purpose: Calibrate Qc with pin insertion depth

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #speadsheet commands
import sys #update paths
import os #import os in order to find relative path
import glob

from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d
pathToParent = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #set
# a variable that equals the relative path of parent directory
path_to_resfit = 'Z:/measurement/resfit'
sys.path.append(path_to_resfit)#path to Fit_Cavity

import fit_resonator.resonator as res
import fit_resonator.fit_S_data as fsd
np.set_printoptions(precision=4,suppress=True)# display numbers with 4 sig.
# figures (digits)


my_dir = 'Z:/measurement/cryores/test_data/Mines'

# Fit individual file results, further post processing required to analyze the
# Qc vs. z_pin data
## File for the pin inserted maximally into the cavity, with indium
## File for the pin inserted partially into the cavity, without indium
## File for the pin flush with the cavity wall, without indium
filename = 'mines_cavity_bare_in_seal_rt_no_header_210426.csv'
# filename = 'mines_cavity_bare_no_in_seal_rt_zpin1_no_header_210427.csv'
# filename = 'mines_cavity_bare_no_in_seal_rt_zpin0_no_header_210427.csv'

fname = my_dir + '/' + filename
print(f'glob.glob(fname): {glob.glob(fname)}')


#############################################
## create Method

fit_type = 'DCM'
MC_iteration = 10
MC_rounds = 1e3
MC_fix = ['w1']
# make your own initial guess: [Qi, Qc, freq, phi] 
# (instead of phi used Qa for CPZM)
# manual_init = [Qi,Qc,freq,phi]
manual_init = None # find initial guess by itself

try: 
    Method = res.FitMethod(fit_type, MC_iteration, MC_rounds=MC_rounds,
                MC_fix=MC_fix, manual_init=manual_init, MC_step_const=0.3)
except:
    print("Failed to initialize method, please change parameters")
    quit()

##############################################################

normalize = 10

### Fit Resonator function without background removal ###
params, conf_array, fig1, chi1, init1 = fsd.fit_resonator(filename=filename,
                                                          Method=Method,
                                                          normalize=normalize,
                                                          dir=my_dir)

### Fit Resonator function with background removal ###
#background_file = 'example_background.csv'
#params1,fig1,chi1,init1 = Fit_Resonator(filename = filename,Method = Method,normalize = normalize,dir = dir,background = background_file)
###############################################
