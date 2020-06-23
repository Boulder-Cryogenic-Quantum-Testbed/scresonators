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

import fit_resonator.fit_TLS_model as tls

np.set_printoptions(precision=4,suppress=True)# display numbers with 4 sig. figures (digits)

                         ## Code Starts Here ##

dir = '/Users/coreyraemcrae/Documents/GitHub/data/epi-GaAs_LEres/GaAs_Die1/Janis_GoldDR/E06_1_7p308GHz_powersweep/E06_1_7p308GHz_HPsweep_13.6mK/' #make sure to use / instead of \
filename = 'power_sweep_params.csv'

initguess = [0.5E5,1E-4,10,0.5] #[Q_hp, Falpha, N_c, beta]

######### Code testing ##########

power = [-31, -33, -35, -37, -39, -41]
lineattenuation = -70
f0 = 7.308E9

filepath = dir+'/'+filename
data = tls.PowerSweepParams.from_csv(filepath)

Qi = data.Qi
Qc = data.Qc

photons = tls.estimate_photons(power, lineattenuation, Qi, Qc, f0)
#print(photons)

#############################################

### Fit TLS model ###
#params,conf_array,fig1,chi1,init1 = tls.fit_powersweep(filename = filename,dir = dir)

###############################################
