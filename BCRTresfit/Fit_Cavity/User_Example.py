# -*- coding: utf-8 -*-
"""
Created on Mon May 14 19:32:43 2018

@author: hung93
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #speadsheet commands
import sys #update paths
import os #import os in order to find relative path

from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d
pathToParent = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #set a variable that equals the relative path of parent directory
sys.path.append(pathToParent)#path to Fit_Cavity
from Fit_Cavity import *
np.set_printoptions(precision=4,suppress=True)# display numbers with 4 sig. figures (digits)

                         ## Code Starts Here ##

dir = "path to folder with your data here" #path to directory with data, make sure to use \\ instead of just \
filename = 'example.txt'
filepath = dir+'\\'+filename

#############################################
## create Method

MC_rounds = 1e3
manual_init = [118.33,165.97,5.3801,0.0000001]#make your own initial guess: [Qi, Qc, freq, phi]
manual_init = None # find initial guess by itself

MC_fix = ['w1']
fit_type = 'DCM'
#try:
Method = Fit_Method(fit_type,1,MC_rounds = MC_rounds,\
             MC_fix = MC_fix,\
             manual_init=manual_init,MC_step_const= 0.3) #mcrounds = 100,000 unless otherwise specified
#except:
#    print("Failed to initialize method, please change parameters")
#    quit()

##############################################################

normalize = 10
params1,fig1,chi1,init1 = Fit_Resonator(filename,filepath,Method,normalize) # Fit Resonator A with method, get fit result=params, fig1 is figure, chi is chi^2, init=initial_guess

fig1.savefig(dir+'\\'+filename+'_'+fit_type+'_fit.png')
###############################################
