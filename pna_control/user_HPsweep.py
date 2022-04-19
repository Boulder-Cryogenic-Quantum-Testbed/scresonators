import sys
sys.path.append(r'C:\Users\Lehnert Lab\Documents\GitHub\measurement\pna_control')
import pna_control as PNA
import numpy as np
import os

# VNA parameters
AVERAGES = 3 # Number of averages for first (highest) power
EDELAY = 73.05 #ns
IFBAND = 1 #kHz
CENTERF = [5.9306,6.3016,6.7436,7.2724,7.9227] #GHz
SPAN = [1,1,1,1,1] #MHz
POINTS = 501

TEMP = 12 #mK
SAMPLEID = 'A01_01' #project ID followed by sample number and die number

STARTPOWER = 0
ENDPOWER = -30
NUMSWEEPS = 7

for i in np.arange(len(CENTERF)):
    OUTPUTFILE = SAMPLEID+'_'+str(CENTERF[i])+'GHz_'+'HPsweep'
    PNA.power_sweep(STARTPOWER, ENDPOWER, NUMSWEEPS, CENTERF[i], SPAN[i], TEMP, AVERAGES, EDELAY, IFBAND, POINTS, OUTPUTFILE,str(CENTERF[i])+'GHz_HPsweep')
    
