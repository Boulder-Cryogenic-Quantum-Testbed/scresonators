import sys
sys.path.append(r'C:\Users\Lehnert Lab\Documents\GitHub\measurement\pna_control')
import pna_control as PNA
import numpy as np
import os

# VNA parameters
AVERAGES = 3 # Number of averages for first (highest) power
EDELAY = 73.05 #ns
IFBAND = 5 #kHz
CENTERF = [5.931,6.302,6.744,7.272,7.923] #GHz
SPAN = 1 #MHz
POINTS = 501
POWER = -30

TEMP = 12 #mK
SAMPLEID = 'A01_01' #project ID followed by sample number and die number

# Make directory
output_path = PNA.timestamp_folder(os.getcwd(),'ressearch')
os.mkdir(output_path)

for i in CENTERF:
    OUTPUTFILE = output_path + '/' + SAMPLEID+'_ressearch'+'_'+str(i)+'GHz'
    PNA.get_data(i, SPAN, TEMP, AVERAGES, POWER, EDELAY, IFBAND, POINTS, OUTPUTFILE)
    
