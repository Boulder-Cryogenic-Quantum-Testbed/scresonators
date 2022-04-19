import sys
sys.path.append(r'C:\Users\Lehnert Lab\Documents\GitHub\measurement\pna_control')
import pna_control as PNA
import os

AVERAGES = 1
POWER = -30 #dB
EDELAY = 73.05 #ns
IFBAND = 5 #kHz
CENTERF = [4.5,5.5,6.5,7.5] #GHz
SPAN = 1000 #MHz
POINTS = 32001

TEMP = 12 #mK
SAMPLEID = 'A01_01' #project ID followed by sample number and die number

# Make directory
output_path = PNA.timestamp_folder(os.getcwd(),'widescan')
os.mkdir(output_path)

for i in CENTERF:
    OUTPUTFILE = output_path + '/' + SAMPLEID+'_widescan'+'_'+str(i)+'GHz'
    PNA.get_data(i, SPAN, TEMP, AVERAGES, POWER, EDELAY, IFBAND, POINTS, OUTPUTFILE)
    
