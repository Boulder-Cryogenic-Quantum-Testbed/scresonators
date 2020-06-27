import numpy as np
import pyvisa
import os
from os import path

def pna_setup(pna, points: int, centerf: float, span: float, ifband: float, power: float, edelay: float, averages: int):
    '''
    set parameters for the PNA for the sweep (number of points, center frequency, span of frequencies, IF bandwidth, power, electrical delay and number of averages)
    '''

    #initial setup for measurement
    if (pna.query('CALC:PAR:CAT:EXT?') != '"Meas,S21"\n'):
        pna.write('CALCulate1:PARameter:DEFine:EXT \'Meas\',S21')
        pna.write('DISPlay:WINDow1:STATE ON')
        pna.write('DISPlay:WINDow1:TRACe1:FEED \'Meas\'')
        pna.write('DISPlay:WINDow1:TRACe2:FEED \'Meas\'')
    #set parameters for sweep
    pna.write('SENSe1:SWEep:POINts {}'.format(points))
    pna.write('SENSe1:FREQuency:CENTer {}GHZ'.format(centerf))
    pna.write('SENSe1:FREQuency:SPAN {}MHZ'.format(span))
    pna.write('SENSe1:BANDwidth {}KHZ'.format(ifband))
    pna.write('SENSe1:SWEep:TIME:AUTO ON')
    pna.write('SOUR:POW1 {}'.format(power))
    pna.write('CALCulate1:CORRection:EDELay:TIME {}NS'.format(edelay))
    pna.write('SENSe1:AVERage:STATe ON')

    #ensure at least 10 averages are taken
    #if(averages < 10):
    #    averages = 10
    if(averages < 1):
        averages = 1
    averages = averages//1
    pna.write('SENSe1:AVERage:Count {}'.format(averages))

def read_data(pna, points, outputfile, power, temp):
    '''
    function to read in data from the pna and output it into a file
    '''

    #read in frequency
    freq = np.linspace(float(pna.query('SENSe1:FREQuency:START?')), float(pna.query('SENSe1:FREQuency:STOP?')), points)

    #read in phase
    pna.write('CALCulate1:FORMat PHASe')
    phase = pna.query_ascii_values('CALCulate1:DATA? FDATA', container=np.array)

    #read in mag
    pna.write('CALCulate1:FORMat MLOG')
    mag = pna.query_ascii_values('CALCulate1:DATA? FDATA', container=np.array)

    #open output file and put data points into the file
    file = open(outputfile[0:-4]+'_'+str(power)+'dB'+'_'+str(temp)+'mK'+'.csv',"w")
    count = 0
    for i in freq:
        file.write(str(i)+','+str(mag[count])+','+str(phase[count])+'\n')
        count = count + 1
    file.close()

def getdata(centerf: float, span: float, temp: float, averages: int = 100, power: float = -30, edelay: float = 40, ifband: float = 5, points: int = 201, outputfile: str = "results.csv"):
    '''
    function to get data and put it into a user specified file
    '''

    #set up the PNA to measure s21 for the specific instrument GPIB0::16::INSTR
    rm = pyvisa.ResourceManager()
    keysight = rm.open_resource('GPIB0::16::INSTR')
    pna_setup(keysight, points, centerf, span, ifband, power, edelay, averages)

    #start taking data for S21
    keysight.write('CALCulate1:PARameter:SELect \'Meas\'')
    keysight.write('FORMat ASCII')

    #wait until the averages are done being taken then read in the data
    count = 10000000
    while(count > 0):
        count = count - 1
    while(True):
        if (keysight.query('STAT:OPER:AVER1:COND?')[1] != "0"):
            break;

    read_data(keysight, points, outputfile, power, temp)

def powersweep(startpower: float, endpower: float, numsweeps: int, centerf: float, span: float, temp: float, averages: float = 100, edelay: float = 40, ifband: float = 5, points: int = 201, outputfile: str = "results.csv"):
    '''
    run a power sweep for specified power range with a certain number of sweeps
    '''

    #create an array with the values of power for each sweep
    sweeps = np.linspace(startpower, endpower, numsweeps)
    stepsize = sweeps[0]-sweeps[1]

    #create a new directory for the output to be put into
    if (path.isdir(outputfile[0:-4]+'_'+'_'+str(temp)+'mK')):
        dircount = 1
        while (True):
            if (not path.isdir(outputfile[0:-4]+'_'+'_'+str(temp)+'mK'+str(dircount))):
                break;
            dircount = dircount + 1
        os.mkdir(outputfile[0:-4]+'_'+'_'+str(temp)+'mK'+str(dircount))
        outputfile = outputfile[0:-4]+'_'+'_'+str(temp)+'mK'+str(dircount) + '/' + outputfile
    else:
        os.mkdir(outputfile[0:-4]+'_'+'_'+str(temp)+'mK')
        outputfile = outputfile[0:-4]+'_'+'_'+str(temp)+'mK' + '/' + outputfile

    #write an output file with conditions
    file = open(outputfile[0:-4]+'_'+str(temp)+'mK_conditions'+'.csv',"w")
    file.write('STARTPOWER: '+str(startpower)+' dB\n')
    file.write('ENDPOWER: '+str(endpower)+' dB\n')
    file.write('NUMSWEEPS: '+str(numsweeps)+'\n')
    file.write('CENTERF: '+str(centerf)+' GHz\n')
    file.write('SPAN: '+str(span)+' MHz\n')
    file.write('TEMP: '+str(temp)+' mK\n')
    file.write('STARTING AVERAGES: '+str(averages)+'\n')
    file.write('EDELAY: '+str(edelay)+' ns\n')
    file.write('IFBAND: '+str(ifband)+' kHz\n')
    file.write('POINTS: '+str(points)+'\n')
    file.close()

    #run each sweep while increasing averages for each power
    for i in sweeps:
        getdata(centerf, span, temp, averages, i, edelay, ifband, points, outputfile)
        averages = averages * ((10**(stepsize/10))**0.5)
