# -*- encoding: utf-8 -*-
"""
Collection of functions defining control of the Keysight PNA instrument

TODO:
    * Collect all functions into a single class
    * Sends commands to the instrument with a socket or other connection
    * Commands of the SCPI variety using the pyvisa interface
    * Write a wrapper that uses the powersweep() function for legacy users

"""
import numpy as np
import pyvisa
import os
from os import path

import time
import pandas as pd

def pna_setup(pna,
              points: int, 
              centerf: float,
              span: float,
              ifband: float,
              power: float,
              edelay: float,
              averages: int,
              sparam : str = 'S12', 
              cal_set : str = None):
    '''
    set parameters for the PNA for the sweep (number of points, center
    frequency, span of frequencies, IF bandwidth, power, electrical delay and
    number of averages)

    XXX: Do not change this order:

    1.  Define a measurement
    2.  Turn on display
    3.  Set the number of points
    4.  Set the center frequency, span
    5.  Turn on sweep time AUTO
    6.  Set the electrical delay
    7.  Turn on interpolation
    8.  Set the calibration
    9.  Set the power
    10. Turn on averaging
    11. Set the IF bandwidth

    '''

    #initial setup for measurement
    if (pna.query('CALC1:PAR:CAT:EXT?') == f'"Meas,{sparam}"\n'):
        pna.write(f'CALCulate1:PARameter:DELete:EXT \'Meas\',{sparam}')
    pna.write(f'CALCulate1:PARameter:DEFine:EXT \'Meas\',{sparam}')
    # pna.write(f'CALCulate1:MEASure:PARameter {sparam}')

    #set parameters for sweep
    pna.write('DISPlay:WINDow1:STATE ON')
    pna.write('DISPlay:WINDow1:TRACe1:FEED \'Meas\'')
    pna.write('DISPlay:WINDow1:TRACe2:FEED \'Meas\'')

    pna.write(f'SENSe1:SWEep:POINts {points}')
    pna.write(f'SENSe1:FREQuency:CENTer {centerf}GHZ')
    pna.write(f'SENSe1:FREQuency:SPAN {span}MHZ')

    pna.write(f'SENSe1:SWEep:TIME:AUTO ON')
    pna.write(f'CALCulate1:CORRection:EDELay:TIME {edelay}NS')

    if cal_set:
        pna.write(f'CALCulate1:CORRection:TYPE \'Full 2 Port(1,2)\'')
        pna.write('SENSe1:CORRection:INTerpolate:state ON')
        cal_cmd = f'SENS1:CORR:CSET:ACT \'{cal_set}\',1'
        # cal_cmd = f'SENS:CORR:CSET:ACT \'{cal_set}\',0'
        # print(f'cal_cmd: {cal_cmd}')
        pna.write(cal_cmd)

    pna.write(f'SOUR1:POW1 {power}')
    pna.write('SENSe1:AVERage:STATe ON')
    pna.write(f'SENSe1:BANDwidth {ifband}KHZ')

    #ensure at least 10 averages are taken
    #if(averages < 10):
    #    averages = 10
    if(averages <= 1):
        averages = 3

    # Convert averages to integer
    averages = averages//1
    pna.write('SENSe1:AVERage:Count {}'.format(averages))

def read_data(pna, points, outputfile, power, temp):
    '''
    function to read in data from the pna and output it into a file
    '''

    #read in frequency
    freq = np.linspace(float(pna.query('SENSe1:FREQuency:START?')),
            float(pna.query('SENSe1:FREQuency:STOP?')), points)

    #read in phase
    pna.write('CALCulate1:FORMat PHASe')
    # pna.write('INITiate:CONTinuous OFF')
    # pna.write('INITiate:IMMediate;*wai')
    phase = pna.query_ascii_values('CALCulate1:DATA? FDATA',
            container=np.array)

    #read in mag
    pna.write('CALCulate1:FORMat MLOG')
    # pna.write('INITiate:CONTinuous OFF')
    # pna.write('INITiate:IMMediate;*wai')
    mag = pna.query_ascii_values('CALCulate1:DATA? FDATA', container=np.array)

    #open output file and put data points into the file
    filename = name_datafile(outputfile, power, temp)
    file = open(filename+'.csv',"w")

    count = 0
    for i in freq:
        file.write(str(i)+','+str(mag[count])+','+str(phase[count])+'\n')
        count = count + 1
    file.close()

def get_data(centerf: float, 
            span: float, 
            temp: float, 
            averages: int = 100, 
            power: float = -30, 
            edelay: float = 40, 
            ifband: float = 5, 
            points: int = 201, 
            outputfile: str = "results.csv",
            # instr_addr : str = 'GPIB::16::INSTR',
            # instr_addr : str = 'TCPIP0::69.254.35.52::islip0::INSTR1',
            instr_addr : str = 'TCPIP0::K-N5222B-21927::hislip0,4880::INSTR',
            sparam : str = 'S12',
            cal_set : str = None,
            setup_only : bool = False):
    '''
    function to get data and put it into a user specified file
    '''

    #set up the PNA to measure s21 for the specific instrument GPIB0::16::INSTR
    rm = pyvisa.ResourceManager()
    GPIB_addr = 'GPIB0::16::INSTR'


    # handle failure to open the GPIB resource #this is an issue when connecting
    # to the PNA-X from newyork rather than ontario
    try:
        keysight = rm.open_resource(instr_addr)
        # keysight = rm.open_resource('GPIB0::16::INSTR')
    except Exception as ex:
        print(f'\n----------\nException:\n{ex}\n----------\n')
        print(f'Trying GPIB address {GPIB_addr} ...')
        keysight = rm.open_resource(GPIB_addr)
        # keysight = rm.open_resource(instr_addr)

    pna_setup(keysight, points, centerf, span, ifband, power, edelay, averages,
              sparam=sparam, cal_set=cal_set)

    if setup_only:
        return

    #start taking data for S21
    keysight.write('INITiate:CONTinuous ON')
    keysight.write('OUTPut:STATe ON')
    keysight.write('CALCulate1:PARameter:SELect \'Meas\'')
    keysight.write('FORMat ASCII')

    #wait until the averages are done being taken then read in the data
    count = 10000000
    cnt = 0
    while(count > 0):
        count = count - 1
    while(True):
        if (keysight.query('STAT:OPER:AVER1:COND?')[1] != "0"):
            cnt += 1
            break;
            
    keysight.query('*OPC?')
    keysight.write('*WAI')
    time.sleep(3.0)
    keysight.write('SYSTem:CHANnels:HOLD')

    read_data(keysight, points, outputfile, power, temp)
    keysight.write('SYSTem:CHANnels:RESume')
    keysight.write('OUTPut:STATe OFF')

def power_sweep(startpower: float, 
                endpower: float, 
                numsweeps: int, 
                centerf: float, 
                span: float, 
                temp: float, 
                averages: float = 100, 
                edelay: float = 40, 
                ifband: float = 5, 
                points: int = 201, 
                outputfile: str = "results.csv",
                meastype: str = 'powersweep',
                sparam : str = 'S12',
                adaptive_averaging : bool = True,
                cal_set : str = None,
                setup_only : bool = False):
    '''
    run a power sweep for specified power range with a certain number of sweeps
    '''

    #create an array with the values of power for each sweep
    sweeps = np.linspace(startpower, endpower, numsweeps)
    stepsize = sweeps[0]-sweeps[1]
    print(f'Measuring {sparam} ...')

    #create a new directory for the output to be put into
    directory_name = timestamp_folder(os.getcwd(), meastype)
    os.mkdir(directory_name)
    outputfile = directory_name + '/' + outputfile

    #write an output file with conditions
    file = open(directory_name+'/'+'conditions.csv',"w")
    if not setup_only:
        file.write('STARTPOWER: '+str(startpower)+' dB\n')
        file.write('ENDPOWER: '+str(endpower)+' dB\n')
        file.write('NUMSWEEPS: '+str(numsweeps)+'\n')
        file.write('CENTERF: '+str(centerf)+' GHz\n')
        file.write('SPAN: '+str(span)+' MHz\n')
        file.write('TEMP: '+f'{temp:.3f}'+' mK\n')
        file.write('STARTING AVERAGES: '+str(averages)+'\n')
        file.write('EDELAY: '+str(edelay)+' ns\n')
        file.write('IFBAND: '+str(ifband)+' kHz\n')
        file.write('POINTS: '+str(points)+'\n')
        file.close()

    #run each sweep
    for i in sweeps:
        print(f'{i} dBm, {averages//1} averages ...')
        get_data(centerf, span, temp, averages, i, edelay, ifband, points,
                outputfile, sparam=sparam, cal_set=cal_set, setup_only=setup_only)
        if adaptive_averaging: 
            averages = averages * ((10**(stepsize/10))**0.5)
    print('Power sweep completed.')


def name_datafile(outputfile: str,
                  power: float,
                  temp: float) -> str:
    # Check that the file does not have an extension, otherwise strip it
    fsplit = outputfile.split('.')
    if len(fsplit) > 1:
      outputfile = fsplit[0]
    # Use f-strings to make the formatting more compact
    filename = f'{outputfile}_{power:.0f}dB_{temp:.0f}_mK.csv'
    # filename = outputfile+'_'+str(power)+'dB'+'_'+str(temp)+'mK.csv'
    filename = filename.replace('.','p')

    return filename
    
def timestamp_folder(dir: str = None, meastype: str='powersweep') -> str:
    """Create a filename and directory structure to annotate the scan.

        Takes a root directory, appends scan type and timestamp.

        Args:
            dir: root directory for the scan
            meastype: type of measurements, eg: 'powersweep' 

        Returns:
            Formatted path eg. dir/5p51414GHz_HPsweep_200713_12_18_04/ 
    """
    now = time.strftime("%y%m%d_%H_%M_%S", time.localtime())

    output = meastype+ '_' + now
    output = output.replace('.','p')
    
    if dir != None:
        output_path = os.path.join(dir,output)
    else:
        output_path = output + '/'
    count=2
    path = output_path
    while os.path.isdir(output_path):
        output_path=path[0:-1]+'_'+ str(count) +'/'
        count = count+1
    return output_path

