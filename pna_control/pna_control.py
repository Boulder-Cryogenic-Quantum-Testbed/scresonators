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
              cal_set : str = None,
              segments : list = None):
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
    # Send a preset command to the VNA and turn off the RF power
    pna.write('SYSTem:FPRESet')
    time.sleep(0.01)
    pna.write('OUTPut:STATe OFF')

    # Initial setup for measurement
    ## Query the exisiting measurements
    measurements = pna.query('CALC1:PAR:CAT:EXTended?')

    ## If any measurements exist, delete them all
    if measurements != 'NO CATALOG':
        pna.write(f'CALCulate1:PARameter:DELete:ALL')
    pna.write(f'CALCulate1:MEASure1:DEFine \"{sparam}\"')
    pna.write(f'CALCulate1:MEASure2:DEFine \"{sparam}\"')

    #set parameters for sweep
    pna.write('DISPlay:WINDow1 ON')
    pna.write('DISPlay:MEAS1:FEED 1')
    pna.write('DISPlay:WINDow2 ON')
    pna.write('DISPlay:MEAS2:FEED 2')


    if segments:
        num_segments = len(segments)
        seg_data = ''.join([s for s in segments])
        pna.write("SENSe1:SWEep:TYPE SEGment")
        pna.write(f'SENSe1:SEGMent:LIST SSTOP, {num_segments}{seg_data}')
    else:
        pna.write("SENSe1:SWEep:TYPE LINear")
        pna.write(f'SENSe1:SWEep:POINts {points}')
        pna.write(f'SENSe1:FREQuency:CENTer {centerf}GHZ')
        pna.write(f'SENSe1:FREQuency:SPAN {span}MHZ')

        pna.write(f'SENSe1:SWEep:TIME:AUTO ON')
        
    pna.write(f'CALCulate1:CORRection:EDELay:TIME {edelay}NS')

    if cal_set:
        pna.write(f'CALCulate1:CORRection:TYPE \'Full 2 Port(1,2)\'')
        pna.write('SENSe1:CORRection:INTerpolate:state ON')
        # XXX: This does not work!
        # cal_cmd = f'SENS1:CORR:CSET:ACT \'{cal_set}\',1'
        cal_cmd = f'SENS:CORR:CSET:ACT \'{cal_set}\',0'
        # print(f'cal_cmd: {cal_cmd}')
        pna.write(cal_cmd)

        # gpoints = pna.query(f'SENSe1:SWEep:POINts?')
        # assert int(gpoints) == points, f'VNA points ({gpoints}) != {points}.'

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

def read_data(pna, points, outputfile, power, temp, segments : list = None):
    '''
    function to read in data from the pna and output it into a file
    '''

    #read in frequency
    cfreq = float(pna.query('SENSe1:FREQuency:CENTER?')) / 1e9

    # This obviates the need for points as an input
    if segments:
        # Read the list of all segments
        freq = np.array([])
        for s in segments:
            ssplit = s.split(',')
            nf = int(ssplit[2])
            f1 = float(ssplit[3])
            f2 = float(ssplit[4])
            f = np.linspace(f1, f2, nf)
            freq = np.hstack((freq, f))
    else:
        gpoints = int(pna.query(f'SENSe1:SWEep:POINts?'))
        freq = np.linspace(float(pna.query('SENSe1:FREQuency:START?')),
                float(pna.query('SENSe1:FREQuency:STOP?')), gpoints)

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
    filename = name_datafile(outputfile, power, temp, cfreq)
    file = open(filename+'.csv', "w")

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
            # instr_addr : str = 'GPIB::16::INSTR', # If using GPIB
            # instr_addr : str = 'TCPIP0::69.254.35.52::islip0::INSTR1', # Old address from JILA lab
            instr_addr : str = 'TCPIP0::K-N5222B-21927::hislip0,4880::INSTR',
            sparam : str = 'S12',
            cal_set : str = None,
            setup_only : bool = False,
            segments : list = None):
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

        ## Attempt to fix the timeout error in averaging command
        keysight.timeout = None
        # keysight = rm.open_resource('GPIB0::16::INSTR')
    except Exception as ex:
        print(f'\n----------\nException:\n{ex}\n----------\n')
        print(f'Trying GPIB address {GPIB_addr} ...')
        keysight = rm.open_resource(GPIB_addr)
        # keysight = rm.open_resource(instr_addr)

    pna_setup(keysight, points, centerf, span, ifband, power, edelay, averages,
              sparam=sparam, cal_set=cal_set, segments=segments)

    if setup_only:
        return

    # start taking data for S21
    keysight.write('INITiate:CONTinuous ON')
    keysight.write('OUTPut:STATe ON')
    # keysight.write('CALCulate1:PARameter:SELect \'M1\'')
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

    read_data(keysight, points, outputfile, power, temp,
              segments=segments)

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
                setup_only : bool = False,
                segments : list = None):
    '''
    run a power sweep for specified power range with a certain number of sweeps
    '''

    #create an array with the values of power for each sweep
    if np.isclose(startpower, endpower):
        print(f'Running only one power {startpower} dBm ...')
        sweeps = [startpower]
        stepsize = 0
    else:
        sweeps = np.linspace(startpower, endpower, numsweeps)
        stepsize = sweeps[0]-sweeps[1]
    print(f'Measuring {sparam} ...')

    #create a new directory for the output to be put into
    directory_name = timestamp_folder(os.getcwd(), centerf, meastype)
    os.mkdir(directory_name)
    outputfile = directory_name + '/' + outputfile

    #write an output file with conditions
    with open(directory_name+'/'+'conditions.csv',"w") as file:
        file.write('# Parameter, Value, Units\n')
        file.write(f'SPARAM, {sparam}, \n')
        file.write(f'CALSET, {cal_set}, \n')
        file.write(f'STARTPOWER, {startpower}, dB\n')
        file.write(f'ENDPOWER, {endpower}, dB\n')
        file.write(f'NUMSWEEPS, {numsweeps}, \n')
        file.write(f'CENTERF, {centerf}, GHz\n')
        file.write(f'SPAN, {span}, MHz\n')
        file.write(f'TEMP, {temp:.3f}, mK\n')
        file.write(f'STARTING AVERAGES, {averages}\n')
        file.write(f'EDELAY, {edelay}, ns\n')
        file.write(f'IFBAND, {ifband}, kHz\n')
        file.write(f'POINTS, {points}, \n')
        file.close()

    #run each sweep
    for i in sweeps:
        print(f'{i} dBm, {averages//1} averages ...')
        get_data(centerf, span, temp, averages, i, edelay, ifband, points,
                outputfile, sparam=sparam, cal_set=cal_set,
                setup_only=setup_only, segments=segments)
        if adaptive_averaging: 
            averages = averages * ((10**(stepsize/10))**0.5)
    print('Power sweep completed.')


def name_datafile(outputfile: str,
                  power: float,
                  temp: float,
                  freq: float) -> str:
    # Check that the file does not have an extension, otherwise strip it
    fsplit = outputfile.split('.')
    if len(fsplit) > 1:
      outputfile = fsplit[0]
    # Use f-strings to make the formatting more compact
    filename = f'{outputfile}_{freq:.3f}GHz_{power:.0f}dB_{temp:.0f}mK'
    filename = filename.replace('.','p')

    return filename
    
def timestamp_folder(dir: str = None, centerf = None, meastype:
        str='powersweep') -> str:
    """Create a filename and directory structure to annotate the scan.

        Takes a root directory, appends scan type and timestamp.

        Args:
            dir: root directory for the scan
            meastype: type of measurements, eg: 'powersweep' 

        Returns:
            Formatted path eg. dir/5p51414GHz_HPsweep_200713_12_18_04/ 
    """
    now = time.strftime("%y%m%d_%H_%M_%S", time.localtime())

    output = meastype+ '_' + f'{centerf:.3f}GHz_' + now
    output = output.replace('.','p')
    
    if dir != None:
        output_path = os.path.join(dir, output)
    else:
        output_path = output + '/'
    count=2
    path = output_path
    while os.path.isdir(output_path):
        output_path=path[0:-1]+'_'+ str(count) +'/'
        count = count+1
    return output_path

