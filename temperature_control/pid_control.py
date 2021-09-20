# -*- encoding: utf-8 -*-
"""
PID Controller using the open loop control on the Janus
gas handling control system.

TODO: 
    * Convert this file to a class with all of the
      routines baked into one place
    * Add error checking and handling
    * Add more robust temperature and time logging
"""

import socket
import simple_pid
import time
import datetime
import subprocess
from multiprocessing import Process, Manager
import glob
import numpy as np

#Make sure you login to the JetWay session on PuTTY
#with user: 'bco' and password: 'aish8Hu8'
#before running this code.
#Also make sure the still heater is set at 0.4 V


def TCP_Send(sck, message):
    length = len(message)
    length = length.to_bytes(4, 'big')
    sck.send(length)
    sck.send(message.encode('ASCII'))

def TCP_Recv(sck):
    buffer = sck.recv(4)
    buffer = int.from_bytes(buffer, 'big')
    data = sck.recv(buffer)
    data = data.decode('ascii')
    return data

def ReadCMN(sck):
    TCP_Send(sck, 'readCMNTemp(9)')
    data = TCP_Recv(sck)
    Z, T, tstamp, status = data.split(',')
    Z = float(Z)
    T = float(T)
    status = int(status)
    tstamp = tstamp.split(' ')
    tstamp = tstamp[0].split('.')[0]
    if not status:
        return Z, T, tstamp
    else:
        print(f'TCP_Send() failed with status: {status}')
        return None, None, None


#This function takes the pid output (Current in mA)
#and converts it to heater settings
def ChooseHeaterSettings(x):
    if x < 0.0316:
        Range = 1
        level = x*100/0.0316
    elif 0.0316 < x <= 0.1:
        Range = 2
        level = x*100/0.1
    elif 0.1 < x <= 0.316:
        Range = 3
        level = x*100/0.316
    elif 0.316 < x <= 1.0:
        Range = 4
        level = x*100/1.0
    elif 1.0 < x <= 3.16:
        Range = 5
        level = x*100/3.16
    else:
        print("DON'T MELT THE FRIDGE")
        Range = 0
        level = 0
    return Range, level

#input is current in mA
def SetCurrent(sck, x):
    Range, level = ChooseHeaterSettings(x)
    TCP_Send(sck, f'setHtrCntrlModeOpenLoop(1,{level},{Range})')
    TCP_Recv(sck)


def TemperatureController(idx, t, pid, fid, sck, out=None):
    if idx == 'temp':
        print(f'Temperature control and measurement ...')
        while 1:
            Z, T, tstamp = ReadCMN(sck)
            tin = t
            if T is not None:
                # Generate the output current
                output = pid(T)
                print(f'{tstamp}, {1e3 * T} mK, {output} mA, {tstamp}, {t} s')
                SetCurrent(sck, output)
                time.sleep(pid.sample_time)
            
                # Write the time stamp, temperature, and impedance to file
                fid.write(f'{tstamp}, {t}, {T}\n')
                tin += sample_time
                out[idx] = [tin, T]
            else:
                out[idx] = None
    else:
        print(f'Initial warmup call ...')
        Z, T, tstamp = ReadCMN(sck)
        tin = t
        if T is not None:
            # Generate the output current
            output = pid(T)
            print(f'{tstamp}, {1e3 * T} mK, {output} mA, {tstamp}, {t} s')
            SetCurrent(sck, output)
            time.sleep(pid.sample_time)
        
            # Write the time stamp, temperature, and impedance to file
            fid.write(f'{tstamp}, {T}, {t}\n')
            tin += sample_time
            return tin, T

        else:
            return None


def MeasurementProcess(idx, measurement_script_path, out):
    p = subprocess.Popen([f'python {measurement_script_path}'])
    out[idx] = p.wait()


if __name__ == '__main__':

    #Here we connect to the GHS with a TCP socket and
    #define some funtions to handle the buffer protocol
    #and reading the MC tempeature
    TCP_IP = 'localhost'
    TCP_PORT = 5559
    
    # Create socket connection to the Janus Gas Handling System
    sck = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sck.connect((TCP_IP, TCP_PORT))
    
    # Path to the pna measurement script
    prepath = 'C:\\Users\\Lehnert Lab\\OneDrive - UCB-O365\\Experiment'
    res_script = 'repeat_resonance_measurement.py'
    measurement_path = f'{prepath}\\Mines_6061_3_temperatures\\{res_script}'
    assert glob.glob(measurement_path) != [], \
            f'{measurement_script_path} not found.'
    

    #trying Ku = 70, Tu = 186 s (this is set by LS372 scanning cycle time)
    #
    #go to https://en.wikipedia.org/wiki/Ziegler%E2%80%93Nichols_method
    #for info on Ku and Tu and how they relate to PID gain settings
    Ku = 70
    Tu =  186#s
    Kp = 0.6*Ku
    Ki = 1.2*Ku/Tu
    Kd = 0.075*Ku*Tu
    #Temperature setpoint must be in Kelvin!
    #sample_time is in seconds
    sample_time = 15
    
    # Iterate over a list of temperatures
    Tstart = 0.03; Tstop = 0.3; NT = 28;
    T_sets = [Tstart] # np.linspace(Tstart, Tstop, NT)
    for T_set in T_sets:
        print(f'PID controller for {T_set*1e3} mK ...')
    
        # T_set = 0.03
        pid = simple_pid.PID(Kp, Ki, Kd, setpoint = T_set, sample_time =
                             sample_time)
    
        # pid.sample_time = 16
        #it is important that pid.sample_time is set to the same value
        #that we wait before calling pid() again
        #
        #I may modify the simple-pid code to make it wait dt = sample_time
        #before returning a new output
        
        #There is a power law relating Current and equilibrium Temperature
        #minus T_base
        # i.e. $T-T_base \prop I^n$ for some n
        #I am inverting this equation and limiting the max current at 33% above
        #this equilibrium value. This is to avoid massive overshoot since
        #time-resolution of the CMN is  poor compared to the timescale at which
        #the temperature increases.  #This data was taken with the still heater
        #in open loop mode, set at 0.4V
        T_base = 0.013 #K (approx. base temperature)
        Max_Current = 1.33*8.373*(T_set-T_base)**(0.720) #mA
        print(f'Max Curent: {Max_Current} mA')
        pid.output_limits = (0, Max_Current)
        
        # Set the output filename and write the results with
        # standard text file IO operations
        dstr = datetime.datetime.today().strftime('%y%m%d')
        eps = 1e-2
        T = 10 * T_set
        t = 0
    
        # Set the log filename, log the temperature, time stamp, and time
        fname = f'temperature_{int(T_set * 1e3)}_mK_log_{dstr}.csv'
    
        # Flag to start measurement run
        start_thermalize_timer = True
        therm_time  = 3. # wait extra 5 minutes to thermalize
        meas_ret = 1
    
        # Open file and start logging time stamps, elapsed time, temperature
        with open(fname, 'w') as fid:
            fid.write('# Time[HH:MM:SS], Time [s], Temperature [mK]\n')
    
            # Continue to run the PID controller as the measurement runs
            try:
                while meas_ret:
                    # Measure the temperature, set the current, write results
                    # to file
                    t, T = TemperatureController('', t, pid, fid, sck, out=None)
                    if t is None:
                        break
    
                    # Check that the temperature has settled
                    if abs((T - T_set) * 1e3) < eps:
                        start_thermalize_timer = True
    
                    # Wait five more minutes
                    if start_thermalize_timer:
                        print('Starting thermalization wait time ...')
                        time.sleep(therm_time)
    
                        # Launch the resonance measurement (power sweep, etc.)
                        # Wait for the process to return
                        print(f'Starting PNA measurement ...')
                        mng = Manager()
                        out = mng.dict()
                        pmeas = Process(target=MeasurementProcess,
                                args=('meas', measurement_path, out))
                        ptemp = Process(target=TemperatureController,
                                args=('temp', t, pid, fid, sck, out))
                        pmeas.start()
                        ptemp.start()
                        pmeas.join()
                        ptemp.join()

                        # Read the time and measurement return code
                        meas_ret = out['meas']
                        t, T = out['temp']

            except KeyboardInterrupt:
                SetCurrent(sck, 0)
                sck.close()
                fid.close()
    
        # Close the file, just in case the context manager does not free it
        fid.close()
        
    # Set the heater current back to 0 mA
    SetCurrent(sck, 0)
    s.close()
