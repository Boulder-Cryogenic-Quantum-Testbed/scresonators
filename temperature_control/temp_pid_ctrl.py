# -*- encoding: utf-8 -*-
"""
PID Controller using the open loop control on the Janus
gas handling control system.

Author: Nick Materise
Date:   210919


TODO: 
    * Add error checking and handling for all steps of the temperature polling
      process and TCP/IP message passing 


Example:
    
    # Modify the temperatures Tstop, Tstart, dT
    # Make sure you login to the JetWay session on PuTTY
    # with user: 'bco' and password: 'aish8Hu8'
    # before running this code.
    # Also make sure the still heater is set at 0.4 V
    # This should all work from the New York control computer

    python temp_pid_ctrl.py


"""

import socket
import simple_pid
import time
import datetime
import subprocess
from multiprocessing import Process, Manager
import glob
import numpy as np

import sys
sys.path.append(r'C:\Users\Lehnert Lab\GitHub\measurement\pna_control')
import pna_control as PNA
import os


class JanusTemperatureController(object):
    """
    Class that implments the Janus temperature controller
    """
    def __init__(self, Tstart, Tstop, dT, *args, **kwargs):
        """
        Class constructor
        """
        # Set the defaults for the TCP address and ports
        self.TCP_IP   = 'localhost'
        self.TCP_PORT = 5559

        # Default thermalization time
        self.therm_time  = 300. # Wait extra 5 minutes to thermalize [s]
        self.T_eps       = 1e-2 # Temperature settling threshold [mK]

        # Create socket connection to the Janus Gas Handling System
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.TCP_IP, self.TCP_PORT))

        # Path to the pna measurement script
        prepath = 'C:\\Users\\Lehnert Lab\\OneDrive - UCB-O365\\Experiment'
        pna_scr = 'repeat_resonance_measurement.py'
        sub_dir = 'Mines_6061_3_temperatures'
        self.path_to_pna_script = f'{prepath}\\{sub_dir}\\{pna_scr}'

        # Update the arguments and the keyword arguments
        # This will overwrite the above defaults with the user-passed kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Set the temperature array
        NT = round((Tstop - Tstart) / dT) + 1
        self.T_sweep_list = np.linspace(Tstart, Tstop, NT)

        # Check that the path exists for the file to run the PNA measurement
        assert glob.glob(self.path_to_pna_script) != [], \
                f'{self.path_to_pna_script} not found.'

        # Print all of the settings before proceeding
        print('\n---------------------------------------------------')
        print(f'JanusTemperatureController class instance members:')
        for k, v in self.__dict__.items():
            print(f'{k} : {v}')
        print('---------------------------------------------------\n')

    def __del__(self):
        """
        Deconstructor to free resources
        """
        # Set the current to zero and close the socket connection
        self.set_current(0.)
        self.socket.close()

    def tcp_send(self, message):
        length = len(message)
        length = length.to_bytes(4, 'big')
        self.socket.send(length)
        self.socket.send(message.encode('ASCII'))
    
    def tcp_rcev(self):
        buffer = self.socket.recv(4)
        buffer = int.from_bytes(buffer, 'big')
        data = self.socket.recv(buffer)
        data = data.decode('ascii')
        return data
    
    def read_cmn(self):
        self.tcp_send('readCMNTemp(9)')
        data = self.tcp_rcev()
        Z, T, tstamp, status = data.split(',')
        Z = float(Z)
        T = float(T)
        status = int(status)
        tstamp = tstamp.split(' ')
        tstamp = tstamp[0].split('.')[0]
        if not status:
            return Z, T, tstamp
        else:
            print(f'tcp_send() failed with status: {status}')
            return None, None, None
    
    def get_heater_rng_lvl(self, x):
        """
        This function takes the pid output (Current in mA)
        nd converts it to heater settings
        """
        if np.isclose(x, 0.):
            Range = 0
            level = 0
        elif x < 0.0316:
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
    
    def set_current(self, x):
        """
        Set the current to x mA
        """
        Range, level = self.get_heater_rng_lvl(x)
        self.tcp_send(f'setHtrCntrlModeOpenLoop(1,{level},{Range})')
        self.tcp_rcev()

    def get_pid_ctrl(self, T_set, sample_time=15):
        """
        Returns a simple_pid PID controller object
        """
        # Trying Ku = 70, Tu = 186 s (this is set by LS372 scanning cycle time)
        # 
        # go to https://en.wikipedia.org/wiki/Ziegler%E2%80%93Nichols_method
        # for info on Ku and Tu and how they relate to PID gain settings
        Ku = 70
        Tu =  186#s
        Kp = 0.6 * Ku
        Ki = 1.2 * Ku / Tu
        Kd = 0.075 * Ku * Tu
        #Temperature setpoint must be in Kelvin!
        #sample_time is in seconds
        stime = self.sample_time if hasattr(self, 'sample_time') \
                else sample_time
        
        print(f'PID controller for {T_set*1e3} mK ...')
        # T_set = 0.03
        pid = simple_pid.PID(Kp, Ki, Kd, setpoint=T_set, sample_time=stime)

        # Set the maximum current
        T_base = 0.013 #K (approx. base temperature)
        Max_Current = 1.33*8.373*(T_set-T_base)**(0.720) #mA
        print(f'Max Curent: {Max_Current} mA')
        pid.output_limits = (0, Max_Current)

        return pid

    
    def temperature_controller(self, idx, t, pid, fid, T_set, out):
        if idx == 'temp':
            print('Temperature control and measurement ...')
            start_thermalize_timer = False
            T = 10 * T_set
            while 1:
                if abs(1e3 * (T - T_set)) < self.T_eps:
                    start_thermalize_timer = True
                
                # Wait five more minutes
                if start_thermalize_timer:
                    print('Starting thermalization timer ...')
                    time.sleep(self.therm_time)
                    out[idx] = [tin, T, tstamp]
                    break

                Z, T, tstamp = self.read_cmn()
                tin = t
                if T is not None:
                    # Generate the output current
                    output = pid(T)
                    print(f'{tstamp}, {1e3 * T} mK, {output} mA, {tin} s')
                    self.set_current(output)
                    time.sleep(pid.sample_time)
                
                    # Write the time stamp, temperature, and impedance to file
                    tin += self.sample_time
                    out[idx] = [tin, T, tstamp]
                else:
                    out[idx] = None
        else:
            print(f'Initial warmup call ...')
            Z, T, tstamp = self.read_cmn()
            tin = t
            if T is not None:
                # Generate the output current
                output = pid(T)
                print(f'{tstamp}, {1e3 * T} mK, {output} mA, {tin} s')
                self.set_current(output)
                time.sleep(pid.sample_time)
            
                # Write the time stamp, temperature, and impedance to file
                tin += self.sample_time
                fid.write(f'{tstamp}, {T}, {tin}\n')
                return tin, T
    
            else:
                return None
    
    def pna_process(self, idx, path_to_scr, T_set, out):
        """
        Performs a PNA-X measurement
        """
        # TODO: pull out these options as class members
        AVERAGES = 1 # Number of averages for first (highest) power
        EDELAY = 79.482 #ns
        IFBAND = 1.0 #kHz
        NUMSWEEPS = 2
        CENTERF = 7.5777
        SPAN = [50] #MHz
        POINTS = 2001
        
        TEMP = T_set * 1e3 #mK
        SAMPLEID = 'M3D6_02_with_InP' #project ID followed by sample number and die number
        
        STARTPOWER = -45
        ENDPOWER = -45

        OUTPUTFILE = SAMPLEID+'_'+str(CENTERF)+'GHz_'+f'LPsweep'
        PNA.powersweep(STARTPOWER, ENDPOWER, NUMSWEEPS, CENTERF, SPAN,
                    TEMP, AVERAGES, EDELAY, IFBAND, POINTS, OUTPUTFILE)

        out[idx] = 0


    def run_temp_sweep(self):
        """
        Execute the temperature sweep
        """
        # Iterate over the temperatures
        for T_set in self.T_sweep_list:
            # Get the pid controller object
            pid = self.get_pid_ctrl(T_set)

            # Set the output filename and write the results with
            # standard text file IO operations
            dstr = datetime.datetime.today().strftime('%y%m%d')
            T = 10 * T_set
            t = 0
        
            # Set the log filename, log the temperature, time stamp, and time
            fname = f'logs/temperature_{int(T_set * 1e3)}_mK_log_{dstr}.csv'
        
            # Flag to start measurement run
            start_thermalize_timer = False
            meas_ret = 1
            path_to_script = self.path_to_pna_script
            assert glob.glob(path_to_script) != [], 'Script not found!!!'

            # Open file and start logging time stamps, elapsed time,
            # temperature
            with open(fname, 'w') as fid:
                fid.write('# Time[HH:MM:SS], Time [s], Temperature [mK]\n')
        
                # Continue to run the PID controller as the measurement runs
                try:
                    while meas_ret:
                        # Measure the temperature, set the current, write
                        # results to file
                        t, T = self.temperature_controller('', t, pid, fid,
                                T_set, None)
                        if t is None:
                            break
        
                        # Check that the temperature has settled
                        if abs(1e3 * (T - T_set)) < Tctrl.T_eps:
                            start_thermalize_timer = True
        
                        # Wait five more minutes
                        if start_thermalize_timer:
                            print('Starting thermalization timer ...')
                            time.sleep(therm_time)
        
                            # Launch the resonance measurement (power sweep,
                            # etc.) Wait for the process to return
                            print('Starting PNA measurement ...')
                            mng = Manager()
                            out = mng.dict()
                            # ptemp = Process(target=self.temperature_controller,
                            #        args=('temp', t, pid, None, T_set, out))
                            pmeas = Process(target=self.pna_process,
                                    args=('meas', path_to_script, T_set, out))
                            # ptemp.start()
                            pmeas.start()
                            # ptemp.join()
                            pmeas.join()
                            print('Finished PNA Measurement.')

                            # # Read the time and measurement return code
                            # t, T = self.temperature_controller('', t, pid, fid,
                            #     T_set, None)
                            # print(f'{tstamp}, {1e3*T} mK, {output} mA, {t} s')

                            # # Write to the log file outside of the process
                            # fid.write(f'{tstamp}, {T}, {t}\n')
                            meas_ret = 0
                            break
                            # meas_ret     = out['meas']

                # Graceful exit on Ctrl-C interrupt by the user
                except (KeyboardInterrupt, Exception):
                    self.set_current(0.)
                    self.socket.close()
                    fid.close()
        
            # Close the file, just in case the context manager does not free it
            fid.close()
            
        # Set the heater current back to 0 mA
        self.set_current(0.)


if __name__ == '__main__':
    # Iterate over a list of temperatures
    # 30 mK -- 300 mK, 10 mK steps
    Tstart = 0.1; Tstop = 0.30; dT = 0.01
    sample_time = 15; T_eps = 1e-2
    therm_time  = 300. # wait extra 5 minutes to thermalize

    # Set the path to the PNA script
    prepath = 'C:\\Users\\Lehnert Lab\\OneDrive - UCB-O365\\Experiment'
    pna_scr = 'repeat_resonance_measurement.py'
    sub_dir = 'Mines_6061_3_temperatures'
    path_to_pna_script = f'{prepath}\\{sub_dir}\\{pna_scr}'

    # Setup the temperature controller class object
    Tctrl = JanusTemperatureController(Tstart, Tstop, dT,
            sample_time=sample_time, T_eps=T_eps, therm_time=therm_time,
            path_to_pna_script=path_to_pna_script)

    # Run the temperature sweep from within the class
    # Tctrl.run_temp_sweep()
    Tctrl.set_current(0.)
    Z, T, tstamp = Tctrl.read_cmn()
    print(f'{tstamp}, {T*1e3} mK')
