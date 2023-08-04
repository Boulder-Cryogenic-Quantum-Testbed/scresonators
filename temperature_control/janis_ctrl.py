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
    # Also make sure the still heater is OFF
    # This should all work from the New York control computer

    python janis_ctrl.py

"""

import socket
import simple_pid
import time
import datetime
import subprocess
from multiprocessing import Process, Manager
import glob
import numpy as np
import errno

import sys
sys.path.append(r'C:\Users\Lehnert Lab\GitHub\measurement\pna_control')
import pna_control as PNA
import os


class JanisCtrl(object):
    """
    Class that implments the Janus temperature controller
    """
    def __init__(self, Tstart, Tstop, dT, *args, **kwargs):
        """
        Class constructor
        """
        # Set the defaults for the TCP address and ports
        self.TCP_IP = 'localhost'
        self.TCP_PORT = 5559
        self.init_socket = True

        # Set the default VNA address
        self.vna_addr = 'TCPIP0::K-N5222B-21927::hislip0,4880::INSTR'

        # Set as True to start the PID controller, then set to False to allow
        # for updates to the PID values from the previous temperature set point
        self.is_pid_init = True
        self.pid_values = None
        self.bypass_janis = False

        # Default thermalization time
        self.therm_time = 300. # Wait extra 5 minutes to thermalize [s]
        self.T_eps = 1e-2 # Temperature settling threshold [mK]
        self.T_sweep_list_spacing = 'linear'
        self.adaptive_averaging = False

        # Set the base temperature of the fridge here
        self.T_base = 0.016
        self.dstr = datetime.datetime.today().strftime('%y%m%d')

        # Dictionary with the channels on the Lakeshore
        self.channel_dict = {'50K'     : 1, '10K'    : 2, '3K'  : 3,
                             'JT'      : 4, 'still'  : 5, 'ICP' : 6,
                             'MC JRS'  : 7, 'Cernox' : 8}

        # Update the arguments and the keyword arguments
        # This will overwrite the above defaults with the user-passed kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Create socket connection to the Janus Gas Handling System
        print(f'self.bypass_janis: {self.bypass_janis}')
        if self.init_socket and (not self.bypass_janis):
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.TCP_IP, self.TCP_PORT))
        elif self.bypass_janis:
            self.socket = None
        else:
            self.socket = None

        # Set the temperature array
        ## Set the number of temperatures
        if hasattr(self, 'NT'):
            NT = self.NT
        else:
            NT = round((Tstop - Tstart) / dT) + 1

        ## Set the temperature list as linearly or logarithmically spaced
        if self.T_sweep_list_spacing == 'linear':
            self.T_sweep_list = np.linspace(Tstart, Tstop, NT)
        elif self.T_sweep_list_spacing == 'linear':
            Tstart_log10 = np.log10(Tstart)
            Tstop_log10 = np.log10(Tstop)
            self.T_sweep_list = np.linspace(Tstart_log10, Tstop_log10, NT)
        else:
            tsls = self.T_sweep_list_spacing
            raise ValueError(f'Temperature sweep spacing {tsls} not supported.')

        # Print all of the members stored in the class
        self.print_class_members()

    def __del__(self):
        """
        Deconstructor to free resources
        """
        # Set the current to zero and close the socket connection
        print('Calling destructor ...')
        if self.socket is not None:
            print('Setting current to 0 ...')
            self.set_current(0.)
            self.socket.close()
            self.socket = None

    def close_socket(self):
        """
        Close the jacob socket connection
        """
        if self.socket is not None:
            print('Setting current to 0 ...')
            self.set_current(0.)
            self.socket.close()
            self.socket = None

    def print_class_members(self):
        """
        Prints all the values of the members stored in the class instance
        """
        # Print all of the settings before proceeding
        print('\n---------------------------------------------------')
        print(f'JanisCtrl class instance members:')
        for k, v in self.__dict__.items():
            print(f'{k} : {v}')
        print('---------------------------------------------------\n')


    def reset_socket(self):
        if self.socket is not None:
            print(f'Resetting socket ...')
            self.socket.close()
            self.socket = None
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.TCP_IP, self.TCP_PORT))
        else:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.TCP_IP, self.TCP_PORT))

    def close_socket(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None


    def tcp_send(self, message):
        length = len(message)
        length = length.to_bytes(4, 'big')
        exceptions = (socket.error, socket.error, KeyboardInterrupt, Exception)
        try:
            self.socket.send(length)
            self.socket.send(message.encode('ASCII'))
        except exceptions as error:
            raise RuntimeError(f'tcp_send: {error}')
    
    def tcp_recv(self):
        exceptions = (socket.error, socket.error, KeyboardInterrupt, Exception)
        try:
            buffer = self.socket.recv(4)
        except exceptions as error:
            raise RuntimeError(f'tcp_recv: {error}')

        buffer = int.from_bytes(buffer, 'big')
        data = self.socket.recv(buffer)
        data = data.decode('ascii')
        return data
    
    def read_cmn(self):
        """
        Reads the CMN temperature sensor and returns the sensor impedance,
        temperature in K and timestamp
        """
        self.tcp_send('readCMNTemp(9)')
        data = self.tcp_recv()
        err = False
        try:
            Z, T, tstamp, status = data.split(',')
        except Exception as e:
            print(f'{e}\ndata: {data}')
            err = True
            
        # Additional error checking
        if err:
            Z = None
            T = None
            status = 1
        else:
            Z = float(Z)
            T = float(T)
            status = int(status)
            tstamp = tstamp.split(' ')
            tstamp = tstamp[0].split('.')[0]

        if not status:
            return Z, T, tstamp
        else:
            print(f'tcp_send(readCMNTemp(9)) failed with status: {status}')
            return None, None, None

    def read_temp(self, channel_name='still'):
        """
        Reads the temperature of one of the channels from the Lakeshore
        """
        if channel_name == 'all':
            for key, ch in self.channel_dict.items():

                self.tcp_send(f'readTemp({ch})')
                data = self.tcp_recv()
                Z, T, tstamp, status = data.split(',')
                tstamp = tstamp.split(' ')
                tstamp = tstamp[0].split('.')[0]
                Z = float(Z)
                T = float(T)
                print(f'{tstamp}, {key}: {T:.4g} K')
        else:
            channel = self.channel_dict[channel_name]
            self.tcp_send(f'readTemp({channel})')
            data = self.tcp_recv()
            Z, T, tstamp, status = data.split(',')
            Z = float(Z)
            T = float(T)
            status = int(status)
            tstamp = tstamp.split(' ')
            tstamp = tstamp[0].split('.')[0]
            msg = f'tcp_send(readTemp({channel})) failed with status: {status}'
            if not status:
                return Z, T, tstamp
            else:
                print(msg)
                return None, None, None

    def read_flow_meter(self):
        """
        Reads the flow meter and returns the flow rate in volts, umol / s, and
        the timestamp
        """
        self.tcp_send('readFlow(1)')
        data = self.tcp_recv()
        flow_V, flow_umol_s1, tstamp, status = data.split(',')
        status = int(status)
        flow_V = float(flow_V)
        flow_umol_s1 = float(flow_umol_s1)
        tstamp = tstamp.split(' ')
        tstamp = tstamp[0].split('.')[0]
        if not status:
            return flow_V, flow_umol_s1, tstamp
        else:
            print(f'tcp_send(readFlow(1)) failed with status: {status}')
            return None, None, None

    def read_pressure(self, channel='all'):
        """
        Reads the temperature of one of the channels from the Lakeshore
        """
        all_channels = [1, 2, 3, 4]
        if channel == 'all':
            for ch in all_channels:

                self.tcp_send(f'readPressure({ch})')
                data = self.tcp_recv()
                P_V, P_mbar, tstamp, status = data.split(',')
                tstamp = tstamp.split(' ')
                tstamp = tstamp[0].split('.')[0]
                P_V    = float(P_V)
                P_mbar = float(P_mbar)
                status = int(status)
                print(f'{tstamp}, G{ch}: {P_mbar:.4e} mbar')
        elif channel in all_channels:

            self.tcp_send(f'readPressure({channel})')
            data = self.tcp_recv()
            P_V, P_mbar, tstamp, status = data.split(',')
            tstamp = tstamp.split(' ')
            tstamp = tstamp[0].split('.')[0]
            P_V    = float(P_V)
            P_mbar = float(P_mbar)
            status = int(status)
            tstamp = tstamp.split(' ')
            tstamp = tstamp[0].split('.')[0]
            msg = f'tcp_send(readPressure({channel})) failed with status: {status}'
            if not status:
                return P_V, P_mbar, tstamp
            else:
                print(msg)
                return None, None, None
        else:
            raise RuntimeError(f'Pressure gauge ({channel}) not recognized')
    
    def get_heater_rng_lvl(self, x):
        """
        This function takes the pid output (Current in mA)
        nd converts it to heater settings
        """
        if abs(x - 0) < 1e-8:
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
        # Extended ranges added on 211111
        elif 3.16 < x <= 10:
            Range = 6
            level = x*100/10
        elif 10 < x <= 31.6:
            Range = 7
            level = x*100/31.6
        elif 31.6 < x <= 100:
            Range = 8
            level = x*100/100
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
        _ = self.tcp_recv()


    def set_still_heater(self, voltage):
        """
        Sets the still heater voltage
        """
        # Check that the voltage is zero
        if voltage < 1e-6:
            print('Turning off still heater')
            self.tcp_send(f'setHtrCntrlModeOpenLoop(2, 0, 0)')
            _ = self.tcp_recv()
        else:
            print(f'Turning on still heater with {voltage} V ...')
            self.tcp_send(f'setHtrCntrlModeOpenLoop(2, {voltage}, {max(voltage, 3)})')
            _ = self.tcp_recv()


    def get_pid_ctrl(self, Tset, sample_time=15):
        """
        Returns a simple_pid PID controller object
        """
        # Trying Ku = 70, Tu = 186 s (this is set by LS372 scanning cycle time)
        # 
        # go to https://en.wikipedia.org/wiki/Ziegler%E2%80%93Nichols_method
        # for info on Ku and Tu and how they relate to PID gain settings
        # if self.is_pid_init:
        Ku = 70.
        Tu = 186 #s -- this is close to Nyquist, 2 * 90 s update time
        # Tu = 90#s
        Kp = 0.6 * Ku
        Ki = 1.2 * Ku / Tu
        Kd = 0.075 * Ku * Tu
        self.pid_values = [Kp, Ki, Kp]
        # else:
        #     self.pid_values = pid.components
        #     Kp, Ki, Kp = self.pid_values
        #     Ku = Ki * Tu / 1.2
        #     Tu = Kd / (0.075 * Ku)

        #Temperature setpoint must be in Kelvin!
        #sample_time is in seconds
        stime = self.sample_time if hasattr(self, 'sample_time') \
                else sample_time
        
        print(f'PID controller for {Tset*1e3} mK ...')
        print(f'PID settings:')
        print(f'Ku: {Ku}\tTu: {Tu}')
        print(f'Kp: {Kp}\tKi: {Ki}\tKd: {Kd}')
        pid = simple_pid.PID(Kp, Ki, Kd, setpoint=Tset, sample_time=stime)

        # Set the maximum current
        # T_base = 0.013 #K (approx. base temperature)
        T_base = self.T_base #K (approx. base temperature)

        # Limits the current to 1/3 of the total range?
        Max_Current = 1.33*8.373*(Tset-T_base)**(0.720) #mA
        print(f'Max Curent: {Max_Current} mA')
        pid.output_limits = (0, Max_Current)

        self.pid = pid
        return pid

    
    def temperature_controller(self, tsidx, tidx, Tidx, t, Tset, fid, out):
        # Reset the socket
        self.reset_socket()
        start_thermalize_timer = False

        # Get the pid controller object
        self.get_pid_ctrl(Tset)

        # Dictionary keys for long inputs
        flo_key = 'flow rate [umol/s]'
        Ikey    = 'Iout [mA]'

        # Read the initial temperature
        Z, T, tstamp = self.read_cmn()

        # Get the initial values from all sensors
        Iout = self.pid(T)
        P, I, D = self.pid.components
        _, flow, _ = self.read_flow_meter()

        print(f'Heating to {Tset * 1e3} mK from {T * 1e3} mK ...')
        tin = t

        # Set to 2 % of the target
        T_eps = 0.02 * Tset
        print(f'T_eps: {T_eps * 1e3} mK')
        while 1:
            diff = abs(1e3 * (T - Tset))
            if  diff < (1e3 * T_eps):
                start_thermalize_timer = True
            
            # Wait five more minutes
            if start_thermalize_timer:
                therm_min = self.therm_time / 60.
                therm_time = self.therm_time - self.pid.sample_time * therm_min 

                # Check that the thermalization time - the sample time > 0
                # Otherwise, tell the user to choose a larger therm_time
                assert therm_time > 0, f'therm_time {therm_time} < 0 s!'
                print(f'Starting thermalization timer {therm_min} min ...')
                tstart = time.time()
                while time.time() < (therm_time + tstart):
                    Z, T, tstamp = self.read_cmn()
                    _, flow, _ = self.read_flow_meter()
                    Iout = self.pid(T)
                    P, I, D = self.pid.components

                    # Wait the update time
                    time.sleep(self.pid.sample_time)
                    tin += self.pid.sample_time
                    out[flo_key] = flow
                    out[tsidx] = tstamp
                    out[Tidx] = T * 1e3
                    out[tidx] = tin 
                    out[Ikey] = Iout
                    out['PID'] = [P, I, D]
                    tsout = out[tsidx]
                    tout = out[tidx]
                    Tout = out[Tidx]
                    self.set_current(Iout)
                    print(f'{tstamp}, {1e3 * T:.2f} mK, {flow:.1f} umol/s, {Iout:.2f} mA, {P:.2f}, {I:.2f}, {D:.2f}, {tout} s')
                    fid.write(f'{tsout}, {tout}, {Tout}, {Iout}, {P}, {I}, {D}, {flow}\n')
                print('Thermalization timer done.')

                # Reset the thermalize_timer flag
                start_thermalize_timer = False
                break
            
            # Get the CMN impedance, temperature, and timestamp
            Z, T, tstamp = self.read_cmn()
            if T is not None:
                # Generate the output current
                Iout = self.pid(T)
                P, I, D = self.pid.components

                # Read the flow meter
                _, flow, _ = self.read_flow_meter()

                print(f'{tstamp}, {1e3 * T:.2f} mK, {flow:.1f} umol/s, {Iout:.2f} mA, {P:.2f}, {I:.2f}, {D:.2f}, {tin} s')
                self.set_current(Iout)

                # Read the current values of the PID controller
                P, I, D = self.pid.components
                time.sleep(self.pid.sample_time)
            
                # Write the time stamp, temperature, and impedance to file
                tin += self.sample_time
                out[flo_key] = flow
                out[tsidx] = tstamp
                out[Tidx] = T * 1e3
                out[tidx] = tin + self.therm_time
                out[Ikey] = Iout
                tsout = out[tsidx]
                tout = out[tidx]
                Tout = out[Tidx]
                out['PID'] = [P, I, D]
            else:
                Iout = None
                flow = None
                out[flo_key] = None
                out[tsidx] = None
                out[Tidx] = None
                out[tidx] = None
                out[Ikey] = None
                out['PID'] = None
                tsout = out[tsidx]
                tout = out[tidx]
                Tout = out[Tidx]

            # Write the timestamp, elapsed time, temperature, current,
            # P, I, D values to file
            fid.write(f'{tsout}, {tout}, {Tout}, {Iout}, {P}, {I}, {D}, {flow}\n')

        # Write the outputs for file writing
        out[tsidx] = tstamp
        out[Tidx] = T * 1e3
        out[tidx] = tin
        out[Ikey] = Iout
        out['PID'] = [P, I, D]
        out[flo_key] = flow
        fid.write(f'{tsout}, {tout}, {Tout}, {Iout}, {P}, {I}, {D}, {flow}\n')


    def estimate_init_adaptive_averages(
            self,
            time_per_sweep : float, 
            powers : np.ndarray,
            total_time_hr : float) -> int:
        """
        Estimates the initial number of averages used by the adaptive averaging
        performed by power_sweep
    
        Arguments:
        ---------
    
        time_per_sweep:     time to perform 1 average [s]
        powers:             array of powers in the power sweep [dBm]
        total_time_hr:      total runtime [hr]
    
        """
        # Check that adaptive averaging is active
        assert self.adaptive_averaging, 'Adaptive averaging not set!'

        # Compute the step size and number of points
        stepsize = abs(powers[1] - powers[0])
        print(f'stepsize: {stepsize}')
        Npts = len(powers)
        fac = (10**(stepsize/10))**0.5
        print(f'fac: {fac}')
        sec2hr = 3600.
    
        # Compute the number of averages
        Navg = total_time_hr / sum([time_per_sweep * fac**i / sec2hr
                                    for i in range(Npts)])
    
        return int(np.ceil(Navg))


    def estimate_time_adaptive_averages(
            self,
            time_per_sweep : float, 
            powers : np.ndarray,
            Navg : int) -> float:
        """
        Estimates the initial number of averages used by the adaptive averaging
        performed by power_sweep
    
        Arguments:
        ---------
    
        time_per_sweep:     time to perform 1 average [s]
        powers:             array of powers in the power sweep [dBm]
        total_time_hr:      total runtime [hr]
    
        """
        # Check that adaptive averaging is active
        assert self.adaptive_averaging, 'Adaptive averaging not set!'

        # Compute the step size and number of points
        stepsize = abs(powers[1] - powers[0])
        print(f'stepsize: {stepsize}')
        Npts = len(powers)
        fac = (10**(stepsize/10))**0.5
        print(f'fac: {fac}')
        sec2hr = 3600.
    
        # Compute the number of averages
        total_time_hr = Navg * sum([time_per_sweep * fac**i / sec2hr
                                    for i in range(Npts)])

        return total_time_hr


    def pna_process(self, idx, Tset, out, prefix='M3D6_02_WITH_1SP_INP',
                    adaptive_averaging=True, cal_set=None, setup_only=False,
                    close_socket_start=True, segments=None):
        """
        Performs a PNA measurement
        """
        if close_socket_start:
            self.close_socket()

        # Get the temperature from the temperature controller
        temp = Tset * 1e3 #mk
        sampleid = f'{prefix}_{self.dstr}' 
        pstr = f'{prefix}_{int(self.vna_startpower)}_{int(self.vna_endpower)}dBm'

        # Preparing to measure frequencies, powers
        if self.vna_numsweeps > 1:
            powers = np.linspace(self.vna_startpower, self.vna_endpower,
                                 self.vna_numsweeps)
            print(f'Measuring at {self.vna_centerf} GHz')
            print(f'IFBW: {self.vna_ifband} kHz')
            print(f'Span: {self.vna_span} MHz')
            print(f'Npoints: {self.vna_points}')
            print(f'Nsweeps: {self.vna_numsweeps}')
            print(f'Powers: {powers} dBm')

            # Note: PNA power sweep assumes the outputfile has .csv as its last
            # four characters and removes them when manipulating strings and
            # directories
            outputfile = sampleid+'_'+str(self.vna_centerf)+'GHz'
            PNA.power_sweep(self.vna_startpower, self.vna_endpower,
                    self.vna_numsweeps, self.vna_centerf, self.vna_span, temp,
                    self.vna_averages, self.vna_edelay, self.vna_ifband,
                    self.vna_points, outputfile, sparam=self.sparam, 
                    meastype=pstr,
                    adaptive_averaging=adaptive_averaging,
                    cal_set=cal_set,
                    setup_only=setup_only,
                    segments=segments)

        else:
            outputfile = sampleid+'_'+str(self.vna_centerf)+'GHz'
            PNA.get_data(centerf = self.vna_centerf,
                         span = self.vna_span,
                         temp = temp,
                         averages = self.vna_averages,
                         power = self.vna_startpower,
                         edelay = self.vna_edelay,
                         ifband = self.vna_ifband,
                         points = self.vna_points,
                         outputfile = outputfile,
                         sparam = self.sparam,
                         cal_set = calset,
                         instr_addr = self.vna_addr)

        out[idx] = 0


    def broadband_sweep(self, sampleid, power, frequency_band=[4e9, 8e9],
                        max_pts=65001, frequency_chunk_size=1e9, cal_set=None):
        """
        Performs a broadband frequency sweep in 1 GHz chunks
        """
        # Read the CMN temperature
        Z, temp, timestamp = self.read_cmn()

        # Determine the number of chunks to measure
        df = frequency_band[1] - frequency_band[0]
        Nchunks = int(np.floor(df / frequency_chunk_size))
        print(f'Sweeping a {df / 1e9} GHz band in {Nchunks} chunks ...')
        f0 = frequency_band[0] / 1e9
        fdata = np.array([])
        smag = np.array([])
        sph = np.array([])
        for n in range(Nchunks):
            # Set the center frequency
            centerf = f0 + 0.5 * frequency_chunk_size / 1e9
            print(f'{f0} GHz to {f0 + (frequency_chunk_size/1e9)} GHz ...')
            PNA.get_data(centerf = centerf,
                         span = frequency_chunk_size / 1e6, # MHz
                         temp = temp * 1e3,
                         averages = self.vna_averages,
                         power = power, 
                         edelay = self.vna_edelay,
                         ifband = self.vna_ifband,
                         points = max(max_pts, self.vna_points), 
                         outputfile = sampleid+'.csv',
                         sparam = self.sparam,
                         cal_set = cal_set,
                         instr_addr = self.vna_addr)

            # Move to the next starting position
            f0 += frequency_chunk_size / 1e9

            # Read data and append to larger data set
            fname = f'{sampleid}_{freq:.3f}GHz_{power:.0f}dB_{temp:.0f}mK'
            fname = filename.replace('.','p')
            data = np.genfromtxt(fname, delimiter=',').T
            fdata = np.hstack((fdata, data[0]))
            smag = np.hstack((smag, data[1]))
            sph = np.hstack((sph, data[2]))

        f0 = frequency_band[0] / 1e9; f1 = frequency_band[1] / 1e9
        fname = f'{sampleid}_{f0:.3f}_{f1:.3f}GHz_{power:.0f}dB_{temp:.0f}mK.csv'
        print(f'Writing all data to {fname_all} ...')
        with open(fname, 'w') as fid:
            fid.write('\n'.join([f'{ff}, {sm}, {sp}' 
                            for ff, sm, sp in zip(fdata, smag, sph)]))


    def run_temp_sweep(self, measure_vna=False, prefix='M3D6_02_WITH_1SP_INP'):
        """
        Execute the temperature sweep
        """
        # Set the output filename and write the results with
        # standard text file IO operations
        dstr =  self.dstr
        flo_key = 'flow rate [umol/s]'

        # Set the log filename, log the temperature, time stamp, and time
        fname = f'logs/temperature_mK_log_{dstr}.csv'
        hdr = '# Time[HH:MM:SS], Time [s], Temperature [mK], Current [mA], P, I, D, Flow Rate [umol/s]\n'

        # Check that file does not exist before writing header
        if not os.path.isfile(fname):
            write_hdr = True
        else:
            write_hdr = False

        # Open file and start logging time stamps, elapsed time,
        # temperature
        with open(fname, 'a') as fid:
            if write_hdr:
                fid.write(hdr)

            # Iterate over the temperatures
            for Tset in self.T_sweep_list:
                # Scale the temperature to mK
                T = 10 * Tset
                t = 0
            
                # Flag to start measurement run
                meas_ret = 1
            
                # Continue to run the PID controller as the measurement runs
                try:
                    while meas_ret:
                        # Measure the temperature, set the current, write
                        # results to file
                        out = {}
                        self.temperature_controller('tstamp [HH:MM:SS]',
                                                    't [s]', 'T [mK]', t,
                                                    Tset, fid, out) 
                        print(f'out:\n{out}')
                        # Update the time stamp, elapsed time, temperature
                        flow = out[flo_key]
                        tsout = out['tstamp [HH:MM:SS]']
                        tout = out['t [s]']
                        Tout = out['T [mK]']
                        Iout = out['Iout [mA]']
                        P, I, D = out['PID']
                        data = f'{tsout}, {tout}, {Tout}, {Iout}, {P}, {I}, {D}, {flow}'
                        fid.write(data + '\n')

                        # Start the PNA measurement
                        if measure_vna:
                            print('Starting PNA measurement ...')
                            self.pna_process('meas', Tset, out, prefix=prefix)

                        # Update the time stamp, elapsed time, temperature
                        print(f'out:\n{out}')
                        tsout = out['tstamp [HH:MM:SS]']
                        tout = out['t [s]']
                        Tout = out['T [mK]']
                        Iout = out['Iout [mA]']
                        P, I, D = out['PID']
                        data = f'{tsout}, {tout}, {Tout}, {Iout}, {P}, {I}, {D}, {flow}'
                        fid.write(data + '\n')

                        print('Finished PNA Measurement.')
                        meas_ret = 0
                        break

                # Graceful exit on Ctrl-C interrupt by the user
                except (KeyboardInterrupt, Exception) as ex:
                    # Write the last result to file
                    flow = out[flo_key]
                    tsout = out['tstamp [HH:MM:SS]']
                    tout = out['t [s]']
                    Tout = out['T [mK]']
                    Iout = out['Iout [mA]']
                    P, I, D = out['PID']
                    data = f'{tsout}, {tout}, {Tout}, {Iout}, {P}, {I}, {D}, {flow}'
                    fid.write(data + '\n')
                    fid.close()
                    print('\n\n-----------------')
                    print(f'Exception:\n{ex}')
                    print('-----------------\n')
                    print('Setting current to 0 ...')
                    if self.socket is not None:
                        self.set_current(0.)
                    else:
                        self.reset_socket()
                        self.set_current(0.)
                    break


    def pna_sweep_multiple_resonators(self, freqs : list, ifbws : list,
            delays : list, spans : list, start_powers : list, 
            end_powers : list, num_powers : list, sample_name : str, avgs :
            list, pts : int=2001, sparam : str='S12'):
        """
        Sweeps the VNA through multiple resonators a fixed or multiple powers
        """
        # Initial measurement setup
        self.sparam = sparam
        self.vna_points = pts

        # Iterate over each frequency
        for idx, f in enumerate(freqs):
            self.vna_edelay = delays[idx]
            self.vna_ifband = ifbws[idx]
            self.vna_centerf = f
            self.vna_averages = avgs[idx]
            self.vna_span = spans[idx]
            self.vna_numsweeps = num_powers[idx]
            self.vna_startpower = start_powers[idx]
            self.vna_endpower = end_powers[idx]
            print(f'Running {sparam} measurement of {f:.2f} GHz resonator ...')
            Z, T, tstamp = self.read_cmn()
            out = {}
            self.pna_process('meas', T, out, prefix=sample_name)

        print(f'PNA multiple resonator measurements complete.')

        self.set_current(0.)
        Z, T, tstamp = self.read_cmn()
        print(f'{tstamp}, {Z} ohms, {T*1e3} mK')


def multiple_resonator_driver(Jctrl : JanisCtrl):
    """
    Script to drive the above function using an instance of the JanisCtrl class
    """
    # Set the sample name
    sample_name = 'RGREF01_01'

    # Set the frequencies
    # freqs = [4.22383,   4.6247,     5.0256, 5.431545, 5.8265775,
    #          6.2312162, 6.6399276, 6.9799, 7.435832, 7.8459646]
    # freqs = [4.22383,   4.6247,     5.0256, 5.8265775,
    #          6.2312162, 6.6399276, 6.9799, 7.435832, 7.8459646]
    # freqs = [4.6247,     5.0256, 5.8265775,
    #          6.2312162, 6.6399276, 6.9799, 7.435832, 7.8459646]
    # freqs = [5.0256, 5.8265775,
    #          6.2312162, 6.6399276, 6.9799, 7.435832, 7.8459646]
    # freqs = [6.6399276, 6.9799, 7.435832, 7.8459646]
    freqs = [5.431545]
    flen = len(freqs)

    # Set the powers
    # start_powers = [15] * flen
    # end_powers = [0] * flen
    # num_powers = [4] * flen
    # start_powers = [-5] * flen
    # end_powers = [-50] * flen
    # num_powers = [10] * flen
    # start_powers = [-50] * flen
    # end_powers = [-70] * flen
    # num_powers = [5] * flen
    start_powers = [-95] * flen
    end_powers = [-95] * flen
    num_powers = [2] * flen

    # Set the frequency spans
    # spans = [0.2, 0.2, 0.2, 0.2, 0.2,
    #          0.2, 0.2,  10., 0.2, 0.2]
    # delays = [60.93,  61., 61.86, 61.85, 61.833,
    #           61.823, 61.82, 61.5, 61.665, 61.672]
    # spans = [0.2, 0.2, 0.2, 0.2, 
    #          0.2, 0.2,  10., 0.2, 0.2]
    # delays = [60.93,  61., 61.86, 61.833,
    #           61.823, 61.82, 61.5, 61.665, 61.672]
    # spans = [0.2, 0.2, 0.2, 
    #          0.2, 0.2,  10., 0.2, 0.2]
    # delays = [61., 61.86, 61.833,
    #           61.823, 61.82, 61.5, 61.665, 61.672]
    # spans = [0.2, 0.2, 
    #          0.2, 0.2,  10., 0.2, 0.2]
    # delays = [61.86, 61.833,
    #           61.823, 61.82, 61.5, 61.665, 61.672]
    # spans = [0.2,  10., 0.2, 0.2]
    # delays = [61.82, 61.5, 61.665, 61.672]
    spans = [0.2]
    delays = [61.85]

    # Set the IF bandwidths and number of averages
    ifbws = [1.] * flen
    # avgs = [3.] * flen
    # avgs = [50.] * flen
    # avgs = [600.] * flen
    avgs = [3000.] * flen

    # Call the class function
    Jctrl.pna_sweep_multiple_resonators(freqs, ifbws, delays, spans,
            start_powers, end_powers, num_powers, sample_name,
            avgs, pts=1001, sparam='S12')


if __name__ == '__main__':
    # Iterate over a list of temperatures
    # 30 mK -- 300 mK, 10 mK steps
    Tstart = 0.03; Tstop = 0.315; dT = 0.015
    # Tstart = 0.255; Tstop = 0.315; dT = 0.015
    # Tstart = 0.150; Tstop = 0.350; dT = 0.1
    # sample_time = 15; T_eps = 0.001 -- up to 240 mK
    sample_time = 15; T_eps = 0.0025 # -- 255 mK and up
    # therm_time  = 5. * 60. # wait an extra 5 minutes to thermalize
    therm_time  = 300. # wait an extra 5 minutes to thermalize

    # Setup the temperature controller class object
    Jctrl = JanisCtrl(Tstart, Tstop, dT,
            sample_time=sample_time, T_eps=T_eps,
            therm_time=therm_time,
            # init_socket=True, bypass_janis=True)
            init_socket=True, bypass_janis=False)

    # Mines 3D #3, bare
    Jctrl.vna_centerf = 8.0214305
    Jctrl.vna_span = 0.5 # MHz
    # Jctrl.vna_edelay = 0.969 #ns
    Jctrl.vna_edelay = 1.039 #ns

    Jctrl.vna_points = 1001

    # Temperature sweep settings
    Jctrl.sparam = 'S12'
    cal_set = 'CryoCal_2SP_INP_8p02G_20220712'

    # First sweep, int power
    Jctrl.vna_averages = 5000
    Jctrl.vna_ifband = 0.1 #khz
    Jctrl.vna_numsweeps = 3
    # Jctrl.vna_startpower = -70
    # Jctrl.vna_endpower = -85
    Jctrl.vna_startpower = -85
    Jctrl.vna_endpower = -95

    Jctrl.print_class_members()

    # Run the temperature sweep from within the class
    Jctrl.set_current(0.)
    Z, T, tstamp = Jctrl.read_cmn()
    print(f'{tstamp}, {Z} ohms, {T*1e3} mK')
    flow_V, flow_umol_s1, tstamp = Jctrl.read_flow_meter()
    print(f'{tstamp}, {flow_V} V, {flow_umol_s1} umol / s')
    Jctrl.read_temp('all')
    Jctrl.read_pressure('all')

    # Mines 3D cavity 9.2 GHz with, without InP
    # sample_name = 'M3D6_02_WITH_2SP_INP'
    # out = {}
    # Jctrl.pna_process('meas', T, out, prefix=sample_name,
    #                 adaptive_averaging=True,
    #                 cal_set=cal_set)
    # Jctrl.set_still_heater(0.4)

    # NYU 2D resonator, Al on InP
    # sample_name = 'NYU2D_AL_INP'
    # Jctrl.run_temp_sweep(measure_vna=True, prefix=sample_name)
    Jctrl.set_current(0.)
    Z, T, tstamp = Jctrl.read_cmn()
    # print(f'{tstamp}, {Z} ohms, {T*1e3} mK')
