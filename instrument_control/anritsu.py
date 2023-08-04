# -*- coding: utf-8 -*-
"""
Anritsu MG3692C Signal Generator control

Author: Nick Materise, Kyle Thompson
Date:   220728

"""

import pyvisa
import sys
sys.path.append(r'C:\Users\Lehnert Lab\GitHub\measurement\pna_control')
import pna_control as pna
import numpy as np
import datetime

class AnritsuCtrl(object):
    """
    Class that implements the Anritsu SCPI control
    """
    def __init__(self, *args, **kwargs):
        """
        Class constructor
        """
        # Default instrument addresses GPIB, TCPIP
        self.anritsu_addr = 'GPIB::5::INSTR'
        self.vna_addr = 'TCPIP0::K-N5222B-21927::hislip0,4880::INSTR'

        # Open the pyvisa resource manager 
        self.rm = pyvisa.ResourceManager()
        self.dstr = datetime.datetime.today().strftime('%y%m%d')

        # Set the precision on the frequency string
        self.fndigits = 3

        # Update the arguments and the keyword arguments
        # This will overwrite the above defaults with the user-passed kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Open the Anritsu instrument object
        self.resource = self.rm.open_resource(self.anritsu_addr)

    def __del__(self):
        """
        Deconstructor to free resources
        """
        if self.rm:
            self.rm.close()

    def print_class_members(self):
        """
        Prints all members in the class
        """
        for k, v in self.__dict__.items():
            print(f'{k} : {v}')

    def vna_process(self, vna_dict : dict, suffix : str = None):
        """
        Performs a PNA measurement

        Parameters:
        ----------

        vna_dict :dict:     dictionary of inputs to the VNA
                            'sample_id' : output file prefix
                            'centerf'   : center frequency [GHz]
                            'span'      : frequency span [MHz]
                            'temp'      : temperature [mK]
                            'avg'       : number of averages
                            'power'     : output power [dBm]
                            'edelay'    : electrical delay [ns]
                            'ifbw'      : IF bandwidth [kHz]
                            'npts'      : number of sample points
                            'sparam'    : S-parameter 'S12', 'S21'
                            'cal_set'   : calibration set
    
        """
        # Get the temperature from the temperature controller
        prefix = vna_dict['sample_id']
        if suffix:
            sampleid = f'{prefix}_{self.dstr}_{suffix}' 
        else:
            sampleid = f'{prefix}_{self.dstr}' 

        # Note: PNA power sweep assumes the outputfile has .csv as its last
        # four characters and removes them when manipulating strings and
        # directories
        outputfile = sampleid+'_'+str(vna_dict['centerf'])+'GHz'
        pna.get_data(centerf = vna_dict['centerf'],
                     span = vna_dict['span'],
                     temp = vna_dict['temp'],
                     averages = vna_dict['avg'],
                     power = vna_dict['power'],
                     edelay = vna_dict['edelay'],
                     ifband = vna_dict['ifbw'],
                     points = vna_dict['npts'],
                     outputfile = outputfile,
                     sparam = vna_dict['sparam'],
                     cal_set = vna_dict['cal_set'],
                     instr_addr = self.vna_addr)

    def write_check(self, cmd : str):
        """
        Writes a command `cmd` and checks for errors
        """
        self.resource.write(cmd)
        err = self.resource.query(':SYST:ERR?')

        # Check that there were no errors
        status, description, = err.split(',')
        status = int(status)

        assert not status, f'Error: {description}'

    def read_check(self, cmd : str, fmt =  float):
        """
        Sends a query command `cmd` and checks for errors
        """
        ret = self.resource.query(cmd)
        err = self.resource.query(':SYST:ERR?')

        # Check that there were no errors
        status, description, = err.split(',')
        status = int(status)

        assert not status, f'Error: {description}'

        return fmt(ret)

    def frequency_sweep(self, sweep_freqs : list, power : float,
            run_vna : bool = False, vna_dict : dict = None):
        """
        Performs sweep from sweep_freqs, at a fixed power

        Parameters:
        ----------

        sweep_freqs     :list:    list of frequencies [GHz]
        power           :float:   fixed power [dBm]
        run_vna         :bool:    run the VNA measurement
        vna_dict        :dict:    parameters to pass to VNA 

        """
        # Set the power
        self.write_check(f'SOUR:POW:LEV:IMM:AMPL {power} dBm')
        print(f'Sweeping frequencies {sweep_freqs} GHz at {power} dBm ...')

        fndigits = self.fndigits

        # Iterate over all frequencies
        for freq in sweep_freqs:
            print(f'Measuring with {freq} GHz ...')
            self.write_check(f'SOUR:FREQ:CW {freq} GHZ') 
            is_output_on = self.read_check('OUTP:STAT?', fmt=int)
            if not is_output_on:
                self.write_check('OUTP:STAT ON')
            if run_vna and vna_dict:
                fout = f'pump_{freq:.{fndigits}f}_GHz_{power}_dBm'
                fout = fout.replace('.', 'p')
                self.vna_process(vna_dict, suffix=fout)

        # Turn off power at the end of the sweep
        self.write_check('OUTP:STAT OFF')

    def power_sweep(self, sweep_powers : list, freq : float,
            run_vna : bool = False, vna_dict : dict = None):
        """
        Performs sweep of power in sweep_powers, at a fixed frequency

        Parameters:
        ----------

        sweep_powers    :list:    list of powers [dBm]
        freq            :float:   fixed frequency [GHz]
        run_vna         :bool:    run the VNA measurement
        vna_dict        :dict:    parameters to pass to VNA 

        """
        # Set the power
        self.write_check(f'SOUR:FREQ:CW {freq} GHZ') 

        print(f'Sweeping powers {sweep_powers} dBm at {freq} GHz ...')
        fndigits = self.fndigits

        # Iterate over all frequencies
        for power in sweep_powers:
            print(f'Measuring with {power} dBm ...')
            self.write_check(f'SOUR:POW:LEV:IMM:AMPL {power} dBm')
            is_output_on = self.read_check('OUTP:STAT?', fmt=int)
            if not is_output_on:
                self.write_check('OUTP:STAT ON')
            if run_vna and vna_dict:
                fout = f'pump_{freq:.{fndigits}f}_GHz_{power}_dBm'
                fout = fout.replace('.', 'p')
                self.vna_process(vna_dict, suffix=fout)

        # Turn off power at the end of the sweep
        self.write_check('OUTP:STAT OFF')

    def power_frequency_sweep_2d(self, sweep_powers : list,
                                sweep_freqs : list,
                                sweep_order : str = 'power_frequency',
                                run_vna : bool = False,
                                vna_dict : dict = None):
        """
        Performs 2D power and frequency from two lists
        
        Parameters:
        ----------

        sweep_powers    :list:    list of powers [dBm]
        sweep_freqs     :list:    list of frequencies [GHz]
        sweep_order     :str:     'power_frequency'
        run_vna         :bool:    run the VNA measurement
        vna_dict        :dict:    parameters to pass to VNA 

        """
        # Check for the sweep order flag
        if sweep_order == 'power_frequency':
            for power in sweep_powers:
                self.frequency_sweep(sweep_freqs, power,
                                     run_vna=run_vna, vna_dict=vna_dict)
        elif sweep_order == 'frequency_power':
            for freq in sweep_freqs:
                self.power_sweep(sweep_powers, freq,
                                 run_vna=run_vna, vna_dict=vna_dict)
        else:
            raise ValueError(f'Sweep order {sweep_order} not recognized.')
