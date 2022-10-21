# -*- encoding: utf-8 -*-
"""
User file for controlling the Janis and PNA instruments

    Make sure you login to the JetWay session on PuTTY
    with user: 'bco' and password: 'aish8Hu8'

"""
import sys
sys.path.append(r'C:\Users\Lehnert Lab\GitHub\measurement\temperature_control')
from janis_ctrl import JanisCtrl
import numpy as np
import time

"""
Temperature sweep settings
XXX: Only change if performing a temperature sweep!
"""

def measure_multiple_resonators(fcs, spans, delays, powers,
        ifbw=1., sparam='S21', npts=1001,
        adaptive_averaging=True, sample_name='',
        runtime=1., cal_set = None):
    """
    Measures multiple resonators sequentially
    """
    # Example inputs to run a temperature sweep
    # Iterate over a list of temperatures
    # 30 mK -- 300 mK, 10 mK steps
    Tstart = 0.03; Tstop = 0.315; dT = 0.015
    sample_time = 15; T_eps = 0.0025 # -- 255 mK and up
    therm_time  = 300. # wait an extra 5 minutes to thermalize
    p1 = powers[0]
    p2 = powers[-1]
    power_steps = len(powers)

    # Delay the start of a sweep by Nstart hours
    h2s = 3600.
    start_delay = 0. # hr

    for fc, span, delay in zip(fcs, spans, delays):

        # Create the JanisCtrl 
        print(f'Measuring {sample_name} at {fc} GHz ...')
        Jctrl = JanisCtrl(Tstart, Tstop, dT,
                sample_time=sample_time, T_eps=T_eps,
                therm_time=therm_time,
                init_socket=True, bypass_janis=False,
                adaptive_averaging=adaptive_averaging)

        """
        Change these settings for each power sweep
        """
        # Jctrl.vna_edelay = 62.06 #ns
        # Jctrl.vna_edelay = 62.275 #ns

        Jctrl.vna_centerf = fc # GHz
        Jctrl.vna_span = span # MHz
        Jctrl.vna_edelay = delay #ns
        Jctrl.vna_points = npts
        Jctrl.sparam = sparam
        Jctrl.vna_ifband = ifbw
        Jctrl.vna_startpower = p1 # dBm
        Jctrl.vna_endpower = p2 # dBm
        Jctrl.vna_numsweeps = power_steps 

        """
        Initial number of averages
        """
        # Only used if adaptive_averaging == False
        Jctrl.vna_averages = 1


        time_per_sweep = Jctrl.vna_points / (1e3 * Jctrl.vna_ifband)
        print(f'powers: {powers}')

        """
        Expected runtime for power sweep
        if using the estimated runtime option
        """
        total_time_hr = runtime
        if Jctrl.adaptive_averaging:
            Navg_adaptive = Jctrl.estimate_init_adaptive_averages(
                               time_per_sweep, 
                               powers,
                               total_time_hr)
            Jctrl.vna_averages = Navg_adaptive



        # Read the MXC temperature from the CMN
        Z, T, tstamp = Jctrl.read_cmn()
        print(f'{tstamp}, {Z} ohms, MXC CMN: {T*1e3:.2f} mK')
        
        # Read the flow rate
        flow_V, flow_umol_s1, tstamp = Jctrl.read_flow_meter()
        print(f'{tstamp}, {flow_V} V, {flow_umol_s1:.2f} umol / s')
        
        # Read and report all temperatures a pressures
        Jctrl.read_temp('all')
        Jctrl.read_pressure('all')
        
        # Enter a sample name and perform the PNA power sweep
        ## Note: adaptive_averaging will increase the averages
        ##       by a factor 10^(dp / 10), where dp is the power
        ##       step in the power sweep -- ~1.7 for dp = 5 dBm
        print(f'Delayed start of {start_delay} hr ...')
        out = {}
        Jctrl.pna_process('meas', T, out, prefix=sample_name,
                          adaptive_averaging=adaptive_averaging,
                          cal_set=cal_set, setup_only=False)
        del Jctrl


# Set the center frequencies, spans, delays, powers
fcs = [6.1122462, 6.497274]
# fcs = [6.497274]
fcs = [# 6.58781,
       # 6.68991,
       7.00279,
       7.3418425]
#spans = [0.2, 0.5]
spans = [# 50.,
        # 30.,
        5.,
        15.]
# delays = [62.06, 62.275]
delays = [62.18] * 4 
powers_hi = np.linspace(-15, -75, 13)
powers_lo = np.linspace(-80, -95, 6)
powers = np.linspace(-15, -95, 41)
# powers = np.hstack((powers_hi, powers_lo))
print(f'powers:\n{powers}')

# Change the sample name
sample_name = 'NYU_AL_INP'

measure_multiple_resonators(fcs, spans, delays, powers,
        ifbw=1., sparam='S21', npts=1001,
        adaptive_averaging=True, sample_name=sample_name,
        runtime=24., cal_set = None)
