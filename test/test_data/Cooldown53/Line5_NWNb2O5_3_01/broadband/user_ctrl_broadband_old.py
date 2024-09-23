# -*- encoding: utf-8 -*-
"""
User file for controlling the Janis and PNA instruments

    Make sure you login to the JetWay session on PuTTY
    with user: 'bco' and password: 'aish8Hu8'

"""
import sys
sys.path.append(r'C:\Users\Lehnert Lab\GitHub\bcqt-ctrl\temperature_control')
from janis_ctrl import JanisCtrl, measure_multiple_resonators
import numpy as np
import time
from datetime import datetime

"""
Temperature sweep settings
XXX: Only change if performing a temperature sweep!
"""

def old_measure_multiple_resonators(fcs, spans, delays, powers,
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
        #print(';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;')
        #print(datetime.now().strftime("%H:%M:%S"))
        # Create the JanisCtrl 

        Jctrl = JanisCtrl(Tstart, Tstop, dT,
                sample_time=sample_time, T_eps=T_eps,
                therm_time=therm_time,
                init_socket=True, bypass_janis=False,
                adaptive_averaging=adaptive_averaging,
                data_dir="5to6_data", output_file="test")

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

# Change the sample name
sample_name = 'NWNb2O5_3_01'

# Set the center frequencies, spans, delays, powers
fcs = [5.9, 6.4]
spans = [500]*len(fcs)
delays = [75.905]*len(fcs)
# powers = np.linspace(-30, -35, 2)
powers = [-5]
n_avgs = 10
file_names = [f"{sample_name}_{fc-span*1e-3:0.3f}GHz_{fc+span*1e-3:0.3f}GHz_{power:.02f}dB".replace(".","p") for fc, span, power in zip(fcs, spans, powers)]
data_dir = [f"{sample_name}_{fc-span*1e-3:0.3f}GHz_{fc+span*1e-3:0.3f}GHz" for fc, span in zip(fcs, spans)]

measure_multiple_resonators(fcs, spans, delays, powers,
                ifbw=1000., sparam='S21', npts=30001,
                adaptive_averaging=False, sample_name=sample_name,
                runtime=0., cal_set=None, data_dirs=data_dir, file_names=file_names,
                Navg_init = n_avgs, is_segmented=False)

