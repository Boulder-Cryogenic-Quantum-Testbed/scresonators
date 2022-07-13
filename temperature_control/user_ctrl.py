# -*- encoding: utf-8 -*-
"""
User file for controlling the Janis and PNA instruments
"""

from janis_ctrl import JanisCtrl
import numpy as np

"""
Temperature sweep settings
"""
# Example inputs to run a temperature sweep
# Iterate over a list of temperatures
# 30 mK -- 300 mK, 10 mK steps
Tstart = 0.03; Tstop = 0.315; dT = 0.015
sample_time = 15; T_eps = 0.0025 # -- 255 mK and up
therm_time  = 300. # wait an extra 5 minutes to thermalize

# Setup the temperature controller class object
## Note: bypass_janis runs the JanisCtrl class without
##       communicating with the JACOB and can only
##       perform measurements with the PNA
adaptive_averaging = True
Jctrl = JanisCtrl(Tstart, Tstop, dT,
        sample_time=sample_time, T_eps=T_eps,
        therm_time=therm_time,
        init_socket=True, bypass_janis=False,
        adaptive_averaging=adaptive_averaging)

"""
Change these settings for each power sweep
"""
# Set the PNA inputs for the power sweep
# Jctrl.vna_centerf = 6.23698 # GHz
Jctrl.vna_centerf = 8.0214305 # GHz
Jctrl.vna_span = 0.5 # MHz
Jctrl.vna_edelay = 0.9773 #ns
Jctrl.vna_points = 1001
Jctrl.sparam = 'S12'
Jctrl.vna_ifband = 1 #khz
Jctrl.vna_startpower = -25 # dBm
Jctrl.vna_endpower = -75 # dBm
Jctrl.vna_numsweeps = 11 
# Jctrl.vna_startpower = -89 # dBm
# Jctrl.vna_endpower = -89 # dBm
# Jctrl.vna_numsweeps = 2
time_per_sweep = Jctrl.vna_points / (1e3 * Jctrl.vna_ifband)
powers = np.linspace(Jctrl.vna_startpower,
                    Jctrl.vna_endpower,
                    Jctrl.vna_numsweeps)
print(f'powers: {powers}')
total_time_hr = 6.
Navg_adaptive = Jctrl.estimate_init_adaptive_averages(
                    time_per_sweep, 
                    powers,
                    total_time_hr)
print(f'Number of averages: {Navg_adaptive}')
# Jctrl.vna_averages = 1000
Jctrl.vna_averages = Navg_adaptive

# sample_name = 'RGSI002_A1g7_6p23698_GHz'
sample_name = 'M3D6_02_WITH_2SP_INP_CRYOCAL'
cal_set = 'CryoCal_2SP_INP_8p02G_20220712'
# cal_set = None

# Print the JanisCtrl class members
Jctrl.print_class_members()

# Set the MXC current to 0 mA
Jctrl.set_current(0.)

# Read the MXC temperature from the CMN
Z, T, tstamp = Jctrl.read_cmn()
print(f'{tstamp}, {Z} ohms, {T*1e3} mK')

# Read the flow rate
flow_V, flow_umol_s1, tstamp = Jctrl.read_flow_meter()
print(f'{tstamp}, {flow_V} V, {flow_umol_s1} umol / s')

# Read and report all temperatures a pressures
Jctrl.read_temp('all')
Jctrl.read_pressure('all')

# Enter a sample name and perform the PNA power sweep
## Note: adaptive_averaging will increase the averages
##       by a factor 10^(dp / 10), where dp is the power
##       step in the power sweep -- ~1.7 for dp = 5 dBm
out = {}
Jctrl.pna_process('meas', T, out, prefix=sample_name,
                  adaptive_averaging=adaptive_averaging,
                  cal_set=cal_set, setup_only=False)

"""
# Temperature sweep, comment out the pna_process above
"""
# Jctrl.run_temp_sweep(measure_vna=True, prefix=sample_name)

# Set the MXC heat to 0 mA just to be safe
# and report the temperature on the CMN before exiting
Jctrl.set_current(0.)
Z, T, tstamp = Jctrl.read_cmn()
print(f'{tstamp}, {Z} ohms, {T*1e3} mK')
