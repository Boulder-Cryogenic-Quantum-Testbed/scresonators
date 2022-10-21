# -*- encoding: utf-8 -*-
"""
User file for controlling the Janis and PNA instruments

    Make sure you login to the JetWay session on PuTTY
    with user: 'bco' and password: 'aish8Hu8'

"""
import sys

# Change to the path where scresonators is installed
sys.path.append(r'C:\Users\Lehnert Lab\GitHub\measurement\temperature_control')
from janis_ctrl import JanisCtrl
import numpy as np
import time

"""
Temperature sweep settings
XXX: Only change if performing a temperature sweep!
"""
# Example inputs to run a temperature sweep
# Iterate over a list of temperatures
# 30 mK -- 300 mK, 10 mK steps
Tstart = 0.03; Tstop = 0.315; dT = 0.015
sample_time = 15; T_eps = 0.0025 # -- 255 mK and up
therm_time  = 300. # wait an extra 5 minutes to thermalize

"""
Use adaptive averaging, after high power sweep(s)
"""
Jctrl = JanisCtrl(Tstart, Tstop, dT,
        sample_time=sample_time, T_eps=T_eps,
        therm_time=therm_time,
        init_socket=True, bypass_janis=False,
        adaptive_averaging=False)

# Read the MXC temperature from the CMN
Z, T, tstamp = Jctrl.read_cmn()
print(f'{tstamp}, {Z} ohms, MXC CMN: {T*1e3} mK')

# Read the flow rate
flow_V, flow_umol_s1, tstamp = Jctrl.read_flow_meter()
print(f'{tstamp}, {flow_V} V, {flow_umol_s1} umol / s')

# Read and report all temperatures a pressures
Jctrl.read_temp('all')
Jctrl.read_pressure('all')
