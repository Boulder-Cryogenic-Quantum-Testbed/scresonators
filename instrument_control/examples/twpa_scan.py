# -*- coding : utf-8 -*-
"""
Anritsu MG3692C Signal Generator control test file

Author: Nick Materise, Kyle Thompson
Date:   220728

"""

import sys
sys.path.append(r'C:\Users\Lehnert Lab\GitHub\measurement\instrument_control')
from anritsu import AnritsuCtrl
import numpy as np
import sys

# Color printing
RED   = '\033[31m'
GREEN = '\033[32m'
CRST  = '\033[0m'


anritsu = AnritsuCtrl()
sweep_freqs = np.linspace(6.3, 7.3, 101)
sweep_powers = np.linspace(-25., -75., 101)
vna_dict = {'sample_id' : 'RGTI_TWPA',
            'centerf' : 6.,
            'span' : 4000.,
            'temp' : 45.,
            'avg' : 3,
            'power' : -25.,
            'edelay' : 69.3,
            'ifbw' : 100.,
            'npts' : 20001,
            'sparam' : 'S21',
            'cal_set' : None}
anritsu.power_frequency_sweep_2d(sweep_powers,
                                 sweep_freqs,
                                 sweep_order='frequency_power',
                                 run_vna=True,
                                 vna_dict=vna_dict)
