# -*- coding : utf-8 -*-
"""
Anritsu MG3692C Signal Generator control test file

Author: Nick Materise, Kyle Thompson
Date:   220728

"""

import sys
# sys.path.append(r'C:\Users\Lehnert Lab\GitHub\measurement\instrument_control')
# Need to test relative path, otherwise try the above
sys.path.append(r'../')
from anritsu import AnritsuCtrl
import numpy as np
import sys

# Color printing
RED   = '\033[31m'
GREEN = '\033[32m'
CRST  = '\033[0m'

def check_test(ret, name):
    """
    Check if a test returned correct or not
    """

    # Print success
    if ret:
        print('>>> ' + name + GREEN + ' PASSED' + CRST)

    # Print failure
    else:
        print('>>> ' + name + RED + ' FAILED' + CRST)

def test_instantiate_class():
   anritsu = AnritsuCtrl()
   anritsu.print_class_members()
   return True

def test_power_sweep():
    anritsu = AnritsuCtrl()
    sweep_powers = [-5, -10, -15]
    freq = 5.
    anritsu.power_sweep(sweep_powers, freq,
           run_vna=False, vna_dict=None)
    return True

def test_frequency_sweep():
    anritsu = AnritsuCtrl()
    sweep_freqs = [5, 10, 15]
    power = -35.
    anritsu.frequency_sweep(sweep_freqs, power,
           run_vna=False, vna_dict=None)
    return True

def test_power_sweep_with_vna():
    anritsu = AnritsuCtrl()
    sweep_powers = [-5, -10, -15]
    freq = 5.
    vna_dict = {'sample_id' : 'M3D6_02_2SP_INP',
                'centerf' : 8.02,
                'span'    : 100.,
                'temp'    : 13.,
                'avg'     : 3,
                'power'   : -35.,
                'edelay'  : 69.3,
                'ifbw'    : 1.,
                'npts'    : 1001,
                'sparam'  : 'S21',
                'cal_set' : None}
    anritsu.power_sweep(sweep_powers, freq,
           run_vna=True, vna_dict=vna_dict)
    return True

def test_frequency_sweep_with_vna():
    anritsu = AnritsuCtrl()
    sweep_freqs = [5, 10, 15]
    power = -35.
    vna_dict = {'sample_id' : 'M3D6_02_2SP_INP',
                'centerf' : 8.02,
                'span'    : 100.,
                'temp'    : 13.,
                'avg'     : 3,
                'power'   : -35.,
                'edelay'  : 69.3,
                'ifbw'    : 1.,
                'npts'    : 1001,
                'sparam'  : 'S21',
                'cal_set' : None}
    anritsu.frequency_sweep(sweep_freqs, power,
           run_vna=True, vna_dict=vna_dict)
    return True

def test_frequency_power_sweep_with_vna():
    anritsu = AnritsuCtrl()
    sweep_freqs = [5, 10, 15]
    sweep_powers = [-35., -40.]
    vna_dict = {'sample_id' : 'M3D6_02_2SP_INP',
                'centerf' : 8.02,
                'span'    : 100.,
                'temp'    : 13.,
                'avg'     : 3,
                'power'   : -35.,
                'edelay'  : 69.3,
                'ifbw'    : 1.,
                'npts'    : 1001,
                'sparam'  : 'S21',
                'cal_set' : None}
    anritsu.power_frequency_sweep_2d(sweep_powers,
                                     sweep_freqs,
                                     sweep_order='frequency_power',
                                     run_vna=True,
                                     vna_dict=vna_dict)
    return True

def run_tests(tests):
    """
    Runs all tests and reports successes and failures
    """
    ret_cnt = 0
    for t in tests:
        print('\n------------------------------------------')
        print(f'Testing {t} ...')
        print('------------------------------------------\n')
        ret = eval(f'{t}()')
        check_test(ret, t)
        if ret: ret_cnt += 1
        print('------------------------------------------')

    print('\n--------------------------------------------')
    print(f'|         {ret_cnt} of {len(tests)} tests passed.             |')
    if ret_cnt != len(tests):
        print(f'{len(tests)-ret_cnt} of {len(tests)} tests failed.')
    print('--------------------------------------------\n')
    

if __name__ == '__main__':
    tests = ['test_instantiate_class',
             # 'test_power_sweep_with_vna',
             # 'test_frequency_sweep_with_vna',
             'test_frequency_power_sweep_with_vna']
    run_tests(tests)
