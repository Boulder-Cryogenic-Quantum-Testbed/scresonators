# -*- coding: utf-8 -*-
"""
Test of the Qi and fc fitting functions on data from NYU 2D Al on InP resonator
"""

import sys
sys.path.append('/home/nick/scresonators/fit_resonator')
from mattis_bardeen_fit import MBFitTemperatureSweep
import numpy as np

# Setup the temperatures and filenames
temperatures = 1e-3 * np.linspace(30, 315, 20)
dpath = '../Resources/sample_temperature_data'
filenames = [f'{dpath}/NYU2D_AL_INP_220131_7.7182GHz_-20.0dB_{T*1e3:.1f}mK.csv'
        for T in temperatures]

mb = MBFitTemperatureSweep(temperatures, filenames, output_fit_figures=None,
        fit_normalization='linear', alpha_sim=1e-3,
        init_fit_guess = {'Tc' : 1.2, 'alpha' : 1e-1, 'lambda' : 65e-9},
        use_jordans_rule=True)

# Plot the Qi data and the fc data
qi_filename = 'NYU_AL_INP_lambda_qi_vs_T.pdf'
fc_filename = 'NYU_AL_INP_lambda_fc_vs_T.pdf'

mb.plot_qi_vs_temperature(qi_filename, use_alpha_sim=True, use_yerrs=True)
mb.plot_fc_vs_temperature(fc_filename, use_alpha_sim=True, use_yerrs=True)

qi_filename = 'NYU_AL_INP_qi_vs_T.pdf'
fc_filename = 'NYU_AL_INP_fc_vs_T.pdf'

mb.plot_qi_vs_temperature(qi_filename, use_alpha_sim=False, use_yerrs=True)
mb.plot_fc_vs_temperature(fc_filename, use_alpha_sim=False, use_yerrs=True)
