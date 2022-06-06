# Lint as: python3
import numpy as np

import resfit.fit_resonator.fit_functions as ff

from resfit.fit_resonator import resonator
"""Tests for analysis."""


def test_resonator():
    #Simple test to ensure object constructor works.
    freqs = np.arange(5,6,0.1)
    s21 = np.arange(0,1,0.01)
    res = resonator.Resonator(freqs, s21, name='test')
    assert res

def test_fit_method():
    fit_type = ff.FittingMethod.DCM
    MC_iteration = 10
    MC_rounds = 1e3
    MC_fix = ['w1']
    manual_init = ff.ModelParams.from_params(Qi=1E6, Qc=5E5, f_res=5.8, phi=0.1)
    fit_method = resonator.FitMethod(fit_type,
               MC_iteration,
               MC_rounds=MC_rounds,
               MC_fix=MC_fix,
               manual_init=manual_init,
               MC_step_const=0.3)
    assert fit_method