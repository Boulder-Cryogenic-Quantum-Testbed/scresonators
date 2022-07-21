import numpy as np
import pytest

import fit_resonator.functions as ff
import fit_resonator.Sdata as fs
import fit_resonator.resonator as res

"""Tests for analysis."""

def test_resonator():
    # Simple test to ensure object constructor works.
    freqs = np.arange(5, 6, 0.1)
    s21 = np.arange(0, 1, 0.01)
    reson = res.Resonator(name='test')
    assert reson


def test_fit_method():
    fit_type = ff.FittingMethod.DCM
    MC_iteration = 10
    MC_rounds = 1e3
    MC_fix = ['w1']
    manual_init = ff.ModelParams.from_params(Qi=1E6, Qc=5E5, f_res=5.8, phi=0.1)
    fit_method = res.fit_method(fit_type,
<<<<<<< HEAD
                                MC_iteration,
                                MC_rounds=MC_rounds,
                                MC_fix=MC_fix,
                                manual_init=manual_init,
                                MC_step_const=0.3)
=======
                                     MC_iteration,
                                     MC_rounds=MC_rounds,
                                     MC_fix=MC_fix,
                                     manual_init=manual_init,
                                     MC_step_const=0.3)
>>>>>>> 23a4b73 (Resonator rewrite)
    assert fit_method


def test_simple():
    reson = res.Resonator()

    # Test with file load into class
<<<<<<< HEAD
    reson.from_file(".s2p", "s12")
=======
    reson.from_file("C:/1work/Research/snp_examples/M3D6_WTH_2SP_INP_-35dBm_19mK_220706.s2p", "s12")
>>>>>>> 23a4b73 (Resonator rewrite)
    fit_type = 'DCM'
    MC_iteration = 10
    MC_rounds = 1e3
    MC_fix = ['w1']
    manual_init = None
    reson.fit_method(fit_type, MC_iteration, MC_rounds=MC_rounds, MC_fix=MC_fix, manual_init=manual_init,
                     MC_step_const=0.3)

    # Test with file provided in fit call
    reson.fit()
<<<<<<< HEAD


def test_raw_res():
    # The object all following code will be called from
    my_resonator = res.Resonator()

    # Load the raw data
    url = 'https://raw.githubusercontent.com/Boulder-Cryogenic-Quantum-Testbed/scresonators/master/cryores/test_data/AWR/AWR_Data.csv'
    raw = np.loadtxt(url, delimiter=',')

    # Test with file load into class
    my_resonator.from_columns(raw)

    # Assign your desired fit method variables
=======
    #reson.fit("C:/1work/Research/snp_examples/M3D6_WTH_2SP_INP_-35dBm_19mK_220706.s2p", measurement="s12")

def test_raw_res():
    url = 'https://raw.githubusercontent.com/Boulder-Cryogenic-Quantum-Testbed/scresonators/master/cryores/test_data/AWR/AWR_Data.csv'
    raw = np.loadtxt(url, delimiter=',')

    reson = res.Resonator(data=raw)

>>>>>>> 23a4b73 (Resonator rewrite)
    fit_type = 'DCM'
    MC_iteration = 10
    MC_rounds = 1e3
    MC_fix = ['w1']
    manual_init = None

<<<<<<< HEAD
    # Pass these to your resonator object
    my_resonator.fit_method(fit_type, MC_iteration, MC_rounds=MC_rounds, MC_fix=MC_fix, manual_init=manual_init,
                     MC_step_const=0.3)

    # Fit!
    my_resonator.fit()
=======
    reson.fit_method(fit_type, MC_iteration, MC_rounds=MC_rounds, MC_fix=MC_fix, manual_init=manual_init,
                     MC_step_const=0.3)
    reson.fit()
>>>>>>> 23a4b73 (Resonator rewrite)
