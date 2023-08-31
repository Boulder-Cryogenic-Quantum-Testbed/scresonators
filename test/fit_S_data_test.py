import fit_resonator.fit as fit
import fit_resonator.plot as plot

import fit_resonator.resonator as res
import numpy as np
import pytest

def test_normalize():
    freqs = np.arange(4, 5, 0.01)
    amps = 10 * np.ones(len(freqs))
    phases = np.ones(len(freqs))
    data = res.from_columns(freqs=freqs,
                                     amps=amps,
                                     phases=phases)
    normed = fit.normalize_data(data)
    linear = 3.1623*np.ones(len(freqs))
    cplx_s21 = 1.7086+2.661j*np.ones(len(freqs))
    np.testing.assert_array_almost_equal(data.linear_amps, linear, decimal=4)
    np.testing.assert_array_almost_equal(normed.complex_s21,
                                         cplx_s21,
                                         decimal=3)

def test_normalized_with_background():
    freqs = np.arange(4, 5, 0.01)
    amps = 20 * np.ones(len(freqs))
    phases = 2 * np.ones(len(freqs))

    bg_amps = 10 * np.ones(len(freqs))
    bg_phases = 1 * np.ones(len(freqs))

    data = res.from_columns(freqs=freqs,
                                     amps=amps,
                                     phases=phases)
    background = res.from_columns(freqs=freqs,
                                           amps=bg_amps,
                                           phases=bg_phases)
    normed = fit.normalize_data(data, background=background)
    cplx_s21 = 1.7086+2.661j*np.ones(len(freqs))
    np.testing.assert_array_almost_equal(normed.complex_s21,
                                         cplx_s21,
                                         decimal=3)

def test_preprocess():
    ph = np.linspace(-3., 3., 100)      # phase with linear slope
    amps = np.linspace(5., 5., 100)     # constant amplitude
    freqs = np.arange(4, 5, 0.01)
    dat = res.from_columns(freqs, amps, ph)
    cplx = fit.normalize_data(dat)
    prepro = fit.preprocess(cplx, 10)
    # test phase angle is zero for all points.
    np.testing.assert_array_almost_equal(np.zeros(100),
                                         np.angle(prepro.complex_s21))
    # test amplitude of all points is 1.
    np.testing.assert_array_almost_equal(np.ones(100),
                                         np.abs(prepro.complex_s21))

def test_extract_near_res():
    amps = np.linspace(5., 5., 100)
    freqs = np.arange(4, 5, 0.01)
    f_res = 4.55
    kappa = 0.3
    ex_x, ex_y = fit.extract_near_res(freqs, amps, f_res, kappa)
    near_x = np.array([4.41, 4.42, 4.43, 4.44, 4.45, 4.46, 4.47, 4.48, 4.49, 4.5 , 4.51,
        4.52, 4.53, 4.54, 4.55, 4.56, 4.57, 4.58, 4.59, 4.6 , 4.61, 4.62,
        4.63, 4.64, 4.65, 4.66, 4.67, 4.68, 4.69, 4.7 ])
    near_y = np.array([5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
        5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.])
    np.testing.assert_array_almost_equal(ex_x, near_x)
    np.testing.assert_array_almost_equal(ex_y, near_y)
    with pytest.raises(Exception):
        f_res = 3.55
        fit.extract_near_res(freqs, amps, f_res, kappa)

def test_data_input():
    filepath = '.s2p'
    data = fit.VNASweep.from_file(filepath, "S21")
    print(data)

def test_data_fit():
    filepath = '.s2p'
    fit_type = 'DCM'
    MC_iteration = 10
    MC_rounds = 1e3
    MC_fix = ['w1']
    manual_init = None
    method = res.FitMethod(fit_type, MC_iteration, MC_rounds=MC_rounds, MC_fix=MC_fix, manual_init=manual_init, MC_step_const=0.3)

    # Fit the data:
    fit.fit(filepath, method, measurement="s12", normalize=10)

