import resfit.fit_resonator.fit_S_data as fsd
import numpy as np
import pytest

def test_normalize():
    freqs = np.arange(4, 5, 0.01)
    amps = 10 * np.ones(len(freqs))
    phases = np.ones(len(freqs))
    data = fsd.VNASweep.from_columns(freqs=freqs,
                                     amps=amps,
                                     phases=phases)
    normed = fsd.normalize_data(data)
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

    data = fsd.VNASweep.from_columns(freqs=freqs,
                                     amps=amps,
                                     phases=phases)
    background = fsd.VNASweep.from_columns(freqs=freqs,
                                           amps=bg_amps,
                                           phases=bg_phases)
    normed = fsd.normalize_data(data, background=background)
    cplx_s21 = 1.7086+2.661j*np.ones(len(freqs))
    np.testing.assert_array_almost_equal(normed.complex_s21,
                                         cplx_s21,
                                         decimal=3)

def test_preprocess():
    ph = np.linspace(-3., 3., 100)      # phase with linear slope
    amps = np.linspace(5., 5., 100)     # constant amplitude
    freqs = np.arange(4, 5, 0.01)
    dat = fsd.VNASweep.from_columns(freqs, amps, ph)
    cplx = fsd.normalize_data(dat)
    prepro = fsd.preprocess(cplx, 10)
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
    ex_x, ex_y = fsd.extract_near_res(freqs, amps, f_res, kappa)
    near_x = np.array([4.41, 4.42, 4.43, 4.44, 4.45, 4.46, 4.47, 4.48, 4.49, 4.5 , 4.51,
        4.52, 4.53, 4.54, 4.55, 4.56, 4.57, 4.58, 4.59, 4.6 , 4.61, 4.62,
        4.63, 4.64, 4.65, 4.66, 4.67, 4.68, 4.69, 4.7 ])
    near_y = np.array([5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
        5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.])
    np.testing.assert_array_almost_equal(ex_x, near_x)
    np.testing.assert_array_almost_equal(ex_y, near_y)
    with pytest.raises(Exception):
        f_res = 3.55
        fsd.extract_near_res(freqs, amps, f_res, kappa)


