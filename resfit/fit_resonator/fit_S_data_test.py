import resfit.fit_resonator.fit_S_data as fsd
import numpy as np
import pytest

def test_normalize():
    freqs = np.arange(4, 5, 0.01)
    amps = 10 * np.ones(len(freqs))
    phases = np.ones(len(freqs))
    data = fsd.VNASweep(freqs=freqs,
                        amps=amps,
                        phases=phases)
    normed = fsd.normalize_data(data)
    linear = 3.1623*np.ones(len(freqs))
    cplx_s21 = 1.7086+2.661j*np.ones(len(freqs))
    np.testing.assert_array_almost_equal(data.linear_amps(), linear, decimal=4)
    np.testing.assert_array_almost_equal(normed.complex_s21,
                                         cplx_s21,
                                         decimal=3)

def test_normalized_with_background():
    freqs = np.arange(4, 5, 0.01)
    amps = 20 * np.ones(len(freqs))
    phases = 2 * np.ones(len(freqs))

    bg_amps = 10 * np.ones(len(freqs))
    bg_phases = 1 * np.ones(len(freqs))

    data = fsd.VNASweep(freqs=freqs,
                        amps=amps,
                        phases=phases)
    background = fsd.VNASweep(freqs=freqs,
                              amps=bg_amps,
                              phases=bg_phases)
    normed = fsd.normalize_data(data, background=background)
    cplx_s21 = 1.7086+2.661j*np.ones(len(freqs))
    np.testing.assert_array_almost_equal(normed.complex_s21,
                                         cplx_s21,
                                         decimal=3)