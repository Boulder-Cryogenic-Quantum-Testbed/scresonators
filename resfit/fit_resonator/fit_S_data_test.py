import resfit.fit_resonator.fit_S_data as fsd
import numpy as np
import pytest

def test_normalize():
    freqs = np.arange(4, 5, 0.01)
    amps = 10 * np.ones(len(freqs))
    phases = np.ones(len(freqs))
    linear = 10 ** (amps / 20)
    data = fsd.VNASweep(freqs=freqs,
                        amps=amps,
                        phases=phases)
    normed = fsd.normalize_data(data)
    assert data.linear_amps()[0] == pytest.approx(3.1623, 0.001)
    assert normed.complex_s21[0] == pytest.approx(1.7086+2.661j, 0.001)

