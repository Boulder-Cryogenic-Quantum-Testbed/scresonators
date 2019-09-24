# Lint as: python3
import numpy as np

from resfit import fit_resonator
"""Tests for analysis."""


def test_resonator():
    #Simple test to ensure object constructor works.
    freqs = np.arange(5,6,0.1)
    s21 = np.arange(0,1,0.01)
    res = fit_resonator.Resonator(freqs, s21, name='test')
    assert res
