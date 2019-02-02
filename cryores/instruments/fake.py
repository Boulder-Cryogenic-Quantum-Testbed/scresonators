from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from labrad.units import dBm, GHz, Value

import cryores.instruments.base as instruments


class FakeVna(instruments.VnaInstrument):
    """A Fake VNA."""

    def __init__(self, peak_frequency_GHz: Optional[float]=None) -> None:
        self.peak_frequency_GHz = peak_frequency_GHz

        # Use an "initialized" flag to make sure the experiments correctly
        # reset the device after each scan.
        self.initialized = False
        self.custom_parameter = 0.0

    def init_device(self):
        """No initialize required."""
        self.initialized = True

    def set_device_parameter(self, parameter: float):
        """Set a device-specific parameter."""
        self.custom_parameter = parameter

    def sweep(
            self,
            frequency_start: Value,
            frequency_end: Value,
            npoints: int,
            power: Value,
            averages: int=1,
            log_or_linear: instruments.LogOrLinear=instruments.LogOrLinear.LOG,
    ) -> pd.DataFrame:

        # Resonator model courtesy of Josh Mutus, based on:
        #   https://arxiv.org/pdf/1304.4533.pdf
        Z_o = 50
        L_1 = .71e-12
        M = 11.9e-12
        L = 288.7e-12
        C_o = 2.5e-12
        C_c = 5e-15
        R = 50E3

        alpha = M / L
        beta = L_1 / L
        gamma = C_c / C_o
        zi = np.sqrt(L / C_o) / Z_o
        omega_o = 1 / np.sqrt(L * C_o)
        q = 1 / (R * omega_o * C_o)

        def S21(f):
            w = 2 * np.pi * f
            Z_in = Z_o + 1j * w * L_1 - 1j * w * M**2 / L
            Z_out = Z_o / (1 + 1j * w * C_c * Z_o)
            Z_c = 1 / (1j * w * C_o + 1 / R)
            term = (M / L - 1j * w * C_c * Z_out)
            numer = -2 * (term / (Z_out + Z_in))
            denom = 1 / (1j * w * L) + 1 / Z_c + 1j * w * C_c / (
                1j * w * C_c * Z_o + 1) + term**2 / (Z_out + Z_in)
            V_ratio = numer / denom
            S_21pre = 1 / (1 + Z_in / Z_o + 1j * w * C_c * Z_in)
            S_21 = S_21pre * (2 + V_ratio * (1j * C_c * Z_in + M / L))
            return S_21

        # Initialize frequencies.
        freqs = np.linspace(frequency_start[GHz], frequency_end[GHz], npoints)

        # Compute S21 magnitude and angle.
        mag = np.absolute(S21(freqs))
        ang = np.angle(S21(freqs))

        self.initialized = False

        # Get a blank DataFrame with the expected columns.
        df = self.initialized_data_frame()

        # Fill in the data.
        df[instruments.COL_FREQ_GHZ] = freqs
        df[instruments.COL_S21_AMP] = mag
        df[instruments.COL_S21_PHASE] = ang

        return df

    def get_parameters(self) -> Dict[str, Any]:
        return {'custom_parameter': self.custom_parameter}
