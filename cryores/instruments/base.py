import abc
import enum
from typing import Any, Dict, List, Optional

import pandas as pd
from labrad.units import dBm, GHz, Value


# Enums are useful for capturing named options.
class LogOrLinear(enum.Enum):
    LOG = 'log'
    LINEAR = 'linear'


class SParameters(enum.Enum):
    S11 = 'S11'
    S12 = 'S12'
    S21 = 'S21'
    S22 = 'S22'


# Dataframe column headers.
COL_FREQ_GHZ = 'Frequency (GHz)'
COL_S21_AMP = 'S21 Amp (dB)'
COL_S21_PHASE = 'S21 Phase (rad)'


class VnaInstrument(abc.ABC):
    """This describes a "contract" for interfacing with a VNA instrument.

    Each of the following methods that are labeled with @abc.abstractmethod
    should be filled in in a separate class that actually implements the
    interaction with a VNA instrument.

    We assume that the instrument starts in factory mode for consistency.
    Alternatively, the "init_device" method should make a best effort at
    putting it into a deterministically consistent state.
    """

    @abc.abstractmethod
    def init_device(self):
        """Initialize the device into a consistent, reproducible state."""

    @abc.abstractmethod
    def sweep(
            self,
            frequency_start: Value,
            frequency_end: Value,
            npoints: int,
            power: Value,
            averages: int=1,
            log_or_linear: LogOrLinear=LogOrLinear.LOG,) -> pd.DataFrame:
        """Sweep frequencies and return data results.
        
        Args:
            averages: The number of sweeps to run.
            power:
            resolution_bandwidth: Resolution bandwidth, typically for internal
                IF filters.

        Returns:
            Pandas Dataframe with the following columns:
                - Frequency_GHz
                - S21_Amp_dB
                - S21_Phase_rad
        """

    @abc.abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Returns device configuration parameters."""

    def initialized_data_frame(self) -> pd.DataFrame:
        """Helper function to return an empty dataframe."""
        return pd.DataFrame(columns=[COL_FREQ_GHZ, COL_S21_AMP, COL_S21_PHASE])
