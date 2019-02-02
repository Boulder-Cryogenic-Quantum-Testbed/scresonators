import abc

import numpy

import instruments.base as base


# Based on
# https://github.com/martinisgroup/servers/blob/master/PNA/agilentN5242A.py
class AgilentN5242A(base.VnaInstrument):
    def __init__(self, agilent_device_api: 'AgilentDeviceApi') -> None:
        pass

    def init_device(self):
        pass

    def sweep(self,
            frequency_start_GHz: List[float],
            frequency_end_GHz: List[float],
            npoints: int) -> np.array:
        pass

        
    # TODO:
    # Page 2253 of manual for calculate SDATA and phase data.
    # Only get SDATA
    # Sweep time


class AgilentDeviceApi(abc.ABC):
    pass
