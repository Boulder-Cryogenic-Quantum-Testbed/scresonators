from types import NoneType
from skrf import Network
from .fitter import Fitter
from .fit_methods.fit_method import FitMethod



"""
only take data in appropriately shaped np arrays
fdata: np shape (,M)
sdata: np shape (M,N,N) for an N-Port device with M frequency points, shape (,M) also works for a 1-Port device 
this removes ambiguity from amplitude/phase data (linear or db & degrees or radians)
and naturally generalizes for Effective Reflection Mode measurements (Two Port required)
"""

class Resonator:
    def __init__(self, fdata=None, sdata = None):
        self.fdata = fdata
        self.sdata = sdata
        self.fitter = None
        self.fit_result = None

    # TODO: Can we use *args to combine the following three functions into one?
    def load_data(self, fdata, sdata):
        """Load data into the resonator object."""
        self.fdata = fdata
        self.sdata = sdata

    def Load_data_from_Network(self, network: Network):
        self.sdata = network.s
        network.frequency.units = 'GHz'
        self.fdata = network.f


    def load_data_from_touchstone(self, touchstone_file: str):
        network = Network(touchstone_file)
        self.sdata = network.s
        network.frequency.units = 'GHz'
        self.fdata = network.f

    #TODO: you should be able to pass a string or a FitMethod object
    def set_fitting_strategy(self, strategy: FitMethod):
        """Set the fitting strategy with a FitMethod object."""
        self.fitter = Fitter(fit_method=strategy)

    def fit(self, manual_init=None, verbose = False):
        """Perform fitting using the selected fitting strategy."""
        if not self.fitter:
            raise ValueError("Fitting strategy not set.")
        if type(self.fdata) == NoneType or type(self.sdata) == NoneType:
            raise ValueError("Data not loaded")

        self.fit_result = self.fitter.fit(self.fdata, self.sdata, manual_init=manual_init, verbose = verbose)
        return self.fit_result

    #Additional resonator functionalities can be added here
