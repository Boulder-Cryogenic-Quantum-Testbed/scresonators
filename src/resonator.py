from .fitter import Fitter
from .fit_methods.fit_method import FitMethod

class Resonator:
    def __init__(self, data=None):
        self.data = data
        self.fitter = None
        self.fit_result = None

    def load_data(self, data):
        """Load data into the resonator object."""
        self.data = data

    def set_fitting_strategy(self, strategy: FitMethod):
        """Set the fitting strategy with a FitMethod object."""
        self.fitter = Fitter(strategy=strategy)

    def fit(self):
        """Perform fitting using the selected fitting strategy."""
        if not self.fitter:
            raise ValueError("Fitting strategy not set.")
        if self.data is None:
            raise ValueError("Data not loaded.")
        
        self.fit_result = self.fitter.fit(self.data)
        return self.fit_result

    # Additional resonator functionalities can be added here