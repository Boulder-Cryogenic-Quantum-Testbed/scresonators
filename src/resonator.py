from .fitter import Fitter
from .fit_methods.fit_method import FitMethod

class Resonator:
    def __init__(self, data=None): ## Why is this default shown as data=None but see line 21 for different representation of a default?
        """Initialize the resonator object"""
        self.data = data
        self.fitter = None
        self.fit_result = None

    def load_data(self, data):
        """Load data into the resonator object. This method is optional as data could have been loaded when the class instance was created"""
        self.data = data ## What format is the data in? Maybe a dictionary based on the method definition for Fitter.fit
        #use pandas dataframes 
        #tocsv
        #fromcsv

    #def save_data(self, data):
        #save data

    def set_fitting_strategy(self, strategy: FitMethod): ## Why not show this strategy as strategy=FitMethod or strategy=None?
        """Set the fitting strategy with a FitMethod object."""
        self.fitter = Fitter(strategy=strategy)

    def fit(self):
        """Perform fitting using the selected fitting strategy."""
        if not self.fitter:
            raise ValueError("Fitting strategy not set.") ## Why is this even an option if I can't run the set_fitting_strategy method without an argument (without a strategy)?
        if self.data is None:
            raise ValueError("Data not loaded.")
        
        self.fit_result = self.fitter.fit(self.data)
        return self.fit_result

    # Additional resonator functionalities can be added here