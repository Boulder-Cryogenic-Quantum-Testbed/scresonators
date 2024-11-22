from abc import ABC, abstractmethod

class FitMethod(ABC):
    """
    Container for data related to fitting method.
    Common attributes and initial configuration for fitting classes.
    """

    def __init__(self,
                 MC_iteration: int = 5,
                 MC_rounds: int = 100,
                 MC_weight: str = 'no',
                 MC_weightvalue: int = 2,
                 MC_fix: list = [],
                 MC_step_const: float = 0.6,
                 manual_init: list = None,
                 vary: list = None):
        
        # Validate and initialize parameters
        self.MC_iteration = MC_iteration
        self.MC_rounds = MC_rounds
        self.MC_weight = MC_weight
        self.MC_weightvalue = MC_weightvalue
        self.MC_step_const = MC_step_const
        self.MC_fix = MC_fix
        self.manual_init = manual_init
        self.vary = vary if vary is not None else [True] * 6

        # Default values for flags indicating parameter variability
        self.change_Q = 'Q' not in MC_fix
        self.change_Qi = 'Qi' not in MC_fix
        self.change_Qc = 'Qc' not in MC_fix
        self.change_w1 = 'w1' not in MC_fix
        self.change_phi = 'phi' not in MC_fix
        self.change_Qa = 'Qa' not in MC_fix

        self.name = None

    @abstractmethod
    def calculate_manual_initial_guess(self):
        """Abstract method to calculate initial guesses from manual input."""
        pass

    @abstractmethod
    def auto_initial_guess(self, xdata, ydata):
        """Abstract method to find initial guess without manual input."""
        pass

    @abstractmethod
    def add_params(self, params_arr):
        """Abstract method to create a set of parameters."""
        pass

    def __repr__(self):
        return ', '.join(f"{key}: {value}" for key, value in vars(self).items())