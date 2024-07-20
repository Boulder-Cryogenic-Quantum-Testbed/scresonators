import numpy as np
import lmfit
import scipy
from .fit_method import FitMethod
from ..utils import find_circle

class DCM(FitMethod):
    def __init__(self):
        pass

    @staticmethod
    def func(x, Q, Qc, f0, phi):
        """DCM fit function."""
        return 1 - Q / Qc * np.exp(1j * phi) / (1 + 1j * (x - f0) / f0 * 2 * Q) ## absolute value of Qc actually

    @staticmethod
    def abs_func(x, Q, Qc, f0, phi):
        return np.abs(1 - Q / Qc * np.exp(1j * phi) / (1 + 1j * (x - f0) / f0 * 2 * Q))

    def create_model(self):
        """Creates an lmfit Model using the static func method."""
        model = lmfit.Model(self.func, independent_vars=['x'])
        return model

    def find_initial_guess(self, x: np.ndarray, y: np.ndarray) -> lmfit.Parameters: ## Redundant? There is a '_estimate_initial_parameters' method in Fitter class
        Q = Qc = 1000  # Starting guess for Q and Qc
        f0_idx = np.abs(y).argmin()
        f0 = x[f0_idx]
        phi = 0
        
        while True:
            try:
                init_guess_params = np.array([Q, Qc, f0, phi])
                param_guesses, _ = scipy.optimize.curve_fit(self.abs_func, x, abs(y), p0=init_guess_params)
                print("Parameter guesses: ", param_guesses)
                break
            except RuntimeError:
                Q *= 10
                Qc *= 10
                print(f"Retrying with Q={Q}, Qc={Qc}")

        Q_guess = param_guesses[0]
        Qc_guess = param_guesses[1]
        f0_guess = param_guesses[2]
        phi_guess = param_guesses[3]

        # Create an lmfit.Parameters object to store initial guesses
        param_guesses = lmfit.Parameters()
        param_guesses.add('Q', value=Q_guess, min=1e3, max=1e6)
        param_guesses.add('Qc', value=Qc_guess, min=1e3, max=1e6)
        param_guesses.add('f0', value=f0_guess, min=f0_guess*0.9, max=f0_guess*1.1)
        param_guesses.add('phi', value=phi_guess, min=-np.pi, max=np.pi)

        return param_guesses
    
    def generate_highres_fit(self, x: np.ndarray, fit_params, num_fit_points=1000):
        
        # Generate a higher-resolution frequency array
        high_res_x = np.linspace(min(x), max(x), num_fit_points)
        # Use the fitted parameters to evaluate the model function at the new high-resolution frequencies
        high_res_y = self.func(high_res_x, fit_params['Q'], fit_params['Qc'], fit_params['f0'], fit_params['phi'])

        return high_res_x, high_res_y