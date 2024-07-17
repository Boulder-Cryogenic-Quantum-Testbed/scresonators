import numpy as np
import lmfit
from .fit_method import FitMethod
from ..utils import find_circle

class DCM(FitMethod):
    def __init__(self):
        pass

    @staticmethod
    def func(x, Q, Qc, f0, phi):
        """DCM fit function."""
        return 1 - Q / Qc * np.exp(1j * phi) / (1 + 1j * (x - f0) / f0 * 2 * Q) ## absolute value of Qc actually
    
    def create_model(self):
        """Creates an lmfit Model using the static func method."""
        model = lmfit.Model(self.func, independent_vars=['x'])
        return model
    

    def find_initial_guess(self, x: np.ndarray, y: np.ndarray) -> lmfit.Parameters: ## Redundant? There is a '_estimate_initial_parameters' method in Fitter class
        x_c, y_c, r = find_circle(np.real(y), np.imag(y))
        z_c = x_c + 1j * y_c
        # Adjust y to the circle's reference frame
        y_adjusted = y - (1 + z_c)
    
        # Use specific criteria or model to calculate initial guesses
        phi_guess = np.angle(-z_c)
        freq_idx = np.argmax(np.abs(y_adjusted))
        f0_guess = x[freq_idx]
        Q_guess = 1e4  # Placeholder guess
        Qc_guess = Q_guess / np.abs(y_adjusted[freq_idx])  # Example calculation

        #Place initial guess intermediate variables into a dictionary
        # guess_intermediate_var = {'circle center': z_c, 'Adjusted complex data': y_adjusted}

        # param_guesses = self.form_initial_guess(x, Q_guess, Qc_guess, f0_guess, phi_guess)

        # Create an lmfit.Parameters object to store initial guesses
        param_guesses = lmfit.Parameters()
        param_guesses.add('Q', value=Q_guess, min=1e3, max=1e6)
        param_guesses.add('Qc', value=Qc_guess, min=1e3, max=1e6)
        param_guesses.add('f0', value=f0_guess, min=f0_guess*0.9, max=f0_guess*1.1)
        param_guesses.add('phi', value=phi_guess, min=-np.pi, max=np.pi)

        return param_guesses

        # return param_guesses, guess_intermediate_var
    
    ## TODO Fix this functionality
    # def form_initial_guess(self, x, Q_guess, Qc_guess, f0_guess, phi_guess):
    #     # Create an lmfit.Parameters object to store initial guesses
    #     param_guesses = lmfit.Parameters()
    #     param_guesses.add('Q', value=Q_guess, min=1e3, max=1e6)
    #     param_guesses.add('Qc', value=Qc_guess, min=1e3, max=1e6)
    #     param_guesses.add('f0', value=f0_guess, min=f0_guess*0.9, max=f0_guess*1.1)
    #     param_guesses.add('phi', value=phi_guess, min=-np.pi, max=np.pi)

    #     return param_guesses
    
    def generate_highres_fit(self, x: np.ndarray, fit_params, num_fit_points=1000):
        
        # Generate a higher-resolution frequency array
        high_res_x = np.linspace(min(x), max(x), num_fit_points)
        # Use the fitted parameters to evaluate the model function at the new high-resolution frequencies
        high_res_y = self.func(high_res_x, fit_params['Q'], fit_params['Qc'], fit_params['f0'], fit_params['phi'])

        return high_res_x, high_res_y