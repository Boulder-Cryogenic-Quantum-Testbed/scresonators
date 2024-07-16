import numpy as np
import lmfit
from .fit_method import FitMethod
from ..utils import find_circle

class DCM(FitMethod):
    def __init__(self):
        pass

    @staticmethod
    def func(x, Q, Qc, w1, phi):
        """DCM fit function."""
        return 1 - Q / Qc * np.exp(1j * phi) / (1 + 1j * (x - w1) / w1 * 2 * Q) ## absolute value of Qc actually
    
    def create_model(self):
        """Creates an lmfit Model using the static func method."""
        model = lmfit.Model(self.func, independent_vars=['x'])
        return model
    

    def find_initial_guess(self, x: np.ndarray, y: np.ndarray) -> lmfit.Parameters: ## Redundant? There is a '_estimate_initial_parameters' method in Fitter class
        x_c, y_c, r = find_circle(np.real(y), np.imag(y))
        z_c = x_c + 1j * y_c
        # Adjust y to the circle's reference frame
        y_adjusted = y - (1 + z_c)
        phi = np.angle(-z_c)

        # Use specific criteria or model to calculate initial guesses
        freq_idx = np.argmax(np.abs(y_adjusted))
        f_0 = x[freq_idx]
        w_0 = 2 * np.pi * f_0
        Q_guess = 1e4  # Placeholder guess
        Qc_guess = Q_guess / np.abs(y_adjusted[freq_idx])  # Example calculation

        # Create an lmfit.Parameters object to store initial guesses
        params = lmfit.Parameters()
        params.add('Q', value=Q_guess, min=1e3, max=1e6)
        params.add('Qc', value=Qc_guess, min=1e3, max=1e6)
        params.add('w1', value=w_0, min=w_0*0.9, max=w_0*1.1)
        params.add('phi', value=phi, min=-np.pi, max=np.pi)

        return params
    
    def generate_highres_fit(self, x: np.ndarray, fit_params, num_fit_points=1000):
        
        # Generate a higher-resolution frequency array
        high_res_x = np.linspace(min(x), max(x), num_fit_points)
        # Use the fitted parameters to evaluate the model function at the new high-resolution frequencies
        high_res_y = self.func(high_res_x, fit_params['Q'], fit_params['Qc'], fit_params['w1'], fit_params['phi'])

        high_res_x_Hz = high_res_x / (2 * np.pi)
        high_res_y_Hz = high_res_y / (2 * np.pi)
        return high_res_x_Hz, high_res_y_Hz