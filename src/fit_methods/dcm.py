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
        return 1 - Q / Qc * np.exp(1j * phi) / (1 + 1j * (x - w1) / w1 * 2 * Q)
    
    def create_model(self):
        """Creates an lmfit Model using the static func method."""
        model = lmfit.Model(self.func, independent_vars=['x'])
        return model
    

    def find_initial_guess(self, x: np.ndarray, y: np.ndarray) -> lmfit.Parameters:
        x_c, y_c, r = find_circle(np.real(y), np.imag(y))
        z_c = x_c + 1j * y_c
        # Adjust y to the circle's reference frame
        y_adjusted = y - (1 + z_c)
        phi = np.angle(-z_c)

        # Use specific criteria or model to calculate initial guesses
        freq_idx = np.argmax(np.abs(y_adjusted))
        f_c = x[freq_idx]
        Q_guess = 1e4  # Placeholder guess
        Qc_guess = Q_guess / np.abs(y_adjusted[freq_idx])  # Example calculation

        # Create an lmfit.Parameters object to store initial guesses
        params = lmfit.Parameters()
        params.add('Q', value=Q_guess, min=1e3, max=1e6)
        params.add('Qc', value=Qc_guess, min=1e3, max=1e6)
        params.add('w1', value=f_c, min=f_c*0.9, max=f_c*1.1)
        params.add('phi', value=phi, min=-np.pi, max=np.pi)

        return params       