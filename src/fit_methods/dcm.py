import numpy as np
import lmfit
from scipy.optimize import curve_fit
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
        # Rough calculation to find loaded quality factor
        mags_min = np.min(y.real)
        mags_max = np.max(y.real)
        mags_halfmax = (mags_max + mags_min) / 2

        crossing_points = []
        for i in range(len(y.real)-1):
            if (y.real[i] - mags_halfmax) * (y.real[i+1] - mags_halfmax) < 0:
                crossing_points.append(i)

        nearest_indices = []
        for point in crossing_points:
            if abs(y.real[point] - mags_halfmax) < abs(y.real[point + 1] - mags_halfmax):
                nearest_indices.append(point)
            else:
                nearest_indices.append(point + 1)

        nearest_values = y.real[nearest_indices]

        print("Nearest indices", nearest_indices)

        ## TODO Use FWHM for Q
        ## Circle diameter is Q/Qc


        f0_idx = np.abs(y).argmin()
        f0 = x[f0_idx]
        phi = 0

        # Create an lmfit.Parameters object to store initial guesses
        param_guesses = lmfit.Parameters()
        param_guesses.add('Q', value=Q, min=1e3, max=1e6)
        param_guesses.add('Qc', value=Qc, min=1e3, max=1e6)
        param_guesses.add('f0', value=f0, min=f0_guess*0.9, max=f0_guess*1.1)
        param_guesses.add('phi', value=phi, min=-np.pi, max=np.pi)

        return param_guesses
    
    def generate_highres_fit(self, x: np.ndarray, fit_params, num_fit_points=1000):
        
        # Generate a higher-resolution frequency array
        high_res_x = np.linspace(min(x), max(x), num_fit_points)
        # Use the fitted parameters to evaluate the model function at the new high-resolution frequencies
        high_res_y = self.func(high_res_x, fit_params['Q'], fit_params['Qc'], fit_params['f0'], fit_params['phi'])

        return high_res_x, high_res_y