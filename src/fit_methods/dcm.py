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
        f0_idx = np.abs(y).argmin()
        f0 = x[f0_idx]
        phi = 0
        
        # Rough calculation to find loaded quality factor
        mags_min = np.min(np.abs(y))
        mags_max = np.max(np.abs(y))
        mags_halfmax = (mags_max + mags_min) / 2

        try:
            multiplier = 0.1
            crossing_pts = np.where(np.abs(np.abs(y) - mags_halfmax) <= (multiplier * mags_max))
            print("np.where crossing pts: ", crossing_pts)
            while(len(crossing_pts[0]) != 2):
                print("Length of cp: ", len(crossing_pts[0]))
                crossing_pts = np.where(np.abs(np.abs(y) - mags_halfmax) <= (multiplier * mags_max))
                multiplier /= 10
                if len(crossing_pts[0]) == 0:
                    print("Crossing pts ValueError conditional: ", crossing_pts)
                    raise ValueError("Better Q-guess didn't work. Using a less precise guess.")
                
            print("Crossing pts: ", crossing_pts[0][0], crossing_pts[0][1])
            kappa = np.abs(x[crossing_pts[0][1]] - x[crossing_pts[0][0]])
            Q = f0 / kappa
        except ValueError:
            kappa_rough_estimate = np.max(x) - np.min(x)
            Q = f0 / kappa_rough_estimate

        ## TODO Use FWHM for Q
        ## Circle diameter is Q/Qc
        _, _, r = find_circle(np.real(y), np.imag(y))
        d = 2 * r
        Qc = d / Q

        # Create an lmfit.Parameters object to store initial guesses
        param_guesses = lmfit.Parameters()
        param_guesses.add('Q', value=Q, min=1e3, max=1e6)
        param_guesses.add('Qc', value=Qc, min=1e3, max=1e6)
        param_guesses.add('f0', value=f0, min=f0*0.9, max=f0*1.1)
        param_guesses.add('phi', value=phi, min=-np.pi, max=np.pi)

        return param_guesses
    
    def generate_highres_fit(self, x: np.ndarray, fit_params, num_fit_points=1000):
        
        # Generate a higher-resolution frequency array
        high_res_x = np.linspace(min(x), max(x), num_fit_points)
        # Use the fitted parameters to evaluate the model function at the new high-resolution frequencies
        high_res_y = self.func(high_res_x, fit_params['Q'], fit_params['Qc'], fit_params['f0'], fit_params['phi'])

        return high_res_x, high_res_y