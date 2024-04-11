import numpy as np
import lmfit
from .fit_method import FitMethod
from ..utils import find_circle

class DCM(FitMethod):
    def __init__(self):
        pass


    def func(self, x, Q, Qc, w1, phi):
        """DCM fit function."""
        return np.array(1 - Q/Qc * np.exp(1j * phi) / (1 + 1j * (x - w1) / w1 * 2 * Q))
    

    def find_initial_guess(self, x: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> tuple:
        y = self._prepare_data(y1, y2)
        x_c, y_c, r = find_circle(np.real(y), np.imag(y))
        z_c = x_c + 1j * y_c
        # Adjust y to the circle's reference frame
        y_adjusted = y - (1 + z_c)
        phi = np.angle(-z_c)

        # Use specific criteria or model to calculate initial guesses
        # Assume a model where we need to guess the resonant frequency and Q factors
        freq_idx = np.argmax(np.abs(y_adjusted))
        f_c = x[freq_idx]
        Q_guess = 1e4  # Placeholder guess
        Qc_guess = Q_guess / np.abs(y_adjusted[freq_idx])  # Example calculation

        init_guess = [Q_guess, Qc_guess, f_c, phi]  # Example initial guess structure

        return init_guess, x_c, y_c, r
    

    def _prepare_data(self, y1: np.ndarray, y2: np.ndarray) -> np.ndarray:
        """Prepares the data by combining real and imaginary parts and possibly inverting."""
        try:
            y = y1 + 1j * y2
            return y
        except Exception as e:
            raise ValueError(f"Error preparing data: {e}")
        

    def min_fit(self, params, xdata, ydata):
        """
        Minimizes parameter values for DCM fitting and transmission data.

        Args:
            params (lmfit.Parameters): Initial guess for the fit parameters.
            xdata (np.ndarray): Frequency data points.
            ydata (np.ndarray): S21 data points.

        Returns:
            tuple: Minimized parameter values and 95% confidence intervals.
        """
        try:
            # Minimize using the DCM specific function
            minner = lmfit.Minimizer(self.func, params, fcn_args=(xdata, ydata))
            result = minner.minimize(method='least_squares')
            
            # Extract minimized parameter values
            fit_params = result.params.valuesdict()
            
            # Calculate 95% confidence intervals for minimized parameters
            conf_intervals = self._compute_confidence_intervals(minner, result)

        except Exception as e:
            print(f"Failed to minimize data for least squares fit: {e}")
            fit_params = None
            conf_intervals = [0.0] * len(params)  # Adjust based on expected number of parameters

        return fit_params, conf_intervals


    def _compute_confidence_intervals(minner, result):
        """
        Computes 95% confidence intervals for the minimized parameters.

        Args:
            minner (lmfit.Minimizer): The minimizer object used for the fit.
            result (lmfit.MinimizerResult): The result of the minimization process.

        Returns:
            list: 95% confidence intervals for the parameters.
        """
        try:
            ci = lmfit.conf_interval(minner, result, sigmas=[2])
            # Format and extract confidence interval data as needed
            conf_array = [0.0] * len(result.params)  # Placeholder for actual CI computation logic
            # Actual CI extraction logic goes here
        except Exception as e:
            print(f"Failed to compute confidence intervals: {e}")
            conf_array = [0.0] * len(result.params)
        
        return conf_array
