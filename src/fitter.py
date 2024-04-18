import numpy as np
import logging

import scipy.optimize as spopt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
from scipy.interpolate import interp1d
from .utils import find_circle, phase_dist, phase_centered, periodic_boundary, normalize


class Fitter:
    def __init__(self, fit_method=None, **kwargs):
        """Initializes the Fitter with a fitting method that includes the fitting function.

        Args:
            fit_method (object): An instance of a fitting method class that contains the `func` method.
        """
        if fit_method is None or not hasattr(fit_method, 'func'):
            raise ValueError("A fitting method with a valid 'func' attribute must be provided.")
        
        self.fit_method = fit_method
        self.func = fit_method.func
        self.normalize = kwargs.get('normalize', 'circle')
        self.MC_rounds = kwargs.get('MC_rounds', 1000)
        self.MC_step_const = kwargs.get('MC_step_const', 0.05)
        self.MC_weight = kwargs.get('MC_weight', True)
        self.MC_fix = kwargs.get('MC_fix', [])
        self.databg = kwargs.get('databg', None)

    def fit(self, freqs: np.ndarray, phases: np.ndarray, amps: np.ndarray, manual_init=None):
        linear_amps = 10 ** (amps / 20)
        phases = np.unwrap(phases)
        xdata = freqs
        ydata = np.multiply(linear_amps, np.exp(1j * phases))

        if self.databg is not None:
            ydata = self.background_removal(ydata)
        elif self.normalize == "linear":
            ydata, _, _, _, _ = self.preprocess_linear(xdata, ydata, self.normalize)
        elif self.normalize == "circle":
            ydata = self.preprocess_circle(xdata, ydata)

        if manual_init:
            init_guess = manual_init
        else:
            init_guess, x_c, y_c, r = self.fit_method.find_initial_guess(xdata, np.real(ydata), np.imag(ydata))

        fit_params, conf_intervals = self.fit_method.min_fit(init_guess, xdata, ydata)

        for _ in range(self.MC_rounds):
            new_params, improved, error = self.monte_carlo_fit(xdata, ydata, fit_params)
            if improved:
                fit_params = new_params

        return fit_params, conf_intervals

    def _extract_near_res(self, x_raw: np.ndarray, y_raw: np.ndarray, f_res: float, kappa: float, extract_factor: int = 1) -> tuple:
        """Extracts portions of the spectrum within a specified width of the resonance frequency.

        This method is intended for internal use to prepare data for fitting processes.

        Args:
            x_raw (np.ndarray): X-values of the spectrum to extract from.
            y_raw (np.ndarray): Y-values of the spectrum to extract from.
            f_res (float): Resonant frequency about which to extract data.
            kappa (float): Width about f_res to extract.
            extract_factor (int): Multiplier for kappa, defines extraction range.
        
        Returns:
            tuple: Two np.ndarray objects containing the extracted X and Y spectrum data around the resonance.
        """
        xstart = f_res - extract_factor / 2 * kappa
        xend = f_res + extract_factor / 2 * kappa

        # Using boolean indexing for a cleaner implementation
        mask = (x_raw > xstart) & (x_raw < xend)
        x_temp = x_raw[mask]
        y_temp = y_raw[mask]

        if len(y_temp) < 5:
            print("Warning: Less than 5 data points to fit data. Not enough points near resonance, attempting to fit anyway.")
        if len(x_temp) == 0:
            raise ValueError("Failed to extract data from designated bandwidth.")

        return x_temp, y_temp
    

    def monte_carlo_fit(self, xdata: np.ndarray, ydata: np.ndarray, parameters: np.ndarray) -> tuple:
        """Performs Monte Carlo optimization to refine fitting parameters.

        Args:
            xdata (np.ndarray): The independent variable data.
            ydata (np.ndarray): The dependent variable data.
            parameters (np.ndarray): Initial guess of the parameters.

        Returns:
            tuple: A tuple containing the optimized parameters, a boolean indicating
                   if the fitting improved, and the final error.
        """
        assert all(param is not None for param in [xdata, ydata, parameters]), "One or more parameters are undefined."
        
        try:
            _, error = self._calculate_weighted_error(xdata, ydata, parameters)
        except Exception as e:
            logging.error(f"Failed to initialize monte_carlo_fit: {e}")
            raise  # Reraising the exception to handle it outside or halt the program

        improved = False  # Flag to track if any improvement was made
        for _ in range(self.MC_rounds):
            new_parameters = self._generate_new_parameters(parameters)
            _, new_error = self._calculate_weighted_error(xdata, ydata, new_parameters)
            
            if new_error < error:
                parameters = new_parameters
                error = new_error
                improved = True  # Indicate an improvement was found

        # Final status message to summarize the outcome
        if improved:
            logging.info('Monte Carlo simulation resulted in better fitting parameters.')
        else:
            logging.info('Monte Carlo simulation did not improve the parameters.')

        return parameters, improved, error


    def _calculate_weighted_error(self, xdata, ydata, parameters, weight_array=None):
        """Calculates weighted error for given parameters."""
        ydata_fit = self.func(xdata, *parameters)
        if self.MC_weight:
            if weight_array is None:
                weight_array = 1 / np.abs(ydata)
        else:
            weight_array = np.ones_like(xdata)
        
        weighted_ydata = weight_array * ydata
        weighted_ydata_fit = weight_array * ydata_fit
        error = np.linalg.norm(weighted_ydata - weighted_ydata_fit) / len(xdata)
        return weighted_ydata, error


    def _generate_new_parameters(self, parameters):
        """Generates a new set of parameters for Monte Carlo simulation."""
        random_factors = self.MC_step_const * (np.random.random_sample(len(parameters)) - 0.5)
        for i, fix in enumerate(['Q', 'Qi', 'Qc', 'w1', 'phi', 'Qa']):
            if fix in self.MC_fix:
                random_factors[i] = 0
        
        new_parameters = parameters * np.exp(random_factors)
        return new_parameters


    def fit_phase(self, f_data, z_data, guesses=None):
        """
        Fits the phase response of a strongly overcoupled resonator in reflection.

        Args:
            f_data (np.ndarray): Frequency data array.
            z_data (np.ndarray): Complex scattering data from which to fit the phase.
            guesses (tuple, optional): Initial guesses for (fr, Ql, delay). If None,
                                       they will be automatically determined.

        Returns:
            tuple: Fitted parameters (fr, Ql, theta, delay).
        """
        phase = np.unwrap(np.angle(z_data))

        # Check for sufficient data coverage
        if np.ptp(phase) <= 0.8 * 2 * np.pi:
            logging.warning("Data does not fully cover a circle. Coverage: {:.1f} rad.".format(np.ptp(phase)))
        
        # Initial parameter estimation
        if guesses is None:
            fr_guess, Ql_guess, delay_guess = self._estimate_initial_parameters(f_data, phase)
        else:
            fr_guess, Ql_guess, delay_guess = guesses
        theta_guess = 0.5 * (np.mean(phase[:5]) + np.mean(phase[-5:]))

        # Sequential fitting process
        p_final = self._sequential_fitting(f_data, phase, fr_guess, Ql_guess, theta_guess, delay_guess)

        return p_final


    def _estimate_initial_parameters(self, f_data, phase):
        """Estimate initial parameters for the fitting process."""
        phase_smooth = gaussian_filter1d(phase, 30)
        fr_guess = f_data[np.argmax(np.gradient(phase_smooth))]
        Ql_guess = 2 * fr_guess / (f_data[-1] - f_data[0])
        delay_guess = -(np.ptp(phase) / (2 * np.pi * (f_data[-1] - f_data[0])))
        return fr_guess, Ql_guess, delay_guess


    def _sequential_fitting(self, f_data, phase, fr_guess, Ql_guess, theta_guess, delay_guess):
        """Performs sequential fitting to refine initial parameter estimates."""
        
        # Define residuals functions for partial fits
        def residuals_Ql(Ql): return self._phase_residuals(f_data, phase, fr=fr_guess, Ql=Ql, theta=theta_guess, delay=delay_guess)
        def residuals_fr_theta(fr, theta): return self._phase_residuals(f_data, phase, fr=fr, Ql=Ql_guess, theta=theta, delay=delay_guess)
        def residuals_delay(delay): return self._phase_residuals(f_data, phase, fr=fr_guess, Ql=Ql_guess, theta=theta_guess, delay=delay)
        def residuals_fr_Ql(fr, Ql): return self._phase_residuals(f_data, phase, fr=fr, Ql=Ql, theta=theta_guess, delay=delay_guess)
        
        # Sequential optimization steps
        Ql_guess = spopt.leastsq(residuals_Ql, Ql_guess)[0]
        fr_guess, theta_guess = spopt.leastsq(residuals_fr_theta, [fr_guess, theta_guess])[0]
        delay_guess = spopt.leastsq(residuals_delay, delay_guess)[0]
        fr_guess, Ql_guess = spopt.leastsq(residuals_fr_Ql, [fr_guess, Ql_guess])[0]
        
        # Final full optimization with all parameters
        p_final = spopt.leastsq(lambda params: self._phase_residuals(f_data, phase, *params),
                                [fr_guess, Ql_guess, theta_guess, delay_guess])

        return p_final[0]
    

    def _phase_residuals(self, f_data, phase, fr, Ql, theta, delay):
        """Calculates residuals for phase fitting."""
        model_phase = phase_centered(f_data, fr, Ql, theta, delay)
        return phase_dist(phase - model_phase)
    

    def fit_delay(self, xdata: np.ndarray, ydata: np.ndarray):
        """
        Finds the cable delay by centering the "circle" and fitting the slope
        of the phase response, iteratively refining the delay estimation.
        """
        
        # Initial circle finding and translation to origin
        xc, yc, _ = find_circle(np.real(ydata), np.imag(ydata))
        z_data = ydata - complex(xc, yc)
        fr, Ql, theta, delay = self.fit_phase(xdata, z_data)

        # Iterative refinement of delay
        delay *= 0.05
        for i in range(5):
            z_data = ydata * np.exp(2j * np.pi * delay * xdata)
            xc, yc, _ = find_circle(np.real(z_data), np.imag(z_data))
            z_data -= complex(xc, yc)
            
            guesses = (fr, Ql, 5e-11)
            fr, Ql, theta, delay_corr = self.fit_phase(xdata, z_data, guesses)
            
            # Condition for stopping iteration
            phase_fit = phase_centered(xdata, fr, Ql, theta, delay_corr)
            residuals = np.unwrap(np.angle(z_data)) - phase_fit
            if self._is_correction_small(xdata, delay_corr, residuals):
                break
            
            delay = self._update_delay(delay, delay_corr)

        if not self._is_correction_small(xdata, delay_corr, residuals, final_check=True):
            logging.warning("Delay could not be fit properly!")
        
        return delay

    def _is_correction_small(self, xdata, delay_corr, residuals, final_check=False):
        """
        Checks if the correction to the delay is smaller than measurable based on residuals.
        """
        condition = 2 * np.pi * (xdata[-1] - xdata[0]) * delay_corr <= np.std(residuals)
        return condition if not final_check else condition and delay_corr > 0

    def _update_delay(self, delay, delay_corr):
        """
        Updates the delay value based on the correction and current delay sign.
        """
        # Logic to adjust delay based on correction sign and magnitude
        # This mirrors the original logic, adjusting for readability and maintainability
        if delay_corr * delay < 0:
            delay = delay * 0.5 if abs(delay_corr) > abs(delay) else delay + delay_corr * 0.1
        else:
            delay += min(delay_corr, delay) if abs(delay_corr) >= 1e-8 else delay_corr
        return delay
    
    def calibrate(self, x_data: np.ndarray, z_data: np.ndarray):
        """
        Finds parameters for normalization of scattering data.
        
        Args:
            x_data (np.ndarray): Independent variable data, typically frequency.
            z_data (np.ndarray): Complex scattering data to be calibrated.
        
        Returns:
            tuple: Calibration parameters including delay_remaining, normalization
                   amplitude (a), phase offset (alpha), resonator phase (theta),
                   phase correction (phi), resonance frequency (fr), and loaded
                   quality factor (Ql).
        """
        # Translate circle to origin
        xc, yc, r = find_circle(np.real(z_data), np.imag(z_data))
        zc = complex(xc, yc)
        z_data2 = z_data - zc

        # Fit phase for off-resonant point
        fr, Ql, theta, delay_remaining = self.fit_phase(x_data, z_data2)
        beta = periodic_boundary(theta - np.pi)
        offrespoint = zc + r * np.exp(1j * beta)
        a, alpha = np.abs(offrespoint), np.angle(offrespoint)
        phi = periodic_boundary(beta - alpha)

        # Adjust radius for normalization
        r /= a

        return delay_remaining, a, alpha, theta, phi, fr, Ql
        
    def preprocess_circle(self, xdata: np.ndarray, ydata: np.ndarray):
        """
        Data Preprocessing using Probst method for cable delay removal and normalization.

        Args:
            xdata (np.ndarray): The frequency data.
            ydata (np.ndarray): The complex S21 data to preprocess.

        Returns:
            np.ndarray: The preprocessed and normalized complex S21 data.
        """

        # Remove cable delay
        delay = self.fit_delay(xdata, ydata)
        z_data = ydata * np.exp(2j * np.pi * delay * xdata)

        # Calibrate and normalize
        delay_remaining, a, alpha, theta, phi, fr, Ql = self.calibrate(xdata, z_data)
        z_norm = normalize(xdata, z_data, delay_remaining, a, alpha)

        return z_norm
    
    def preprocess_linear(self, xdata: np.ndarray, ydata: np.ndarray, normalize: int):
        """
        Preprocesses S21 data linearly. Removes cable delay and normalizes 
        phase/magnitude of S21 by linear fit of a specified number of endpoints.

        Args:
            xdata (np.ndarray): The frequency data.
            ydata (np.ndarray): The complex S21 data to preprocess.
            normalize (int): Number of endpoints to use for normalization.

        Returns:
            tuple: Preprocessed S21 data, phase slope, phase intercept,
                   magnitude slope, and magnitude intercept.
        """
        if normalize * 2 > len(ydata):
            raise ValueError(
                "Not enough points to normalize. Please decrease the 'normalize' value or include more data points near resonance.")

        # Unwrap phase for linear preprocessing
        phase = np.unwrap(np.angle(ydata))

        # Normalize phase using linear fit
        slope, intercept, _, _, _ = linregress(
            np.append(xdata[:normalize], xdata[-normalize:]),
            np.append(phase[:normalize], phase[-normalize:])
        )

        # Adjust phase to remove cable delay and rotate off-resonant point to (1, 0i)
        adjusted_phase = phase - (slope * xdata + intercept)
        y_adjusted = np.abs(ydata) * np.exp(1j * adjusted_phase)

        # Normalize magnitude using linear fit
        y_db = 20 * np.log10(np.abs(ydata))
        mag_slope, mag_intercept, _, _, _ = linregress(
            np.append(xdata[:normalize], xdata[-normalize:]),
            np.append(y_db[:normalize], y_db[-normalize:])
        )
        adjusted_magnitude = 10 ** ((y_db - (mag_slope * xdata + mag_intercept)) / 20)

        preprocessed_data = adjusted_magnitude * np.exp(1j * adjusted_phase)

        return preprocessed_data, slope, intercept, mag_slope, mag_intercept
    

    def background_removal(self, linear_amps: np.ndarray, phases: np.ndarray):
        """
        Removes background signal by interpolating and adjusting amplitude and phase,
        using stored background data.

        Args:
            linear_amps (np.ndarray): Measured linear amplitudes to be corrected.
            phases (np.ndarray): Measured phases to be corrected.

        Returns:
            np.ndarray: Corrected complex S21 data with background removed.
        """
        if not self.databg:
            raise ValueError("Background data ('databg') not provided.")

        # Extract background data
        x_bg = self.databg.freqs
        linear_amps_bg = self.databg.linear_amps
        phases_bg = self.databg.phases

        # Create interpolation functions for background amplitude and phase
        fmag = interp1d(x_bg, linear_amps_bg, kind='cubic', fill_value="extrapolate")
        fang = interp1d(x_bg, phases_bg, kind='cubic', fill_value="extrapolate")

        # Correct measured data using interpolated background
        linear_amps_corrected = np.divide(linear_amps, fmag(self.databg.freqs))
        phases_corrected = np.subtract(phases, fang(self.databg.freqs))

        # Return corrected data as complex S21 values
        return np.multiply(linear_amps_corrected, np.exp(1j * phases_corrected))