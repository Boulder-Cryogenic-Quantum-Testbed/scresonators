import numpy as np
import logging
import lmfit
import scipy.optimize as spopt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
from scipy.interpolate import interp1d
from .utils import find_circle, phase_dist, phase_centered, periodic_boundary, normalize


## FOR TESTING ONLY
import matplotlib.pyplot as plt
import scresonators.src.plotter as plotter
## FOR TESTING ONLY


class Fitter:
    # TODO Verify the format of data throughout this whole class
    # TODO Make 'Resonator' an umbrella class of 'Fitter' such that 'Fitter' inherits parameters from 'Resonator'
    # TODO Implement 'trim_S21_wings' method to get rid of frequency data that is too far away from the resonance
    def __init__(self, fit_method=None, **kwargs):
        """Initializes the Fitter with a fitting method that includes the fitting function.

        Args:
            fit_method (object): An instance of a fitting method class that contains the `func` method.

            keyword arguments:
                preprocess: str
                    choice of preprocess: 'circle' or 'linear'. Defaults to 'circle'

                normalize: int

                MC_rounds: int
                in each MC iteration, number of rounds of randomly choose parameter

                MC_step_const: int
                    randomly choose number in range MC_step_const*[-0.5~0.5]
                    for fitting. Exp(0.5)=1.6, and this is used for Qi,... . However, the res. frequency, theta, amplitude are usually fixed during Monte Carlo.   

                MC_weight: bool
                    True or False, weight the extract_factor fitting range, True uses 1/|S21| weight, which we call iDCM

                MC_fix: list of str
                    'Amp','f0','theta','phi','Qc', 'Q' for DCM, 'Qi' for INV    

                databg: class?


        """
        if fit_method is None or not hasattr(fit_method, 'func'): ## Why are you looking for func as an attribute instead of a method? Because func is an attribute of an instance of the class DCM.
            #Because fit_method might not be a class (or instance thereof)
            raise ValueError("A fitting method with a valid 'func' attribute must be provided.")
        
        self.fit_method = fit_method
        self.preprocess = kwargs.get('preprocess', 'circle') ## If preprocess is not set equal to something else in the argument line, the default value is 'circle'
        self.normalize = kwargs.get('normalize', 4)
        self.MC_rounds = kwargs.get('MC_rounds', 1000) ## We might not want to use Monte Carlo stuff (MC)
        self.MC_step_const = kwargs.get('MC_step_const', 0.05)
        self.MC_weight = kwargs.get('MC_weight', False)
        self.MC_fix = kwargs.get('MC_fix', [])
        self.databg = kwargs.get('databg', None) ## databg is an instance of a class?

    def load_data(self, freqs, amps_dB, phases):
        self.xdata = freqs
        self.amps_dB = amps_dB
        self.phases = phases

        # Check if each argument is a NumPy array
        if not isinstance(self.xdata, np.ndarray):
            raise TypeError("Frequency data must be a NumPy array")
        if not isinstance(self.amps_dB, np.ndarray):
            raise TypeError("Experimental S21 magnitude data must be a NumPy array")
        if not isinstance(self.phases, np.ndarray):
            raise TypeError("Experimental S21 phase data must be a NumPy array")

        self.amps_linear = 10 ** (amps_dB / 20)
        self.ydata = np.multiply(self.amps_linear, np.exp(1j * phases))

    def fit(self, preprocessing_guesses=None, manual_init=None, verbose=False): ## How can we preprocess such that this function can take pandas dataframes as input? 
        """Fit resonator data using the provided method using lmfit's Model.fit
        """
        self.phases = np.unwrap(self.phases)

        if self.databg: ## not utilizing this feature at the moment
            self.background_removal()
        elif self.preprocess == "circle":
            self.preprocess_circle(preprocessing_guesses) ## Do we want to pass 'manual_init' or 'preprocessing_guesses' along?
        elif self.preprocess == "linear":
            # TODO: implement
            # ydata, _, _, _, _ = self.preprocess_linear(xdata, ydata, self.normalize)
            pass
        
        # Setup the initial parameters or use provided manual_init
        if manual_init:
            params = manual_init
            pass
        else:
            params = self.fit_method.find_initial_guess(self.xdata, self.ydata)
        
        # Create the model and fit
        model = self.fit_method.create_model() ## Creating model with instance of ____ class (instance is "fit_method") - at the moment only DCM is implemented. The blank on the left is "DCM"
        result = model.fit(self.ydata, params, x=self.xdata, method='leastsq') 
        ## Above line is calling "fit" method from lmfit package. "model" is the instance of the lmfit.Model class
        if verbose: print(result.fit_report()) ## Calling methods within lmfit package
        if verbose: print(result.ci_report()) 
        
        # TODO: implement confidence intervals
        conf_intervals = self._bootstrap_conf_intervals(model, result.params) 
        
        # # TODO: implement monte carlo
        # # Using Monte Carlo to explore parameter space if enabled
        # if self.MC_weight:
        #     emcee_kwargs = {
        #         'steps': self.MC_rounds,
        #         'thin':10,
        #         'burn': int(self.MC_rounds * 0.3),
        #         'is_weighted': self.MC_weight,
        #         'workers': 1
        #     }
        #     emcee_result = model.fit(data=ydata, params=result.params, x=xdata, method='emcee', fit_kws=emcee_kwargs)            
        #     if verbose:
        #         print(emcee_result.fit_report())
        #     return emcee_result.params, conf_intervals           
        
        return result, conf_intervals
    
    def _bootstrap_conf_intervals(self, model, params, iterations=1000):
        """
        This method finds confidence intervals for each of the parameters in 'params'
        
        Args:
            model (instance of a class): instance of lmfit.Model of chosen fit function
            ydata (np.ndarray): complex S21 data, an array of complex numbers
            params (dictionary of dictionaries): initial guess parameters
            iterations (integer): number of iterations. Defaults to 1000.
        
        Returns:
            conf_intervals (dictionary): dictionary containing confidence intervals for each type of fitted parameter in 'params' ##unsure about this...
        """
        sampled_params = []
        for _ in range(iterations):
            indices = np.random.randint(0, len(self.ydata), len(self.ydata))
            sample_ydata = self.ydata[indices]
            res = model.fit(sample_ydata, params, x=self.ydata[indices])
            sampled_params.append(res.params)
            
        conf_intervals = {key: np.percentile([p[key].value for p in sampled_params], [5, 95]) for key in params}
        return conf_intervals
    
    
    def preprocess_circle(self, preprocessing_guesses=None):
        """
        Data Preprocessing using the Probst 2015 method for cable delay removal and normalization.

        Args:
            xdata (np.ndarray): frequency data (Hz), an array of floats
            ydata (np.ndarray): The complex S21 data to preprocess, an array of complex numbers
            preprocessing_guesses (tuple, optional): Initial guesses for (fr, Ql, delay). If None,
                                       they will be automatically determined.


        Returns:
            np.ndarray: The preprocessed and normalized complex S21 data.

        Citation:
            Review of Scientific Instruments 86, 024706 (2015); doi: 10.1063/1.4907935
        """
        
        self.preprocess_circle_ydata = self.ydata
        ## plot_preprocessing_steps PLOT HERE using self.ydata
        ## TESTING PLOT
        # plot1 = plotter.Plotter()
        # plot1.load_data(xdata, self.y_data)
        # layout = [
        #     ["main", "main", "mag"],
        #     ["main", "main", "ang"]
        # ]
        # fig1, ax_dict1 = plt.subplot_mosaic(layout, figsize=(12, 8))
        # plot1.plot_before_fit(fig1, ax_dict1, figure_title='S21 in fit_delay after circle translation')
        ## TESTING PLOT

        # Remove cable delay 
        delay = self.fit_delay(self.xdata, self.ydata, preprocessing_guesses)
        print("Delay from 'fit_delay': ", delay/1e-9, "ns")

        self.preprocess_circle_z_data = self.ydata * np.exp(2j * np.pi * delay * self.xdata)

        ## plot_preprocessing_steps PLOT HERE using self.-preprocess_circle_z_data
        ## TESTING PLOT
        # plot2 = plotter.Plotter()
        # plot2.load_data(xdata, self.z_data)
        # layout = [
        #     ["main", "main", "mag"],
        #     ["main", "main", "ang"]
        # ]
        # fig2, ax_dict2 = plt.subplot_mosaic(layout, figsize=(12, 8))
        # plot2.plot_before_fit(fig2, ax_dict2, figure_title='S21 in fit_delay after circle translation')
        ## TESTING PLOT

        # Calibrate and normalize
        delay_remaining, a, alpha, theta, phi, fr, Ql = self.calibrate(self.xdata, self.preprocess_circle_z_data) 
        
        self.preprocess_circle_z_norm = normalize(self.xdata, self.preprocess_circle_z_data, delay_remaining, a, alpha)

        self.ydata = self.preprocess_circle_z_norm
        ## plot_preprocessing_steps PLOT HERE with self.preprocess_circle_z_norm
        ## TESTING PLOT
        # plot3 = plotter.Plotter()
        # plot3.load_data(xdata, self.z_norm)
        # layout = [
        #     ["main", "main", "mag"],
        #     ["main", "main", "ang"]
        # ]
        # fig3, ax_dict3 = plt.subplot_mosaic(layout, figsize=(12, 8))
        # plot3.plot_before_fit(fig3, ax_dict3, figure_title='S21 in fit_delay after circle translation')
        ## TESTING PLOT
    
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
    

    def background_removal(self): ## not worried about this for V1.0?
        """
        Removes background signal by interpolating and adjusting amplitude and phase,
        using stored background data.

        Args:
            amps_linear (np.ndarray): Measured linear amplitudes to be corrected.
            phases (np.ndarray): Measured phases to be corrected.

        Returns:
            np.ndarray: Corrected complex S21 data with background removed.
        """
        if not self.databg:
            raise ValueError("Background data ('databg') not provided.")
        
        # databg_type = type(self.databg)
        # if databg_type != np.ndarray:
        #     raise TypeError(f"Background data ('databg') is {databg_type} data type. It should be np.ndarray")

        # Extract background data
        x_bg = self.databg.freqs ## Is databg a class or a dictionary?
        amps_linear_bg = self.databg.amps_linear 
        phases_bg = self.databg.phases

        # Create interpolation functions for background amplitude and phase
        fmag = interp1d(x_bg, amps_linear_bg, kind='cubic', fill_value="extrapolate")
        fang = interp1d(x_bg, phases_bg, kind='cubic', fill_value="extrapolate")

        # Correct measured data using interpolated background
        amps_linear_corrected = np.divide(self.amps_linear, fmag(self.databg.freqs))
        phases_corrected = np.subtract(self.phases, fang(self.databg.freqs))

        # Return corrected data as complex S21 values
        self.ydata = np.multiply(amps_linear_corrected, np.exp(1j * phases_corrected))
        

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


    def fit_phase(self, f_data, z_data, guesses=None):
        """
        Fits the phase response of a strongly overcoupled resonator in reflection.

        Args:
            f_data (np.ndarray): frequency data (Hz), an array of floats
            z_data (np.ndarray): Complex scattering data from which to fit the phase, an array of complex numbers
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
        """Estimate initial parameters for the phase fitting process.
        
        Args: 
            f_data (np.ndarray): array of frequency data (Hz), an array of floats
            phase (np.ndarray): phase data (radians), an array of floats

        Returns:
            fr_guess (float): guess of resonant frequency (Hz)
            Ql_guess (float): guess of loaded quality factor 
            delay_guess (float): guess of cable time delay (s) ## not sure about this...
        """
        phase_smooth = gaussian_filter1d(phase, 30) 
        fr_guess = f_data[np.argmax(np.gradient(phase_smooth))]
        Ql_guess = 2 * fr_guess / (f_data[-1] - f_data[0])
        delay_guess = -(np.ptp(phase) / (2 * np.pi * (f_data[-1] - f_data[0])))
        return fr_guess, Ql_guess, delay_guess

    def _sequential_fitting(self, f_data: np.ndarray, phase: np.ndarray, fr_guess: float, Ql_guess: float, theta_guess: float, delay_guess: float):
        """Refines initial parameter estimates using sequential fitting.
        
        Args: 
            f_data (np.ndarray): frequency data (Hz), an array of floats
            phase (np.ndarray): phase data (radians), an array of floats
            fr_guess (float): guess of resonant frequency (Hz)
            Ql_guess (float): guess of loaded quality factor
            theta_guess (float): guess of asymmetry (radians) ## not sure about this...
            delay_guess (float): guess of cable time delay (s) ## not sure about this...
        
        Returns:
            p_final (dictionary): initial parameter guesses ## not sure about this...
        """

        def residuals_Ql(Ql):
            return np.array(self._phase_residuals(f_data, phase, fr=fr_guess, Ql=Ql, theta=theta_guess, delay=delay_guess), dtype=np.float64)

        def residuals_fr_theta(params):
            fr, theta = params
            return np.array(self._phase_residuals(f_data, phase, fr=fr, Ql=Ql_guess, theta=theta, delay=delay_guess), dtype=np.float64)

        def residuals_delay(delay):
            return np.array(self._phase_residuals(f_data, phase, fr=fr_guess, Ql=Ql_guess, theta=theta_guess, delay=delay), dtype=np.float64)

        def residuals_fr_Ql(params):
            fr, Ql = params
            return np.array(self._phase_residuals(f_data, phase, fr=fr, Ql=Ql, theta=theta_guess, delay=delay_guess), dtype=np.float64)
        
        def residuals_final(params):
            fr, Ql, theta, delay = params
            return np.array(self._phase_residuals(f_data, phase, fr=fr, Ql=Ql, theta=theta, delay=delay), dtype=np.float64)

        # Ensure the initial guesses are float arrays
        initial_guesses = np.array([fr_guess, Ql_guess], dtype=np.float64)

        # Perform the least squares fits
        ## TODO utilize the return value 'ier' that tells if the solution was found or not
        Ql_guess = spopt.leastsq(residuals_Ql, Ql_guess)[0][0]
        fr_theta_placeholder = spopt.leastsq(residuals_fr_theta, [fr_guess, theta_guess])[0]
        fr_guess, theta_guess = fr_theta_placeholder[0], fr_theta_placeholder[1]
        delay_guess = spopt.leastsq(residuals_delay, delay_guess)[0][0]
        fr_Ql_placeholder = spopt.leastsq(residuals_fr_Ql, initial_guesses)[0]
        fr_guess, Ql_guess = fr_Ql_placeholder[0], fr_Ql_placeholder[1]

        # Final optimization for all parameters together
        try:
            all_params_initial = np.array([fr_guess, Ql_guess, theta_guess, delay_guess], dtype=np.float64)
            p_final = spopt.leastsq(residuals_final, all_params_initial)[0]
        except Exception as e:
            logging.error(f"Error during optimization: {e}")
            raise

        return p_final


    def _phase_residuals(self, f_data: np.ndarray, phase: np.ndarray, fr: float, Ql: float, theta: float, delay: float) -> np.ndarray:
        """
        Calculates the residuals for phase fitting.

        Args:
            f_data (np.ndarray): The frequency data points, an array of floats.
            phase (np.ndarray): The unwrapped phase data points, an array of floats.
            fr (float): The resonant frequency guess.
            Ql (float): The loaded quality factor guess.
            theta (float): The phase offset guess.
            delay (float): The delay guess.

        Returns:
            np.ndarray: The residuals of the phase model compared to the actual phase data, as a NumPy array of floats.
        """
        # Calculate the model phase using the provided parameters
        model_phase = phase_centered(f_data, fr, Ql, theta, delay)

        # Compute the phase difference as residuals
        residuals = phase_dist(phase - model_phase)

        # Ensure the residuals are returned as a numpy array of type float
        return residuals.astype(np.float64)


    
    def fit_delay(self, xdata: np.ndarray, ydata: np.ndarray, guesses=None):
        """
        Finds the cable delay by centering the "circle" and fitting the slope
        of the phase response, iteratively refining the delay estimation.

        Args: 
            xdata (np.ndarray): frequency data (Hz), an array of floats.
            ydata (np.ndarray): complex scattering data, an array of complex numbers.
            guesses (tuple, optional): Initial guesses for (fr, Ql, delay). If None,
                                       they will be automatically determined.

        Returns:
            delay (float): cable delay time (s) ## not sure about this...
        """

        # Initial circle finding and translation to origin
        xc, yc, _ = find_circle(np.real(ydata), np.imag(ydata))  ## Find circle in complex coordinates
        z_data = ydata - complex(xc, yc) ## Translate circle center to origin
        
        self.fit_delay_zdata1 = z_data
        ## plot_preprocessing_steps PLOT HERE with self.fit_delay_zdata1
        ## TESTING PLOT
        # plot2 = plotter.Plotter()
        # plot2.load_data(xdata, z_data)
        # layout = [
        #     ["main", "main", "mag"],
        #     ["main", "main", "ang"]
        # ]
        # fig2, ax_dict2 = plt.subplot_mosaic(layout, figsize=(12, 8))
        # plot2.plot_before_fit(fig2, ax_dict2, figure_title='S21 in fit_delay after circle translation')
        ## TESTING PLOT

        fr, Ql, theta, delay = self.fit_phase(xdata, z_data, guesses) ## fit_phase function should include initial guesses!

        self.fit_delay_zdata2 = ydata * np.exp(2j * np.pi * delay * xdata)
        print("Initial Delay: ", delay*1e9, "ns")
        
        # Iterative refinement of delay
        delay *= 0.05 ## hard coded value
        for i in range(5):
            z_data = ydata * np.exp(2j * np.pi * delay * xdata)
            xc, yc, _ = find_circle(np.real(z_data), np.imag(z_data))
            z_data -= complex(xc, yc)
            
            guesses = (fr, Ql, 5e-11) ## hard coded 5e-11
            fr, Ql, theta, delay_corr = self.fit_phase(xdata, z_data, guesses)
            print("Delay Correction: ", delay_corr*1e9, "ns")
            # Condition for stopping iteration
            phase_fit = phase_centered(xdata, fr, Ql, theta, delay_corr)
            residuals = np.unwrap(np.angle(z_data)) - phase_fit
            if self._is_correction_small(xdata, delay_corr, residuals):
                break
            
            delay = self._update_delay(delay, delay_corr)

        self.fit_delay_zdata3 = z_data
        ## plot_preprocessing_steps PLOT HERE with self.fit_delay_zdata2
        ## TESTING PLOT
        # plot3 = plotter.Plotter()
        # plot3.load_data(xdata, z_data)
        # layout = [
        #     ["main", "main", "mag"],
        #     ["main", "main", "ang"]
        # ]
        # fig3, ax_dict3 = plt.subplot_mosaic(layout, figsize=(12, 8))
        # plot3.plot_before_fit(fig3, ax_dict3, figure_title='S21 in fit_delay after delay refinement')
        ## TESTING PLOT

        # if not self._is_correction_small(xdata, delay_corr, residuals, final_check=True):
        #     logging.warning("Delay could not be fit properly!")
        
        return delay

    def _is_correction_small(self, xdata, delay_corr, residuals, final_check=False): ## Consider moving into 'fit_delay' method 
        """
        Checks if the correction to the delay is smaller than measurable based on residuals.
        """
        condition = 2 * np.pi * (xdata[-1] - xdata[0]) * delay_corr <= np.std(residuals)
        return condition if not final_check else condition and delay_corr > 0

    def _update_delay(self, delay, delay_corr): ##Consider moving into 'fit_delay' method 
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
        ## TODO UPDATE FOR CONSISTENCY
            ## needs data types on return data
            ## needs clarification of units on return data
        Args:
            x_data (np.ndarray): Independent variable data, typically frequency (Hz).
            z_data (np.ndarray): Complex scattering data to be calibrated.
        
        Returns: ## all units below are not verified
            tuple: Calibration parameters
                - delay_remaining (s)
                - a: normalization amplitude (dB)
                - alpha: phase offset (radians)
                - theta: resonator phase (radians)
                - phi: phase correction (radians)
                - fr: resonance frequency (Hz)
                - Ql: loaded quality factor
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

        return delay_remaining, a, alpha, theta, phi, fr, Ql