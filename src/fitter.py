import numpy as np
import logging
import lmfit
import scipy.optimize as spopt
from scipy.ndimage import gaussian_filter
from scipy.stats import linregress
from scipy.interpolate import interp1d
from .utils import *
from math import fmod


class Fitter:
    def __init__(self, fit_method=None, **kwargs):
        """Initializes the Fitter with a fitting method that includes the fitting function.

        Args:
            fit_method (object): An instance of a fitting method class that contains the `func` method.
        """
        if fit_method is None or not hasattr(fit_method, 'func'):
            raise ValueError("A fitting method with a valid 'func' attribute must be provided.")
        
        self.fit_method = fit_method
        self.remove_elec_delay = kwargs.get('remove_delay', True)
        self.preprocess_circle = kwargs.get('preprocess_circle', True)
        self.preprocess_linear = kwargs.get('preprocess_linear', False)
        self.normalize = kwargs.get('normalize', 4)
        self.MC_rounds = kwargs.get('MC_rounds', 1000)
        self.MC_step_const = kwargs.get('MC_step_const', 0.05)
        self.MC_weight = kwargs.get('MC_weight', False)
        self.MC_fix = kwargs.get('MC_fix', [])
        self.databg = kwargs.get('databg', None)
        self.plot_results = kwargs.get('plotstyle', None)#for later implementation of optional plotting post fitting
        self.delay_guess = kwargs.get('delay_guess', None)
        self.Ql_guess = None
        self.fr_guess = None
        self.theta_0 = None
        self.phi = None
        self.off_res_point = kwargs.get('off_res_point', 1+0*1j)


    def fit(self, fdata, sdata, manual_init=None, verbose=False):
        """Fit resonator data using the provided method and lmfit's Model fit"""
        #fdata: numpy array of the frequency data
        #sdata: complex valued numpy array of the scattering parameter data


        ##########################################
        #PREPROCESSING
        ##########################################
        if self.databg:
            #this feature is untested
            sdata = self.background_removal(sdata)
        if self.preprocess_linear == True:
            #TODO: this step needs fixing
            sdata, _, _, _, _ = self.preprocess_linear(fdata, sdata, self.normalize)
        if self.remove_elec_delay == True:
            delay = self.find_delay(fdata, sdata)
            sdata = remove_delay(fdata, sdata, delay)
        if self.preprocess_circle == True:
            #rotate and scale the off-resonant point to a prescribed anchor point
            sdata = self.anchor_to_point(fdata, sdata)






        ##############################################################################
        #Initial guess for fitting parameters
        ##############################################################################
        # Setup the initial parameters or use provided manual_init
        if manual_init:
            params = self.manual_init
        else:
            params = self.fit_method.find_initial_guess(self = self.fit_method ,fdata = fdata,sdata = sdata)
            #very weird that self = self.fit_method needs to be passed


        #####################################################
        #The actual fit, implemented with the lmfit package
        #####################################################
        model = self.fit_method.create_model(self = self.fit_method)
        #this creates an lmfit.Model() object defined by the FitMethod
        result = model.fit(sdata, params, f=fdata, method='leastsq') #lmfit.Model.fit(), not Fitter.fit()
        if verbose: print(result.fit_report())

        
        # Using Monte Carlo to explore parameter space if enabled
        #may want to delete this
        if self.MC_weight:
            emcee_kwargs = {
                'steps': self.MC_rounds,
                'thin':10,
                'burn': int(self.MC_rounds * 0.3),
                'is_weighted': self.MC_weight,
                'workers': 1
            }
            emcee_result = model.fit(data=sdata, params=result.params, x=fdata, method='emcee', fit_kws=emcee_kwargs)
            if verbose:
                print(emcee_result.fit_report())
            return emcee_result.params

        params = self.fit_method.extractQi(self = self.fit_method, params = result.params)
        return params
    
    
    def preprocess_circle(self, fdata: np.ndarray, sdata: np.ndarray):
        """
        Data Preprocessing using Probst method for cable delay removal and normalization.

        Depreciated.

        Args:
            fdata (np.ndarray): The frequency data.
            sdata (np.ndarray): The complex S21 data to preprocess.

        Returns:
            np.ndarray: The preprocessed and normalized complex S21 data.
        """

        # Remove cable delay
        delay = self.find_delay(fdata, sdata)
        z_data = remove_delay(fdata, sdata, delay)

        # Calibrate and normalize
        delay_remaining, a, alpha, theta, phi, fr, Ql = self.calibrate(fdata, z_data)
        z_norm = normalize(fdata, z_data, delay_remaining, a, alpha)

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

        Untested

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

    def find_delay(self, fdata: np.ndarray, sdata: np.ndarray):
        """
        Modified version of the Criclefit method described in Probst.

        Adjusts electrical delay such that the offset phase most closely fits an arctangent.

        Args:
            fdata: numpy array of the frequency data
            sdata: numpy array of the scattering data

        Returns:
            delay: float representing the electrical delay
        """
        params = lmfit.Parameters()
        guess_params = self.fit_method.find_initial_guess(self = self.fit_method ,fdata = fdata,sdata = sdata)

        if self.delay_guess == None:
            delay_guess = self.initial_guess_delay(fdata, sdata)/1.25
        elif self.delay_guess == 0:
            delay_guess = 1e-16*self.guess_delay(fdata, sdata)
        else:
            delay_guess = self.delay_guess


        params.add(name = 'fr', value = guess_params['f0'].value)
        params.add(name = 'Ql', value = guess_params['Q'].value)
        params.add(name = 'delay', value = delay_guess)
        params.add(name = 'theta_0', value = 0)


        #running the minimizer several times also improves results
        for iter in range(30):
            #alternate running the least squares minimization & brute force only varying the delay?
            min_result = lmfit.minimize(fcn=self.arctan_deviation, params=params, args=(fdata, sdata),
                                    method='leastsq', max_nfev = 10000000)

            params = min_result.params

            #TODO: we should have a condition on the residuals to break this loop early
        electrical_delay = params['delay'].value
        self.Ql_guess = params['Ql'].value
        self.fr_guess = params['fr'].value
        self.theta_0 = params['theta_0'].value

        #Plot the sloped arctan as a verification step
        '''
        import matplotlib.pyplot as plt
        sdata_new = remove_delay(fdata, sdata, electrical_delay)
        xc, yc, r = find_circle(np.real(sdata_new), np.imag(sdata_new))

        offset_phase = params['theta_0'].value+2*np.arctan(2*params['Ql'].value*(1-fdata/params['fr'].value))

        plt.plot(fdata, np.unwrap(np.angle(sdata_new-(xc+1j*yc))), label = 'data')
        plt.plot(fdata, offset_phase, label = 'min fit')
        plt.ylabel('offset phase (rad.)')
        plt.xlabel('frequency (a.u.)')
        plt.legend()
        plt.show()
        '''

        return electrical_delay

    def guess_delay(self, fdata: np.ndarray, sdata: np.ndarray):
        """
        Linear fit of the phase to be used as an initial guess of the electrical delay.

        Args:
            fdata: numpy array of the frequency data
            sdata: numpy array of the scattering data

        Returns:
            delay: float representing the electrical delay
        """
        phase = np.unwrap(np.angle(sdata))
        lrg_result = linregress(fdata, phase)
        delay_guess = lrg_result.slope/(-2*np.pi)

        return delay_guess

    def initial_guess_delay(self, fdata: np.ndarray, sdata: np.ndarray):
        '''
        Just fit the phase while discarding data in the linewidth.

        Args:
            fdata: numpy array of the frequency data
            sdata: numpy array of the scattering data

        Returns:
            delay: float representing the electrical delay
        '''
        filtered_data = gaussian_filter(sdata, sigma=3)  # sigma may need to be changed for noisy data
        gradS = np.gradient(filtered_data, fdata)
        gradSmagnitude = np.abs(gradS)

        chiFunction = partitionFrequencyBand(fdata, gradS, keep = 'above', cutoff = 0.05)
        fc_index = np.argmax(gradSmagnitude)

        freq_arrays = np.split(fdata, [fc_index])
        data_arrays = np.split(sdata, [fc_index])
        chi_arrays = np.split(chiFunction, [fc_index])

        #TODO: trim data and perform two separate fits, average values for delay_guess

        delay_fit = []
        lrg_result = [0,0]
        trimmed_freq = [0,0]
        for n in range(2):
            #trim the data
            trim = np.nonzero(chi_arrays[n])
            trimmed_freq[n] = np.delete(freq_arrays[n], trim)
            trimmed_data = np.delete(data_arrays[n], trim)

            #fit trimmed freq and data
            trimmed_phase = np.unwrap(np.angle(trimmed_data))
            lrg_result[n] = linregress(trimmed_freq[n], trimmed_phase)
            delay_guess = lrg_result[n].slope / (-2 * np.pi)
            delay_fit = np.append(delay_fit, delay_guess)
        avg_delay_guess = (delay_fit[0]+delay_fit[1])/2

        #plot the fits as a check
        '''
        import matplotlib.pyplot as plt
        plt.plot(fdata, np.unwrap(np.angle(sdata)))
        plt.plot(trimmed_freq[0], lrg_result[0].intercept+lrg_result[0].slope*trimmed_freq[0], color = 'k', linestyle = 'dashed')
        plt.plot(trimmed_freq[1], lrg_result[1].intercept + lrg_result[1].slope * trimmed_freq[1], color='k',
                 linestyle='dashed')
        plt.show()
        '''

        print(f'initial delay guess: {avg_delay_guess}')
        return avg_delay_guess

    def find_delay_circlefit(self, fdata: np.ndarray, sdata: np.ndarray):
        '''
        Finds the electrical delay using the circle fit method described in Probst et al.

        The delay is varied to minimize the deviation of the scattering data from an ideal circle in the complex plane.
        REVIEW OF SCIENTIFIC INSTRUMENTS 86, 024706 (2015). Results are unsatisfactory, but may be improved by adopting
        the heuristics present in self.find_delay

        Args:
            fdata: numpy array of the frequency data
            sdata: numpy array of the scattering data

        Returns:
            delay: float representing the electrical delay
        '''

        phase_data = np.unwrap(np.angle(sdata))
        #the initial guess just needs to get the scale right, this should be good enough
        delay_guess = self.guess_delay(fdata, sdata)
        print(f'delay initial guess: {delay_guess}')

        # make an lmfit.Parameters object and add a single lmfit.Parameter object representing the electrical delay
        delay_params = lmfit.Parameters()
        delay_params.add('delay', delay_guess, max = delay_guess+0.2*np.abs(delay_guess), min = delay_guess-0.2*np.abs(delay_guess), brute_step = np.abs(delay_guess)/10000)

        min_result = lmfit.minimize(fcn = self.circle_deviation, params = delay_params, args = (fdata, sdata), method = 'leastsq')
        electrical_delay = min_result.params['delay'].value

        return electrical_delay

    def arctan_deviation(self, params: lmfit.Parameters, fdata: np.ndarray, sdata: np.ndarray):
        '''
        An objective function to be minimized to find the electrical delay.

        The delay value passed through params is removed from the scattering data. Then the best fit circle is found for
        the resulting data. Its center is translated to the origin. If the delay removed corresponds to the actual
        electrical delay then the phase of this transformed data is an arctangent.

        Args:
            params: lmfit.Parameters object to store parameters for the arctangent function and the delay
            fdata: numpy array of the frequency data
            sdata: numpy array of the scattering data

        Returns:
            residuals: numpy array of the residuals comparing the transformed data to the arctangent described by params
        '''
        fr = params['fr'].value
        Ql = params['Ql'].value
        t_delay = params['delay'].value
        theta_0 = params['theta_0'].value

        sdata = remove_delay(fdata, sdata, t_delay)
        xc, yc, r = find_circle(np.real(sdata), np.imag(sdata))
        sdata = sdata - (xc+1j*yc)

        offset_phase = np.unwrap(np.angle(sdata))

        residuals = np.abs(offset_phase - (theta_0+2*np.arctan(2*Ql*(1-fdata/fr))))
        return residuals

    def circle_deviation(self, params: lmfit.Parameters, fdata: np.ndarray, sdata: np.ndarray):
        '''
        this is the objective function to be minimized to find the electrical delay in the Probst circlefit method
        '''
        # the signature must be: fcn(params, *args, **kws)
        # the data is passed into minimize with args = (fdata, sdata) as a tuple
        # electrical delay is the only param, but the data is passed through *args
        # this must return an array of residuals r^2-(x_n-x_c)^2-(y_n-y_c)^2

        # unpack param value
        delay = params['delay'].value
        # remove that amount of delay
        adjusted_sdata = remove_delay(fdata, sdata, delay)
        # grab x any y components of sdata
        # calculate r, x_c, and y_c from the circlefit method
        x_c, y_c, r = find_circle(np.real(sdata), np.imag(sdata))
        # calculate and return array of terms like r**2-(x_n-x_c)**2-(y_n-y_c)**2
        return np.abs(r**2-(np.real(adjusted_sdata)-x_c)**2-(np.imag(adjusted_sdata)-y_c)**2)

    def find_off_res_point(self, fdata, sdata):
        """
        This replaces the poorly named calibration function
        Args:
            fdata: numpy array of the frequency data
            sdata: numpy array of the scattering data

        Returns:
            the off-resonant point as a complex float
            additionally sets self.phi
        """
        if self.theta_0 is not None:
            beta = fmod(self.theta_0+np.pi, np.pi)
            xc, yc, r = find_circle(np.real(sdata), np.imag(sdata))
            self.phi = fmod(np.pi - self.theta_0 + np.angle(xc+1j*yc), np.pi)
            print(f'phi (preprocessing): {self.phi}')
            return xc+ 1j*yc+ r*np.exp(1j*beta)
        else:
            #this block is untested
            #TODO: fit the offset phase to an arctan (without delay, that is not the place for this function)
            orpmodel = lmfit.Model(sloped_arctan)
            guess_params = self.fit_method.find_initial_guess(self=self.fit_method, fdata=fdata, sdata=sdata)

            orparams = lmfit.Parameters()
            orparams.add(name = 'Ql', value = guess_params['Q'].value)
            orparams.add(name = 'fr', value = guess_params['f0'].value)
            orparams.add(name = 'delay', value = 0, vary = False)
            orparams.add(name = 'theta_0', value = 0)

            orp_results = orpmodel.fit(np.unwrap(np.angle(sdata)), orparams, f=fdata)
            theta_0 = orp_results.params['theta_0'].value
            beta = fmod(theta_0 + np.pi, np.pi)
            xc, yc, r = find_circle(np.real(sdata), np.imag(sdata))
            self.phi = fmod(np.pi - theta_0 + np.angle(xc + 1j * yc), np.pi)
            print(f'phi (preprocessing): {self.phi}')
            return xc + 1j * yc + r * np.exp(1j * beta)

    def anchor_to_point(self, fdata: np.ndarray, sdata: np.ndarray, anchor_point = None):
        """
        Rotates and scales the scattering data to send the off-resonant point to an anchor point (e.g. 1+0*1j).

        Args:
            fdata: numpy array of the frequency data
            sdata: numpy array of the scattering data
            anchor_point: optional complex float to specify the desired off-resonant point (defaults to 1+0j)

        Returns:
            sdata: numpy array of the transformed scattering data
        """
        if anchor_point == None:
            anchor_point = self.off_res_point

        Old_ORP = self.find_off_res_point(fdata, sdata)
        sdata = anchor_point*sdata/Old_ORP
        return sdata
