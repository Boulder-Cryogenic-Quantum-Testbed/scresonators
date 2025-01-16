import numpy as np
import scipy.optimize as spopt
from scipy.ndimage import gaussian_filter1d
import logging
from scipy import stats
import plot as fp
from utilities import find_circle
from scipy.interpolate import interp1d


class DataProcessor: 
    def __init__(self, resonator, normalize_pts, preprocess_method):

        self.resonator = resonator
        self.normalize_pts = normalize_pts
        self.preprocess_method = preprocess_method

        self.freqs = resonator.freqs
        self.ydata = resonator.S21_data

        self.filename = resonator.filename
        self.dir = resonator.dir

    def process_data(self):
        # Process according to selected method

        if self.preprocess_method == "linear":
            y_data = self.preprocess_linear(
                self.freqs, self.ydata, self.normalize_pts
            )
        elif self.preprocess_method == "circle":
            y_data = self.preprocess_circle(self.freqs, self.ydata, self.normalize_pts)
        
        return self.freqs, y_data

    def phase_centered(self, f, fr, Ql, theta, delay=0.):
        """
            Yields the phase response of a strongly overcoupled (Qi >> Qc) resonator
            in reflection which corresponds to a circle centered around the origin.
            Additionally, a linear background slope is accounted for if needed.

            Args:
                fr: Resonance frequency
                Ql: Loaded quality factor (and since Qi >> Qc also Ql = Qc)
                theta: Offset phase
                delay (opt.): Time delay between output and input signal leading to
                            linearly frequency dependent phase shift
            Returns:
                Phase response (float)
        """
        return theta - 2 * np.pi * delay * (f - fr) + 2. * np.arctan(2. * Ql * (1. - f / fr))
    
    def phase_dist(self, angle):
        """
            Maps angle [-2pi, +2pi] to phase distance on circle [0, pi]
            """
        return np.pi - np.abs(np.pi - np.abs(angle))
    
    def fit_phase(self, f_data, z_data, guesses=None):
        """
            Fits the phase response of a strongly overcoupled (Qi >> Qc) resonator
            in reflection which corresponds to a circle centered around the origin.

            Args:
                z_data: Scattering data of which the phase should be fit. Data must be
                    distributed around origin ("circle-like").
                guesses (opt.): If not given, initial guesses for the fit parameters
                            will be determined. If given, should contain useful
                            guesses for fit parameters as a tuple (fr, Ql, delay)

            Returns:
                fr: Resonance frequency
                Ql: Loaded quality factor
                theta: Offset phase
                delay: Time delay between output and input signal leading to linearly
                    frequency dependent phase shift
            """
        phase = np.unwrap(np.angle(z_data))

        # For centered circle roll-off should be close to 2pi. If not warn user.
        if np.max(phase) - np.min(phase) <= 0.8 * 2 * np.pi:
            logging.warning(
                "Data does not cover a full circle (only {:.1f}".format(
                    np.max(phase) - np.min(phase)
                )
                + " rad). Increase the frequency span around the resonance?"
            )
            roll_off = np.max(phase) - np.min(phase)
        else:
            roll_off = 2 * np.pi

        # Set useful starting parameters
        if guesses is None:
            # Use maximum of derivative of phase as guess for fr
            phase_smooth = gaussian_filter1d(phase, 30)
            phase_derivative = np.gradient(phase_smooth)
            fr_guess = f_data[np.argmax(np.abs(phase_derivative))]
            Ql_guess = 2 * fr_guess / (f_data[-1] - f_data[0])
            # Estimate delay from background slope of phase (substract roll-off)
            slope = phase[-1] - phase[0] + roll_off
            delay_guess = -slope / (2 * np.pi * (f_data[-1] - f_data[0]))
        else:
            fr_guess, Ql_guess, delay_guess = guesses
        # This one seems stable and we do not need a manual guess for it
        theta_guess = 0.5 * (np.mean(phase[:5]) + np.mean(phase[-5:]))

        # Fit model with less parameters first to improve stability of fit

        def residuals_Ql(params):
            Ql, = params
            return residuals_full((fr_guess, Ql, theta_guess, delay_guess))

        def residuals_fr_theta(params):
            fr, theta = params
            return residuals_full((fr, Ql_guess, theta, delay_guess))

        def residuals_delay(params):
            delay, = params
            return residuals_full((fr_guess, Ql_guess, theta_guess, delay))

        def residuals_fr_Ql(params):
            fr, Ql = params
            return residuals_full((fr, Ql, theta_guess, delay_guess))

        def residuals_full(params):
            return self.phase_dist(
                phase - self.phase_centered(f_data, *params)
            )

        p_final = spopt.leastsq(residuals_Ql, [Ql_guess])
        Ql_guess, = p_final[0]
        p_final = spopt.leastsq(residuals_fr_theta, [fr_guess, theta_guess])
        fr_guess, theta_guess = p_final[0]
        p_final = spopt.leastsq(residuals_delay, [delay_guess])
        delay_guess, = p_final[0]
        p_final = spopt.leastsq(residuals_fr_Ql, [fr_guess, Ql_guess])
        fr_guess, Ql_guess = p_final[0]
        p_final = spopt.leastsq(residuals_full, [fr_guess, Ql_guess, theta_guess, delay_guess])

        return p_final[0]
    
    def fit_delay(self, xdata: np.ndarray, ydata: np.ndarray):
        """
            Finds the cable delay by repeatedly centering the "circle" and fitting
            the slope of the phase response.
            """

        # Translate data to origin
        xc, yc, r0 = find_circle(np.real(ydata), np.imag(ydata))
        z_data = ydata - complex(xc, yc)
        fr, Ql, theta, delay = self.fit_phase(xdata, z_data)

        # Do not overreact (see end of for loop)
        delay *= 0.05
        delay_corr = 0
        residuals = 0
        # Iterate to improve result for delay
        for i in range(10):
            # Translate new best fit data to origin
            z_data = ydata * np.exp(2j * np.pi * delay * xdata)
            xc, yc, r0 = find_circle(np.real(z_data), np.imag(z_data))
            z_data -= complex(xc, yc)

            # Find correction to current delay
            guesses = (fr, Ql, 5e-11)
            fr, Ql, theta, delay_corr = self.fit_phase(xdata, z_data, guesses)

            # Stop if correction would be smaller than "measurable"
            phase_fit = self.phase_centered(xdata, fr, Ql, theta, delay_corr)
            residuals = np.unwrap(np.angle(z_data)) - phase_fit
            if 2 * np.pi * (xdata[-1] - xdata[0]) * delay_corr <= np.std(residuals):
                break

            # Avoid overcorrection that makes procedure switch between positive
            # and negative delays
            if delay_corr * delay < 0:  # different sign -> be careful
                if abs(delay_corr) > abs(delay):
                    delay *= 0.5
                else:
                    delay += 0.1 * np.sign(delay_corr) * 5e-11
            else:  # same direction -> can converge faster
                if abs(delay_corr) >= 1e-8:
                    delay += min(delay_corr, delay)
                elif abs(delay_corr) >= 1e-9:
                    delay *= 1.1
                else:
                    delay += delay_corr

        if 2 * np.pi * (xdata[-1] - xdata[0]) * delay_corr > np.std(residuals):
            logging.warning("Delay could not be fit properly!")

        return delay
    
    def periodic_boundary(self, angle):
        # Maps arbitrary angle to interval [-np.pi, np.pi)
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def calibrate(self, x_data: np.ndarray, z_data: np.ndarray):
        """
            Finds the parameters for normalization of the scattering data. See
            Sij of port classes for explanation of parameters.
        """

        # Translate circle to origin
        xc, yc, r = find_circle(np.real(z_data), np.imag(z_data))
        zc = complex(xc, yc)
        z_data2 = z_data - zc

        # Find off-resonant point by fitting offset phase
        # (centered circle corresponds to lossless resonator in reflection)
        fr, Ql, theta, delay_remaining = self.fit_phase(x_data, z_data2)
        angle = theta - np.pi
        beta = self.periodic_boundary(angle)
        offrespoint = zc + r * np.cos(beta) + 1j * r * np.sin(beta)
        a = np.absolute(offrespoint)
        alpha = np.angle(offrespoint)
        phi = self.periodic_boundary(beta - alpha)

        # Store radius for later calculation
        r /= a

        return delay_remaining, a, alpha, theta, phi, fr, Ql

    def normalize(self, z_data, a, alpha):
        """
            Transforms scattering data into canonical position with off-resonant
            point at (1, 0) (does not correct for rotation phi of circle around
            off-resonant point).
            """
        z_norm = (z_data / a) * np.exp(1j * (-alpha))
        return z_norm

    def preprocess_linear(self, xdata: np.ndarray, ydata: np.ndarray, normalize: int):
        """
        Data Preprocessing. Get rid of cable delay and normalize phase/magnitude of S21 by linear fit of normalize # of endpoints
        """

        if normalize * 2 > len(ydata):
            print("Not enough points to normalize, please lower value of normalize variable or take more points near resonance")

        # Check for bad linear preprocessing outputs
        # Redirect to circle preprocessing
        phase = np.unwrap(np.angle(ydata))

        # normalize phase of S21 using linear fit
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.append(xdata[0:normalize], xdata[-normalize:]),
                                                                    np.append(phase[0:normalize], phase[-normalize:]))

        angle = np.subtract(phase, slope * xdata)  # remove cable delay
        y_test = np.multiply(np.abs(ydata), np.exp(1j * angle))
        

        angle = np.subtract(angle, intercept)  # rotate off resonant point to (1,0i) in complex plane
        y_test = np.multiply(np.abs(ydata), np.exp(1j * angle))
        
        # normalize magnitude of S21 using linear fit
        y_db = np.log10(np.abs(ydata)) * 20
        slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(
            np.append(xdata[0:normalize], xdata[-normalize:]), np.append(y_db[0:normalize], y_db[-normalize:]))
        magnitude = np.subtract(y_db, slope2 * xdata + intercept2)
        magnitude = 10 ** (magnitude / 20)

        preprocessed_data = np.multiply(magnitude, np.exp(1j * angle))

        return preprocessed_data

    def preprocess_freq_dep_atten(self, xdata: np.ndarray, ydata: np.ndarray, normalize_pts=None):
        """
            we are modeling this frequency dependent **attenuation** as a magn-only effect
            that has a constant and a linear term, so it looks like: 
                    S_21_data = S_21_ideal * (A + B_0 * f)
                    
            then we simply remove it by dividing that term out
                    S_21_preprocessed = S_21_data / (A + B_0 * f)
        
        """
        
        # by default use 10% on each side of the trace
        # so a 50 point trace has 5 points on each side to do a linear fit
        if normalize_pts is None:
            normalize_pts = int(len(ydata) * 0.10)
        magn = np.abs(ydata)

        # normalize phase of S21 using linear fit
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.append(xdata[0:normalize_pts], xdata[-normalize_pts:]),
                                                                    np.append(magn[0:normalize_pts], magn[-normalize_pts:]))
        
        preprocessed_data = ydata / (intercept + slope*xdata)
        
        return preprocessed_data, slope, intercept
    
    def preprocess_circle(self, xdata: np.ndarray, ydata: np.ndarray, normalize_pts : int):
        """
        Data Preprocessing. Use Probst method to get rid of cable delay and normalize phase/magnitude of S21 by circle fit
        """
        # # Unwrap the phase
        # phase = np.unwrap(np.angle(ydata))
        # ydata = np.abs(ydata) * np.exp(1j * phase)

        # remove freq dependence in magnitude  (freq-dep attenuators)
        ydata_linear_removed, intercept, slope = self.preprocess_freq_dep_atten(xdata, ydata, normalize_pts)


        # remove freq dependence in phase  (cable delay)
        delay = self.fit_delay(xdata, ydata_linear_removed)
        z_data = ydata_linear_removed * np.exp(2j * np.pi * delay * xdata)


        # calibrate and normalize
        delay_remaining, a, alpha, theta, phi, fr, Ql = self.calibrate(xdata, z_data)
        z_norm = self.normalize(z_data, a, alpha)

        return z_norm
    
    def background_removal(databg, linear_amps: np.ndarray, 
                       phases: np.ndarray, output_path: str):
        x_bg = databg.freqs
        linear_amps_bg = databg.linear_amps
        phases_bg = databg.phases

        ybg = np.multiply(linear_amps_bg, np.exp(1j * phases_bg))

        fmag = interp1d(x_bg, linear_amps_bg, kind='cubic')
        fang = interp1d(x_bg, phases_bg, kind='cubic')

        fp.plot2(databg.freqs, databg.linear_amps, x_bg, linear_amps_bg, "VS_mag", output_path)
        fp.plot2(databg.freqs, databg.phases, x_bg, phases_bg, "VS_ang", output_path)
        
        linear_amps = np.divide(linear_amps, linear_amps_bg)
        phases = np.subtract(phases, phases_bg)

        return np.multiply(linear_amps, np.exp(1j * phases))