import numpy as np
import cavity_functions as ff
from utilities import * 
from .fit_method import FitMethod
import scipy.optimize as spopt
from uncertainties import ufloat, umath
import lmfit

class DCM(FitMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'DCM'
        self.func = ff.cavity_DCM

    def func(self, x, Q, Qc, w1, phi): 
        #DCM fit function
        return np.array(1-Q/Qc*np.exp(1j*phi)/(1 + 1j*(x-w1)/w1*2*Q))
    
    def calculate_manual_initial_guess(self):
        """DCM-specific initial guess calculation."""
        init = self.manual_init
        Qc = init[1] / np.exp(1j * init[3])
        init[0] = 1 / (1 / init[0] + np.real(1 / Qc))
        kappa = init[2] / init[0]
        freq = init[2]
        return init, kappa, freq
    
    def auto_initial_guess(self, x, y1, y2):
        """ Return init, xc, yc, r """

        try:
            # recombine transmission S21 from real and complex parts
            y = y1 + 1j * y2
        except:
            raise ValueError(">Problem initializing data in find_initial_guess(), please make",
                " sure data is of correct format")

        try:
            # find circle that matches the data
            x_c, y_c, r = find_circle(y1, y2)
            # define complex number to house circle center location data
            z_c = x_c + 1j * y_c
        except:
            raise ValueError(">Problem in function find_circle, please make sure data is of ",
                "correct format")
    
        try:
            ## move gap of circle to (0,0)
            # Center point P at (0,0)
            ydata = y - 1
            # Shift guide circle to match data shift
            z_c = z_c - 1
        except:
            raise ValueError(">Error when trying to shift data into canonical position",
                " minus 1")
        
        try:
            # determine the angle to the center of the fitting circle from origin
            phi = np.angle(-z_c)
            freq_idx = np.argmax(np.abs(ydata))
            f_c = x[freq_idx]

            # rotate resonant freq to minimum
            ydata = ydata * np.exp(-1j * phi)
            z_c = z_c * np.exp(-1j * phi)

        except:
            raise ValueError(">Error when trying to shift data according to phi in ",
                "find_initial_guess")
        
        if f_c < 0:
            raise ValueError(">Resonance frequency is negative. Please only input ",
                "positive frequencies.")
        
        try:
            # diameter of the circle found from getting distance from (0,0) to 
            # resonance frequency data point (possibly should use fit circle)
            Q_Qc = np.max(np.abs(ydata))
            # y_temp = |ydata|-(diameter/sqrt(2))
            y_temp = np.abs(np.abs(ydata) - np.max(np.abs(ydata)) / 2 ** 0.5)

            # find min value in y_temp on one half of circle from resonance
            _, idx1 = find_nearest(y_temp[0:freq_idx], 0)
            # find min value in y_temp on other half of circle from resonance
            _, idx2 = find_nearest(y_temp[freq_idx:], 0)
            # add index of resonance frequency to get correct index for idx2
            idx2 = idx2 + freq_idx
            # bandwidth of frequencies
            kappa = abs((x[idx1] - x[idx2]))

            Q = f_c / kappa
            Qc = Q / Q_Qc
            # fits parameters for the 3 terms given in p0 
            # (this is where Qi and Qc are actually guessed)
            popt, pcov = spopt.curve_fit(ff.one_cavity_peak_abs, x, 
                                         np.abs(ydata), p0=[Q, Qc, f_c], 
                                         bounds=(0, [np.inf] * 3))
            Q = popt[0]
            Qc = popt[1]
            init_guess = [Q, Qc, f_c, phi]

        except Exception as e:
            print(e)
            raise RuntimeError(">Failed to find initial guess for method DCM.",
                      " Please manually initialize a guess")
        
        return init_guess, x_c, y_c, r

    def add_params(self, params_arr):

        params = lmfit.Parameters()
        params.add('Q', value=params_arr[0], vary=self.change_Q, min=params_arr[0] * 0.5, max=params_arr[0] * 1.5)
        params.add('Qc', value=params_arr[1], vary=self.change_Qc, min=params_arr[1] * 0.8, max=params_arr[1] * 1.2)
        params.add('w1', value=params_arr[2], vary=self.change_w1, min=params_arr[2] * 0.9, max=params_arr[2] * 1.1)
        params.add('phi', value=params_arr[3], vary=self.change_phi, min=-np.pi, max=np.pi)

        return params
    
    def min_one_cavity(self, parameter, x, data=None):
        #fit function call for DCM fitting method
        Q = parameter['Q']
        Qc = parameter['Qc']
        w1 = parameter['w1']
        phi = parameter['phi']

        model = self.cavity_DCM(x, Q, Qc, w1,phi)
        real_model = model.real
        imag_model = model.imag
        real_data = data.real
        imag_data = data.imag

        resid_re = real_model - real_data
        resid_im = imag_model - imag_data
        return np.concatenate((resid_re,resid_im))
    
    def min_fit(self, params, xdata, ydata):

        """
            Minimizes parameter values for the given function and transmission data

            Args:
                params: guess for correct values of the fit parameters
                xdata: array of frequency data points
                ydata: array of S21 data points
                Method: instance of Method class

            Returns:
                minimized parameter values, 95% confidence intervals for those parameter values
        """

        try:
            
            minner = lmfit.Minimizer(self.fit_method.min_one_cavity, params, fcn_args=(xdata, ydata))
            result = minner.minimize(method='least_squares')
            fit_params = result.params
            parameter = fit_params.valuesdict()

            # extracts the actual value for each parameter and puts it in the fit_params list
            fit_params = [value for _, value in parameter.items()]

        except:

            print(">Failed to minimize data for least squares fit")
            print(">Confidence intervals unknown and given as 0.0")
            fit_params = None
            conf_array = [0, 0, 0, 0, 0, 0]
            return fit_params, conf_array
        
        try:

            p_names = []

            for parameter in params:
                if parameter not in self.fit_method.MC_fix:
                    p_names.append(parameter)

            
            ci = lmfit.conf_interval(minner, result, p_names=p_names, sigmas=[2])

            # confidence interval for Q
            Q_conf = max(np.abs(ci['Q'][1][1] - ci['Q'][0][1]), np.abs(ci['Q'][1][1] - ci['Q'][2][1]))

            # confidence interval for Qc
            Qc_conf = max(np.abs(ci['Qc'][1][1] - ci['Qc'][0][1]), np.abs(ci['Qc'][1][1] - ci['Qc'][2][1]))
            # Ignore one-sided conf test
            if np.isinf(Qc_conf):
                Qc_conf = min(np.abs(ci['Qc'][1][1] - ci['Qc'][0][1]), np.abs(ci['Qc'][1][1] - ci['Qc'][2][1]))
            # confidence interval for 1/Re[1/Qc]
            Qc_Re = 1 / np.real(np.exp(1j * fit_params[3]) / ci['Qc'][1][1])
            Qc_Re_neg = 1 / np.real(np.exp(1j * fit_params[3]) / ci['Qc'][0][1])
            Qc_Re_pos = 1 / np.real(np.exp(1j * fit_params[3]) / ci['Qc'][2][1])
            Qc_Re_conf = max(np.abs(Qc_Re - Qc_Re_neg), np.abs(Qc_Re - Qc_Re_pos))
            # Ignore one-sided conf test
            if np.isinf(Qc_Re_conf):
                Qc_Re_conf = min(np.abs(Qc_Re - Qc_Re_neg), np.abs(Qc_Re - Qc_Re_pos))

            # confidence interval for phi
            phi_conf = max(np.abs(ci['phi'][1][1] - ci['phi'][0][1]), 
                            np.abs(ci['phi'][1][1] - ci['phi'][2][1]))
            # Ignore one-sided conf test
            if np.isinf(phi_conf):
                phi_conf = min(np.abs(ci['phi'][1][1] - ci['phi'][0][1]), 
                                np.abs(ci['phi'][1][1] - ci['phi'][2][1]))

            # confidence interval for resonance frequency
            w1_conf = max(np.abs(ci['w1'][1][1] - ci['w1'][0][1]), 
                            np.abs(ci['w1'][1][1] - ci['w1'][2][1]))
            # Ignore one-sided conf test
            if np.isinf(w1_conf):
                w1_conf = min(np.abs(ci['w1'][1][1] - ci['w1'][0][1]), 
                                np.abs(ci['w1'][1][1] - ci['w1'][2][1]))
                    
            # confidence interval for Qi
            Q_ufloat = ufloat(ci['Q'][1][1], Q_conf)
            Qc_ufloat = ufloat(ci['Qc'][1][1], Qc_conf)
            phi_ufloat = ufloat(ci['phi'][1][1], phi_conf)
            Qi_ufloat = (1/Q_ufloat - umath.cos(phi_ufloat)/Qc_ufloat) ** -1

            # Array of confidence intervals
            conf_array = [Q_conf, Qi_ufloat.s, Qc_conf, Qc_Re_conf, phi_conf, w1_conf]
                
        except Exception as e:
            print(e)
            print(">Failed to find confidence intervals for least squares fit")
            conf_array = [0, 0, 0, 0, 0, 0]
            
        return fit_params, conf_array
    
    def monte_carlo_fit(self, xdata, ydata, parameter):

        # check if all parameters are defined
        assert xdata is not None, "xdata is not defined"
        assert ydata is not None, "ydata is not defined"
        assert parameter is not None, "parameter is not defined"

        try:
            ydata_1stfit = self.fit_method.func(xdata, *parameter)  # set of S21 data based on initial guess parameters

            ## weight condition
            if self.fit_method.MC_weight == 'yes':
                weight_array = 1 / abs(ydata)  # new array of inversed magnitude ydata
            else:
                weight_array = np.full(len(xdata), 1)  # new array of len(xdata) all slots filled with 1

            weighted_ydata = np.multiply(weight_array,
                                        ydata)  # array filled with 1s if MC_weight='yes' and exact same array as ydata otherwise
            weighted_ydata_1stfit = np.multiply(weight_array,
                                                ydata_1stfit)  # array with values (ydata^(-1))*ydata_1stfit if MC_weight='yes' and exact same array as ydata_1stfit otherwise
            error = np.linalg.norm(weighted_ydata - weighted_ydata_1stfit) / len(
                xdata)  # first error #finds magnitude of (weighted_ydata-weighted_ydata_1stfit) and divides by length (average magnitude)
            error_0 = error

        except:
            raise ValueError(">Failed to initialize monte_carlo_fit(), please check parameters")
        # Fix condition and Monte Carlo Method with random number Generator

        counts = 0
        try:
            while counts < self.fit_method.MC_rounds:
                counts = counts + 1
                # generate an array of 4 random numbers between -0.5 and 0.5 in the format [r,r,r,r] where r is each of the random numbers times the step constant
                random = self.fit_method.MC_step_const * (np.random.random_sample(len(parameter)) - 0.5)
                if 'Q' in self.fit_method.MC_fix:
                    random[0] = 0
                if 'Qi' in self.fit_method.MC_fix:
                    random[0] = 0
                if 'Qc' in self.fit_method.MC_fix:
                    random[1] = 0
                if 'w1' in self.fit_method.MC_fix:
                    random[2] = 0
                if 'phi' in self.fit_method.MC_fix:
                    random[3] = 0
                if 'Qa' in self.fit_method.MC_fix:
                    random[3] = 0
                    
                # Generate new parameter to test
                random[3] = random[3] * 0.1
                random = np.exp(random)
                new_parameter = np.multiply(parameter, random)
                new_parameter[3] = np.mod(new_parameter[3], 2 * np.pi)

                # new set of data with new parameters
                ydata_MC = self.fit_method.func(xdata, *new_parameter)
                # check new error with new set of parameters
                weighted_ydata_MC = np.multiply(weight_array, ydata_MC)
                new_error = np.linalg.norm(weighted_ydata_MC - weighted_ydata) / len(xdata)
                if new_error < error:
                    parameter = new_parameter
                    error = new_error
        except:
            raise RuntimeError(">Error in while loop of monte_carlo_fit")
        ## If finally gets better fit then plot ##
        if error < error_0:
            stop_MC = False
            print('Monte Carlo fit got better fitting parameters')
            if self.fit_method.manual_init != None:
                print('>User input parameters getting stuck in local minimum, please input more accurate parameters')
        else:
            stop_MC = True
        return parameter, stop_MC, error