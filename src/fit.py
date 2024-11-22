import os
import lmfit
import numpy as np
import cavity_functions as ff
import plot as fp
from uncertainties import ufloat, umath
from utilities import *

class Fitter:
    
    def __init__(self, fit_method, preprocessor):
        """
        Fitter class to handle the fitting of resonator data.

        Args:
            fit_method (FitMethod): Instance of the fit method to be used (e.g., DCMFitMethod).
            preprocessor (DataPreprocessor): Instance of the DataPreprocessor for data preprocessing.
        """
        self.fit_method = fit_method
        self.preprocessor = preprocessor

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
                if self.fit_method.name != 'CPZM':
                    random[3] = random[3] * 0.1
                random = np.exp(random)
                new_parameter = np.multiply(parameter, random)
                if self.fit_method.name  != 'CPZM':
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

    def min_fit(self, params, xdata, ydata):
        """Minimizes parameter values for the given function and transmission data

        Args:
            params: guess for correct values of the fit parameters
            xdata: array of frequency data points
            ydata: array of S21 data points
            Method: instance of Method class

        Returns:
            minimized parameter values, 95% confidence intervals for those parameter values
        """
        
        fit_method_name = self.fit_method.name
        try:
            if fit_method_name == 'DCM' or fit_method_name == 'PHI':
                minner = lmfit.Minimizer(ff.min_one_Cavity_dip, params, fcn_args=(xdata, ydata))
            elif fit_method_name == 'DCM REFLECTION':
                minner = lmfit.Minimizer(ff.min_one_Cavity_DCM_REFLECTION, params, fcn_args=(xdata, ydata))
            elif fit_method_name == 'INV':
                minner = lmfit.Minimizer(ff.min_one_Cavity_inverse, params, fcn_args=(xdata, ydata))
            elif fit_method_name == 'CPZM':
                minner = lmfit.Minimizer(ff.min_one_Cavity_CPZM, params, fcn_args=(xdata, ydata))

            else:
                raise ValueError(">Method is not defined. Please choose a method: DCM, ",
                    "DCM REFLECTION, PHI, INV or CPZM")

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
            if fit_method_name == 'DCM' or fit_method_name == 'PHI' or fit_method_name == 'DCM REFLECTION':
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

                if fit_method_name == 'PHI':
                    Qi_ufloat = (1/Q_ufloat - umath.fabs(1/Qc_ufloat)) ** -1
                else:
                    Qi_ufloat = (1/Q_ufloat - umath.cos(phi_ufloat)/Qc_ufloat) ** -1

                # Array of confidence intervals
                conf_array = [Q_conf, Qi_ufloat.s, Qc_conf, Qc_Re_conf, phi_conf, w1_conf]

            elif fit_method_name == 'INV':
                ci = lmfit.conf_interval(minner, result, p_names=p_names, sigmas=[2])
                # confidence interval for Qi
                if 'Qi' in p_names:
                    Qi_conf = max(np.abs(ci['Qi'][1][1] - ci['Qi'][0][1]), 
                                np.abs(ci['Qi'][1][1] - ci['Qi'][2][1]))
                else:
                    Qi_conf = 0
                # confidence interval for Qc
                if 'Qc' in p_names:
                    Qc_conf = max(np.abs(ci['Qc'][1][1] - ci['Qc'][0][1]), 
                                np.abs(ci['Qc'][1][1] - ci['Qc'][2][1]))
                else:
                    Qc_conf = 0
                # confidence interval for phi
                if 'phi' in p_names:
                    phi_conf = max(np.abs(ci['phi'][1][1] - ci['phi'][0][1]), 
                                np.abs(ci['phi'][1][1] - ci['phi'][2][1]))
                    # Ignore one-sided conf test
                    if np.isinf(phi_conf):
                        phi_conf = min(np.abs(ci['phi'][1][1] - ci['phi'][0][1]), 
                                    np.abs(ci['phi'][1][1] - ci['phi'][2][1]))
                else:
                    phi_conf = 0
                # confidence interval for resonance frequency
                if 'w1' in p_names:
                    w1_conf = max(np.abs(ci['w1'][1][1] - ci['w1'][0][1]), 
                                np.abs(ci['w1'][1][1] - ci['w1'][2][1]))
                    # Ignore one-sided conf test
                    if np.isinf(w1_conf):
                        w1_conf = min(np.abs(ci['w1'][1][1] - ci['w1'][0][1]), 
                                    np.abs(ci['w1'][1][1] - ci['w1'][2][1]))
                else:
                    w1_conf = 0
                # Array of confidence intervals
                conf_array = [Qi_conf, Qc_conf, phi_conf, w1_conf]
            else:
                ci = lmfit.conf_interval(minner, result, p_names=p_names, sigmas=[2])
                # confidence interval for Qi
                if 'Qi' in p_names:
                    Qi_conf = max(np.abs(ci['Qi'][1][1] - ci['Qi'][0][1]), 
                                np.abs(ci['Qi'][1][1] - ci['Qi'][2][1]))
                else:
                    Qi_conf = 0
                # confidence interval for Qc
                if 'Qc' in p_names:
                    Qc = ci['Qi'][1][1] / ci['Qc'][1][1]
                    Qc_neg = ci['Qi'][0][1] / ci['Qc'][0][1]
                    Qc_pos = ci['Qi'][2][1] / ci['Qc'][2][1]
                    Qc_conf = max(np.abs(Qc - Qc_neg), np.abs(Qc - Qc_neg))
                else:
                    Qc_conf = 0
                # confidence interval for Qa
                if 'Qa' in p_names:
                    Qa = ci['Qi'][1][1] / ci['Qa'][1][1]
                    Qa_neg = ci['Qi'][2][1] / ci['Qa'][2][1]
                    Qa_pos = ci['Qi'][0][1] / ci['Qa'][0][1]
                    Qa_conf = max(np.abs(Qa - Qa_neg), np.abs(Qa - Qa_neg))
                else:
                    Qa_conf = 0
                # confidence interval for resonance frequency
                if 'w1' in p_names:
                    w1_conf = max(np.abs(ci['w1'][1][1] - ci['w1'][0][1]), 
                                np.abs(ci['w1'][1][1] - ci['w1'][2][1]))
                    # Ignore one-sided conf test
                    if np.isinf(w1_conf):
                        min(np.abs(ci['w1'][1][1] - ci['w1'][0][1]), 
                            np.abs(ci['w1'][1][1] - ci['w1'][2][1]))
                else:
                    w1_conf = 0
                # Array of confidence intervals
                conf_array = [Qi_conf, Qc_conf, Qa_conf, w1_conf]
        except Exception as e:
            print(e)
            print(">Failed to find confidence intervals for least squares fit")
            conf_array = [0, 0, 0, 0, 0, 0]
            
        return fit_params, conf_array

    def fit(self):

        x_initial, y_initial = self.preprocessor.freqs, self.preprocessor.ydata

        # normalize data
        slope = 0
        intercept = 0
        slope2 = 0
        intercept2 = 0

        xdata, ydata = self.preprocessor.process_data()
        x_raw, y_raw = xdata, ydata

         # Step 1 -- Find initial guess if not specified and extract part of data close to resonance

        init = [0] * 4 # store initial guess parameters

        # When user manually initializes a guess initialize the following variable
        manual_init = self.fit_method.manual_init
        if manual_init is not None: 
            try: 
                if len(manual_init) == 4:
                    init, kappa, freq = self.fit_method.calculate_manual_initial_guess()
                else: 
                    print(manual_init)
                    raise ValueError(">Manual input wrong format, please follow the correct "
                      "format of 4 parameters in an array")
            except Exception as e:
                print(f'Excepction {e}')
                print(f'Loaded manual_init: {manual_init}')
                raise ValueError("Problem loading manually initialized parameters, please "
                    "make sure parameters are all numbers")
        else:
            # generate initial guess parameters from data when user does not manually initialze guess
            y1data = np.real(ydata)
            y2data = np.imag(ydata)
            init, x_c, y_c, r = self.fit_method.auto_initial_guess(xdata, y1data, y2data)
            
            freq = init[2]
            kappa = init[2] / init[0]

            if self.fit_method.name  == 'CPZM':
                kappa = init[4]
                init = init[0:4]


        # Extract data near resonate frequency to fit
        xdata, ydata = x_raw, y_raw #extract_near_res(x_raw, y_raw, freq, kappa, extract_factor=1)  # xdata is new set of data to be fit, within extract_factor
        # times the bandwidth, ydata is S21 data to match indices with xdata

        if self.fit_method.name == 'INV':
            ydata = ydata ** -1  # Inverse S21

        # Step Two. Fit Both Re and Im data
        # create a set of Parameters
        ## Monte Carlo Loop to check for local minimums
        # define parameters from initial guess for John Martinis and monte_carlo_fit
        try:
            params = self.fit_method.add_params(init)
        except Exception as e:
            print(f'Exception {e}')
            raise ValueError(">Failed to define parameters, please make sure parameters are of correct format")
        
        # Fit data to least squares fit for respective fit type
        fit_params, conf_array = self.min_fit(params, xdata, ydata)

        if manual_init is None and fit_params is None:
            raise RuntimeError(">Failed to minimize function for least squares fit")
        if fit_params is None:
            fit_params = manual_init

        # setup for while loop
        MC_counts = 0
        error = [10]
        stop_MC = False
        continue_condition = (MC_counts < self.fit_method.MC_iteration) and (stop_MC == False)
        output_params = []

        while continue_condition:

            # run a Monte Carlo fit on just minimized data to test if parameters trapped in local minimum
            MC_param, stop_MC, error_MC = self.monte_carlo_fit(xdata, ydata, fit_params)
            error.append(error_MC)
            if error[MC_counts] < error_MC:
                stop_MC = True

            output_params.append(MC_param)
            MC_counts = MC_counts + 1

            continue_condition = (MC_counts < self.fit_method.MC_iteration) and (stop_MC == False)

            if continue_condition == False:
                output_params = output_params[MC_counts - 1]

        error = min(error)

        # if monte carlo fit got better results than initial minimization, run a minimization on the monte carlo parameters
        if output_params[0] != fit_params[0]:
            params2 = self.fit_method.add_params(output_params)
            output_params, conf_array = self.min_fit(params2, xdata, ydata)

        if manual_init is None and fit_params is None:
            raise RuntimeError(">Failed to minimize function for least squares fit")
        
        if fit_params is None:
            fit_params = manual_init

        # Check that bandwidth is not equal to zero
        if len(xdata) == 0:
            if manual_init is not None:
                print(">Length of extracted data equals zero thus bandwidth is incorrect, "
                    "most likely due to initial parameters being too far off")
                print(">Please enter a new set of manual initial guess data "
                    "or run an auto guess")
            else:
                raise ValueError(">Length of extracted data equals zero thus bandwidth is incorrect, "
                    "please manually input a guess for parameters")
                
        # set the range to plot for 1 3dB bandwidth
        if self.fit_method.name == 'CPZM':
            Q = 1 / (1 / output_params[0] + output_params[1] / output_params[0])
            kappa = output_params[2] / Q
        else:
            kappa = output_params[2] / output_params[0]
        xstart = output_params[2] - kappa / 2  # starting resonance to add to fit
        xend = output_params[2] + kappa / 2
        extract_factor = [xstart, xend]

        # plot fit
        output_path = fp.name_folder(self.preprocessor.dir, self.fit_method.name)
        
        #title = f'{Method.method} fit for {filename}'
        title = f'{self.fit_method.name} Method Fit'
        figurename = f"{self.fit_method.name} with Monte Carlo Fit and Raw data\nPower: {self.preprocessor.file_name}"
        fig = fp.PlotFit(x_raw, y_raw, x_initial, y_initial, slope, intercept, 
                    slope2, intercept2, output_params, self.fit_method, 
                    error, figurename, x_c, y_c, r, output_path, conf_array, 
                    extract_factor, title=title, manual_params=self.fit_method.manual_init)

        
        fig.savefig(fp.name_plot(self.preprocessor.file_name, str(self.fit_method.name), output_path, 
                                    format=f'.png'), format=f'png')
            
        return output_params, conf_array, error, init, output_path

                        

