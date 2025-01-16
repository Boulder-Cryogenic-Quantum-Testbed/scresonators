import numpy as np
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

        # Extract data near resonate frequency to fit
        xdata, ydata = x_raw, y_raw 


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
        fit_params, conf_array = self.fit_method.min_fit(params, xdata, ydata)

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
            MC_param, stop_MC, error_MC = self.fit_method.monte_carlo_fit(xdata, ydata, fit_params)
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
            output_params, conf_array = self.fit_method.min_fit(params2, xdata, ydata)

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
        kappa = output_params[2] / output_params[0]
        xstart = output_params[2] - kappa / 2  # starting resonance to add to fit
        xend = output_params[2] + kappa / 2
        extract_factor = [xstart, xend]

        # plot fit
        output_path = fp.name_folder(self.preprocessor.dir, self.fit_method.name)
        
        #title = f'{Method.method} fit for {filename}'
        title = f'{self.fit_method.name} Method Fit'
        figurename = f"{self.fit_method.name} with Monte Carlo Fit and Raw data\nPower: {self.preprocessor.filename}"
        fig = fp.PlotFit(x_raw, y_raw, x_initial, y_initial, slope, intercept, 
                    slope2, intercept2, output_params, self.fit_method, 
                    error, figurename, x_c, y_c, r, output_path, conf_array, 
                    extract_factor, title=title, manual_params=self.fit_method.manual_init)

        
        fig.savefig(fp.name_plot(self.preprocessor.filename, str(self.fit_method.name), output_path, 
                                    format=f'.png'), format=f'png')
        
        output_dict = { 
                       "output_params" : output_params, 
                       "conf_array" : conf_array, 
                       "error" : error, 
                       "init" : init, 
                       "output_path" : output_path,
                       }
            
        return output_dict

                        

