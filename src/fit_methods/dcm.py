import numpy as np
import cavity_functions as ff
from utilities import * 
from .fit_method import FitMethod
import lmfit

class DCM(FitMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'DCM'
        self.func = ff.cavity_DCM

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
