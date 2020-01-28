"""Adapted from @mullinski and @hung93"""
import datetime
import numpy as np
from matplotlib import pyplot as plt

from resfit.fit_resonator import fit_functions


class Resonator: # object is defined in init below
    """
    Object for representing a resonator fit.

    Args:
      freq: list of frequencies (units of GHz).
      S21: complex S21 for each frequency.
      name (optional): name of scan.
      date (optional): date of scan.
      temp (optional): temperature of scan (in Kelvin).
      bias (optional): bias present during scan (in volts?).
    """
    def __init__(self, 
                 freq: np.ndarray, 
                 S21: np.ndarray, 
                 name: str = '', 
                 date: datetime.datetime = None,
                 temp: float = None,
                 bias: float = None):
        self.name = name 
        self.freq = np.asarray(freq)
        self.date = date if date is not None else None
        self.temp = temp if temp is not None else None
        self.bias = float(bias)*1000 if bias is not None else None
        self.center_freq = (freq.max()+freq.min())/2
        self.span = (freq.max()-freq.min())

        self.S21 = S21
        self.phase = np.angle(self.S21)
        self.method = None #code will put DCM or INV method after fitting
        self.DCMparams = None # later fit results
        self.INVparams = None # later fit results

    def load_params(self, method: str, params: list, chi: any): 
        """
        Loads model parameters for a corresponding fit technique.

        Args:
          method: One of DCM, PHI, DCM REFLECTION, INV, CPZM. Described
            in the readme.
          params: model fit parameters.
          chi: TODO(mutus) desicribe this argument. What is this?

        """
        if self.method == None: 
            self.method = []
            self.fc = params[3]
            if method == 'DCM':
                self.method.append("DCM")
                self.DCMparams = DCMparams(params, chi)
                self.compare = fit_raw_compare(self.freq,self.S21,self.DCMparams.all,'DCM')
            elif method == 'PHI':
                self.method.append("PHI")
                self.DCMparams= DCMparams(params, chi)
                self.compare = fit_raw_compare(self.freq,self.S21,self.DCMparams.all,'DCM')
            if method == 'DCM REFLECTION':
                self.method.append("DCM REFLECTION")
                self.DCMparams= DCMparams(params, chi)
                self.compare = fit_raw_compare(self.freq,self.S21,self.DCMparams.all,'DCM')
            elif method == 'INV':
                self.method.append("INV")
                self.INVparams = INVparams(params, chi)
                self.compare = fit_raw_compare(self.freq,self.S21,self.INVparams.all,'INV')
            elif method == 'CPZM':
                self.method.append("CPZM")
                self.CPZMparams = CPZMparams(params, chi)
            else:
                print('Please input DCM, DCM REFLECTION, PHI, INV or CPZM')
        else:
            if method not in self.method:
                self.method.append(method)

                if method == 'DCM':
                    self.method.append("DCM")
                    self.DCMparams = DCMparams(params, chi)

                elif method == 'PHI':
                    self.method.append("PHI")
                    self.DCMparams = DCMparams(params, chi)

                elif method == 'DCM REFLECTION':
                    self.method.append("DCM REFLECTION")
                    self.DCMparams = DCMparams(params, chi)

                elif method == 'INV':
                    self.method.append("INV")
                    self.INVparams = INVparams(params, chi)

                elif method == 'CPZM':
                    self.method.append("CPZM")
                    self.CPZMparams = CPZMparams(params, chi)
            else:
                print("repeated load parameter")

    def reload_params(self, method: str, params: list, chi: any):
        """
        Reloads model parameters for a corresponding fit technique.

        Args:
          method: One of DCM, PHI, DCM REFLECTION, INV, CPZM. Described
            in the readme.
          params: model fit parameters.
          chi: TODO(mutus) desicribe this argument. What is this?

        """
        if  method in self.method:
            print(self.name + ' changed params')
            self.fc = params[3]
            if method == 'DCM REFLECTION':
                self.DCMparams= DCMparams(params, chi)
                self.compare = fit_raw_compare(self.freq,self.S21,self.DCMparams.all,'DCM')
            elif method == 'INV':

                self.INVparams = INVparams(params, chi)
                self.compare = fit_raw_compare(self.freq,self.S21,self.INVparams.all,'INV')
            elif method == 'CPZM':
                self.method.append("CPZM")
                self.CPZMparams = CPZMparams(params, chi)
        else:
            print('no')

    def power_calibrate(self, p): # optional input calibration: linear function of input S21(frequency) to sample cables, meas. at room temp.

        assert (self.DCMparams is not None) or (self.INVparams is not None),'Please load parameters first'
        p = np.array(p)
        x = self.power
        f = self.fc
        self.corrected_power = p[0]*f+p[1]*x+p[2]+x
        hbar = 1.05*10**-34
        f = 2*np.pi*f*10**9
        p = 10**(self.corrected_power/10-3)
        if 'DCM' in self.method :

            Q = self.DCMparams.Q
            Qc = self.DCMparams.Qc
            self.DCMparams.num_photon = 2*p/hbar/f**2*Q**2/Qc ### calculate number of photon
        if 'INV' in self.method :
            Q = self.INVparams.Q
            Qc = self.INVparams.Qc
            self.INVparams.num_photon = 2*p/hbar/f**2*Q**2/Qc

    def fit(self,**kwargs):
        #define method
        method_default = FitMethod("INV")
        for key, value in kwargs.items():
            if key == 'MC_iteration':
                method_default.MC_iteration = value
            elif key == 'MC_rounds':
                method_default.MC_rounds = value
            elif key == 'MC_weight':
                method_default.MC_weight = value
            elif key == 'MC_weightvalue':
                method_default.MC_weightvalue = value
            elif key == 'MC_fix':
                method_default.MC_fix = value
            elif key == 'MC_step_const':
                method_default.MC_step_const = value
            elif key == 'find_circle':
                method_default.find_circle = value
            elif key == 'manual_init':
                method_default.manual_init = value
            elif key == 'vary':
                method_default.vary = value
        if 'INV_init_guess' in kwargs.keys():
            method_default.manual_init = kwargs.pop("INV_init_guess")

        #run Fit_Resonator with data from self and method just created for INV
        params_INV,fig_INV,chi_INV,init_INV = Fit_Resonator(self,method_default)

        #Do the same thing again with DCM this time
        init_DCM = convert_params('INV',params_INV)
        method_default.change_method("DCM")
        if 'DCM_init_guess' in kwargs.keys():
            method_default.manual_init = kwargs.pop("DCM_init_guess")
        else:
            method_default.manual_init = None
        params_DCM,fig_DCM,chi_DCM,init_DCM = Fit_Resonator(self,method_default)
        init_INV = convert_params('DCM',params_DCM)
        method_default.manual_init = init_DCM
        params_DCM2,fig_DCM2,chi_DCM2,init_DCM2 = Fit_Resonator(self,method_default)

        method_default.change_method("INV")
        method_default.manual_init = init_INV
        params_INV2,fig_INV2,chi_INV2,init_INV2 = Fit_Resonator(self,method_default)
        plt.close('all')
        if chi_DCM > chi_DCM2:
            DCM_list = [params_DCM2,fig_DCM2]
            self.load_params("DCM",params_DCM2,chi_DCM2)
        else:
            DCM_list = [params_DCM,fig_DCM]
            self.load_params("DCM",params_DCM,chi_DCM)


        if chi_INV > chi_INV2:
            INV_list = [params_INV2,fig_INV2]
            self.load_params("INV",params_INV2,chi_INV2)
        else:
            INV_list = [params_INV,fig_INV]
            self.load_params("INV",params_INV,chi_INV)

        return DCM_list,INV_list,method_default


class DCMparams(object): # DCM fitting results
    #TODO(mutus) change these to attr.dataclasses
    def __init__(self,params,chi):
        self.Qc = params[2]
        self.Q = params[1]
        Qc = params[2]*np.exp(1j*params[4])
        Qi = (params[1]**-1-abs(np.real(Qc**-1)))**-1
        self.ReQc = 1/np.real(Qc**-1)
        self.Qi = Qi
        self.chi = chi
        self.fc = params[3]
        self.phi = ((params[4]+np.pi)%(2*np.pi)-np.pi)/np.pi*180
        self.A = params[0]
        self.theta = params[5]
        self.all = params

class INVparams(object): # INV fitting results
    #TODO(mutus) change these to attr.dataclasses
    def __init__(self,params,chi):
        self.Qc = params[2]
        self.Qi = params[1]
        Q = 1/(params[1]**-1+params[2]**-1)
        self.Q = Q
        self.chi = chi
        self.fc = params[3]
        self.phi = ((params[4]+np.pi)%(2*np.pi)-np.pi)/np.pi*180
        self.A = params[0]
        self.theta = params[5]
        self.all = params


class CPZMparams(object):
    #TODO(mutus) change these to attr.dataclasses
    def __init__(self,params,chi):
        self.Qc = params[2]
        self.Qi = params[1]
        self.Qa = params[4]
        self.chi = chi
        self.fc = params[3]

class FitMethod(object):
    """
    method: str
            "DCM" or 'INV' or 'CPZM'

        fitting range:
                    number ->  number * FW2M (Full width at half twice min). if not =1, changes the FW2M fitting range
                    list -> from list[0] to list[1]
                    'all' -> all

    MC_iteration: int
                  Number of iteration of 1) least square fit + 2) Monte Carlo

    MC_rounds: int
               in each MC iteration, number of rounds of randomly choose parameter

    MC_weigh: str
               'no' or 'yes', weight the extract_factor fitting range, yes uses 1/|S21| weight, which we call iDCM

    MC_weightvalue: int
                    multiplication factor for weighing, such as 2 for twice the 1/|S21| weight.

    MC_fix: list of str
            'Amp','w1','theta','phi','Qc', 'Q' for DCM, 'Qi' for INV

    MC_step_const: int
                  randomly choose number in range MC_step_const*[-0.5~0.5]
                  for fitting. Exp(0.5)=1.6, and this is used for Qi,... . However, the res. frequency, theta, amplitude are usually fixed during Monte Carlo.

    find_circle: bool
                 true=> find initial guess from circle (better) or false if find from linewidth

    manual_init: None or list of 6 float number
                 manual input initial guesses
                 DCM: [amplitude, Q, Qc, freq, phi, theta]
                 INV: [amplitude, Qi, Qc, freq, phi, theta]
    vary: None or list of 6 booleans
          vary parameter in least square fit (which parameters change = true)
"""
    def __init__(self,
                 method: fit_functions.FittingMethod,
                 MC_iteration: int = 5,
                 MC_rounds: int = 100,
                 MC_weight: str = 'no',
                 MC_weightvalue: int = 2,
                 MC_fix: list = ['Amp','w1','theta'],
                 MC_step_const: float = 0.6,
                 find_circle: bool = True,
                 manual_init: fit_functions.ModelParams=None,
                 vary: bool = None):
        self.method = method
        if method.name == 'DCM':
            self.func = fit_functions.cavity_DCM
        elif method.name == 'DCM_REFLECTION':
            self.func = fit_functions.cavity_DCM_REFLECTION
        elif method.name == 'PHI':
            self.func = fit_functions.cavity_DCM
        elif method.name == 'INV':
            self.func = fit_functions.cavity_inverse
        elif method.name == 'CPZM':
            self.func = fit_functions.cavity_CPZM
        self.MC_rounds = MC_rounds
        self.MC_iteration = MC_iteration
        self.MC_weight = MC_weight
        self.MC_weightvalue = MC_weightvalue
        self.MC_step_const = MC_step_const
        self.MC_fix = MC_fix
        self.find_circle = find_circle
        self.manual_init = manual_init
        self.vary =  vary if vary is not None else [True]*6

    def change_method(self, method):
        if self.method == method:
            print("Fit method does not change")
        else:
            self.method = method

            if method.name == 'DCM':
                self.func = fit_functions.Cavity_DCM
            if method.name == 'PHI':
                self.func = fit_functions.Cavity_DCM
            elif method.name == 'DCM_REFLECTION':
                self.func = fit_functions.Cavity_DCM_REFLECTION
            elif method.name == 'INV':
                self.func = fit_functions.Cavity_inverse
            elif method.name == 'CPZM':
                self.func = fit_functions.Cavity_CPZM
