# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 10:53:51 2018

@author: hung93
"""

import numpy as np
import matplotlib.pyplot as plt
from .fitS21 import Cavity_DCM,Cavity_inverse,Cavity_CPZM,fit_raw_compare,Fit_Resonator,convert_params #look for fitS21 in subdir., get functions
class resonator(object): # object is defined in init below

    """
    required input: freq (GHz), I, Q, power (dbm), electrical delay (ns)
            ex: resonator(xdata,y1data,y2data,-50,80.15)
    function: load_params, add_temp, power_calibrate

            *load_params: After fitting, load fitting result to resonator object
                         input: method ("DCM" or "INV"), fitting result, chi square
                ex: resonator.load_parameter('DCM',[1,5000,8000,4.56,0.01,1.5], 0.05)

            *add_temp: add temperature (mK) to resonator
                ex: resonator.add_temp(20)

            *power_calibrate: calibrate input line, get input into device
                              input [freq_dependence, power_dependence, attenuators]
                ex: resonator.power_calibrate([0,0,0])

    Load_params.method=DCM -> Diameter Correction Method of Khalil, Stoutimore, Wellstood, Osborn. , DOI: 10.1063/1.3692073
    Load_params.method=INV -> use Inverse S21 method of Megrant, Neill, ... Martinis, Cleland, https://doi.org/10.1063/1.3693409


    """
    def __init__(self, freq, magnitude, phase, name = '', date = None,temp = None,bias = None):#frequency, S21, pwr, delay, name = '', date = None,temp = None,bias = None):
        self.name = name #start definitions of class (as part of a full directory of definitions)
        self.freq = np.asarray(freq)
        #self.I = np.asarray(I)
        #self.Q = np.asarray(Q)
        self.date = date if date is not None else None
        self.temp = temp if temp is not None else None
        self.bias = float(bias)*1000 if bias is not None else None
        self.center_freq = (freq.max()+freq.min())/2
        self.span = (freq.max()-freq.min())

        #S21 = I + 1j*Q
        #S21 = np.multiply(S21,np.exp(1j*delay*2*np.pi*freq))
        self.S21 = np.multiply(magnitude,np.exp(1j*phase))
        self.phase = np.angle(self.S21)
        self.method = None #code will put DCM or INV method after fitting
#        self.uphase = np.unwrap(self.phase) #Unwrap the 2pi phase jumps
#        self.mag = np.abs(self.S21) #Units are volts.
#        self.logmag = 20*np.log10(self.mag) #Units are dB (20 because V->Pwr)
        self.DCMparams = None # later fit results
        self.INVparams = None # later fit results
#        self.corrected_power = None
    def load_params(self, method, params, chi): #process data using load_params.method :load_params.method, and later put fit results also under load_params.DCMparams (if it is DCM method)
        if self.method == None: #
            self.method = []
            self.fc = params[3]
            if method == 'DCM':
                self.method.append("DCM")
                self.DCMparams= DCMparams_class(params, chi)
                self.compare = fit_raw_compare(self.freq,self.S21,self.DCMparams.all,'DCM')
            elif method == 'INV':
                self.method.append("INV")
                self.INVparams = INVparams_class(params, chi)
                self.compare = fit_raw_compare(self.freq,self.S21,self.INVparams.all,'INV')
            elif method == 'CPZM':
                self.method.append("CPZM")
                self.CPZMparams = CPZMparams_class(params, chi)
            else:
                print('Please input DCM or INV or CPZM')
        else:
            if method not in self.method:
                self.method.append(method)

                if method == 'DCM':
                    self.method.append("DCM")
                    self.DCMparams= DCMparams_class(params, chi)

                elif method == 'INV':
                    self.method.append("INV")
                    self.INVparams = INVparams_class(params, chi)

                elif method == 'CPZM':
                    self.method.append("CPZM")
                    self.CPZMparams = CPZMparams_class(params, chi)
            else:
                print("repeated load parameter")

    def reload_params(self, method, params, chi):
        if  method in self.method:
            print(self.name + ' changed params')
            self.fc = params[3]
            if method == 'DCM':
                self.DCMparams= DCMparams_class(params, chi)
                self.compare = fit_raw_compare(self.freq,self.S21,self.DCMparams.all,'DCM')
            elif method == 'INV':

                self.INVparams = INVparams_class(params, chi)
                self.compare = fit_raw_compare(self.freq,self.S21,self.INVparams.all,'INV')
            elif method == 'CPZM':
                self.method.append("CPZM")
                self.CPZMparams = CPZMparams_class(params, chi)
        else:
            print('no')
    def add_temp(self,temp): # temperature (any unit)
        self.temp = temp

    def power_calibrate(self,p): # optional input calibration: linear function of input S21(frequency) to sample cables, meas. at room temp.

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
        method_default = Fit_Method("INV")
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
#        self.load_params('INV',params_INV,chi_INV)

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


class DCMparams_class(object): # DCM fitting results
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

class INVparams_class(object): # INV fitting results
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


class CPZMparams_class(object):
    def __init__(self,params,chi):
        self.Qc = params[2]
        self.Qi = params[1]
        self.Qa = params[4]
        self.chi = chi
        self.fc = params[3]
class Fit_Method(object):
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
    def __init__(self, method,MC_iteration = 5, MC_rounds=100,\
                 MC_weight = 'no',MC_weightvalue = 2,\
                 MC_fix = ['Amp','w1','theta'],MC_step_const= 0.6,\
                 find_circle = True,manual_init=None,vary = None):
        assert method in ['DCM','INV','CPZM'],"Wrong Method, DCM,INV "
        assert (manual_init == None) or (type(manual_init)==list and len(manual_init)==4),'Wrong manual_init, None or len = 6'
        self.method = method
        if method == 'DCM':
            self.func = Cavity_DCM
        elif method == 'INV':
            self.func = Cavity_inverse
        elif method == 'CPZM':
            self.func = Cavity_CPZM
        self.MC_rounds = MC_rounds
        self.MC_iteration = MC_iteration
        self.MC_weight = MC_weight
        self.MC_weightvalue = MC_weightvalue
        self.MC_step_const = MC_step_const
        self.MC_fix = MC_fix
        self.find_circle = find_circle
        self.manual_init = manual_init
        self.vary =  vary if vary is not None else [True]*6

    def change_method(self,method):
        assert method in ['DCM','INV','CPZM'],"Wrong Method, DCM,INV "
        if self.method == method:
            print("Fit method does not change")
        else:
            self.method = method

            if method == 'DCM':
                self.func = Cavity_DCM
            elif method == 'INV':
                self.func = Cavity_inverse
            elif method == 'CPZM':
                self.func = Cavity_CPZM
