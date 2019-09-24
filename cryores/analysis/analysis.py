"""Adapted from @mullinski and @hung93"""
import numpy as np
from matplotlib import pyplot as plt

import cryores.analysis.base as analysis
import cryores.experiments.data.base as data
import cryores.instruments.base as instruments

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
    def __init__(self, freq, S21, name = '', date = None,temp = None,bias = None):#frequency, S21, pwr, delay, name = '', date = None,temp = None,bias = None):
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
        self.S21 = S21
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
            elif method == 'PHI':
                self.method.append("PHI")
                self.DCMparams= DCMparams_class(params, chi)
                self.compare = fit_raw_compare(self.freq,self.S21,self.DCMparams.all,'DCM')
            if method == 'DCM REFLECTION':
                self.method.append("DCM REFLECTION")
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
                print('Please input DCM, DCM REFLECTION, PHI, INV or CPZM')
        else:
            if method not in self.method:
                self.method.append(method)

                if method == 'DCM':
                    self.method.append("DCM")
                    self.DCMparams= DCMparams_class(params, chi)

                elif method == 'PHI':
                    self.method.append("PHI")
                    self.DCMparams= DCMparams_class(params, chi)

                elif method == 'DCM REFLECTION':
                    self.method.append("DCM REFLECTION")
                    self.DCMparams= DCMparams_class(params, chi)

                elif method == 'INV':
                    self.method.append("INV")
                    self.INVparams = INVparams_class(params, chi)

                elif method == 'CPZM':
                    self.method.append("CPZM")
                    self.CPZMparams = CPZMparams_class(params, chi)
            else:
                print("repeated load parameter")



class ResonatorMinAnalyzer(analysis.Analyzer):
    def analyze(self, dataset: data.Dataset) -> analysis.AnalysisResults:
        # Simply grab the minimum value from the data.
        return analysis.AnalysisResults(
            lowest_resonance=dataset.data[instruments.COL_S21_AMP].min())
