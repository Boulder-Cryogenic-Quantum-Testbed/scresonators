"""Analytic fit functions"""
import numpy as np

def cavity_DCM(x, Q, Qc, w1, phi):
    #DCM fit function
    return np.array(1-Q/Qc*np.exp(1j*phi)/(1 + 1j*(x-w1)/w1*2*Q))

def cavity_DCM_REFLECTION(x, Q, Qc, w1, phi):
    #DCM REFLECTION fit function
    return np.array(1-2*Q/Qc*np.exp(1j*phi)/(1 + 1j*(x-w1)/w1*2*Q))

def cavity_inverse(x, Qi, Qc, w1, phi):
    #Inverse fit function
    return np.array((1 + Qi/Qc*np.exp(1j*phi)/(1 + 1j*2*Qi*(x-w1)/w1)))

def cavity_CPZM(x, Qi, Qic, w1, Qia):
    #CPZM fit function
    return np.array(\
        (1 + 2*1j*Qi*(x-w1)/w1)/(1 + Qic +1j*Qia + 1j*2*Qi*(x-w1)/w1))

def one_cavity_peak_abs(x, Q, Qc, w1):
    #Ideal resonator fit function
    return np.abs(Q/Qc/(1 + 1j*(x-w1)/w1*2*Q))

def one_cavity_peak_abs_REFLECTION(x, Q, Qc, w1):
    #Ideal resonator fit function
    return np.abs(2*Q/Qc/(1 + 1j*(x-w1)/w1*2*Q))

#############################################################################
def fit_raw_compare(x,y,params,method):
    #Compare fit to raw data
    yfit = np.zeros(len(x))
    if method == 'DCM':
        yfit = cavity_DCM(x,*params)
    if method == 'INV':
        yfit = cavity_inverse(x,*params)
    ym = np.abs(y-yfit)/np.abs(y)
    return ym
############################################################################
    ## Least square fit model for one cavity, dip-like S21
def min_one_Cavity_dip(parameter, x, data=None):
    #fit function call for DCM fitting method
    Q = parameter['Q']
    Qc = parameter['Qc']
    w1 = parameter['w1']
    phi = parameter['phi']

    model = cavity_DCM(x, Q, Qc, w1,phi)
    real_model = model.real
    imag_model = model.imag
    real_data = data.real
    imag_data = data.imag

    resid_re = real_model - real_data
    resid_im = imag_model - imag_data
    return np.concatenate((resid_re,resid_im))

########################################################

    ## Least square fit model for one cavity, dip-like S21
def min_one_Cavity_DCM_REFLECTION(parameter, x, data=None):
    #fit function call for DCM fitting method
    Q = parameter['Q']
    Qc = parameter['Qc']
    w1 = parameter['w1']
    phi = parameter['phi']

    model = cavity_DCM_REFLECTION(x, Q, Qc, w1,phi)
    real_model = model.real
    imag_model = model.imag
    real_data = data.real
    imag_data = data.imag

    resid_re = real_model - real_data
    resid_im = imag_model - imag_data
    return np.concatenate((resid_re,resid_im))

########################################################
    ## Least square fit model for one cavity, dip-like "Inverse" S21
def min_one_Cavity_inverse(parameter, x, data=None):
    #fit function call for INV fitting method
    Qc = parameter['Qc']
    Qi = parameter['Qi']
    w1 = parameter['w1']
    phi = parameter['phi']

    model = cavity_inverse(x, Qi, Qc, w1,phi)
    real_model = model.real
    imag_model = model.imag
    real_data = data.real
    imag_data = data.imag

    resid_re = real_model - real_data
    resid_im = imag_model - imag_data
    return np.concatenate((resid_re,resid_im))


#####################################################################
def min_one_Cavity_CPZM(parameter, x, data=None):
    #fit function call for CPZM fitting method
    Qc = parameter['Qc']
    Qi = parameter['Qi']
    w1 = parameter['w1']
    Qa = parameter['Qa']

    model = cavity_CPZM(x, Qi, Qc, w1,Qa)
    real_model = model.real
    imag_model = model.imag
    real_data = data.real
    imag_data = data.imag

    resid_re = real_model - real_data
    resid_im = imag_model - imag_data
    return np.concatenate((resid_re,resid_im))

