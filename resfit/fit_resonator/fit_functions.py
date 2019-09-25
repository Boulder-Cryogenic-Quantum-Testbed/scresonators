"""Analytic fit functions"""

#TODO(mutus) create an enum for all the fit models.

import numpy as np

def Cavity_DCM(x, Q, Qc, w1,phi):
    #DCM fit function
    return np.array(1-Q/Qc*np.exp(1j*phi)/(1 + 1j*(x-w1)/w1*2*Q))

def Cavity_DCM_REFLECTION(x, Q, Qc, w1,phi):
    #DCM REFLECTION fit function
    return np.array(1-2*Q/Qc*np.exp(1j*phi)/(1 + 1j*(x-w1)/w1*2*Q))

def Cavity_inverse(x, Qi,Qc, w1,phi):
    #Inverse fit function
    return np.array(\
        (1 + Qi/Qc*np.exp(1j*phi)/(1 + 1j*2*Qi*(x-w1)/w1)))

def Cavity_CPZM(x, Qi,Qic,w1,Qia):
    #CPZM fit function
    return np.array(\
        (1 + 2*1j*Qi*(x-w1)/w1)/(1 + Qic +1j*Qia + 1j*2*Qi*(x-w1)/w1))

def One_Cavity_peak_abs(x, Q, Qc, w1):
    #Ideal resonator fit function
    return np.abs(Q/Qc/(1 + 1j*(x-w1)/w1*2*Q))

def One_Cavity_peak_abs_REFLECTION(x, Q, Qc, w1):
    #Ideal resonator fit function
    return np.abs(2*Q/Qc/(1 + 1j*(x-w1)/w1*2*Q))