import numpy as np
import pandas as pd

def DCM_S21(freqs, f0, Q, Qc, phi):
    s21 = 1-(Q*np.exp(-1j*phi)/Qc)/(1+2j*Q*(freqs-f0)/f0)
    return s21

#generate sample data
f0 = 4.5 #GHz
Qi = 400000
Qc = 200000
phi = 0
Qtot = 1/(1/Qi+np.cos(phi)/Qc)
linewidth = 2*f0/(Qtot)
delay =  80#ns


fstart = f0-15*linewidth
fstop = f0+15*linewidth
npoints = 501
freqs = np.linspace(fstart, fstop, npoints)


S21 = np.exp(-2j*np.pi*freqs*delay)*DCM_S21(freqs, f0, Qtot, Qc, phi)

#### quick hack to make a s2p dictionary and dataframea  
magn_lin, phase_rad = np.abs(S21), np.angle(S21)

magn_dB = np.log10(magn_lin) * 20

s2p_dict = {
    "Frequency" : freqs,
    "S21 magn_dB" : magn_dB,
    "S21 phase_rad" : phase_rad,    
    }

s2p_df = pd.DataFrame(s2p_dict)

display(s2p_df)
