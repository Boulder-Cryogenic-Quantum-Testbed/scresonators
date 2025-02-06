import numpy as np

def Refl_S21(freqs, f0, Q, Qc):
    s21 = 1-(2*Q/Qc)/(1-2j*Q*(freqs-f0)/f0)
    return s21

#generate sample data
f0 = 4.5 #GHz
Qi = 900000
Qc = 500000
Qtot = 1/(1/Qi+1/Qc)
linewidth = 2*f0/(Qtot)


fstart = f0-5*linewidth
fstop = f0+5*linewidth
npoints = 401
freqs = np.linspace(fstart, fstop, npoints)

S21 = Refl_S21(freqs, f0, Qtot, Qc)
#TODO: give this a nonzero electrical delay & arbitrary phase shift to test preprocessing mathods