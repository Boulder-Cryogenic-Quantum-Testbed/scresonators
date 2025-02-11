# %%

import sys
from pathlib import Path

display(Path("../src").absolute())

sys.path.append("../src")
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
#import skrf as rf
from src.fit_methods.dcm import DCM
from src.resonator import Resonator
from example_data_dcm import s2p_df
from plotting import plotres
from src.utilities import remove_delay, find_circle
import lmfit
from matplotlib.patches import Circle

if s2p_df["Frequency"][0] < 1e9:
    print("Converting GHz to Hz")
    s2p_df["Frequency"] *= 1e9
  
display(s2p_df)
freqs = s2p_df["Frequency"].values
magn_dB = s2p_df["S21 magn_dB"].values
magn_lin = 10**(magn_dB/20)
phase_rad = s2p_df["S21 phase_rad"].values

S21 = magn_lin * np.exp(1j*phase_rad)


fig, ax = plotres.plot_s2p_df(s2p_df)
# s2p_df["Frequency"] *= 1e9

# %%
myres = Resonator(df=s2p_df, verbose=True)
myres.initialize_fit_method(fit_name="DCM")   # provide string of fit name instead of the imported DCM
#myres.fitter.delay_guess=75e

results = myres.fit()   # moved verbose to be an attribute of myres
# print(f'Qtot: {Qtot}')

delay = myres.data_processor.fit_delay(freqs, magn_lin)
z_data = remove_delay(freqs, S21, delay)
print(f'delay: {delay}')

xc, yc, R = find_circle(np.real(z_data), np.imag(z_data))

# %%

# %%

#Begin results plot
fig, ax = plotres.makeSummaryFigure()
plotres.summaryPlot(freqs, S21, color = 'blue', label = 'data')
plotres.summaryPlot(freqs, DCM.fit_function(freqs, results), color = 'k', linestyle = 'dashed', label='fit')
plotres.summaryPlot(freqs, z_data, color = 'green', label = 'delay removed')
#ax['smith'].add_patch(Circle((xc, yc),R, fill = True, color = 'red'))
#plotres.annotateParam(results['Qi'])
#plotres.annotateParam(results['f0'])
#plotres.annotateParam(results['Qc'])
plotres.displayAllParams(results)
fig.suptitle('Plotting Test')
plt.show()
