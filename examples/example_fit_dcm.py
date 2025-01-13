import numpy as np
import matplotlib.pyplot as plt
#import skrf as rf
from src.fit_methods.dcm import DCM
from src.resonator import Resonator
from example_data_dcm import freqs, S21, Qtot
from plotting import plotres
from src.utils import remove_delay, find_circle
import lmfit
from matplotlib.patches import Circle


myres = Resonator()
myres.load_data(fdata = freqs, sdata = S21)
myres.set_fitting_strategy(strategy = DCM)
#myres.fitter.delay_guess=75
results = myres.fit(verbose = True)
print(f'Qtot: {Qtot}')

delay = myres.fitter.find_delay(freqs, S21)
z_data = remove_delay(freqs, S21, delay)
print(f'delay: {delay}')

xc, yc, R = find_circle(np.real(z_data), np.imag(z_data))


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
