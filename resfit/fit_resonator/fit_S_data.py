# -*- coding: utf-8 -*-
"""
Created on Fri May 11 19:37:41 2018

@author: hung93
"""
import attr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
import inflect
import inspect
import matplotlib.pylab as pylab
from scipy import optimize
from scipy import stats


import resfit.fit_resonator.fit_functions as ff

params = {'legend.fontsize': 10,
          'figure.figsize': (10, 8),
         'axes.labelsize': 18,
         'axes.titlesize':18,
         'xtick.labelsize':18,
         'ytick.labelsize':18,
         'lines.markersize' : 1,
         'lines.linewidth' : 2,
         'font.size': 15.0 }
pylab.rcParams.update(params)

np.set_printoptions(precision=4,suppress=True)
p = inflect.engine() # search ordinal

def make_objective(x, y , func):
    """Create an objective function for fitting."""
    def objective(params):
        """Calculates the difference."""
        fit_y = func(x, *params)
        return np.sum(np.abs(fit_y - y)**2)
    return objective


@attr.s
class VNASweep:
    """A container to hold data from a vna frequency sweep."""
    freqs = attr.ib(type=np.ndarray)
    amps = attr.ib(type=np.ndarray)
    phases = attr.ib(type=np.ndarray)
    linear_amps = attr.ib(type=np.ndarray)

    @classmethod
    def from_csv(cls, csv):
        """Load data from csv file."""
        data = np.loadtxt(csv, delimiter=',')
        freqs = data.T[0]
        amps = data.T[1]
        phases = data.T[2]
        linear_amps = 10**(amps/20)
        return cls(freqs=freqs,amps=amps,phases=phases,linear_amps=linear_amps)

    @classmethod
    def from_columns(cls, freqs, amps, phases):
        """Load data from columns provided individually."""
        linear_amps = 10 ** (amps / 20)
        return cls(freqs=freqs, amps=amps, phases=phases, linear_amps=linear_amps)


@attr.s
class ComplexData:
    """Container for normalized data"""
    freqs = attr.ib(type=np.ndarray)
    complex_s21 = attr.ib(type=np.ndarray)

def normalize_data(data: VNASweep,
                   background: VNASweep = None)->ComplexData:
    """Normalize and exponentiate data.

    Also performs background subtraction if applicable.

    Args:
        freq: Frequencies of scan. Units of GHz.
        amplitude: Amplitude data from vna, units of dB
        phase: Phase data from vna, in radians.
        background (optional): 3 column array, frequencies, amplitudes and
          phase.

    Returns:
        Normalized data complex S21.
    """

    xdata = data.freqs
    linear_amps = data.linear_amps
    phases = data.phases
    complex_data = np.multiply(linear_amps, np.exp(1j * phases))

    if background:
        amps_background = background.linear_amps
        phases_background = background.phases

        amps_subtracted = np.divide(linear_amps, amps_background)
        phases_subtracted = np.subtract(phases, phases_background)
        complex_data = np.multiply(amps_subtracted,
                                   np.exp(1j * phases_subtracted))

    complex_data = ComplexData(freqs=xdata, complex_s21=complex_data)
    return complex_data

def preprocess(cplx_data: ComplexData, normalize: int)->ComplexData:
    """Preprocess data by removing cable delay and normalizing S21.

    Args:
        cplx_data: Complex data, frequency vs. A*exp(j*phi)
        normalize: Number points of at beginning and end of trace
          for normalization.

    Returns:
        Data with cable delay removed and normalized amplitudes.
    """

    x_initial = cplx_data.freqs
    y_initial = cplx_data.complex_s21

    if normalize * 2 > len(y_initial):
        raise ValueError(
            'Not enough points to normalize, please lower value of normalize variable or take more points near resonance')

    # normalize phase of S21 using linear fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        np.append(x_initial[0:10], x_initial[-10:]),
        np.append(np.angle(y_initial[0:normalize]),
                  np.angle(y_initial[-normalize:])))
    angle = np.subtract(np.angle(y_initial), slope * x_initial)  # remove cable delay
    angle = np.subtract(angle, intercept)  # rotate off resonant point to (1,0i) in complex plane

    # normalize magnitude of S21 using linear fit
    y_db = np.log10(np.abs(y_initial)) * 20
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(
        np.append(x_initial[0:normalize], x_initial[-normalize:]),
        np.append(y_db[0:normalize], y_db[-normalize:]))
    magnitude = np.subtract(y_db, slope2 * x_initial + intercept2)
    magnitude = 10 ** (magnitude / 20)

    y_raw = np.multiply(magnitude, np.exp(1j * angle))
    preped_data = ComplexData(x_initial, y_raw)
    return preped_data

def extract_near_res(x_raw: np.ndarray,
                     y_raw: np.ndarray,
                     f_res: float,
                     kappa: float,
                     extract_factor: int=1)->ComplexData:
    """Extracts portions of spectrum of kappa within resonance.

    Args:
        x_raw: X-values of spectrum to extract from.
        y_raw: Y-values of spectrum to extract from.
        f_res: Resonant frequency about which to extract data.
        kappa: Width about f_res to extract.
        extract_factor: Multiplier for kappa.

    Returns:
        Extracted spectrum kappa about f_res.
    """
    xstart = f_res - extract_factor/2*kappa #starting resonance to add to fit
    xend = f_res + extract_factor/2*kappa #final resonance to add to fit
    x_temp = []
    y_temp = []
    # xdata is new set of data to be fit, within extract_factor times the bandwidth, ydata is S21 data to match indices with xdata
    for i, freq in enumerate(x_raw):
        if (freq > xstart and freq< xend):
            x_temp.append(freq)
            y_temp.append(y_raw[i])

    if len(y_temp) < 5:
        print("Less than 5 Data points to fit data, not enough points near resonance, attempting to fit anyway")
    if len(x_temp) < 1:
        raise Exception(">Failed to extract data from designated bandwidth")

    return np.asarray(x_temp), np.asarray(y_temp)

def fit_resonator(data_array: np.ndarray,
                  manual_init: ff.ModelParams,
                  method: ff.FittingMethod,
                  normalize: int,
                  background: VNASweep = None):

    data = VNASweep.from_columns(freqs=data_array.T[0],
                                 amps=data_array.T[1],
                                 phases=data_array.T[2])

    normed_data = normalize_data(data, background=background)
    print("Loaded the data!")

    prepped_data = preprocess(normed_data, normalize=normalize)


    print('method name: ', method.name)

    if not manual_init: # TODO(mutus) Take care of case with no initial guess.
        raise Exception("Please provide an initial guess")

    if method.name in ['DCM', 'DCM_REFLECTION', 'PHI']:
        ranges = [[manual_init.Q*0.5, manual_init.Q*1.5],
                  [manual_init.Qc*0.5, manual_init.Qc*1.5],
                  [manual_init.f_res*0.8, manual_init.f_res*1.1],
                  [-1.0*manual_init.phi, manual_init.phi]]
        if method.name == 'DCM':
            obj = make_objective(prepped_data.freqs,
                                 prepped_data.complex_s21,
                                 ff.cavity_DCM)
            args = inspect.getfullargspec(ff.cavity_DCM).args[1:]
        elif method.name == 'DCM_REFLECTION':
            obj = make_objective(prepped_data.freqs,
                                 prepped_data.complex_s21,
                                 ff.cavity_DCM_REFLECTION)
            args = inspect.getfullargspec(ff.cavity_DCM_REFLECTION).args[1:]
        elif method.name == 'PHI':
            obj = make_objective(prepped_data.freqs,
                                 prepped_data.complex_s21,
                                 ff.cavity_phiRM)
            args = inspect.getfullargspec(ff.cavity_phiRM).args[1:]
    elif method.name in ['INV', 'CZPM']:
        ranges = [[manual_init.Qi*0.5, manual_init.Qi*1.5],
                  [manual_init.Qc*0.5, manual_init.Qc*1.5],
                  [manual_init.f_res*0.6, manual_init.f_res*1.4],
                  [-1.0*manual_init.phi, manual_init.phi*0.9]]
        if method.name == 'INV':
            obj = make_objective(prepped_data.freqs,
                                 prepped_data.complex_s21**-1,
                                 ff.cavity_inverse)
            args = inspect.getfullargspec(ff.cavity_inverse).args[1:]
        elif method.name == 'CZPM':
            obj = make_objective(prepped_data.freqs,
                                 prepped_data.complex_s21,
                                 ff.cavity_CPZM)
            args = inspect.getfullargspec(ff.cavity_CPZM).args[1:]


    ret = optimize.differential_evolution(obj,
                                          ranges,
                                          maxiter=1000,
                                          popsize=200,
                                          tol=0.005)
    return dict(zip(args, ret.x))

def plot(x,y,name,output_path,x_c=None,y_c=None,r=None,p_x=None,p_y=None):
    #plot any given set of x and y data
    fig = plt.figure('raw_data',figsize=(10, 10))
    gs = GridSpec(2,2)
    ax = plt.subplot(gs[0:2,0:2]) ## plot
    #plot axies
    #ax.axvline(x=0, linewidth=1, color='grey', linestyle = '--')
    #ax.axhline(y=0, linewidth=1, color='grey', linestyle = '--')
    ax.plot(x,y,'bo',label = 'raw data',markersize = 3)
    #plot guide circle if it applies
    if x_c!=None and y_c!=None and r!=None:
        circle = Circle((x_c, y_c), r, facecolor='none',\
                    edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
        ax.add_patch(circle)
    #plot a red point to represent something if it applies (resonance or off resonance for example)
    if p_x!=None and p_y!=None:
        ax.plot(p_x,p_y,'*',color = 'red',markersize = 5)
    fig.savefig(output_path+name+'.png')

#########################################################################

def plot2(x,y,x2,y2,name,output_path):
    #plot any given set of x and y data
    fig = plt.figure('raw_data',figsize=(10, 10))
    gs = GridSpec(2,2)
    ax = plt.subplot(gs[0:2,0:2]) ## plot
    ax.plot(x,y,'bo',label = 'raw data',markersize = 3)
    ax.plot(x2,y2,'bo',label = 'raw data',markersize = 3, color = 'red')
    fig.savefig(output_path+name+'.png')
