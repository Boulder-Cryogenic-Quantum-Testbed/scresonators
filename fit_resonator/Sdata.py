import lmfit
from lmfit import Minimizer

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

import scipy.optimize as spopt
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy import stats

import attr
import numpy as np
import inflect
import time
import sys
import os
import logging
from git import Repo
import csv

import fit_resonator.functions as ff

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)  # Path to Fit_Cavity

params = {'legend.fontsize': 20,
          'figure.figsize': (10, 8),
          'axes.labelsize': 20,
          'axes.titlesize': 20,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'lines.markersize': 1,
          'lines.linewidth': 2,
          'font.size': 20
          }

pylab.rcParams.update(params)

np.set_printoptions(precision=4, suppress=True)
p = inflect.engine()  # search ordinal


def extract_near_res(x_raw: np.ndarray,
                     y_raw: np.ndarray,
                     f_res: float,
                     kappa: float,
                     extract_factor: int = 1):
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
    # starting resonance to add to fit
    xstart = f_res - extract_factor / 2 * kappa
    # final resonance to add to fit
    xend = f_res + extract_factor / 2 * kappa
    x_temp = []
    y_temp = []
    # xdata is new set of data to be fit, within extract_factor times the 
    # bandwidth, ydata is S21 data to match indices with xdata
    for i, freq in enumerate(x_raw):
        if (freq > xstart and freq < xend):
            x_temp.append(freq)
            y_temp.append(y_raw[i])

    if len(y_temp) < 5:
        print("Less than 5 Data points to fit data, not enough points near",
            "resonance, attempting to fit anyway")
    if len(x_temp) < 1:
        raise Exception(">Failed to extract data from designated bandwidth")

    return np.asarray(x_temp), np.asarray(y_temp)


def convert_params(from_method, params):
    if from_method == 'DCM':
        Qc = params[2] / np.cos(params[4])
        Qi = params[1] * Qc / (Qc - params[1])
        Qc_INV = params[2]
        Qi_INV = Qi / (1 + np.sin(params[4]) / Qc_INV / 2)
        return [1 / params[0], Qi_INV, Qc_INV, params[3], -params[4],
                 -params[5]]
    elif from_method == 'INV':
        Qc_DCM = params[2]
        Q_DCM = (np.cos(params[4]) / params[2] + 1 / params[1]) ** -1
        return [1 / params[0], Q_DCM, Qc_DCM, params[3], -params[4], -params[5]]


def find_circle(x, y):
    """Given a set of x,y data return a circle that fits data using LeastSquares
      Circle Fit Randy Bullock (2017)

    Args:
        x: Array of x position of data in complex plane (real)
        y: Array of y position of data in complex plane (imaginary)

    Returns:
        x (matrix1) and y (matrix2) center coordinates of the circle, and the 
        radius of the circle "R"
    """
    N = 0
    xavg = 0
    yavg = 0
    for i in range(0, len(x)):
        N = N + 1
        xavg = xavg + x[i]
    for i in range(0, len(y)):
        yavg = yavg + y[i]

    xavg = xavg / N
    yavg = yavg / N

    xnew = []
    ynew = []
    Suu = 0
    Svv = 0
    for i in range(0, len(x)):
        xnew.append(x[i] - xavg)
        Suu = Suu + (x[i] - xavg) * (x[i] - xavg)
    for i in range(0, len(y)):
        ynew.append(y[i] - yavg)
        Svv = Svv + (y[i] - yavg) * (y[i] - yavg)

    Suv = 0
    Suuu = 0
    Svvv = 0
    Suvv = 0
    Svuu = 0
    for i in range(0, len(xnew)):
        Suv = Suv + xnew[i] * ynew[i]
        Suuu = Suuu + xnew[i] * xnew[i] * xnew[i]
        Svvv = Svvv + ynew[i] * ynew[i] * ynew[i]
        Suvv = Suvv + xnew[i] * ynew[i] * ynew[i]
        Svuu = Svuu + ynew[i] * xnew[i] * xnew[i]
    Suv2 = Suv

    matrix1 = 0.5 * (Suuu + Suvv)
    matrix2 = 0.5 * (Svvv + Svuu)

    # row reduction for row 1
    Suv = Suv / Suu
    matrix1 = matrix1 / Suu

    # row subtraction for row 2 by row 1
    Svv = Svv - (Suv * Suv2)
    matrix2 = matrix2 - (Suv2 * matrix1)

    # row reduction for row 2
    matrix2 = matrix2 / Svv

    # row subtraction for row 1 by row 2
    matrix1 = matrix1 - (Suv * matrix2)

    # at this point matrix1 is x_c and matrix2 is y_c
    alpha = (matrix1 * matrix1) + (matrix2 * matrix2) + (Suu + Svv) / N
    R = alpha ** (0.5)

    matrix1 = matrix1 + xavg
    matrix2 = matrix2 + yavg

    return matrix1, matrix2, R


#########################################################################

def find_initial_guess(x, y1, y2, Method, output_path, plot_extra):
    """Determines an initial guess for the parameters

    Args:
        x: frequency data
        y1: real part of transmission data
        y2: imaginary part of transmission data
        Method: method class
        output_path: place to output any plots generated
        plot_extra: boolean that determines if extra plots will be output

    Returns: initial guess for parameters, x coordinate for center of fit 
    circle, y coordinate for center of fit circle, radius of fit circle

    """
    try:
        # recombine transmission S21 from real and complex parts
        y = y1 + 1j * y2
        # inverse transmission such that y = S21^(-1)
        if Method.method == 'INV':
            y = 1 / y
        # redefine y1 and y2 to account for possibility they were inversed above
        y1 = np.real(y)
        y2 = np.imag(y)
    except:
        print(">Problem initializing data in find_initial_guess(), please make",
              " sure data is of correct format")
        quit()

    try:
        # find circle that matches the data
        x_c, y_c, r = find_circle(y1, y2)
        # define complex number to house circle center location data
        z_c = x_c + 1j * y_c
    except:
        print(">Problem in function find_circle, please make sure data is of ",
              "correct format")
        quit()

    if plot_extra:
        try:
            plot(np.real(y), np.imag(y), "circle", output_path, np.real(z_c), 
                 np.imag(z_c), r)
        except:
            print(">Error when trying to plot raw data and circle fit in ",
                  "find_initial_guess")
            quit()

    try:
        ## move gap of circle to (0,0)
        # Center point P at (0,0)
        ydata = y - 1
        # Shift guide circle to match data shift
        z_c = z_c - 1
    except:
        print(">Error when trying to shift data into canonical position",
              " minus 1")
        quit()

    try:
        # determine the angle to the center of the fitting circle from origin
        if Method.method == 'INV':
            phi = np.angle(z_c)
        else:
            phi = np.angle(-z_c)

        freq_idx = np.argmax(np.abs(ydata))
        f_c = x[freq_idx]

        if plot_extra:
            # plot data with guide circle
            plot(np.real(ydata),
                 np.imag(ydata),
                 "resonance",
                 output_path,
                 np.real(z_c),
                 np.imag(z_c),
                 r,
                 np.real(ydata[freq_idx]),
                 np.imag(ydata[freq_idx]))
        # rotate resonant freq to minimum
        ydata = ydata * np.exp(-1j * phi)

        z_c = z_c * np.exp(-1j * phi)
        if plot_extra:
            # plot shifted data with guide circle
            plot(np.real(ydata), np.imag(ydata), "phi", output_path, 
                 np.real(z_c), np.imag(z_c), r,
                 np.real(ydata[freq_idx]), np.imag(ydata[freq_idx]))
    except:
        print(">Error when trying to shift data according to phi in ",
              "find_initial_guess")
        quit()

    try:
        if f_c < 0:
            print(">Resonance frequency is negative. Please only input ",
                  "positive frequencies.")
            quit()
    except:
        print(">Cannot find resonance frequency in find_initial_guess")
        quit()

    if Method.method == 'DCM' or Method.method == 'PHI':
        try:
            # diameter of the circle found from getting distance from (0,0) to 
            # resonance frequency data point (possibly should use fit circle)
            Q_Qc = np.max(np.abs(ydata))
            # y_temp = |ydata|-(diameter/sqrt(2))
            y_temp = np.abs(np.abs(ydata) - np.max(np.abs(ydata)) / 2 ** 0.5)

            # find min value in y_temp on one half of circle from resonance
            _, idx1 = find_nearest(y_temp[0:freq_idx], 0)
            # find min value in y_temp on other half of circle from resonance
            _, idx2 = find_nearest(y_temp[freq_idx:], 0)
            # add index of resonance frequency to get correct index for idx2
            idx2 = idx2 + freq_idx
            # bandwidth of frequencies
            kappa = abs((x[idx1] - x[idx2]))

            Q = f_c / kappa
            Qc = Q / Q_Qc
            guess = Q, Qc, f_c
            # fits parameters for the 3 terms given in p0 
            # (this is where Qi and Qc are actually guessed)
            popt, pcov = spopt.curve_fit(ff.one_cavity_peak_abs, x, 
                                         np.abs(ydata), p0=[Q, Qc, f_c], 
                                         bounds=(0, [np.inf] * 3))
            Q = popt[0]
            Qc = popt[1]
            init_guess = [Q, Qc, f_c, phi]
        except Exception as e:
            print(e)
            if Method.method == 'DCM':
                print(">Failed to find initial guess for method DCM.",
                      " Please manually initialize a guess")
            else:
                print(">Failed to find initial guess for method PHI.",
                      " Please manually initialize a guess")
            quit()

    elif Method.method == 'DCM REFLECTION':
        try:
            # diameter of the circle found from getting distance from (0,0) 
            # to resonance frequency data point (possibly should use fit circle)
            Q_Qc = np.max(np.abs(ydata)) / 2
            y_temp = np.abs(np.abs(ydata) - np.max(np.abs(ydata)) / 2 ** 0.5)

            # find min value in y_temp on one half of circle from resonance
            _, idx1 = find_nearest(y_temp[0:freq_idx], 0)
            # find min value in y_temp on other half of circle from resonance
            _, idx2 = find_nearest(y_temp[freq_idx:], 0)
            # add index of resonance frequency to get correct index for idx2
            idx2 = idx2 + freq_idx
            # bandwidth of frequencies
            kappa = abs((x[idx1] - x[idx2]))

            Q = f_c / kappa
            Qc = Q / Q_Qc
            # fits parameters for the 3 terms given in p0 
            # (this is where Qi and Qc are actually guessed)
            popt, pcov = spopt.curve_fit(ff.one_cavity_peak_abs_REFLECTION, x, 
                                         np.abs(ydata), p0=[Q, Qc, f_c],
                                         bounds=(0, [np.inf] * 3))
            Q = popt[0]
            Qc = popt[1]
            init_guess = [Q, Qc, f_c, phi]
        except:
            print(">Failed to find initial guess for method DCM REFLECTION. ",
                  "Please manually initialize a guess")
            quit()

    elif Method.method == 'INV':

        try:
            # diameter of the circle found from getting distance from (0,0) 
            # to resonance frequency
            Qi_Qc = np.max(np.abs(ydata))
            # y_temp = |ydata|-(diameter/sqrt(2))
            y_temp = np.abs(np.abs(ydata) - np.max(np.abs(ydata)) / 2 ** 0.5)

            # find min value in y_temp on one half of circle from resonance
            _, idx1 = find_nearest(y_temp[0:freq_idx], 0)
            # find min value in y_temp on other half of circle from resonance
            _, idx2 = find_nearest(y_temp[freq_idx:], 0)
            # add index of resonance frequency to get correct index for idx2
            idx2 = idx2 + freq_idx
            # bandwidth of frequencies
            kappa = abs((x[idx1] - x[idx2]))

            Qi = f_c / (kappa)
            Qc = Qi / Qi_Qc
            # fits parameters for the 3 terms given in p0 
            # (this is where Qi and Qc are actually guessed)
            popt, pcov = spopt.curve_fit(ff.one_cavity_peak_abs, x, 
                                         np.abs(ydata), p0=[Qi, Qc, f_c], 
                                         bounds=(0, [np.inf] * 3))
            Qi = popt[0]
            Qc = popt[1]
            init_guess = [Qi, Qc, f_c, phi]
        except:
            print(">Failed to find initial guess for method INV. ",
                  "Please manually initialize a guess")
            quit()

    elif Method.method == 'CPZM':
        try:
            Q_Qc = np.max(np.abs(ydata))
            y_temp = np.abs(np.abs(ydata) - np.max(np.abs(ydata)) / 2 ** 0.5)

            _, idx1 = find_nearest(y_temp[0:freq_idx], 0)
            _, idx2 = find_nearest(y_temp[freq_idx:], 0)
            idx2 = idx2 + freq_idx
            kappa = abs((x[idx1] - x[idx2]))

            Q = f_c / kappa
            Qc = Q / Q_Qc
            popt, pcov = spopt.curve_fit(ff.one_cavity_peak_abs, x, 
                                         np.abs(ydata), p0=[Q, Qc, f_c], 
                                         bounds=(0, [np.inf] * 3))
            Q = popt[0]
            Qc = popt[1]
            Qa = -1 / np.imag(Qc ** -1 * np.exp(-1j * phi))
            Qc = 1 / np.real(Qc ** -1 * np.exp(-1j * phi))
            Qi = (1 / Q - 1 / Qc) ** -1
            Qic = Qi / Qc
            Qia = Qi / Qa
            init_guess = [Qi, Qic, f_c, Qia, kappa]
        except:
            print(">Failed to find initial guess for method CPZM. ",
                  "Please manually initialize a guess")
            quit()
    else:
        print(">Method is not defined. Please choose a method: DCM, ",
              "DCM REFLECTION, PHI, INV or CPZM")
        quit()
    return init_guess, x_c, y_c, r


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    val = array[idx]
    return val, idx


def PlotFit(x,
            y,
            x_initial,
            y_initial,
            slope,
            intercept,
            slope2,
            intercept2,
            params,
            Method,
            func,
            error,
            figurename,
            x_c,
            y_c,
            radius,
            output_path,
            conf_array,
            extract_factor=None,
            title="Fit",
            manual_params=None,
            dfac: int = 1,
            msizes: list = [12, 7],
            xstr: str = 'Frequency [GHz]',
            fsize: float = 16.):
    """
    Plots data and outputs fit parameters to a file

    Args:
        x: cropped frequency data
        y: cropped S21 data after normalization
        x_initial: original frequency data before normalization
        y_initial: original S21 data before normalization
        slope: slope of the normalization line for magnitude
        intercept: intercept of normalization line for magnitude
        slope2: slope of the normalization line for phase
        intercept2: intercept of normalization line for phase
        params: parameters generated from fit function
        Method: Method class instance
        func: function used to generate data for plotting
              (DCM/DCM REFLECTION/INV/PHI/CPZM)
        error: error from Monte Carlo Fit function
        figurename: name the plot should have
        x_c: center x coordinate for fit circle
        y_c: center y coordinate for fit circle
        radius: radius of fit circle
        output_path: file path where the plots will be saved
        conf_array: array with 95% confidence interval values
        extract_factor: contains starting and ending frequency values
        title: title the plot will have
        manual_params: user input manual initial guess parameters for fit
        dfac : decimation factor on the plotting of points in the resonance
               circle Im S21 vs. Re S21
        xstr : x-axis string label
        fsize : fontsize for axes numbers and labels

    Returns:
        plot output to file
    """
    # close plot if still open
    plt.close(figurename)
    # generate an even distribution of 5000 frequency points between 
    # the min and max of x for graphing purposes
    if extract_factor == None:
        x_fit = np.linspace(x.min(), x.max(), 5000)
    elif isinstance(extract_factor, list) == True:
        x_fit = np.linspace(extract_factor[0], extract_factor[1], 5000)
    # plug in the 5000 x points to respective fit function to create set of 
    # S21 data for graphing
    y_fit = func(x_fit, *params)

    fig = plt.figure(figurename, figsize=(15, 10))
    fig.set_tight_layout(True)
    gs = GridSpec(11, 10)
    ax1 = plt.subplot(gs[0:4, 6:10])
    # normalized angle
    ax2 = plt.subplot(gs[4:8, 6:10])
    # IQ plot
    ax = plt.subplot(gs[0:11, 0:6])

    # Marker sizes
    msize1, msize2 = msizes

    # add title
    if len(title) > 77:
        plot_title = title[0:40] + "\n" + title[40:76] + '...'
        plt.gcf().text(0.05, 0.9, plot_title, fontsize=30)
    if len(title) > 40:
        plot_title = title[0:40] + "\n" + title[40:]
        plt.gcf().text(0.05, 0.9, plot_title, fontsize=30)
    else:
        plt.gcf().text(0.05, 0.92, title, fontsize=30)

    # manual parameters
    textstr = ''
    if manual_params != None:
        if func == ff.cavity_inverse:
            textstr = r'Manually input parameters:' + '\n' + r'$Q_c$ = ' + '%s' % float(
                '{0:.5g}'.format(manual_params[1])) + \
                      '\n' + r'$Q_i$ = ' + '%s' % float('{0:.5g}'.format(manual_params[0])) + \
                      '\n' + r'$f_c$ = ' + '%s' % float('{0:.5g}'.format(manual_params[2])) + ' GHz' \
                      '\n' + r'$\phi$ = ' + '%s' % float('{0:.5g}'.format(manual_params[3])) + ' radians'
        elif func == ff.cavity_CPZM:
            textstr = r'Manually input parameters:' + '\n' + r'$Q_c$ = ' + '%s' % float(
                '{0:.5g}'.format(manual_params[0] * manual_params[1] ** -1)) + \
                      '\n' + r'$Q_i$ = ' + '%s' % float('{0:.5g}'.format(manual_params[0])) + \
                      '\n' + r'$Q_a$ = ' + '%s' % float('{0:.5g}'.format(manual_params[0] * manual_params[3] ** -1)) + \
                      '\n' + r'$f_c$ = ' + '%s' % float('{0:.5g}'.format(manual_params[2])) + ' GHz'
        else:
            Qc = manual_params[1] / np.exp(1j * manual_params[3])
            Qi = (manual_params[0] ** -1 - abs(np.real(Qc ** -1))) ** -1
            textstr = r'Manually input parameters:' + '\n' + 'Q = ' + '%s' % float('{0:.5g}'.format(manual_params[0])) + \
                      '\n' + r'1/Re[1/$Q_c$] = ' + '%s' % float('{0:.5g}'.format(1 / np.real(1 / Qc))) + \
                      '\n' + r'$Q_c$ = ' + '%s' % float('{0:.5g}'.format(manual_params[1])) + \
                      '\n' + r'$Q_i$ = ' + '%s' % float('{0:.5g}'.format(Qi)) + \
                      '\n' + r'$f_c$ = ' + '%s' % float('{0:.5g}'.format(manual_params[2])) + ' GHz' + \
                      '\n' + r'$\phi$ = ' + '%s' % float('{0:.5g}'.format(manual_params[3])) + ' radians'
        plt.gcf().text(0.1, 0.7, textstr, fontsize=15)
    else:
        pass

    if isinstance(extract_factor, list) == True:
        x_fit_full = np.linspace(x.min(), x.max(), 5000)
        y_fit_full = func(x_fit_full, *params)

        x_fit = np.copy(x_fit_full)
        y_fit = np.copy(y_fit_full)

    if func == ff.cavity_inverse:
        ax1.set_ylabel('Mag[1/S21]')
        ax2.set_ylabel('Ang[1/S21]')
    else:
        ax1.set_ylabel('Mag[S21]')
        ax2.set_ylabel('Ang[S21]')

    for i in range(len(x_initial)):
        x_initial[i] = round(x_initial[i], 8)

    """
    ax1.plot(x_initial[0::dfac],
             np.log10(np.abs(y_initial[0::dfac])) * 20, 'bo',
             label='raw data',
             markersize=msize2)
    ax1.plot(x, x * slope2 + intercept2, '--', color='tab:gray',
             label='normalize line', linewidth=2)
    ax1.set_xlim(left=x[0], right=x[-1])
    ax1.set_xlabel(xstr, fontsize=17)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(fsize)

    ax2.plot(x_initial[0::dfac], np.angle(y_initial[0::dfac]), 'bo',
             label='raw data', markersize=msize2)
    ax2.plot(x, x * slope + intercept, '--', color='tab:gray',
             label='normalize line', linewidth=2)
    ax2.set_xlim(left=x[0], right=x[-1])
    ax2.set_xlabel(xstr, fontsize=17)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(fsize)
    """

    # Decimate the x and y-data
    x = x[0::dfac]
    y = y[0::dfac]

    ax1.plot(x, np.log10(np.abs(y)) * 20, 'bo',
            label='normalized data',
            markersize=msize2)
    ax1.plot(x_fit, np.log10(np.abs(y_fit)) * 20, 'r-',
             lw=3, label='fit function')
    ax1.set_xlim(left=x[0], right=x[-1])
    ax1.set_xlabel(xstr, fontsize=17)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(fsize)

    ax2.plot(x, np.angle(y), 'bo', label='normalized data', markersize=msize2)
    ax2.plot(x_fit, np.angle(y_fit), 'r-', label='fit function', lw=3)
    ax2.set_xlim(left=x[0], right=x[-1])
    ax2.set_xlabel(xstr, fontsize=17)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(fsize)

    line1 = ax.plot(np.real(y), np.imag(y), 'bo',
                    label='normalized data', markersize=msize2)
    line2 = ax.plot(np.real(y_fit), np.imag(y_fit), 'r-', label='fit function',
                    linewidth=3)
    if x_c == 0 and y_c == 0 and radius == 0:
        pass
    else:
        pass

    # plot resonance
    if func == ff.cavity_inverse:
        resonance = (1 + params[0] / params[1] * np.exp(1j * params[3]) / (
                1 + 1j * 2 * params[0] * (params[2] - params[2]) / params[2]))
    elif func == ff.cavity_DCM:
        resonance = 1 - params[0] / params[1] * np.exp(1j * params[3])
    elif func == ff.cavity_DCM_REFLECTION:
        resonance = (1 - 2 * params[0] / params[1] * np.exp(1j * params[3]) / (
                1 + 1j * (params[2] - params[2]) / params[2] * 2 * params[0]))
    elif func == ff.cavity_CPZM:
        resonance = 1 / (1 + params[1] + 1j * params[3])
    else:
        resonance = 1 + 1j * 0
    ax.plot(np.real(resonance), np.imag(resonance), '*', color='red', label=
    'resonance', markersize=10)
    ax1.plot(params[2], np.log10(np.abs(resonance)) * 20, '*', color='red', 
             label='resonance', markersize=msize1)
    ax2.plot(params[2], np.angle(resonance), '*', color='red', label=
    'resonance', markersize=msize1)

    plt.axis('square')
    plt.ylabel('Im[S21]')
    plt.xlabel("Re[S21]")
    if func == ff.cavity_inverse:
        plt.ylabel('Im[$S_{21}^{-1}$]')
        plt.xlabel("Re[$S_{21}^{-1}$]")
    leg = plt.legend()

    # get the individual lines inside legend and set line width
    for line in leg.get_lines():
        line.set_linewidth(10)

    try:
        if params:
            if func == ff.cavity_inverse:
                if params[0] < 0:
                    print("Qi is less than zero. Please make sure data is of"
                          " correct format: decibals (log10*20 version), and "
                          "radians. Otherwise, it is quite likely that the "
                          "resonator being fit is not a Notch type resonator. "
                          "For reflection type geometry, "
                          "please use DCM REFLECTION.")
                if conf_array[0] > 10 ** -10 and conf_array[0] != float('inf'):
                    Qi = params[0] - params[0] % (10 ** int(np.log10(conf_array[0]) - 1))
                else:
                    Qi = params[0]
                if conf_array[1] > 10 ** -10 and conf_array[1] != float('inf'):
                    Qc = params[1] - params[1] % (10 ** int(np.log10(conf_array[1]) - 1))
                else:
                    Qc = params[1]
                if conf_array[2] > 10 ** -10 and conf_array[2] != float('inf'):
                    phi = params[3] - params[3] % (10 ** int(np.log10(conf_array[2]) - 1))
                else:
                    phi = params[3]
                if conf_array[3] > 10 ** -10 and conf_array[3] != float('inf'):
                    f_c = params[2] - params[2] % (10 ** int(np.log10(conf_array[3]) - 1))
                else:
                    f_c = params[2]
                textstr = r'$Q_i$ = ' + '%s' % float('{0:.10g}'.format(Qi)) + r"$\pm$" + '%s' % float(
                    '{0:.1g}'.format(conf_array[0])) + \
                          '\n' + r'$Q_c^*$ = ' + '%s' % float('{0:.10g}'.format(Qc)) + r"$\pm$" + '%s' % float(
                    '{0:.1g}'.format(conf_array[1])) + \
                          '\n' + r'$\phi$ = ' + '%s' % float('{0:.10g}'.format(phi)) + r"$\pm$" + '%s' % float(
                    '{0:.1g}'.format(conf_array[2])) + ' radians' + \
                          '\n' + r'$f_c$ = ' + '%s' % float('{0:.10g}'.format(f_c)) + r"$\pm$" + '%s' % float(
                    '{0:.1g}'.format(conf_array[3])) + ' GHz'
                plt.gcf().text(0.7, 0.11, textstr, fontsize=18)
            elif func == ff.cavity_CPZM:
                if params[0] < 0:
                    print("Qi is less than zero. Please make sure data is "
                          "of correct format: decibals (log10*20 version), "
                          "and radians. Otherwise, it is quite likely that the "
                          "resonator being fit is not a Notch type resonator. "
                          "For reflection type geometry, "
                          "please use DCM REFLECTION.")
                if conf_array[0] > 10 ** -10 and conf_array[0] != float('inf'):
                    Qi = params[0] - params[0] % (10 ** int(np.log10(conf_array[0]) - 1))
                else:
                    Qi = params[0]
                if conf_array[1] > 10 ** -10 and conf_array[1] != float('inf'):
                    Qc = (params[0] * params[1] ** -1) - (params[0] * params[1] ** -1) % (
                            10 ** int(np.log10(conf_array[1]) - 1))
                else:
                    Qc = (params[0] * params[1] ** -1)
                if conf_array[2] > 10 ** -10 and conf_array[2] != float('inf'):
                    Qa = (params[0] * params[3] ** -1) - (params[0] * params[3] ** -1) % (
                            10 ** int(np.log10(conf_array[2]) - 1))
                else:
                    Qa = (params[0] * params[3] ** -1)
                if conf_array[3] > 10 ** -10 and conf_array[3] != float('inf'):
                    f_c = params[2] - params[2] % (10 ** int(np.log10(conf_array[3]) - 1))
                else:
                    f_c = params[2]
                reports = ['$Q_i$', '$Q_c$', '$Q_a$', '$f_c$']
                p_ref = [Qi, Qc, Qa, f_c]
                textstr = ''

                for val in reports:
                    textstr += val + f' = {p_ref[reports.index(val)]:.10f}' + r'$\pm$' + f'{conf_array[reports.index(val)]:.1f} UNITS'

                plt.gcf().text(0.7, 0.11, textstr, fontsize=18)

            else:
                Qc = params[1] / np.exp(1j * params[3])
                if Method.method == 'PHI':
                    Qi = (params[0] ** -1 - np.abs(Qc ** -1)) ** -1
                else:
                    Qi = (params[0] ** -1 - np.real(Qc ** -1)) ** -1

                if Qi < 0 and Method.method != 'DCM REFLECTION':
                    print(
                        "Qi is less than zero. Please make sure data is of correct format: decibals (log10*20 version), and radians. Otherwise, it is quite likely that the resonator being fit is not a Notch type resonator. Other types of resonators will not work with this code.")
                if 1 / np.real(1 / Qc) < 0:
                    print("Warning: 1/Real[1/Qc] is less than 0. Calculating Qi anyway")
                    if conf_array[0] > 10 ** -10 and conf_array[0] != float('inf'):
                        Q = params[0] - params[0] % (10 ** int(np.log10(conf_array[0]) - 1))
                    else:
                        Q = params[0]
                    if conf_array[1] > 10 ** -10 and conf_array[1] != float('inf'):
                        Qi = Qi - Qi % (10 ** int(np.log10(conf_array[1]) - 1))
                    if conf_array[2] > 10 ** -10 and conf_array[2] != float('inf'):
                        Qc = params[1] - params[1] % (10 ** int(np.log10(conf_array[2]) - 1))
                    else:
                        Qc = params[1]
                    if conf_array[3] > 10 ** -10 and conf_array[3] != float('inf'):
                        Qc_Re = (1 / np.real(1 / (params[1] / np.exp(1j * params[3])))) - (
                                1 / np.real(1 / (params[1] / np.exp(1j * params[3])))) % (
                                        10 ** int(np.log10(conf_array[3]) - 1))
                    else:
                        Qc_Re = (1 / np.real(1 / (params[1] / np.exp(1j * params[3]))))
                    if conf_array[4] > 10 ** -10 and conf_array[4] != float('inf'):
                        phi = params[3] - params[3] % (10 ** int(np.log10(conf_array[4]) - 1))
                    else:
                        phi = params[3]
                    if conf_array[5] > 10 ** -10 and conf_array[5] != float('inf'):
                        f_c = params[2] - params[2] % (10 ** int(np.log10(conf_array[5]) - 1))
                    else:
                        f_c = params[2]
                    textstr = 'Q = ' + '%s' % float('{0:.10g}'.format(Q)) + r"$\pm$" + '%s' % float(
                        '{0:.1g}'.format(conf_array[0])) + \
                              '\n' + r'$Q_i$ = ' + '%s' % float('{0:.10g}'.format(Qi)) + r"$\pm$" + '%s' % float(
                        '{0:.1g}'.format(conf_array[1])) + \
                              '\n' + r'$Q_c$ = ' + '%s' % float('{0:.10g}'.format(Qc)) + r"$\pm$" + '%s' % float(
                        '{0:.1g}'.format(conf_array[2])) + \
                              '\n' + r'$\phi$ = ' + '%s' % float('{0:.10g}'.format(phi)) + r"$\pm$" + '%s' % float(
                        '{0:.2g}'.format(conf_array[4])) + ' radians' + \
                              '\n' + r'$f_c$ = ' + '%s' % float('{0:.7g}'.format(f_c)) + r"$\pm$" + '%s' % float(
                        '{0:.1g}'.format(conf_array[5])) + ' GHz'
                    plt.gcf().text(0.7, 0.09, textstr, fontsize=18)
                    Qc_str = r'1/Re[1/$Q_c$] = ' + '%s' % float('{0:.10g}'.format(Qc_Re)) + r"$\pm$" + '%s' % float(
                        '{0:.1g}'.format(conf_array[3]))
                    plt.gcf().text(0.7, 0.245, Qc_str, fontsize=18, color='red')

                else:
                    if conf_array[0] > 10 ** -10 and conf_array[0] != float('inf'):
                        Q = params[0] - params[0] % (10 ** int(np.log10(conf_array[0]) - 1))
                    else:
                        Q = params[0]
                    if conf_array[1] > 10 ** -10 and conf_array[1] != float('inf'):
                        Qi = Qi - Qi % (10 ** int(np.log10(conf_array[1]) - 1))
                    if conf_array[2] > 10 ** -10 and conf_array[2] != float('inf'):
                        Qc = params[1] - params[1] % (10 ** int(np.log10(conf_array[2]) - 1))
                    else:
                        Qc = params[1]
                    if conf_array[3] > 10 ** -10 and conf_array[3] != float('inf'):
                        Qc_Re = (1 / np.real(1 / (params[1] / np.exp(1j * params[3])))) - (
                                1 / np.real(1 / (params[1] / np.exp(1j * params[3])))) % (
                                        10 ** int(np.log10(conf_array[3]) - 1))
                    else:
                        Qc_Re = (1 / np.real(1 / (params[1] / np.exp(1j * params[3]))))
                    if conf_array[4] > 10 ** -10 and conf_array[4] != float('inf'):
                        phi = params[3] - params[3] % (10 ** int(np.log10(conf_array[4]) - 1))
                    else:
                        phi = params[3]
                    if conf_array[5] > 10 ** -10 and conf_array[5] != float('inf'):
                        f_c = params[2] - params[2] % (10 ** int(np.log10(conf_array[5]) - 1))
                    else:
                        f_c = params[2]

                    reports = ['Q', r'$Q_i$', r'$Q_c$', r'1/Re[1/$Q_c$]', r'$\phi$', r'$f_c$']
                    p_ref = [Q, Qi, Qc, Qc_Re, phi, f_c]
                    textstr = ''
                    for val in reports:
                        textstr += val + f' = {p_ref[reports.index(val)]:0.10g}'

                        if conf_array[reports.index(val)] > 0:
                            textstr += r'$\pm$' + f'{conf_array[reports.index(val)]:5.3f}'

                        if val == r'$\phi$':
                            textstr += ' radians'
                        elif val == r'$f_c$':
                            textstr += ' GHz'
                        textstr += '\n'
                    plt.gcf().text(0.63, 0.05, textstr, fontsize=18)

            # write to output csv file
            with open(output_path + "fit_params.csv", "w", newline='') as file:
                writer = csv.writer(file)
                if func == ff.cavity_inverse:
                    fields = ['Qi', 'Qc*', 'phi', 'fc']
                    vals = [[float('{0:.10g}'.format(Qi)), float('{0:.10g}'.format(Qc)), float('{0:.10g}'.format(phi)),
                             float('{0:.10g}'.format(f_c))],
                            [float('{0:.1g}'.format(conf_array[0])), float('{0:.1g}'.format(conf_array[1])),
                             float('{0:.1g}'.format(conf_array[2])), float('{0:.1g}'.format(conf_array[3]))]]

                elif func == ff.cavity_CPZM:
                    fields = ['Qi', 'Qc', 'Qa', 'fc']
                    vals = [[float('{0:.10g}'.format(Qi)), float('{0:.10g}'.format(Qc)), float('{0:.10g}'.format(Qa)),
                             float('{0:.10g}'.format(f_c))],
                            [float('{0:.1g}'.format(conf_array[0])), float('{0:.1g}'.format(conf_array[1])),
                             float('{0:.1g}'.format(conf_array[2])), float('{0:.1g}'.format(conf_array[3]))]]
                else:

                    fields = ['Q', 'Qi', 'Qc', '1/Re[1/Qc]', 'phi', 'fc']
                    vals = [[float('{0:.10g}'.format(Q)), float('{0:.10g}'.format(Qi)), float('{0:.10g}'.format(Qc)),
                             float('{0:.10g}'.format(Qc_Re)), float('{0:.10g}'.format(phi)), float('{0:.10g}'.format(f_c))],
                            [float('{0:.1g}'.format(conf_array[0])), float('{0:.1g}'.format(conf_array[1])),
                             float('{0:.1g}'.format(conf_array[2])), float('{0:.1g}'.format(conf_array[3])),
                             float('{0:.1g}'.format(conf_array[4])), float('{0:.1g}'.format(conf_array[5]))]]
                writer.writerow(fields)
                writer.writerows(vals)
                file.close()
    except:
        print(">Error when trying to write parameters on plot")
        quit()
    return fig


def monte_carlo_fit(xdata=None, ydata=None, parameter=None, Method=None):
    try:
        ydata_1stfit = Method.func(xdata, *parameter)  # set of S21 data based on initial guess parameters

        ## weight condition
        if Method.MC_weight == 'yes':
            weight_array = 1 / abs(ydata)  # new array of inversed magnitude ydata
        else:
            weight_array = np.full(len(xdata), 1)  # new array of len(xdata) all slots filled with 1

        weighted_ydata = np.multiply(weight_array,
                                     ydata)  # array filled with 1s if MC_weight='yes' and exact same array as ydata otherwise
        weighted_ydata_1stfit = np.multiply(weight_array,
                                            ydata_1stfit)  # array with values (ydata^(-1))*ydata_1stfit if MC_weight='yes' and exact same array as ydata_1stfit otherwise
        error = np.linalg.norm(weighted_ydata - weighted_ydata_1stfit) / len(
            xdata)  # first error #finds magnitude of (weighted_ydata-weighted_ydata_1stfit) and divides by length (average magnitude)
        error_0 = error
    except:
        print(">Failed to initialize monte_carlo_fit(), please check parameters")
        quit()
    # Fix condition and Monte Carlo Method with random number Generator

    counts = 0
    try:
        while counts < Method.MC_rounds:
            counts = counts + 1
            # generate an array of 4 random numbers between -0.5 and 0.5 in the format [r,r,r,r] where r is each of the random numbers times the step constant
            random = Method.MC_step_const * (np.random.random_sample(len(parameter)) - 0.5)
            if 'Q' in Method.MC_fix:
                random[0] = 0
            if 'Qi' in Method.MC_fix:
                random[0] = 0
            if 'Qc' in Method.MC_fix:
                random[1] = 0
            if 'w1' in Method.MC_fix:
                random[2] = 0
            if 'phi' in Method.MC_fix:
                random[3] = 0
            if 'Qa' in Method.MC_fix:
                random[3] = 0
            # Generate new parameter to test
            if Method.method != 'CPZM':
                random[3] = random[3] * 0.1
            random = np.exp(random)
            new_parameter = np.multiply(parameter, random)
            if Method.method != 'CPZM':
                new_parameter[3] = np.mod(new_parameter[3], 2 * np.pi)

            # new set of data with new parameters
            ydata_MC = Method.func(xdata, *new_parameter)
            # check new error with new set of parameters
            weighted_ydata_MC = np.multiply(weight_array, ydata_MC)
            new_error = np.linalg.norm(weighted_ydata_MC - weighted_ydata) / len(xdata)
            if new_error < error:
                parameter = new_parameter
                error = new_error
    except:
        print(">Error in while loop of monte_carlo_fit")
        quit()
    ## If finally gets better fit then plot ##
    if error < error_0:
        stop_MC = False
        print('Monte Carlo fit got better fitting parameters')
        if Method.manual_init != None:
            print('>User input parameters getting stuck in local minimum, please input more accurate parameters')
    else:
        stop_MC = True
    return parameter, stop_MC, error


@attr.s
class VNASweep(object):
    """A container to hold data from a vna frequency sweep."""

    freqs = attr.ib(type=np.ndarray)
    amps = attr.ib(type=np.ndarray)
    phases = attr.ib(type=np.ndarray)
    linear_amps = attr.ib(type=np.ndarray)
    s_col = None

    @classmethod
    def from_file(cls, filepath, data_column=None, fscale=1e9):
        if data_column is not None:
            cls.s_col = data_column
        filename, extension = os.path.splitext(filepath)
        if extension.startswith('.s') and extension.endswith('p'):
            try:
                snp_file = open(filepath, 'r')
            except OSError as e:
                print(f'ERROR {e} when opening file')
                print(f'Data file: {filepath} could not be found/read')
            file, inline, options, frequency_units, data_format = cls.header_parse(cls, file=snp_file)
            freqs, amps, phases, linear_amps = cls.data_parse(cls, inline, frequency_units, data_format, file, options)
            print(frequency_units)
            if frequency_units == 'hz':
                fscale = fscale / 1e9
            elif frequency_units == 'mhz':
                fscale = fscale / 1e3
            elif frequency_units == 'khz':
                fscale = fscale / 1e6
            freqs = freqs / fscale

            return cls(freqs=freqs, amps=amps, phases=phases, linear_amps=linear_amps)
        elif 'txt' in extension or 'csv' in extension:
            try:
                txt_file = open(filepath, 'r')
                file, line, options, frequency_units, data_format = cls.header_parse(cls, file=txt_file)
                data_lines = []
                while line:
                    if 'END' in line:
                        break
                    data_lines.append(line)
                    line = file.readline().strip()
                data = np.loadtxt(data_lines, delimiter=',')
            except Exception as e:
                print(f'Exception: **{e}** encountered when attempting to load data file as .txt/.csv')
                print(f'Are you using a comma as your delimiter?')

            freqs = data.T[0] / fscale
            amps = data.T[1]
            phases = data.T[2] * np.pi / 180
            linear_amps = 10 ** (amps / 20)
            return cls(freqs=freqs, amps=amps, phases=phases, linear_amps=linear_amps)

    def header_parse(self, file):
        data_format = None
        frequency_units = None
        comment_line = ['!']
        option_line = ['#']
        metadata = ['s21', 's11', 's12', 's22']
        dformats = ['db', 'ma', 'ri']
        nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '.']

        options = []
        inline = file.readline()
        while any(comment in inline.lower() for comment in comment_line) \
                or any(option_lead in inline for option_lead in option_line) \
                or not any(number in inline.lower()[0] for number in nums):
            if any(option_lead in inline.lower() for option_lead in option_line) \
                    or any(dformat in inline.lower() for dformat in dformats) \
                    or any(measure in inline.lower() for measure in metadata):
                options.append(inline)

            inline = file.readline()

        for val in options:
            if 'db' in val.lower():
                data_format = 'db'
            elif 'ma' in val.lower():
                data_format = 'ma'
            elif 'ri' in val.lower():
                data_format = 'ri'

            if 'hz' in val.lower():
                frequency_units = 'hz'
            elif 'khz' in val.lower():
                frequency_units = 'khz'
            elif 'mhz' in val.lower():
                frequency_units = 'mhz'
            elif 'ghz' in val.lower():
                frequency_units = 'ghz'

        return file, inline, options, frequency_units, data_format

    def data_parse(self, line, frequency_units, data_format, file, options):
        row = line.split()
        data_rows = [3, 4]
        if len(row) == 0:
            print("Data not found in file.")
            quit()

        if len(row) > 3:
            # If too many rows, use info from header to pull correct column
            # s_col has potential focus column
            if isinstance(self.s_col, int):
                data_rows[0] = self.s_col
                data_rows[1] = self.c_col + 1
            elif isinstance(self.s_col, (tuple, list, np.ndarray)):
                data_rows[0] = self.s_col[0]
                data_rows[1] = self.s_col[1]
            elif isinstance(self.s_col, str):
                for metadata in options:
                    if 'Measurements: ' in metadata:
                        measurements = metadata.rsplit('Measurements: ')[1].strip('.:\n').lower().split(', ')
                        idx = (measurements.index(self.s_col.lower()) * 2) + 1
                        data_rows[0] = idx
                        data_rows[1] = idx + 1
                        break
            else:
                print("Could not interpret which data columns to use, using default")

        freqs = np.array(float(row[0]))
        if data_format == "db":
            amps = np.array(float(row[data_rows[0]]))
            phases = np.array(float(row[data_rows[1]]))
            line = file.readline().strip()

            while line:
                row = line.split()
                freqs = np.append(freqs, float(row[0]))
                amps = np.append(amps, float(row[data_rows[0]]))
                phases = np.append(phases, float(row[data_rows[1]]))
                line = file.readline().strip()
            phases = phases * np.pi / 180
            linear_amps = 10 ** (amps / 20)

        elif data_format == "ma":
            linear_amps = np.array(float(row[data_rows[0]]))
            phases = np.array(float(row[data_rows[1]]))
            line = file.readline().strip()

            while line:
                row = line.split()
                freqs = np.append(freqs, float(row[0]))
                linear_amps = np.append(linear_amps, float(row[data_rows[0]]))
                phases = np.append(phases, float(row[data_rows[1]]))
                line = file.readline().strip()

            phases = phases * np.pi / 180
            amps = np.log10(linear_amps) * 20

        elif data_format == "ri":
            real = np.array(float(row[data_rows[0]]))
            imaginary = np.array(float(row[data_rows[1]]))
            line = file.readline().strip()

            while line:
                row = line.split()
                freqs = np.append(freqs, float(row[0]))
                real = np.append(real, float(row[data_rows[0]]))
                imaginary = np.append(imaginary, float(row[data_rows[1]]))
                line = file.readline().strip()
            linear_amps = np.absolute(real + imaginary)
            phases = np.angle(real + 1j * imaginary, deg=True)
            amps = np.log10(linear_amps) * 20

        else:
            print("Data type in file not supported. Please use DB, MA, or RI.")
            quit()

        if frequency_units == "hz":
            freqs = freqs / 10 ** 9
        elif frequency_units == "khz":
            freqs = freqs / 10 ** 6
        elif frequency_units == "mhz":
            freqs = freqs / 10 ** 3
        elif frequency_units != "ghz":
            print(
                "Units for the frequency not found. Please include units for frequency in the header of the file.")
        return freqs, amps, phases, linear_amps

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


def phase_centered(f, fr, Ql, theta, delay=0.):
    """
        Yields the phase response of a strongly overcoupled (Qi >> Qc) resonator
        in reflection which corresponds to a circle centered around the origin.
        Additionally, a linear background slope is accounted for if needed.

        Args:
            fr: Resonance frequency
            Ql: Loaded quality factor (and since Qi >> Qc also Ql = Qc)
            theta: Offset phase
            delay (opt.): Time delay between output and input signal leading to
                        linearly frequency dependent phase shift
        Returns:
            Phase response (float)
        """
    return theta - 2 * np.pi * delay * (f - fr) + 2. * np.arctan(2. * Ql * (1. - f / fr))


def phase_dist(angle):
    """
        Maps angle [-2pi, +2pi] to phase distance on circle [0, pi]
        """
    return np.pi - np.abs(np.pi - np.abs(angle))


def fit_phase(f_data, z_data, guesses=None):
    """
        Fits the phase response of a strongly overcoupled (Qi >> Qc) resonator
        in reflection which corresponds to a circle centered around the origin
        (cf‌. phase_centered()).

        Args:
            z_data: Scattering data of which the phase should be fit. Data must be
                  distributed around origin ("circle-like").
            guesses (opt.): If not given, initial guesses for the fit parameters
                          will be determined. If given, should contain useful
                          guesses for fit parameters as a tuple (fr, Ql, delay)

        Returns:
            fr: Resonance frequency
            Ql: Loaded quality factor
            theta: Offset phase
            delay: Time delay between output and input signal leading to linearly
                 frequency dependent phase shift
        """
    phase = np.unwrap(np.angle(z_data))

    # For centered circle roll-off should be close to 2pi. If not warn user.
    if np.max(phase) - np.min(phase) <= 0.8 * 2 * np.pi:
        logging.warning(
            "Data does not cover a full circle (only {:.1f}".format(
                np.max(phase) - np.min(phase)
            )
            + " rad). Increase the frequency span around the resonance?"
        )
        roll_off = np.max(phase) - np.min(phase)
    else:
        roll_off = 2 * np.pi

    # Set useful starting parameters
    if guesses is None:
        # Use maximum of derivative of phase as guess for fr
        phase_smooth = gaussian_filter1d(phase, 30)
        phase_derivative = np.gradient(phase_smooth)
        fr_guess = f_data[np.argmax(np.abs(phase_derivative))]
        Ql_guess = 2 * fr_guess / (f_data[-1] - f_data[0])
        # Estimate delay from background slope of phase (substract roll-off)
        slope = phase[-1] - phase[0] + roll_off
        delay_guess = -slope / (2 * np.pi * (f_data[-1] - f_data[0]))
    else:
        fr_guess, Ql_guess, delay_guess = guesses
    # This one seems stable and we do not need a manual guess for it
    theta_guess = 0.5 * (np.mean(phase[:5]) + np.mean(phase[-5:]))

    # Fit model with less parameters first to improve stability of fit

    def residuals_Ql(params):
        Ql, = params
        return residuals_full((fr_guess, Ql, theta_guess, delay_guess))

    def residuals_fr_theta(params):
        fr, theta = params
        return residuals_full((fr, Ql_guess, theta, delay_guess))

    def residuals_delay(params):
        delay, = params
        return residuals_full((fr_guess, Ql_guess, theta_guess, delay))

    def residuals_fr_Ql(params):
        fr, Ql = params
        return residuals_full((fr, Ql, theta_guess, delay_guess))

    def residuals_full(params):
        return phase_dist(
            phase - phase_centered(f_data, *params)
        )

    p_final = spopt.leastsq(residuals_Ql, [Ql_guess])
    Ql_guess, = p_final[0]
    p_final = spopt.leastsq(residuals_fr_theta, [fr_guess, theta_guess])
    fr_guess, theta_guess = p_final[0]
    p_final = spopt.leastsq(residuals_delay, [delay_guess])
    delay_guess, = p_final[0]
    p_final = spopt.leastsq(residuals_fr_Ql, [fr_guess, Ql_guess])
    fr_guess, Ql_guess = p_final[0]
    p_final = spopt.leastsq(residuals_full, [
        fr_guess, Ql_guess, theta_guess, delay_guess
    ])

    return p_final[0]


def fit_delay(xdata: np.ndarray, ydata: np.ndarray):
    """
        Finds the cable delay by repeatedly centering the "circle" and fitting
        the slope of the phase response.
        """

    # Translate data to origin
    xc, yc, r0 = find_circle(np.real(ydata), np.imag(ydata))
    z_data = ydata - complex(xc, yc)
    # Find first estimate of parameters
    fr, Ql, theta, delay = fit_phase(xdata, z_data)

    # Do not overreact (see end of for loop)
    delay *= 0.05

    # Iterate to improve result for delay
    for i in range(5):
        # Translate new best fit data to origin
        z_data = ydata * np.exp(2j * np.pi * delay * xdata)
        xc, yc, r0 = find_circle(np.real(z_data), np.imag(z_data))
        z_data -= complex(xc, yc)

        # Find correction to current delay
        guesses = (fr, Ql, 5e-11)
        fr, Ql, theta, delay_corr = fit_phase(xdata, z_data, guesses)

        # Stop if correction would be smaller than "measurable"
        phase_fit = phase_centered(xdata, fr, Ql, theta, delay_corr)
        residuals = np.unwrap(np.angle(z_data)) - phase_fit
        if 2 * np.pi * (xdata[-1] - xdata[0]) * delay_corr <= np.std(residuals):
            break

        # Avoid overcorrection that makes procedure switch between positive
        # and negative delays
        if delay_corr * delay < 0:  # different sign -> be careful
            if abs(delay_corr) > abs(delay):
                delay *= 0.5
            else:
                # delay += 0.1*delay_corr
                delay += 0.1 * np.sign(delay_corr) * 5e-11
        else:  # same direction -> can converge faster
            if abs(delay_corr) >= 1e-8:
                delay += min(delay_corr, delay)
            elif abs(delay_corr) >= 1e-9:
                delay *= 1.1
            else:
                delay += delay_corr

    if 2 * np.pi * (xdata[-1] - xdata[0]) * delay_corr > np.std(residuals):
        logging.warning(
            "Delay could not be fit properly!"
        )

    return delay


def periodic_boundary(angle):
    """
        Maps arbitrary angle to interval [-np.pi, np.pi)
        """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def calibrate(x_data: np.ndarray, z_data: np.ndarray):
    """
        Finds the parameters for normalization of the scattering data. See
        Sij of port classes for explanation of parameters.
        """

    # Translate circle to origin
    xc, yc, r = find_circle(np.real(z_data), np.imag(z_data))
    zc = complex(xc, yc)
    z_data2 = z_data - zc

    # Find off-resonant point by fitting offset phase
    # (centered circle corresponds to lossless resonator in reflection)
    fr, Ql, theta, delay_remaining = fit_phase(x_data, z_data2)
    theta_interval = periodic_boundary(theta)
    beta = periodic_boundary(theta - np.pi)
    offrespoint = zc + r * np.cos(beta) + 1j * r * np.sin(beta)
    a = np.absolute(offrespoint)
    alpha = np.angle(offrespoint)
    phi = periodic_boundary(beta - alpha)

    # Store radius for later calculation
    r /= a

    return delay_remaining, a, alpha, theta, phi, fr, Ql


def normalize(f_data, z_data, delay, a, alpha):
    """
        Transforms scattering data into canonical position with off-resonant
        point at (1, 0) (does not correct for rotation phi of circle around
        off-resonant point).
        """
    z_norm = (z_data / a) * np.exp(1j * (-alpha))
    return z_norm


def preprocess_linear(xdata: np.ndarray, ydata: np.ndarray, normalize: int, output_path: str, plot_extra):
    """
    Data Preprocessing. Get rid of cable delay and normalize phase/magnitude of S21 by linear fit of normalize # of endpoints
    """
    if plot_extra:
        plot(np.real(ydata), np.imag(ydata), "Normalize_1", output_path)

    if normalize * 2 > len(ydata):
        print(
            "Not enough points to normalize, please lower value of normalize variable or take more points near resonance")
        quit()

    # Check for bad linear preprocessing outputs
    # Redirect to circle preprocessing
    phase = np.unwrap(np.angle(ydata))

    # normalize phase of S21 using linear fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.append(xdata[0:normalize], xdata[-normalize:]),
                                                                   np.append(phase[0:normalize], phase[-normalize:]))

    angle = np.subtract(phase, slope * xdata)  # remove cable delay
    y_test = np.multiply(np.abs(ydata), np.exp(1j * angle))
    if plot_extra:
        plot(np.real(y_test), np.imag(y_test), "Normalize_2", output_path)

    angle = np.subtract(angle, intercept)  # rotate off resonant point to (1,0i) in complex plane
    y_test = np.multiply(np.abs(ydata), np.exp(1j * angle))
    if plot_extra:
        plot(np.real(y_test), np.imag(y_test), "Normalize_3", output_path)

    # normalize magnitude of S21 using linear fit
    y_db = np.log10(np.abs(ydata)) * 20
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(
        np.append(xdata[0:normalize], xdata[-normalize:]), np.append(y_db[0:normalize], y_db[-normalize:]))
    magnitude = np.subtract(y_db, slope2 * xdata + intercept2)
    magnitude = 10 ** (magnitude / 20)

    preprocessed_data = np.multiply(magnitude, np.exp(1j * angle))
    if plot_extra:
        plot(np.real(preprocessed_data), np.imag(preprocessed_data), "Normalize_4", output_path)

    return preprocessed_data, slope, intercept, slope2, intercept2


def preprocess_circle(xdata: np.ndarray, ydata: np.ndarray, output_path: str, plot_extra):
    """
    Data Preprocessing. Use Probst method to get rid of cable delay and normalize phase/magnitude of S21 by circle fit
    """
    # # Unwrap the phase
    # phase = np.unwrap(np.angle(ydata))
    # ydata = np.abs(ydata) * np.exp(1j * phase)

    if plot_extra:
        plot(np.real(ydata), np.imag(ydata), "Normalize_1", output_path)

    # remove cable delay
    delay = fit_delay(xdata, ydata)
    z_data = ydata * np.exp(2j * np.pi * delay * xdata)

    if plot_extra:
        plot(np.real(z_data), np.imag(z_data), "Normalize_2", output_path)

    # calibrate and normalize
    delay_remaining, a, alpha, theta, phi, fr, Ql = calibrate(xdata, z_data)
    z_norm = normalize(xdata, z_data, delay_remaining, a, alpha)

    if plot_extra:
        plot(np.real(z_norm), np.imag(z_norm), "Normalize_3", output_path)

    return z_norm


def background_removal(databg: VNASweep, linear_amps: np.ndarray, 
                       phases: np.ndarray, output_path: str):
    x_bg = databg.freqs
    linear_amps_bg = databg.linear_amps
    phases_bg = databg.phases

    ybg = np.multiply(linear_amps_bg, np.exp(1j * phases_bg))

    fmag = interp1d(x_bg, linear_amps_bg, kind='cubic')
    fang = interp1d(x_bg, phases_bg, kind='cubic')

    plot2(databg.freqs, databg.linear_amps, x_bg, linear_amps_bg, "VS_mag", output_path)
    plot2(databg.freqs, databg.phases, x_bg, phases_bg, "VS_ang", output_path)

    linear_amps = np.divide(linear_amps, linear_amps_bg)
    phases = np.subtract(phases, phases_bg)

    return np.multiply(linear_amps, np.exp(1j * phases))


# Fit data to least squares fit for respective fit type
def min_fit(params, xdata, ydata, Method):
    """Minimizes parameter values for the given function and transmission data

    Args:
        params: guess for correct values of the fit parameters
        xdata: array of frequency data points
        ydata: array of S21 data points
        Method: instance of Method class

    Returns:
        minimized parameter values, 95% confidence intervals for those parameter values
    """
    try:
        if Method.method == 'DCM' or Method.method == 'PHI':
            minner = Minimizer(ff.min_one_Cavity_dip, params, fcn_args=(xdata, ydata))
        elif Method.method == 'DCM REFLECTION':
            minner = Minimizer(ff.min_one_Cavity_DCM_REFLECTION, params, fcn_args=(xdata, ydata))
        elif Method.method == 'INV':
            minner = Minimizer(ff.min_one_Cavity_inverse, params, fcn_args=(xdata, ydata))
        elif Method.method == 'CPZM':
            minner = Minimizer(ff.min_one_Cavity_CPZM, params, fcn_args=(xdata, ydata))

        result = minner.minimize(method='least_squares')

        fit_params = result.params
        parameter = fit_params.valuesdict()
        # extracts the actual value for each parameter and puts it in the fit_params list
        fit_params = [value for _, value in parameter.items()]
    except:
        print(">Failed to minimize data for least squares fit")
        print(">Confidence intervals unknown and given as 0.0")
        fit_params = None
        conf_array = [0, 0, 0, 0, 0, 0]
        return fit_params, conf_array
    try:
        p_names = []
        for parameter in params:
            if parameter not in Method.MC_fix:
                p_names.append(parameter)
        if Method.method == 'DCM' or Method.method == 'PHI' or Method.method == 'DCM REFLECTION':
            ci = lmfit.conf_interval(minner, result, p_names=p_names, sigmas=[2])

            # confidence interval for Q
            if 'Q' in p_names:
                Q_conf = max(np.abs(ci['Q'][1][1] - ci['Q'][0][1]), np.abs(ci['Q'][1][1] - ci['Q'][2][1]))
            else:
                Q_conf = 0
            # confidence interval for Qi
            if 'Q' in p_names and 'Qc' in p_names and 'Qi' not in Method.MC_fix:
                if Method.method == 'PHI':
                    Qi = ((ci['Q'][1][1]) ** -1 - np.abs(ci['Qc'][1][1] ** -1 * np.exp(1j * fit_params[3]))) ** -1
                    Qi_neg = Qi - (
                                (ci['Q'][0][1]) ** -1 - np.abs(ci['Qc'][2][1] ** -1 * np.exp(1j * fit_params[3]))) ** -1
                    Qi_pos = Qi - (
                                (ci['Q'][2][1]) ** -1 - np.abs(ci['Qc'][0][1] ** -1 * np.exp(1j * fit_params[3]))) ** -1
                else:
                    Qi = ((ci['Q'][1][1]) ** -1 - np.real(ci['Qc'][1][1] ** -1 * np.exp(1j * fit_params[3]))) ** -1
                    Qi_neg = Qi - ((ci['Q'][0][1]) ** -1 - np.real(
                        ci['Qc'][2][1] ** -1 * np.exp(1j * fit_params[3]))) ** -1
                    Qi_pos = Qi - ((ci['Q'][2][1]) ** -1 - np.real(
                        ci['Qc'][0][1] ** -1 * np.exp(1j * fit_params[3]))) ** -1
                Qi_conf = max(np.abs(Qi_neg), np.abs(Qi_pos))
            else:
                Qi_conf = 0
            # confidence interval for Qc
            if 'Qc' in p_names:
                Qc_conf = max(np.abs(ci['Qc'][1][1] - ci['Qc'][0][1]), np.abs(ci['Qc'][1][1] - ci['Qc'][2][1]))
                # Ignore one-sided conf test
                if np.isinf(Qc_conf):
                    Qc_conf = min(np.abs(ci['Qc'][1][1] - ci['Qc'][0][1]), np.abs(ci['Qc'][1][1] - ci['Qc'][2][1]))
                # confidence interval for 1/Re[1/Qc]
                Qc_Re = 1 / np.real(np.exp(1j * fit_params[3]) / ci['Qc'][1][1])
                Qc_Re_neg = 1 / np.real(np.exp(1j * fit_params[3]) / ci['Qc'][0][1])
                Qc_Re_pos = 1 / np.real(np.exp(1j * fit_params[3]) / ci['Qc'][2][1])
                Qc_Re_conf = max(np.abs(Qc_Re - Qc_Re_neg), np.abs(Qc_Re - Qc_Re_pos))
                # Ignore one-sided conf test
                if np.isinf(Qc_Re_conf):
                    Qc_Re_conf = min(np.abs(Qc_Re - Qc_Re_neg), np.abs(Qc_Re - Qc_Re_pos))
            else:
                Qc_conf = 0
            # confidence interval for phi
            if 'phi' in p_names:
                phi_neg = ci['phi'][0][1]
                phi_pos = ci['phi'][2][1]
                phi_conf = max(np.abs(ci['phi'][1][1] - ci['phi'][0][1]), 
                               np.abs(ci['phi'][1][1] - ci['phi'][2][1]))
                # Ignore one-sided conf test
                if np.isinf(phi_conf):
                    phi_conf = min(np.abs(ci['phi'][1][1] - ci['phi'][0][1]), 
                                   np.abs(ci['phi'][1][1] - ci['phi'][2][1]))
            else:
                phi_conf = 0
            # confidence interval for resonance frequency
            if 'w1' in p_names:
                w1_neg = ci['w1'][0][1]
                w1_pos = ci['w1'][2][1]
                w1_conf = max(np.abs(ci['w1'][1][1] - ci['w1'][0][1]), 
                              np.abs(ci['w1'][1][1] - ci['w1'][2][1]))
                # Ignore one-sided conf test
                if np.isinf(w1_conf):
                    w1_conf = min(np.abs(ci['w1'][1][1] - ci['w1'][0][1]), 
                                  np.abs(ci['w1'][1][1] - ci['w1'][2][1]))
            else:
                w1_conf = 0
            # Array of confidence intervals
            conf_array = [Q_conf, Qi_conf, Qc_conf, Qc_Re_conf, phi_conf, w1_conf]
        elif Method.method == 'INV':
            ci = lmfit.conf_interval(minner, result, p_names=p_names, sigmas=[2])
            # confidence interval for Qi
            if 'Qi' in p_names:
                Qi_conf = max(np.abs(ci['Qi'][1][1] - ci['Qi'][0][1]), 
                              np.abs(ci['Qi'][1][1] - ci['Qi'][2][1]))
            else:
                Qi_conf = 0
            # confidence interval for Qc
            if 'Qc' in p_names:
                Qc_conf = max(np.abs(ci['Qc'][1][1] - ci['Qc'][0][1]), 
                              np.abs(ci['Qc'][1][1] - ci['Qc'][2][1]))
            else:
                Qc_conf = 0
            # confidence interval for phi
            if 'phi' in p_names:
                phi_conf = max(np.abs(ci['phi'][1][1] - ci['phi'][0][1]), 
                               np.abs(ci['phi'][1][1] - ci['phi'][2][1]))
                # Ignore one-sided conf test
                if np.isinf(phi_conf):
                    phi_conf = min(np.abs(ci['phi'][1][1] - ci['phi'][0][1]), 
                                   np.abs(ci['phi'][1][1] - ci['phi'][2][1]))
            else:
                phi_conf = 0
            # confidence interval for resonance frequency
            if 'w1' in p_names:
                w1_conf = max(np.abs(ci['w1'][1][1] - ci['w1'][0][1]), 
                              np.abs(ci['w1'][1][1] - ci['w1'][2][1]))
                # Ignore one-sided conf test
                if np.isinf(w1_conf):
                    w1_conf = min(np.abs(ci['w1'][1][1] - ci['w1'][0][1]), 
                                  np.abs(ci['w1'][1][1] - ci['w1'][2][1]))
            else:
                w1_conf = 0
            # Array of confidence intervals
            conf_array = [Qi_conf, Qc_conf, phi_conf, w1_conf]
        else:
            ci = lmfit.conf_interval(minner, result, p_names=p_names, sigmas=[2])
            # confidence interval for Qi
            if 'Qi' in p_names:
                Qi_conf = max(np.abs(ci['Qi'][1][1] - ci['Qi'][0][1]), 
                              np.abs(ci['Qi'][1][1] - ci['Qi'][2][1]))
            else:
                Qi_conf = 0
            # confidence interval for Qc
            if 'Qc' in p_names:
                Qc = ci['Qi'][1][1] / ci['Qc'][1][1]
                Qc_neg = ci['Qi'][0][1] / ci['Qc'][0][1]
                Qc_pos = ci['Qi'][2][1] / ci['Qc'][2][1]
                Qc_conf = max(np.abs(Qc - Qc_neg), np.abs(Qc - Qc_neg))
            else:
                Qc_conf = 0
            # confidence interval for Qa
            if 'Qa' in p_names:
                Qa = ci['Qi'][1][1] / ci['Qa'][1][1]
                Qa_neg = ci['Qi'][2][1] / ci['Qa'][2][1]
                Qa_pos = ci['Qi'][0][1] / ci['Qa'][0][1]
                Qa_conf = max(np.abs(Qa - Qa_neg), np.abs(Qa - Qa_neg))
            else:
                Qa_conf = 0
            # confidence interval for resonance frequency
            if 'w1' in p_names:
                w1_conf = max(np.abs(ci['w1'][1][1] - ci['w1'][0][1]), 
                              np.abs(ci['w1'][1][1] - ci['w1'][2][1]))
                # Ignore one-sided conf test
                if np.isinf(w1_conf):
                    min(np.abs(ci['w1'][1][1] - ci['w1'][0][1]), 
                        np.abs(ci['w1'][1][1] - ci['w1'][2][1]))
            else:
                w1_conf = 0
            # Array of confidence intervals
            conf_array = [Qi_conf, Qc_conf, Qa_conf, w1_conf]
    except Exception as e:
        print(e)
        print(">Failed to find confidence intervals for least squares fit")
        conf_array = [0, 0, 0, 0, 0, 0]
    return fit_params, conf_array


def fit(resonator):
    """Function to fit resonator data

    Args:
        Resonator class object

    Returns:
        final minimized parameter values
        95% confidence interval values for those parameters
        main figure output by the plotting function
        error of Monte Carlo Fit
        initial guess parameters
    """

    filepath = resonator.filepath
    Method = resonator.method_class
    normalize = resonator.normalize
    data = resonator.data
    background = resonator.background
    background_array = resonator.background_array
    plot_extra = resonator.plot_extra
    preprocess_method = resonator.preprocess_method

    # read in data from file
    if filepath is not None:
        os_path = os.path.split(filepath)
        dir = os_path[0]
        filename = os_path[1]
        if dir == '':
            dir = ROOT_DIR
    else:
        dir = ROOT_DIR
        filename = 'scresonators'

    # separate data by column
    try:
        xdata = data.freqs
        linear_amps = data.linear_amps
        phases = np.unwrap(data.phases)
        ydata = np.multiply(linear_amps, np.exp(1j * phases))

    except Exception as e:
        print(f'Exception: {e}')
        print("When trying to read data from VNASweep class")
        quit()

    # make a folder to put all output in
    result = time.localtime(time.time())
    output = str(result.tm_year)
    output = str(Method.method) + '_' + output
    if len(str(result.tm_mon)) < 2:
        output = output + '0' + str(result.tm_mon)
    else:
        output = output + str(result.tm_mon)
    if len(str(result.tm_mday)):
        output = output + '0' + str(result.tm_mday) + '_' + str(result.tm_hour)\
             + '_' + str(result.tm_min) + '_' + str(result.tm_sec)
    else:
        output = output + str(result.tm_mday) + '_' + str(result.tm_hour) + '_' \
            + str(result.tm_min) + '_' + str(result.tm_sec)

    output_path = dir + '/' + output + '/'

    count = 2
    path = output_path
    while os.path.isdir(output_path):
        output_path = path[0:-1] + '_' + str(count) + '/'
        count = count + 1
    os.mkdir(output_path)

    # original data before normalization
    x_initial = xdata
    y_initial = ydata

    # normalize data
    slope = 0
    intercept = 0
    slope2 = 0
    intercept2 = 0
    if resonator.databg is not None:
        ydata = background_removal(resonator.databg, linear_amps, phases, output_path)
    elif preprocess_method == "linear":
        t_ydata, t_slope, t_intercept, t_slope2, t_intercept2 = preprocess_linear(xdata, ydata, normalize, output_path,
                                                                                  plot_extra)
        # Logic check for error'd linear preprocessing
        if t_ydata == "circle":
            preprocess_method = "circle"
            ydata = preprocess_circle(xdata, ydata, output_path, plot_extra)
        else:
            ydata, slope, intercept, slope2, intercept2 = t_ydata, t_slope, t_intercept, t_slope2, t_intercept2


    elif preprocess_method == "circle":
        ydata = preprocess_circle(xdata, ydata, output_path, plot_extra)
    else:
        pass
    print(f'preprocess_method: {preprocess_method}')

    # a copy of data before modification for plotting
    y_raw = ydata
    x_raw = xdata

    # Init function variables
    manual_init = Method.manual_init
    change_Q, change_Qi, change_Qc, change_w1, change_phi, change_Qa = True, True, True, True, True, True
    if 'Q' in Method.MC_fix:
        change_Q = False
    if 'Qi' in Method.MC_fix:
        change_Qi = False
    if 'Qc' in Method.MC_fix:
        change_Qc = False
    if 'w1' in Method.MC_fix:
        change_w1 = False
    if 'phi' in Method.MC_fix:
        change_phi = False
    if 'Qa' in Method.MC_fix:
        change_Qa = False
    y1data = np.real(ydata)
    y2data = np.imag(ydata)

    # Step one. Find initial guess if not specified and extract part of data close to resonance

    if len(x_raw) < 20:
        print(">Not enough data points to run code. Please have at least 20 data points.")
        quit()

    # place to store initial guess parameters
    init = [0] * 4
    # when user manually initializes a guess initialize the following variables
    if manual_init is not None:
        try:
            if len(manual_init) == 4:

                # bandwidth for frequency values
                print(f'init: {init}')

                # FIXME: This is a bug in the manual_init != None case
                #        It will result in a divide by zero exception, printed
                #        below by the try - except block
                #        Solution is to move kappa into if
                # kappa = init[2] / (init[0])

                # If method is DCM or PHI, set parameter 1 equal to Q which is 1/(1/Qi + 1/Qc) aka. convert from Qi
                if Method.method == 'DCM' or Method.method == "DCM REFLECTION" or Method.method == 'PHI':
                    Qc = manual_init[1] / np.exp(1j * manual_init[3])
                    if Method.method == 'PHI':
                        manual_init[0] = 1 / (1 / manual_init[0] + np.abs(1 / Qc))
                    else:
                        manual_init[0] = 1 / (1 / manual_init[0] + np.real(1 / Qc))
                    kappa = manual_init[2] / manual_init[0]
                elif Method.method == 'CPZM':
                    Q = 1 / (1 / manual_init[0] + 1 / manual_init[1])
                    kappa = manual_init[2] / Q
                    manual_init[1] = manual_init[0] / manual_init[1]
                    manual_init[3] = manual_init[0] / manual_init[3]

                init = manual_init
                freq = init[2]

                # set initial guess circle variables to zero so circle does not appear in plots
                x_c, y_c, r = 0, 0, 0
                print("Manual initial guess")
            else:
                print(manual_init)
                print(">Manual input wrong format, please follow the correct "
                      "format of 4 parameters in an array")
                quit()
        except Exception as e:
            print(f'Excepction {e}')
            print(f'Loaded manual_init: {manual_init}')
            print("Problem loading manually initialized parameters, please "
                  "make sure parameters are all numbers")
            quit()
    else:
        # generate initial guess parameters from data when user does not manually initialze guess
        init, x_c, y_c, r = find_initial_guess(xdata, y1data, y2data, Method, output_path,
                                               plot_extra)  # resonance frequency
        freq = init[2]
        # f_0/Qi is kappa
        kappa = init[2] / init[0]
        if Method.method == 'CPZM':
            kappa = init[4]
            init = init[0:4]

    # Extract data near resonate frequency to fit
    xdata, ydata = extract_near_res(x_raw, y_raw, freq, kappa,
                                    extract_factor=1)  # xdata is new set of data to be fit, within extract_factor
    # times the bandwidth, ydata is S21 data to match indices with xdata

    if Method.method == 'INV':
        ydata = ydata ** -1  # Inverse S21

    # Step Two. Fit Both Re and Im data
    # create a set of Parameters
    ## Monte Carlo Loop to check for local minimums
    # define parameters from initial guess for John Martinis and monte_carlo_fit
    try:
        # initialize parameter class, min is lower bound, max is upper bound, 
        # vary = boolean to determine if parameter varies during fit
        params = lmfit.Parameters()
        if Method.method == 'DCM' or Method.method == 'DCM REFLECTION' or Method.method == 'PHI':
            params.add('Q', value=init[0], vary=change_Q, min=init[0] * 0.5, max=init[0] * 1.5)
        elif Method.method == 'INV' or Method.method == 'CPZM':
            params.add('Qi', value=init[0], vary=change_Qi, min=init[0] * 0.8, max=init[0] * 1.2)
        params.add('Qc', value=init[1], vary=change_Qc, min=init[1] * 0.8, max=init[1] * 1.2)
        params.add('w1', value=init[2], vary=change_w1, min=init[2] * 0.9, max=init[2] * 1.1)
        if Method.method == 'CPZM':
            params.add('Qa', value=init[3], vary=change_Qa, min=-init[3] * 1.1, max=init[3] * 1.1)
        else:
            params.add('phi', value=init[3], vary=change_phi, min=-np.pi, max=np.pi)
    except Exception as e:
        print(f'Exception {e}')
        print(">Failed to define parameters, please make sure parameters are of correct format")
        quit()

    # Fit data to least squares fit for respective fit type
    fit_params, conf_array = min_fit(params, xdata, ydata, Method)

    if manual_init is None and fit_params is None:
        print(">Failed to minimize function for least squares fit")
        quit()
    if fit_params is None:
        fit_params = manual_init

    # setup for while loop
    MC_counts = 0
    error = [10]
    stop_MC = False
    continue_condition = (MC_counts < Method.MC_iteration) and (stop_MC == False)
    output_params = []

    while continue_condition:

        # run a Monte Carlo fit on just minimized data to test if parameters trapped in local minimum
        MC_param, stop_MC, error_MC = monte_carlo_fit(xdata, ydata, fit_params, Method)
        error.append(error_MC)
        if error[MC_counts] < error_MC:
            stop_MC = True

        output_params.append(MC_param)
        MC_counts = MC_counts + 1

        continue_condition = (MC_counts < Method.MC_iteration) and (stop_MC == False)

        if continue_condition == False:
            output_params = output_params[MC_counts - 1]

    error = min(error)

    # if monte carlo fit got better results than initial minimization, run a minimization on the monte carlo parameters
    if output_params[0] != fit_params[0]:
        params2 = lmfit.Parameters()  # initialize parameter class, min is lower bound, max is upper bound, vary = boolean to determine if parameter varies during fit
        if Method.method == 'DCM' or Method.method == 'DCM REFLECTION' or Method.method == 'PHI':
            params2.add('Q', value=output_params[0], vary=change_Q, min=output_params[0] * 0.5,
                        max=output_params[0] * 1.5)
        elif Method.method == 'INV' or Method.method == 'CPZM':
            params2.add('Qi', value=output_params[0], vary=change_Qi, min=output_params[0] * 0.8,
                        max=output_params[0] * 1.2)
        params2.add('Qc', value=output_params[1], vary=change_Qc, min=output_params[1] * 0.8,
                    max=output_params[1] * 1.2)
        params2.add('w1', value=output_params[2], vary=change_w1, min=output_params[2] * 0.9,
                    max=output_params[2] * 1.1)
        if Method.method == 'CPZM':
            params2.add('Qa', value=output_params[3], vary=change_Qa, min=output_params[3] * 0.9,
                        max=output_params[3] * 1.1)
        else:
            params2.add('phi', value=output_params[3], vary=change_phi, min=output_params[3] * 0.9,
                        max=output_params[3] * 1.1)
        output_params, conf_array = min_fit(params2, xdata, ydata, Method)

    if manual_init is None and fit_params is None:
        print(">Failed to minimize function for least squares fit")
        quit()
    if fit_params is None:
        fit_params = manual_init

    # Check that bandwidth is not equal to zero
    if len(xdata) == 0:
        if manual_init is not None:
            print(">Length of extracted data equals zero thus bandwidth is incorrect, "
                  "most likely due to initial parameters being too far off")
            print(">Please enter a new set of manual initial guess data "
                  "or run an auto guess")
        else:
            print(">Length of extracted data equals zero thus bandwidth is incorrect, "
                  "please manually input a guess for parameters")
        quit()

    # set the range to plot for 1 3dB bandwidth
    if Method.method == 'CPZM':
        Q = 1 / (1 / output_params[0] + output_params[1] / output_params[0])
        kappa = output_params[2] / Q
    else:
        kappa = output_params[2] / output_params[0]
    xstart = output_params[2] - kappa / 2  # starting resonance to add to fit
    xend = output_params[2] + kappa / 2
    extract_factor = [xstart, xend]

    # plot fit
    if Method.method == 'DCM':
        title = 'Diameter Correction Method Fit'
        figurename = " DCM with Monte Carlo Fit and Raw data\nPower: " + filename
        fig = PlotFit(x_raw, y_raw, x_initial, y_initial, slope, intercept, 
                      slope2, intercept2, output_params, Method, ff.cavity_DCM, 
                      error, figurename, x_c, y_c, r, output_path, conf_array, 
                      extract_factor, title=title, manual_params=Method.manual_init)
    elif Method.method == 'PHI':
        try:
            title = 'PHI fit for ' + filename
            figurename = "PHI with Monte Carlo Fit and Raw data\nPower: " + filename
            fig = PlotFit(x_raw, y_raw, x_initial, y_initial, slope, intercept, 
                      slope2, intercept2, output_params, Method, ff.cavity_DCM, 
                      error, figurename, x_c, y_c, r, output_path, conf_array, 
                      extract_factor, title=title, manual_params=Method.manual_init)
        except Exception as e:
            print(f'Exception: {e}')
            print(f'Failed to plot PHI fit for {data}')
            quit()
    elif Method.method == 'DCM REFLECTION':
        try:
            title = 'DCM REFLECTION fit for ' + filename
            figurename = " DCM REFLECTION with Monte Carlo Fit and Raw data\nPower: " + filename
            fig = PlotFit(x_raw, y_raw, x_initial, y_initial, slope, intercept, 
                      slope2, intercept2, output_params, Method, ff.cavity_DCM_REFLECTION,
                      error, figurename, x_c, y_c, r, output_path, conf_array, 
                      extract_factor, title=title, manual_params=Method.manual_init)
        except:
            print(">Failed to plot DCM fit for data")
            quit()
    elif Method.method == 'INV':
        try:
            title = 'INV fit for ' + filename
            figurename = " Inverse with MC Fit and Raw data\nPower: " + filename
            fig = PlotFit(x_raw, y_raw, x_initial, y_initial, slope, intercept, 
                      slope2, intercept2, output_params, Method, ff.cavity_inverse, 
                      error, figurename, x_c, y_c, r, output_path, conf_array, 
                      extract_factor, title=title, manual_params=Method.manual_init)
        except Exception as e:
            print(f'Exception: {e}')
            print(f'Failed to plot INV fit for {data}')
            quit()
    elif Method.method == 'CPZM':
        try:
            title = 'CPZM fit for ' + filename
            figurename = " CPZM with MC Fit and Raw data\nPower: " + filename
            fig = PlotFit(x_raw, y_raw, x_initial, y_initial, slope, intercept, 
                      slope2, intercept2, output_params, Method, ff.cavity_CPZM, 
                      error, figurename, x_c, y_c, r, output_path, conf_array, 
                      extract_factor, title=title, manual_params=Method.manual_init)
        except Exception as e:
            print(f'Exception: {e}')
            print(f'Failed to plot CPZM fit for {data}')
            quit()

    fig.savefig(name_plot(filename, str(Method.method), output_path),
                format='pdf')
    
    ######################### CODE TO BE MOVED TO INDEPENDENT FUNCTION IN SOFTWARE REDESIGN #####################

    repo = Repo(ROOT_DIR)
    sha = repo.head.object.hexsha
    # write input parameters to metadata file
    with open(output_path + "metadata.csv", "w", newline='') as file:
        writer = csv.writer(file)
        fields = ['Method', 'MC_iteration', 'MC_rounds',
                  'MC_weight', 'MC_weightvalue', 'MC_fix',
                  'MC_step_const', 'manual_init',
                  'preprocess_method', 'Current Git Commit']
        vals = [resonator.method_class.method, resonator.method_class.MC_iteration, 
                resonator.method_class.MC_rounds,resonator.method_class.MC_weight, 
                resonator.method_class.MC_weightvalue, resonator.method_class.MC_fix,
                resonator.method_class.MC_step_const, resonator.method_class.manual_init,
                resonator.method_class.preprocess_method, sha]
        writer.writerow(fields)
        writer.writerow(vals)
        file.close()

    #############################################################################################################

    return output_params, conf_array, fig, error, init


def plot(x,
         y,
         name,
         output_path,
         x_c=None,
         y_c=None,
         r=None,
         p_x=None,
         p_y=None):
    # plot any given set of x and y data
    fig = plt.figure('raw_data', figsize=(10, 10))
    gs = GridSpec(2, 2)
    ax = plt.subplot(gs[0:2, 0:2])  ## plot
    # plot axies
    ax.plot(x, y, 'bo', label='raw data', markersize=3)
    # plot guide circle if it applies
    if x_c is not None and y_c is not None and r is not None:
        circle = Circle((x_c, y_c), r, facecolor='none', \
                        edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
        ax.add_patch(circle)
    # plot a red point to represent something if it applies (resonance or off resonance for example)
    if p_x is not None and p_y is not None:
        ax.plot(p_x, p_y, '*', color='red', markersize=5)
    fig.savefig(output_path + name + '.pdf', format='pdf')


def plot2(x, y, x2, y2, name, output_path):
    # plot any given set of x and y data
    fig = plt.figure('raw_data', figsize=(10, 10))
    gs = GridSpec(2, 2)
    ax = plt.subplot(gs[0:2, 0:2])  ## plot
    ax.plot(x, y, 'bo', label='raw data', markersize=3)
    ax.plot(x2, y2, 'bo', label='raw data', markersize=3, color='red')
    fig.savefig(output_path + name + '.pdf', format='pdf')


def name_folder(dir, strmethod):
    result = time.localtime(time.time())
    output = strmethod + '_' + str(result.tm_year)
    if len(str(result.tm_mon)) < 2:
        output = output + '0' + str(result.tm_mon)
    else:
        output = output + str(result.tm_mon)
    if len(str(result.tm_mday)):
        output = output + '0' + str(result.tm_mday) + '_' + str(result.tm_hour) \
            + '_' + str(result.tm_min) + '_' + str(result.tm_sec)
    else:
        output = output + str(result.tm_mday) + '_' + str(result.tm_hour) + '_' \
            + str(result.tm_min) + '_' + str(result.tm_sec)
    if dir is not None:
        output_path = dir + '/' + output + '/'
    else:
        output_path = output + '/'
    count = 2
    path = output_path
    while os.path.isdir(output_path):
        output_path = path[0:-1] + '_' + str(count) + '/'
        count = count + 1
    return output_path


def name_plot(filename, strmethod, output_path, format='.pdf'):
    if filename.endswith('.csv'):
        filename = filename[:-4]
    filename = filename.replace('.', 'p')
    return output_path + strmethod + '_' + filename + format
