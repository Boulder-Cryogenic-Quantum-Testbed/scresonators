# -*- encoding: utf-8 -*-
"""
User code to fit S21 data for the Mines Al 6061 3D Cavity
Data collection information:
---------------------------
Collector: Nick Materise
VNA: Keysight PNA-X
Date collected: 210430
Purpose: Calibrate Qc with pin insertion depth
---------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #speadsheet commands
import sys #update paths
import os #import os in order to find relative path
import glob
from matplotlib.gridspec import GridSpec
import scipy.optimize
from scipy.interpolate import interp1d
import scipy.special
pathToParent = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 

# Set a variable that equals the relative path of parent directory
path_to_resfit = 'Z:/measurement/resfit'
sys.path.append(path_to_resfit)
import fit_resonator.resonator as res
import fit_resonator.fit_S_data as fsd
np.set_printoptions(precision=4,suppress=True)# display numbers with 4 sig.


def in2mm(x):
    """ inches to millimeters """
    return x * 25.4

def mm2in(x):
    """ millimeters to inches """
    return x / 25.4


def get_beta_tm_0m(f0, r0, m=1):
    """
    Compute the propagation constant for the TM_0m mode of the SMA port
    """
    # Compute and check the sign of the term inside the square root
    ## Hardcode the speed of light in SI units
    c0 = 2.99792458e8

    ## mth zero of the zeroth order Bessel function of the first kind
    p0m = scipy.special.jn_zeros(0, m)[m-1]
    beta2 = (2*np.pi*f0 / c0)**2 - (p0m / r0)**2
    assert (beta2 < 0), f'beta^2 ({beta2}) > 0, mode not evanescent.'
    beta = 1j*np.sqrt(-beta2)

    return beta


def trim_s21_wings(fname_in : str, Ntrim : list, minpts : int = 100,
                   use_asymm : bool = False):
    """
    Trim the front and back of the data from fname, write to the same location
    with the appended _trimmed path returned and the data
    """
    # Read data from file
    data_in = np.genfromtxt(fname_in, delimiter=',')

    # Trim ends by different amounts
    ## Check trim length is much less than the total length
    if sum(Ntrim) > data_in.shape[0] - minpts:
        raise ValueError(f'Ntrim ({Ntrim}) too long for min pts {minpts}.')
    
    # Remove from front and back of data
    print(f'data_in.shape: {data_in.shape}')
    print(f'Ntrim: {Ntrim}')
    data_out = data_in[Ntrim[0]:-(Ntrim[1]+1), :]
    print(f'data_out.shape: {data_out.shape}')

    # Write the results to file
    fext = fname_in.split('.')[-1]
    fname_out = fname_in.split('.')[0] + f'_trimmed.{fext}'
    with open(fname_out, 'w') as fid:
        fid.write('\n'.join(['%.8g, %.8g, %.8g' % (f, sdb, sph) 
        for f, sdb, sph in zip(data_out[:,0], data_out[:,1], data_out[:,2])]))

    return fname_out, data_out


def fit_single_res(filename, filter_points=[0,0]):
    """
    Fit a single resonator from file
    """
    # Current directory for the data under inspection
    my_dir = os.getcwd() # 'Z:/measurement/cryores/test_data/Mines'
    fname = my_dir + '/' + filename
    
    #########
    # Update the data by trimming the wings
    if sum(filter_points) > 0:
        fname, sdata = trim_s21_wings(fname, filter_points) 
    filename = fname.split('/')[-1]
    print(f'glob.glob(fname): {glob.glob(fname)}')
    
    #############################################
    ## create Method
    fit_type = 'DCM'
    MC_iteration = 50
    MC_rounds = 1e3
    MC_fix = ['w1']

    # make your own initial guess: [Qi, Qc, freq, phi] 
    # (instead of phi used Qa for CPZM)
    # manual_init = [Qi,Qc,freq,phi]
    manual_init = None # find initial guess by itself
    # fmin = sdata[:,0][np.argmin(sdata[:,1])]
    # manual_init = [2000.0, 150000.0, fmin / 1e9, 1.5]
    
    # Setup the method for fitting
    try: 
        Method = res.FitMethod(fit_type, MC_iteration, MC_rounds=MC_rounds,
                    MC_fix=MC_fix, manual_init=manual_init, MC_step_const=0.3)
    except:
        print("Failed to initialize method, please change parameters")
        quit()
    
    ##############################################################
    normalize = 10
    
    ### Fit Resonator function without background removal ###
    params, conf_array, fig1, chi1, init1 = fsd.fit_resonator(
                                filename=filename,
                                Method=Method, normalize=normalize,
                                dir=my_dir, plot_extra=False)
    
    print(f'params:\n{params}')
    print(f'conf_array:\n{conf_array}')
    print(f'chi1:\n{chi1}')
    print(f'init1:\n{init1}')

    return params, chi1


def fit_qc_vs_z(filenames, zvals, r0=1e-3, filter_points=None):
    """
    Fits Qc vs. z from a list of files
    Note: r0 is the radius of the circular waveguide feeding the cavity
    """
    # Iterate over the filenames, save the values of f0, Qc, fit error
    Npts = len(filenames)
    Qc = np.zeros(Npts); f0 = np.zeros(Npts)
    errs = np.zeros(Npts)
    betas = np.zeros(Npts, dtype=np.complex128)
    for idx, filename in enumerate(filenames):
        ## Compute the resonator fits and their errors
        filter_pts = filter_points[idx] if filter_points is not None else [0]*2
        params, err = fit_single_res(filename, filter_points=filter_pts)
        Qc[idx] = params[1]
        f0[idx] = params[2] * 1e9
        errs[idx] = err

        ## Compute beta and check with 
        betas[idx] = get_beta_tm_0m(f0[idx], r0, m=1)
        print(f'beta: {betas[idx]}')

    # Save the data to file
    print(f'Saving data to file ...')
    df = pd.DataFrame(np.vstack((betas, Qc)))
    df.to_csv('cavity_rt_qc_vs_z.csv')

    # Check that we have enough data to perform a fit of ln Qc vs. z
    if len(filenames) > 2:
        ## Fit the slope and plot the best fit
        def fitfun(x, a, b):
            return a * x + b

        ## Curve-fit results
        popt, pcov = scipy.optimize.curve_fit(fitfun, zvals, np.log(Qc))
        logQc_fit = fitfun(zvals, *popt)

        ## Plot and save the results to file
        print(f'Generating and saving plot ...')
        fsize = 20
        print(f'betas:{betas}')
        print(f'-j z betas + beta0: {-1j*zvals*betas + popt[0]}')
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        ax.scatter(zvals * 1e3, np.log(Qc), marker='s', s=50.0)
        ax.plot(zvals * 1e3, -1j * zvals * betas+popt[1], '--', color='orange',
                label=r'$-j\beta z+\beta_0$')
        ax.plot(zvals * 1e3, logQc_fit, '-.', color='green',
                label='Linear Fit')
        ax.set_xlabel(r'$z$ [mm] offset from cavity wall', fontsize=fsize)
        ax.set_ylabel(r'$\ln\ Q_c$', fontsize=fsize)
        plt.legend(loc='best')
        fig.savefig('cavity_rt_qc_vs_z.pdf', format='pdf')


if __name__ == '__main__':
    
    # Send a list of filenames to iterate over and fit the Qc vs. z data
    ## Pin values in inches
    zvals_in = np.array([-0.125, -0.048, -0.0235, 0, 0.026, 0.0485, 0.0606])
    zvals_in = np.array([0, 0.026, 0.0485, 0.0606])
    # zvals_in = np.array([-0.0235, -0.048, -0.125])
    zvals_in = np.sort(zvals_in)

    ## Convert to mm
    zvals_mm = -1e-3 * in2mm(zvals_in)

    ## Use the inches data to set the filenames
    zvals_strs = ['%.4f' % z if z > 0  else '0' if z == 0. else 'n%.4f' % abs(z)
                    for z in zvals_in]
    zvals_strs = ['_'.join(z.split('.')) for z in zvals_strs]
    print(f'zvals_strs:\n{zvals_strs}')
    filenames = ['mines_cavity_bare_no_in_seal_rt_zpin_%sin_210430.csv' % z
                 for z in zvals_strs]

    # Si/SiOx loaded sample
    filenames = ['mines_cavity_bare_in_seal_rt_new_sma_etch_with_ptfe_si_siox_zpi.csv']

    # Latest pin + dielectric measurment before Si/SiOx loaded
    filenames = ['mines_cavity_bare_no_in_seal_rt_new_sma_etch_with_ptfe_zpin_n0X.csv']

    # filenames = ['mines_cavity_bare_no_in_seal_rt_zpin_%sin_210430.csv' %
    #              zvals_strs[2]]

    # Set the number of points to trim from each data set
    # Ordered as [-0.125, -0.048, -0.0235, 0, 0.026, 0.0485, 0.0606]
    filter_points=[[0]*2, [0]*2, [0]*2, [650]*2, [0]*2, [0]*2, [0]*2]
    filter_points=[[650]*2, [0]*2, [0]*2, [0]*2]
    filter_points=[[200, 200]]

    fit_qc_vs_z(filenames, zvals_mm, filter_points=filter_points)
