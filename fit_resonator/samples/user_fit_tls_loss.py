# -*- encoding: utf-8 -*-
"""
User code to fit S21 data for the Mines Al 6061 3D Cavity
Data collection information:
---------------------------
Collector: Dave Pappas
VNA: Keysight PNA
Date collected: 210430
Purpose: Collect power and power dependence of cavity resonance
         with a SiOx sample loaded
---------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #speadsheet commands
import datetime
import sys #update paths
import os #import os in order to find relative path
import glob
from matplotlib.gridspec import GridSpec
import scipy.optimize
from scipy.interpolate import interp1d
import scipy.special
import re as regex
import uncertainties
pathToParent = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 

# XXX: You may need to add measurement/resfit to your path if it is not
#      installed in a standard location.
# Set a variable that equals the relative path of parent directory
path_to_resfit = 'C:\\Users\\Lehnert Lab\\Github\\measurement'
pglob = glob.glob(path_to_resfit)
assert len(pglob), f'Path: {path_to_resfit} does not exist'
sys.path.append(path_to_resfit)

import fit_resonator.resonator as res
import fit_resonator.Sdata as fsd
np.set_printoptions(precision=4,suppress=True)# display numbers with 4 sig.


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
    dot_split = fname_in.split('.')
    fext      = dot_split[-1]
    if len(dot_split) > 1:
        fname_out = '.'.join(dot_split[0:-1]) + f'_trimmed.{fext}'
    else:
        fname_out = dot_split[0] + f'_trimmed.{fext}'
    with open(fname_out, 'w') as fid:
        fid.write('\n'.join(['%.8g, %.8g, %.8g' % (f, sdb, sph) 
        for f, sdb, sph in zip(data_out[:,0], data_out[:,1], data_out[:,2])]))

    return fname_out, data_out


def fit_single_res(filename, filter_points=[0,0], preprocess_method='linear',
                   use_gauss_filt=False, use_matched_filt=False,
                   use_elliptic_filt=False, use_mov_avg_filt=False,
                   fname_ref=None):
    """
    Fit a single resonator from file
    """
    # Current directory for the data under inspection
    my_dir = os.getcwd() # 'Z:/measurement/cryores/test_data/Mines'
    fname = my_dir + '/' + filename
    
    #########
    # Update the data by trimming the wings
    if sum(filter_points) > 0:
        print(f'Trimming data to {filter_points} ...')
        fname, sdata = trim_s21_wings(fname, filter_points) 

    filename = fname # fname.split('/')[-1]

    print('-------------')
    print(filename)
    
    #############################################
    ## create Method
    fit_type = 'DCM'
    MC_iteration = 10
    MC_rounds = 1e3
    MC_fix = []

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
    params, conf_intervals, fig1, chi1, init1 = fsd.fit(
                                filename,
                                Method,
                                normalize,
                                plot_extra=False,
                                preprocess_method=preprocess_method)
    
    return params, chi1, conf_intervals


def fit_qiqcfc_vs_power(filenames, powers, filter_points=None,
                        preprocess_method='linear', phi0=0.,
                        use_gauss_filt=False, use_matched_filt=False,
                        use_elliptic_filt=False, filt_idxs=None,
                        use_mov_avg_filt=False, fname_ref=None):
    """
    Fits multiple resonances at different powers for a given power
    """
    # Iterate over the filenames, save the values of fc, Qc, fit error
    Npts = len(filenames)
    Qc = np.zeros(Npts); fc = np.zeros(Npts);
    Qi = np.zeros(Npts); Q = np.zeros(Npts);
    Qc_err = np.zeros(Npts); fc_err = np.zeros(Npts); Qi_err = np.zeros(Npts)
    navg = np.zeros(Npts); errs = np.zeros(Npts)
    for idx, filename in enumerate(filenames):
        ## Compute the resonator fits and their errors
        filter_pts = filter_points[idx] if filter_points is not None else [0]*2
        use_matched_filt_chk = use_matched_filt and (idx in filt_idxs)
        use_elliptic_filt_chk = use_elliptic_filt and (idx in filt_idxs)
        use_mov_avg_filt_chk = use_mov_avg_filt and (idx in filt_idxs)
        params, err, conf_int = fit_single_res(filename,
                                filter_points=filter_pts,
                                preprocess_method=preprocess_method,
                                use_gauss_filt=use_gauss_filt,
                                use_matched_filt=use_matched_filt_chk,
                                use_elliptic_filt=use_elliptic_filt_chk,
                                use_mov_avg_filt=use_mov_avg_filt_chk,
                                fname_ref=fname_ref)

        # Qcj = params[1] / np.exp(1j*params[3])
        Qcj = params[1] * np.exp(1j*(params[3] + phi0))
        Qij = 1. / (1. / params[0] - np.real(1. / Qcj))

        # Total quality factor
        Q[idx] = params[0]

        # Store the quality factors, resonance frequencies
        Qc[idx] = np.real(Qcj)
        Qi[idx] = Qij
        fc[idx] = params[2]
        errs[idx] = err
        navg[idx] = power_to_navg(powers[idx], Qi[idx], Qc[0], fc[0])

        # Store each quantity's 95 % confidence intervals
        Qi_err[idx] = conf_int[1]
        Qc_err[idx] = conf_int[2]
        fc_err[idx] = conf_int[5]

        print(f'navg: {navg[idx]} photons')
        print(f'Q: {Q[idx]} +/- {conf_int[0]}')
        print(f'Qi: {Qi[idx]} +/- {Qi_err[idx]}')
        print(f'Qc: {Qc[idx]} +/- {Qc_err[idx]}')
        print(f'fc: {fc[idx]} +/- {fc_err[idx]} GHz')
        print('-------------\n')
        plt.close('all')

    # Save the data to file
    df = pd.DataFrame(np.vstack((powers, navg, fc, Qi, Qc, Q,
                    errs, Qi_err, Qc_err, fc_err)).T,
            columns=['Power [dBm]', 'navg', 'fc [GHz]', 'Qi', 'Qc', 'Q',
                'error', 'Qi error', 'Qc error', 'fc error'])
                                                                               
    dstr = datetime.datetime.today().strftime('%y%m%d_%H_%M_%S')
    df.to_csv(f'qiqcfc_vs_power_{dstr}.csv')

    return df


def fit_delta_tls(Qi, T, fc, Qc, p, display_scales={'QHP' : 1e5,
                'nc' : 1e7, 'Fdtls' : 1e-6}):
    """
    Fit the loss using the expression

    delta_tls = F * delta0_tls * tanh(hbar w_c / 2 kB T) (1 + <n> / nc)^-1/2

    """
    # Convert the inputs to the correct format
    h      = 6.626069934e-34
    hbar   = 1.0545718e-34
    kB     = 1.3806485e-23
    fc_GHz = fc if np.any(fc >= 1e9) else fc * 1e9
    TK     = T if T <= 400e-3 else T * 1e-3
    delta  = 1. / Qi
    hw0    = hbar * 2 * np.pi * fc_GHz
    hf0    = h * fc_GHz
    kT     = kB * TK

    # Extract the scale factors for the fit parameters
    QHP_max   = display_scales['QHP']
    nc_max    = display_scales['nc']
    Fdtls_max = display_scales['Fdtls']

    navg = power_to_navg(p, Qi, Qc, fc)
    labels = [r'$10^{%.2g}$' % x for x in np.log10(navg)]
    print(f'<n>: {labels}')
    print(f'T: {TK} K')
    print(f'fc_GHz: {fc_GHz} Hz')

    def fitfun5(n, Fdtls, nc, dHP, beta, dqp0):
    # def fitfun(n, Fdtls, nc, dHP):
       # num = Fdtls * np.tanh(hw0 / (2 * kT))
       num = Fdtls * np.tanh(hf0 / (2 * kT))
       den = np.sqrt(1. + n / nc)
       dqp = dqp0 * (n / nc)**beta
       return num / den + dHP + dqp

    def fitfun4(n, Fdtls, nc, QHP, beta):
    # def fitfun(n, Fdtls, nc, dHP):
       num = Fdtls * np.tanh(hw0 / (2 * kT))
       den = (1. + n / nc)**beta
       return num / den + 1./QHP

    def fitfun(n, Fdtls, nc, dHP):
       num = Fdtls * np.tanh(hw0 / (2 * kT))
       den = np.sqrt(1. + n / nc)
       return num / den + dHP

    # Fit with Levenberg-Marquardt
    # x0 = [1e-6, 1e6, 1e4]
    # popt, pcov = scipy.optimize.curve_fit(fitfun, navg, delta, p0=x0)
    #     F*d^0TLS,  nc,    dHP,  beta
    x0 = [     1e-6,  1e2,  np.max(Qi), 0.1]
    # x0 = [     3e-6,  1.2e6,  19600, 0.17]
    bounds = ((1e-10, 1e1,   1e3, 1e-3),\
              (1e-3,  1e10,  1e8, 1.))
    popt, pcov = scipy.optimize.curve_fit(fitfun4, navg, delta, p0=x0)
                                          # bounds=bounds)

    # Read off the fit values and covariances
    # Fdtls, nc, dHP = popt
    # errs = np.sqrt(np.diag(pcov))
    # Fdtls_err, nc_err, dHP_err = errs
    Fdtls, nc, QHP, beta = popt
    errs = np.sqrt(np.diag(pcov))
    Fdtls_err, nc_err, QHP_err, beta_err = errs

    # Uncertainty formatting
    ## Rounding to n significant figures
    round_sigfig = lambda x, n \
            : round(x, n - int(np.floor(np.log10(abs(x)))) - 1)

    Fdtls_err = round_sigfig(Fdtls_err, 1)
    nc_err    = round_sigfig(nc_err, 1)
    QHP_err   = round_sigfig(QHP_err, 1)
    beta_err  = round_sigfig(beta_err, 1)

    ## Uncertainty objects
    Fdtls_un = uncertainties.ufloat(Fdtls, Fdtls_err)
    nc_un    = uncertainties.ufloat(nc, nc_err)
    QHP_un   = uncertainties.ufloat(QHP, QHP_err)
    beta_un  = uncertainties.ufloat(beta, beta_err)

    # Build a string with the results
    ## Formatted uncertainties
    Fdtls_latex = f'{Fdtls_un:L}'
    nc_latex = f'{nc_un:L}'
    QHP_latex = f'{QHP_un:L}'
    beta_latex = f'{beta_un:L}'

    ## Latex strings
    Fdtls_str = r'$F\delta^{0}_{TLS}: %s$'  % Fdtls_latex
    nc_str    = r'$n_c: %s$' % nc_latex
    QHP_str   = r'$Q_{HP}: %s$' % QHP_latex
    beta_str   = r'$\beta: %s$' % beta_latex
    delta_fit_str = Fdtls_str + '\n' + nc_str \
            + '\n' + QHP_str + '\n' + beta_str
    # delta_fit_str = Fdtls_str + '\n' + nc_str
    print(delta_fit_str)

    return Fdtls, nc, QHP, Fdtls_err, nc_err, QHP_err, \
            fitfun4(navg, *popt), delta_fit_str
            # fitfun4(nout, Fdtls, nc, QHP, beta), delta_fit_str, pout
            # fitfun(navg, Fdtls, nc, dHP), delta_fit_str


def get_powers_from_file(time_fname, temp_fname):
    """
    Returns the powers after reading a file
    """
    powers = np.genfromtxt(time_fname, delimiter=',').T
    times, temperatures = np.genfromtxt(temp_fname, delimiter=',', dtype=str).T
    temperatures = np.asarray(temperatures)
    times = np.asarray([str(t) for t in times])
    return times, temperatures, powers


def power_to_navg(power_dBm, Qi, Qc, fc, Z0_o_Zr=1.):
    """
    Converts power to photon number following Eq. (1) of arXiv:1801.10204
    and Eq. (3) of arXiv:1912.09119
    """
    # Physical constants, Planck's constant J s
    h = 6.62607015e-34
    hbar = 1.0545718e-34

    # Convert dBm to W
    Papp = 10**(power_dBm / 10) # * 1e-3
    # hb_wc2 = np.pi * h * (fc_GHz * 1e9)**2
    fc_GHz = fc * 1e9
    hb_wc2 = hbar * (2 * np.pi * fc_GHz)**2

    # Return the power as average number of photons
    Q = 1. / ((1. / Qi) + (1. / Qc))
    navg = (2. / hb_wc2) * (Q**2 / Qc) * Papp

    return navg


def power_sweep_fit_drv(atten=[0, -60], powers_in=None,
                        filenames_in=None,
                        temperature='100mK',
                        plot_from_file=False, use_error_bars=True,
                        temp_correction='', phi0=0., use_gauss_filt=True,
                        use_matched_filt=False, use_elliptic_filt=False,
                        use_mov_avg_filt=False, loss_scale=None,
                        preprocess_method='linear',
                        ds = {'QHP' : 1e4, 'nc' : 1e6, 'Fdtls' : 1e-6}):
    """
    Driver for fitting the power sweep data for a given set of data
    """
    if np.any(powers_in):
        powers = np.copy(powers_in)
    else:
        powers = np.linspace(15, -105, 25)

    print(f'powers: {powers}')

    tstr = '14_mKpcsv'
    if filenames_in:
        filenames = filenames_in
    else:
        filenames = [f'M3D6_02_WITH_2SP_INP_220630_8_{int(p)}dB_{tstr}.csv' 
                for p in powers]
    filt_idxs = [] # list(range(10, powers.size))
    fname_ref = filenames[0]
    filter_points = [[0, 0] for _ in filenames]
    fpts = [[400, 400]]
    # filter_points[-1:] = fpts
    print(f'filter_points:\n{filter_points}')
    dstr = datetime.datetime.today().strftime('%y%m%d')
    err_str = '_error_bars'  if use_error_bars else ''
    cal_str = temp_correction + '_'
    fsize = 20
    csize = 5

    # Plot the results after gathering all of the fits
    if plot_from_file:
        df = pd.read_csv('qiqcfc_vs_power_210811_17_23_44.csv')
    else:
        df = fit_qiqcfc_vs_power(filenames, powers,
                filter_points=filter_points,
                preprocess_method=preprocess_method,
                phi0=phi0, use_gauss_filt=use_gauss_filt,
                use_matched_filt=use_matched_filt,
                use_elliptic_filt=use_elliptic_filt,
                use_mov_avg_filt=use_mov_avg_filt,
                filt_idxs=filt_idxs,
                fname_ref=fname_ref)

    # Extract the powers, quality factors, resonance frequencies, and 95 %
    # confidence intervals
    Qi = df['Qi']
    Qc = df['Qc']
    Q  = df['Q']
    navg = df['navg']
    delta = 1. / Qi
    fc = df['fc [GHz]']
    Qi_err = df['Qi error']
    Qc_err = df['Qc error']
    delta_err = Qi_err / Qi**2
    fc_err = df['fc error']

    # Add attenuation to powers
    powers += sum(atten)

    def pdBm_to_navg_ticks(p):
    	n = power_to_navg(powers[0::2], Qi[0::2], Qc[0], fc[0])
    	labels = [r'$10^{%.2g}$' % x for x in np.log10(n)]
    	print(f'labels:\n{labels}')
    	return labels

    # Fit the TLS loss
    # tcmp = regex.compile('[0-9]+.[0-9]+')
    
    tcmp = regex.compile('[0-9]+')
    T = float(tcmp.match(tstr).group())
    doff = 0
    if doff > 0:
        Fdtls, nc, QHP, Fdtls_err, nc_err, QHP_err, delta_fit, delta_fit_str \
                = fit_delta_tls(Qi[0:-doff], T, fc[0], Qc[0], powers[0:-doff],\
                display_scales=ds)
    else:
        Fdtls, nc, QHP, Fdtls_err, nc_err, QHP_err, delta_fit, delta_fit_str \
                = fit_delta_tls(Qi, T, fc[0], Qc[0], powers,\
                display_scales=ds)

    if loss_scale:
        delta /= loss_scale
        delta_err /= loss_scale
        delta_fit /= loss_scale

    print('\n')
    print(f'F * d0_tls: {Fdtls:.2g} +/- {Fdtls_err:.2g}')
    print(f'nc: {nc:.2g} +/- {nc_err:.2g}')
    print('\n')

    #the results
    ## Plot the resonance frequenices
    fig_fc, ax_fc = plt.subplots(1, 1, tight_layout=True)
    ax_fc.set_xlabel('Power [dBm]', fontsize=fsize)
    ax_fc.set_ylabel('Resonance Frequency [GHz]', fontsize=fsize)
    ax_fc_top = ax_fc.twiny()

    ## Plot the internal and external quality factors separately
    fig_qc, ax_qc = plt.subplots(1, 1, tight_layout=True)
    fig_qi, ax_qi = plt.subplots(1, 1, tight_layout=True)
    fig_qiqc, ax_qiqc = plt.subplots(1, 1, tight_layout=True)
    fig_d, ax_d = plt.subplots(1, 1, tight_layout=True)
    ax_qc_top = ax_qc.twiny()
    ax_qi_top = ax_qi.twiny()
    ax_qiqc_top = ax_qiqc.twiny()
    ax_d_top  = ax_d.twiny()

    # Plot with / without error bars
    if use_error_bars:
        markers = ['o', 'd', '>', 's', '<', 'h', '^', 'p', 'v']
        colors  = plt.rcParams['axes.prop_cycle'].by_key()['color']
        ax_fc.errorbar(powers, fc, yerr=fc_err, marker='o', ls='', ms=10,
                capsize=csize)
        ax_qc.errorbar(powers, Qc, yerr=Qc_err, marker='o', ls='', ms=10,
                capsize=csize)
        ax_qiqc.errorbar(powers, Qi, yerr=Qi_err, marker='h', ls='', ms=10,
                capsize=csize, color=colors[5],
                label=r'$Q_i$ Cooldown 26 (01/29/22)')
        ax_qiqc.errorbar(powers, Qc, yerr=Qc_err, marker='^', ls='', ms=10,
                capsize=csize, color=colors[6],
                label=r'$Q_c$ Cooldown 26 (01/29/22)')
        ax_qi.errorbar(powers, Qi, yerr=Qi_err, marker='o', ls='', ms=10,
                capsize=csize)
        if doff > 0:
            ax_d.errorbar(powers[0:-doff], delta[0:-doff],
                    yerr=delta_err[0:-doff], marker='d', ls='', color=colors[1],
                    ms=10, capsize=csize)
            ax_d.plot(powers[0:-doff], delta_fit, ls='-', label=delta_fit_str,
                    color=colors[1])
        else:
            ax_d.errorbar(powers, delta,
                    yerr=delta_err, marker='d', ls='', color=colors[1],
                    ms=10, capsize=csize)
            ax_d.plot(powers, delta_fit, ls='-', label=delta_fit_str,
                    color=colors[1])


    else:
        ax_fc.plot(powers, fc, marker='o', ms=10, ls='')
        ax_qc.plot(powers, Qc, marker='o', ms=10, ls='')
        ax_qi.plot(powers, Qi, marker='o', ms=10, ls='')
        ax_d.plot(powers, delta, marker='o', ms=10, ls='')

    ax_qc.set_xlabel('Power [dBm]', fontsize=fsize)
    ax_qi.set_xlabel('Power [dBm]', fontsize=fsize)
    ax_qiqc.set_xlabel('Power [dBm]', fontsize=fsize)
    ax_d.set_xlabel('Power [dBm]', fontsize=fsize)
    ax_qc.set_ylabel(r'$Q_c$', fontsize=fsize)
    ax_qi.set_ylabel(r'$Q_i$', fontsize=fsize)
    ax_qiqc.set_ylabel(r'$Q_i, Q_c$', fontsize=fsize)

    if loss_scale:
        ax_d.set_ylabel(r'$Q_i^{-1}\times 10^{%d}$' \
                         % int(np.log10(loss_scale)), fontsize=fsize)
    else:
        ax_d.set_ylabel(r'$Q_i^{-1}$', fontsize=fsize)

    power_str = f'{atten[0]} dB ext, {atten[1]} dB int attenuation'
    ax_qc.set_title(power_str, fontsize=fsize)
    ax_fc.set_title(power_str, fontsize=fsize)
    ax_qi.set_title(power_str, fontsize=fsize)
    ax_qiqc.set_title('Measured at Base Temperature ~17 mK', fontsize=fsize)

    # Set the second (top) axis labels
    ax_qc_top.set_xlabel(r'Power [$\left<{n}\right>$]', fontsize=fsize)
    ax_qi_top.set_xlabel(r'Power [$\left<{n}\right>$]', fontsize=fsize)
    ax_qiqc_top.set_xlabel(r'[$\left<{n}\right>$]', fontsize=fsize)
    ax_d_top.set_xlabel(r'$\left<{n}\right>$', fontsize=fsize)
    ax_fc_top.set_xlabel(r'Power [$\left<{n}\right>$]', fontsize=fsize)
    
    # Update the tick labels to display the photon numbers
    ax_qc.set_xticks(powers[0::2])
    ax_qi.set_xticks(powers[0::2])
    ax_qiqc.set_xticks(powers[0::2])
    ax_d.set_xticks(powers[0::2])
    ax_fc.set_xticks(powers[0::2])
    
    ax_qc_top.set_xticks(ax_qc.get_xticks())
    ax_qi_top.set_xticks(ax_qi.get_xticks())
    ax_qiqc_top.set_xticks(ax_qi.get_xticks())
    ax_d_top.set_xticks(ax_d.get_xticks())
    ax_fc_top.set_xticks(ax_fc.get_xticks())
    
    ax_qc_top.set_xbound(ax_qc.get_xbound())
    ax_qi_top.set_xbound(ax_qi.get_xbound())
    ax_qiqc_top.set_xbound(ax_qi.get_xbound())
    ax_d_top.set_xbound(ax_d.get_xbound())
    ax_fc_top.set_xbound(ax_fc.get_xbound())
    
    ax_qc_top.set_xticklabels(pdBm_to_navg_ticks(ax_qc.get_xticks()))
    ax_qi_top.set_xticklabels(pdBm_to_navg_ticks(ax_qi.get_xticks()))
    ax_qiqc_top.set_xticklabels(pdBm_to_navg_ticks(ax_qi.get_xticks()))
    ax_d_top.set_xticklabels(pdBm_to_navg_ticks(ax_d.get_xticks()), rotation=45)
    ax_fc_top.set_xticklabels(pdBm_to_navg_ticks(ax_fc.get_xticks()))

    ax_d.set_xticklabels(ax_d.get_xticks(), rotation=45)

    

    # Set the legends
    qiqc_lbls, qiqc_hdls = ax_qiqc.get_legend_handles_labels()
    ax_qiqc.legend(qiqc_lbls, qiqc_hdls, loc=(0.4, 0.4), fontsize=fsize)
    d_lbls, d_hdls = ax_d.get_legend_handles_labels()
    ax_d.legend(d_lbls, d_hdls, loc='upper right', fontsize=fsize)

    ## Save all figures to file
    fig_fc.savefig(f'fc_vs_power{cal_str}{dstr}_{temperature}{err_str}.pdf',
            format='pdf')
    fig_qc.savefig(f'qc_vs_power{cal_str}{dstr}_{temperature}{err_str}.pdf',
            format='pdf')
    fig_qiqc.savefig(f'qiqc_vs_power{cal_str}{dstr}_{temperature}{err_str}.pdf',
            format='pdf')
    fig_qi.savefig(f'qi_vs_power{cal_str}{dstr}_{temperature}{err_str}.pdf',
            format='pdf')
    fig_d.savefig(f'tand_vs_power{cal_str}{dstr}_{temperature}{err_str}.pdf',
            format='pdf')

    plt.close('all')

if __name__ == '__main__':
    # Set the temperature
    temperature = '13mK'

    # Set the input powers
    powers_in = np.linspace(15, -105, 25)
    powers_in = np.hstack(([-82.5, -87.5, -92.5, -97.5, -98.0, -99.0], powers_in))
    powers_in = np.sort(powers_in)[::-1]
    tstr = '14_mK'

    # Set the filename strings
    filenames_in = [f'M3D6_02_WITH_2SP_INP_220630_8_{p:.1f}dB_{tstr}pcsv.csv' 
                for p in powers_in]

    # Run the power sweep
    power_sweep_fit_drv(atten=[0, -70], temperature=temperature,
                        powers_in=powers_in,
                        filenames_in=filenames_in, plot_from_file=False,
                        use_error_bars=True, temp_correction='', phi0=0.,
                        use_gauss_filt=False, use_matched_filt=False,
                        use_elliptic_filt=False, use_mov_avg_filt=False,
                        loss_scale=1e-5, preprocess_method='circle',
                        ds = {'QHP' : 1e5, 'nc' : 1e0, 'Fdtls' : 1e-6})
