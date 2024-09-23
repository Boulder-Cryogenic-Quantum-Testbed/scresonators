# %%

'''
    helper_fit.py
'''

# print("    loading helper_fit.py")

# %% 

import glob, os, sys, time
import datetime, regex
import uncertainties

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd

import fit_resonator.resonator as res
import fit_resonator.fit as fsd
import helper_misc as hm

# %%

def fit_single_res(filename, filter_points=[0,0], preprocess_method='linear',
                   use_gauss_filt=False, use_matched_filt=False,
                   use_elliptic_filt=False, use_mov_avg_filt=False,
                   fname_ref=None, data_dir=None, save_dcm_plot=False,
                   manual_init=None):
    """
    Fit a single resonator from file
    """
    # Current directory for the data under inspection
    if data_dir is None:
        my_dir = os.getcwd() 
        fname = my_dir + '\\' + filename
    else:
        fname = filename
        
    #########
    # Update the data by trimming the wings
    if sum(filter_points) > 0:
        print(f'Trimming data to {filter_points} ...')
        fname, sdata = trim_s21_wings(fname, filter_points) 

    filename = fname #.split('/')[-1]

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
    # manual_init = None # find initial guess by itself
    # fmin = sdata[:,0][np.argmin(sdata[:,1])]
    # manual_init = [2000.0, 150000.0, fmin / 1e9, 1.5]
    normalize = 10

    myres = res.Resonator()
    myres.from_file(filename)
    myres.preprocess_method = preprocess_method
    myres.normalize = normalize
    myres.save_dcm_plot = save_dcm_plot
    myres.plot = 'png'
    myres.fit_dir = data_dir + '\\scres_fits\\'
    
    # Setup the method for fitting
    try: 
        myres.fit_method(fit_type, MC_iteration, MC_rounds=MC_rounds,
                    MC_fix=MC_fix, manual_init=manual_init, MC_step_const=0.3)
    except Exception as ex:
        print(f'Exception:\n{ex}')
        quit()
    
    ##############################################################
    
    ### Fit Resonator function without background removal ###
    params, conf_intervals, err, init1, fig = fsd.fit(myres) 
    return params, err, conf_intervals, fig



def fit_qiqcfc_vs_power(filenames, powers, filter_points=None,
                        preprocess_method='linear', phi0=0.,
                        use_gauss_filt=False, use_matched_filt=False,
                        use_elliptic_filt=False, filt_idxs=None,
                        use_mov_avg_filt=False, fname_ref=None,
                        data_dir='', show_plots=False, save_dcm_plot=False,
                        manual_init_list=None):
    """
    Fits multiple resonances at different powers for a given power
    """
    # Iterate over the filenames, save the values of fc, Qc, fit error
    Npts = len(filenames)
    Q, Qc, Qi, fc, navg = np.zeros(Npts), np.zeros(Npts), np.zeros(Npts), np.zeros(Npts), np.zeros(Npts)
    Qc_err, Qi_err, fc_err, errs = np.zeros(Npts), np.zeros(Npts), np.zeros(Npts), np.zeros(Npts)
    
    for idx, filename in enumerate(filenames):
        ## Compute the resonator fits and their errors
        manual_init = manual_init_list[idx]
        filter_pts = filter_points[idx] if filter_points is not None else [0]*2
        use_matched_filt_chk = use_matched_filt and (idx in filt_idxs)
        use_elliptic_filt_chk = use_elliptic_filt and (idx in filt_idxs)
        use_mov_avg_filt_chk = use_mov_avg_filt and (idx in filt_idxs)
        params, err, conf_int, fig = fit_single_res(filename,
                                               filter_points=filter_pts,
                                               preprocess_method=preprocess_method,
                                               use_gauss_filt=use_gauss_filt,
                                               use_matched_filt=use_matched_filt_chk,
                                               use_elliptic_filt=use_elliptic_filt_chk,
                                               use_mov_avg_filt=use_mov_avg_filt_chk,
                                               fname_ref=fname_ref, 
                                               save_dcm_plot=save_dcm_plot,
                                               manual_init=manual_init,
                                               data_dir=data_dir)
        # Qcj = params[1] / np.exp(1j*params[3])
        Qcj = params[1] * np.exp(1j*(params[3] + phi0))
        Qij = 1. / (1. / params[0] - np.real(1. / Qcj))

        # Total quality factor
        Q[idx] = params[0]

        fscale = 1e9 if params[2] > 1e9 else 1

        # Store the quality factors, resonance frequencies
        Qc[idx] = np.real(Qcj)
        Qi[idx] = Qij
        fc[idx] = params[2] / fscale
        errs[idx] = err
        navg[idx] = power_to_navg(powers[idx], Qi[idx], Qc[0], fc[0])

        # Store each quantity's 95 % confidence intervals
        Qi_err[idx] = conf_int[1]
        Qc_err[idx] = conf_int[2]
        fc_err[idx] = conf_int[5] / fscale

        print(f'navg: {navg[idx]} photons')
        print(f'Q: {Q[idx]} +/- {conf_int[0]}')
        print(f'Qi: {Qi[idx]} +/- {Qi_err[idx]}')
        print(f'Qc: {Qc[idx]} +/- {Qc_err[idx]}')
        print(f'fc: {fc[idx]} +/- {fc_err[idx]} GHz')
        print('-------------\n')
        if show_plots is False:
            plt.close('all')
        else:
            fig.show()

    # Save the data to file
    df = pd.DataFrame(np.vstack((powers, navg, fc, Qi, Qc, Q,
                    errs, Qi_err, Qc_err, fc_err)).T,
            columns=['Power [dBm]', 'navg', 'fc [GHz]', 'Qi', 'Qc', 'Q',
                'error', 'Qi error', 'Qc error', 'fc error'])
                                     
    dstr = datetime.datetime.today().strftime('%y%m%d_%H_%M_%S')
    
    filename = f'qiqcfc_vs_power_{dstr}.csv'
    df.to_csv(data_dir + filename)

    return df



def fit_delta_tls(Qi, T, fc, Qc, p, display_scales={'QHP' : 1e5,
                'nc' : 1e7, 'Fdtls' : 1e-6}, QHP_fix=False, Qierr=None):
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

    navg = np.abs(power_to_navg(p, Qi, Qc, fc))
    labels = [r'$10^{%.2g}$' % x for x in np.log10(navg)]
    print(f'<n>: {labels}')
    print(f'T: {TK} K')
    print(f'fc_GHz: {fc_GHz} Hz')

    def fitfun4(n, Fdtls, nc, QHP, beta):
       num = Fdtls * np.tanh(hw0 / (2 * kT))
       den = (1. + n / nc)**beta
       return num / den + 1. / QHP

    if QHP_fix:
        QHPidx = np.argmax(Qi)
        QHP = Qi[QHPidx]
        QHP_err = Qierr[QHPidx]

    def fitfun3(n, Fdtls, nc, beta):
       num = Fdtls * np.tanh(hw0 / (2 * kT))
       den = (1. + n / nc)**beta
       return num / den + 1. / QHP

    def fitfun(n, Fdtls, nc, dHP):
       num = Fdtls * np.tanh(hw0 / (2 * kT))
       den = np.sqrt(1. + n / nc)
       return num / den + dHP

    # Fit with Levenberg-Marquardt
    # x0 = [1e-6, 1e6, 1e4]
    # popt, pcov = sp.optimize.curve_fit(fitfun, navg, delta, p0=x0)
    #     F*d^0TLS,  nc,    dHP,  beta
    x0 = [     1e-6,  1e2,  np.max(Qi), 0.1]
    # x0 = [     3e-6,  1.2e6,  19600, 0.17]
    bounds = ((1e-10, 1e1,   1e3, 1e-3),\
              (1e-3,  1e10,  1e8, 1.))
    if QHP_fix:
        x0 = [1e-6,  1e2, 0.1]
        # bnds = ((1e-10, 1e1, 1e-3), (1e-3, 1e7, 1.))
        popt, pcov = sp.optimize.curve_fit(fitfun3, navg, 
                delta, p0=x0) # , bounds=bnds)
        Fdtls, nc, beta = popt
        errs = np.sqrt(np.diag(pcov))
        Fdtls_err, nc_err, beta_err = errs
    else:
        x0 = [     1e-6,  1e2,  np.max(Qi), 0.1]
        popt, pcov = sp.optimize.curve_fit(fitfun4, navg, delta, p0=x0)
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

    print(f'QHP: {QHP:.2f}+/-{QHP_err:.2f}')

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

    if QHP_fix:
        return Fdtls, nc, QHP, Fdtls_err, nc_err, QHP_err, \
            fitfun3(navg, *popt), delta_fit_str
            # fitfun4(nout, Fdtls, nc, QHP, beta), delta_fit_str, pout
            # fitfun(navg, Fdtls, nc, dHP), delta_fit_str
    else:
        return Fdtls, nc, QHP, Fdtls_err, nc_err, QHP_err, \
            fitfun4(navg, *popt), delta_fit_str
            # fitfun4(nout, Fdtls, nc, QHP, beta), delta_fit_str, pout
            # fitfun(navg, Fdtls, nc, dHP), delta_fit_str



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



def power_to_navg(power_dBm, Qi, Qc, fc, Z0_o_Zr=1.):
    """
    Converts power to photon number following Eq. (1) of arXiv:1801.10204
    and Eq. (3) of arXiv:1912.09119
    """
    # Physical constants, Planck's constant J s
    h = 6.62607015e-34
    hbar = 1.0545718e-34

    # Convert dBm to W
    Papp = 10**((power_dBm - 30) / 10) # * 1e-3
    # hb_wc2 = np.pi * h * (fc_GHz * 1e9)**2
    fscale = 1. if fc > 1e9 else 1e9
    fc_GHz = fc * fscale
    hb_wc2 = hbar * (2 * np.pi * fc_GHz)**2

    # Return the power as average number of photons
    Q = 1. / ((1. / Qi) + (1. / Qc))
    navg = (2. / hb_wc2) * (Q**2 / Qc) * Papp

    return navg



def power_sweep_fit_drv(atten=[0, -60], sample_name=None,
                        powers_in=None,
                        all_paths=None,
                        temperature=0.014,
                        plot_from_file=False, use_error_bars=True,
                        temp_correction='', phi0=0., use_gauss_filt=True,
                        use_matched_filt=False, use_elliptic_filt=False,
                        use_mov_avg_filt=False, loss_scale=None,
                        preprocess_method='linear',
                        ds = {'QHP' : 1e4, 'nc' : 1e6, 'Fdtls' : 1e-6}, data_dir=None,
                        plot_twinx=True, plot_fit=False, QHP_fix=False, show_plots=False,
                        save_dcm_plot=False, manual_init_list=None, show_dbm=False):
    """
    Driver for fitting the power sweep data for a given set of data
    """
    if np.any(powers_in):
        powers = np.copy(powers_in)
    else:
        # powers = np.linspace(15, -105, 25)
        print("Invalid input powers")
        return
        

    print(f'powers: {powers}')

    if all_paths is not None:
        filenames = all_paths
    filt_idxs = [] # list(range(10, powers.size))
    fname_ref = filenames[0]
    filter_points = [[0, 0] for _ in filenames]
    fpts = [[400, 400]]
    # filter_points[-1:] = fpts
    print(f'filter_points:\n{filter_points}')
    dstr = datetime.datetime.today().strftime('%y_%m_%d')
    err_str = '_error_bars'  if use_error_bars else ''
    cal_str = temp_correction + '_'
    fsize = 20
    csize = 5

    # Plot the results after gathering all of the fits
    df = fit_qiqcfc_vs_power(filenames, powers,
            filter_points=filter_points,
            preprocess_method=preprocess_method,
            phi0=phi0, use_gauss_filt=use_gauss_filt,
            use_matched_filt=use_matched_filt,
            use_elliptic_filt=use_elliptic_filt,
            use_mov_avg_filt=use_mov_avg_filt,
            filt_idxs=filt_idxs,
            fname_ref=fname_ref,
            data_dir=data_dir,
            save_dcm_plot=save_dcm_plot,
            manual_init_list=manual_init_list)

    # Extract the powers, quality factors, resonance frequencies, and 95 %
    # confidence intervals
    Qi = np.asarray(df['Qi'])
    Qc = df['Qc']
    Q  = df['Q']
    navg = df['navg']
    delta = 1. / Qi
    fc = df['fc [GHz]']
    Qi_err = np.asarray(df['Qi error'])
    Qc_err = df['Qc error']
    delta_err = Qi_err / Qi**2
    fc_err = df['fc error']

    # Add attenuation to powers
    powers += sum(atten)

    def pdBm_to_navg_ticks(p):
        n = np.abs(power_to_navg(powers[0::2], Qi[0::2], Qc[0], fc[0]))
        labels = [r'$10^{%.2g}$' % x for x in np.log10(n)]
        print(f'labels:\n{labels}')
        return labels

    # Fit the TLS loss
    # tcmp = regex.compile('[0-9]+.[0-9]+')
    tcmp = regex.compile('[0-9]+')
    T = temperature # float(tcmp.match(temperature).group())
    doff = 0
    if plot_fit:
        if doff > 0:
            Fdtls, nc, QHP, Fdtls_err, nc_err, QHP_err, delta_fit, \
                    delta_fit_str \
                    = fit_delta_tls(Qi[0:-doff], T, fc[0], \
                    Qc[0], powers[0:-doff],\
                    display_scales=ds, QHP_fix=QHP_fix, Qierr=Qi_err)
        else:
            Fdtls, nc, QHP, Fdtls_err, nc_err, QHP_err, \
                    delta_fit, delta_fit_str \
                    = fit_delta_tls(Qi, T, fc[0], Qc[0], powers,\
                    display_scales=ds, QHP_fix=QHP_fix, Qierr=Qi_err)

        if loss_scale:
            delta_fit /= loss_scale

        print('\n')
        print(f'F * d0_tls: {Fdtls:.2g} +/- {Fdtls_err:.2g}')
        print(f'nc: {nc:.2g} +/- {nc_err:.2g}')
        print('\n')

    if loss_scale:
        delta /= loss_scale
        delta_err /= loss_scale


    #the results
    ## Plot the resonance frequenices
    fig_fc, ax_fc = plt.subplots(1, 1, tight_layout=True)
    ax_fc.set_xlabel('Power [dBm]', fontsize=fsize)
    ax_fc.set_ylabel('Res Freq Shift From High Power [GHz]', fontsize=fsize)
    ax_fc_top = ax_fc.twiny()

    ## Plot the internal and external quality factors separately
    fig_qc, ax_qc = plt.subplots(1, 1, tight_layout=True)
    fig_qi, ax_qi = plt.subplots(1, 1, tight_layout=True)
    fig_qiqc, ax_qiqc = plt.subplots(1, 1, tight_layout=True)
    fig_d, ax_d = plt.subplots(1, 1, tight_layout=True)

    if not plot_twinx:
        powers = np.abs(power_to_navg(powers, Qi, Qc[0], fc[0]))

    # Plot with / without error bars
    if use_error_bars:
        markers = ['o', 'd', '>', 's', '<', 'h', '^', 'p', 'v']
        colors  = plt.rcParams['axes.prop_cycle'].by_key()['color']
        ax_fc.errorbar(powers, fc - fc[0] , yerr=fc_err, marker='o', ls='', ms=10,
                capsize=csize)
        ax_qc.errorbar(powers, Qc, yerr=Qc_err, marker='o', ls='', ms=10,
                capsize=csize)
        ax_qiqc.errorbar(powers, Qi, yerr=Qi_err, marker='h', ls='', ms=10,
                capsize=csize, color=colors[5],
                label=r'$Q_i$')
        ax_qiqc.errorbar(powers, Qc, yerr=Qc_err, marker='^', ls='', ms=10,
                capsize=csize, color=colors[6],
                label=r'$Q_c$')
        ax_qi.errorbar(powers, Qi, yerr=Qi_err, marker='o', ls='', ms=10,
                capsize=csize)
        if doff > 0:
            ax_d.errorbar(powers[0:-doff], delta[0:-doff],
                    yerr=delta_err[0:-doff], marker='d', ls='',
                    color=colors[1], ms=10, capsize=csize)
            if plot_fit:
                ax_d.plot(powers[0:-doff], delta_fit, ls='-',
                        label=delta_fit_str, color=colors[1])
        else:
            ax_d.errorbar(powers, delta,
                    yerr=delta_err, marker='d', ls='', color=colors[1],
                    ms=10, capsize=csize)
            if plot_fit:
                ax_d.plot(powers, delta_fit, ls='-', label=delta_fit_str,
                    color=colors[1])

    else:
        ax_fc.plot(powers, fc - fc[0] , marker='o', ms=10, ls='')
        ax_qc.plot(powers, Qc, marker='o', ms=10, ls='')
        ax_qi.plot(powers, Qi, marker='o', ms=10, ls='')
        ax_d.plot(powers, delta, marker='o', ms=10, ls='')
        
        
    if show_dbm:
        for x, y, text in zip(powers, delta, powers_in):
            ax_d.text(x, y, f"{text} dBm", size=8, rotation_mode="anchor", rotation=45)

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
    ax_qiqc.set_title(power_str, fontsize=fsize)
    ax_d.set_title(power_str, fontsize=fsize)

    # Set the second (top) axis labels
    if plot_twinx:
        ax_d_top  = ax_d.twiny()
        ax_qc_top = ax_qc.twiny()
        ax_qi_top = ax_qi.twiny()
        ax_qiqc_top = ax_qiqc.twiny()

        ax_qc.set_xlabel('Power [dBm]', fontsize=fsize)
        ax_qi.set_xlabel('Power [dBm]', fontsize=fsize)
        ax_qiqc.set_xlabel('Power [dBm]', fontsize=fsize)
        ax_d.set_xlabel('Power [dBm]', fontsize=fsize)

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
        ax_d_top.set_xticklabels(pdBm_to_navg_ticks(ax_d.get_xticks()))
        ax_fc_top.set_xticklabels(pdBm_to_navg_ticks(ax_fc.get_xticks()))

        set_xaxis_rot(ax_d_top, 45)
        set_xaxis_rot(ax_d, 45)

    else:
        ax_qc.set_xscale('log')
        ax_qi.set_xscale('log')
        ax_qiqc.set_xscale('log')
        ax_fc.set_xscale('log')
        ax_d.set_xscale('log')
        
        
        # ax_qc.set_yscale('log')
        ax_qi.set_yscale('log')
        ax_qiqc.set_yscale('log')
        # ax_fc.set_yscale('log')
        # ax_d.set_yscale('log')

        # Turn on all ticks
        ax_qc.get_xaxis().get_major_formatter().labelOnlyBase = False
        ax_qi.get_xaxis().get_major_formatter().labelOnlyBase = False
        ax_qiqc.get_xaxis().get_major_formatter().labelOnlyBase = False
        ax_fc.get_xaxis().get_major_formatter().labelOnlyBase = False
        ax_d.get_xaxis().get_major_formatter().labelOnlyBase = False

        ax_qc.set_xlabel(r'Power [$\left<{n}\right>$]', fontsize=fsize)
        ax_qi.set_xlabel(r'Power [$\left<{n}\right>$]', fontsize=fsize)
        ax_qiqc.set_xlabel(r'[$\left<{n}\right>$]', fontsize=fsize)
        ax_fc.set_xlabel(r'Power [$\left<{n}\right>$]', fontsize=fsize)
        ax_d.set_xlabel(r'$\left<{n}\right>$', fontsize=fsize)


    # Set the legends
    qiqc_lbls, qiqc_hdls = ax_qiqc.get_legend_handles_labels()
    ax_qiqc.legend(qiqc_lbls, qiqc_hdls, fontsize=fsize)
    d_lbls, d_hdls = ax_d.get_legend_handles_labels()
    ax_d.legend(d_lbls, d_hdls, loc='upper right', fontsize=fsize)

    # add note to filename if plot has labels on it
    dbm_label = "_labeled_powers" if show_dbm is True else ""
        
    fc_str = str(hm.get_frequency_from_filename(filenames[0])).replace(".","p")
    fsuffix = f"_{fc_str}GHz_{temperature}mK_{dstr}{dbm_label}.png"
    
    
    ## Save all figures to file
    if data_dir is None:
        fprefix = sample_name + '_' if sample_name else ''  # saved in script directory
    else:
        fprefix = data_dir + "fits\\"
        hm.check_and_make_dir(fprefix)  # saved in data_dir under "fits"
        
        fprefix_2 = f"reports\\{sample_name}_{fc_str}GHz\\"
        hm.check_and_make_dir(fprefix_2)  # a copy saved in main directory under "reports"
    
    
    fig_fc_title = 'fc_vs_power'+fsuffix
    fig_qc_title = 'qc_vs_power'+fsuffix
    fig_qiqc_title = 'qiqc_vs_power'+fsuffix
    fig_qi_title = 'qi_vs_power'+fsuffix
    fig_d_title = 'tand_vs_power'+fsuffix
    
    fig_title_size = 24
    fig_fc.suptitle(fig_fc_title.replace(".png",""), fontsize=fig_title_size)
    fig_qc.suptitle(fig_qc_title.replace(".png",""), fontsize=fig_title_size)
    fig_qiqc.suptitle(fig_qiqc_title.replace(".png",""), fontsize=fig_title_size)
    fig_qi.suptitle(fig_qi_title.replace(".png",""), fontsize=fig_title_size)
    fig_d.suptitle(fig_d_title.replace(".png",""), fontsize=fig_title_size)
    
    for fig in [fig_fc, fig_qc, fig_qc, fig_qiqc, fig_qi, fig_d]:
        fig.tight_layout()
        
    fig_fc.savefig(fprefix+fig_fc_title, format='png')
    fig_qc.savefig(fprefix+fig_qc_title, format='png')
    fig_qiqc.savefig(fprefix+fig_qiqc_title, format='png')
    fig_qi.savefig(fprefix+fig_qi_title, format='png')
    fig_d.savefig(fprefix+fig_d_title, format='png')
        
    fig_fc.savefig(fprefix_2+fig_fc_title, format='png')
    fig_qc.savefig(fprefix_2+fig_qc_title, format='png')
    fig_qiqc.savefig(fprefix_2+fig_qiqc_title, format='png')
    fig_qi.savefig(fprefix_2+fig_qi_title, format='png')
    fig_d.savefig(fprefix_2+fig_d_title, format='png')
    
    if show_plots is False:
        plt.close('all')



def set_xaxis_rot(ax, angle=45.):
    """
    Rotate x-axis labels
    """
    for tick in ax.get_xticklabels():
        tick.set_rotation(angle)



def plot_multiple_cooldown_power_sweeps(atten=[0, -60], sub_QHP=False,
                                        loss_scale=None):
    """
    Plots all of the data as function of temperature for different powers on the
    same set of figures
    """
    # Hardcode the powers used in each cooldown
    powers_cd23 = np.linspace(-15, -95, 17)
    powers_cd24 = np.linspace(-15, -95, 17)

    # Power sweep data for cooldowns 23 and 24
    # Cooldown 23, unknown base temperature, PNA-X
    # Cooldown 24, 16 mK base temperature, PNA
    df_cd23_filename = 'qiqcfc_boe_segmented.csv'
    df_cd24_filename = 'qiqcfc_boe_linear.csv'

    # NYU Al on InP
    cd_str = 'linear_segmented'
    filenames = [df_cd23_filename, df_cd24_filename]
    cd_labels = ['Segmented, 5/41/5',
                 'Linear, 401']
    powers = [powers_cd23+sum(atten), 
              powers_cd24+sum(atten)]
    ds_1 = {'QHP' : 1e5, 'nc' : 1e0, 'Fdtls' : 1e-6}
    ds_2 = {'QHP' : 1e5, 'nc' : 1e0, 'Fdtls' : 1e-6}
    ds = [ds_1, ds_2]

    long_idx = np.argmax([len(p) for p in powers])
    print(f'long_idx: {long_idx}')
    long_powers = powers[long_idx]
    print(f'long_powers: {long_powers}')

    ## Plot the resonance frequenices
    fsize = 20; csize = 5; lsize = 10 
    fig_fc, ax_fc = plt.subplots(1, 1, tight_layout=True)

    ## Plot the internal and external quality factors separately
    fig_qc, ax_qc = plt.subplots(1, 1, tight_layout=True)
    fig_qi, ax_qi = plt.subplots(1, 1, tight_layout=True)
    fig_d, ax_d = plt.subplots(1, 1, tight_layout=True)
    fig_qiqc, ax_qiqc = plt.subplots(1, 1, tight_layout=True)

    # Iterate over all powers, and filenames
    markers = ['o', 'd', '>', 's', '<', 'h', '^', 'p', 'v']
    ls      = ['--', ':', '-.']
    lllen   = len(ls)
    mlen    = len(markers) // 2
    mrk_qi  = markers[0::2]
    mrk_qc  = markers[1::2]
    colors  = plt.rcParams['axes.prop_cycle'].by_key()['color']
    clen    = len(colors) // 2
    clr_qi  = colors[0::2]
    clr_qc  = colors[1::2]
    for pidx, p in enumerate(powers):
        # Read the data from each of the power files
        df = pd.read_csv(filenames[pidx])

        # Extract the powers, quality factors, resonance frequencies, and 95 %
        # confidence intervals
        p      = p #[stidx:endidx]
        Q      = df['Q'].to_numpy() #[stidx:endidx]
        Qi     = df['Qi'].to_numpy() #[stidx:endidx]
        Qc     = df['Qc'].to_numpy() #[stidx:endidx]
        fc     = df['fc [GHz]'].to_numpy() #[stidx:endidx]
        Qi_err = df['Qi error'].to_numpy() #[stidx:endidx]
        Qc_err = df['Qc error'].to_numpy() #[stidx:endidx]
        fc_err = df['fc error'].to_numpy() #[stidx:endidx]
        navg   = df['navg'].to_numpy() #[stidx:endidx]

        print(f'\n{cd_labels[pidx]}\n--------------\n')
        print(f'powers: {p}')
        print(f'len(powers): {len(p)}')
        print(f'len(fc): {len(fc)}')

        d = 1. / Qi
        d_err = Qi_err / Qi**2

        # Hardcode the temperature in mK for now
        T = 13.
        pp = p
        dd = d
        dd_err = d_err
        Fdtls, nc, QHP, Fdtls_err, nc_err, QHP_err, delta_fit, delta_fit_str \
                = fit_delta_tls(Qi, T, fc[0], Qc[0], pp,
                        display_scales=ds[pidx])

        Fdtls_str = r'$F\delta^{0}_{TLS}$: %.2g$\pm$%.2g' % (Fdtls, Fdtls_err)
        QHP_str   = ',\n' + r'$Q_{HP}$: %.3g$\pm$%.2g' % (QHP, QHP_err)
        print(f'delta_other: {1./QHP}')

        # Compute the average number of photons for each input power
        n = power_to_navg(pp, Qi, Qc[0], fc[0], Z0_o_Zr=1.)
        print(f'n: {n}')

        # Subtract off the high power loss
        if sub_QHP:
            dd -= 1. / QHP
            delta_fit -= 1./QHP

        # Scale the results by some power of 10
        if loss_scale:
            d /= loss_scale
            d_err /= loss_scale
            loss_str = r'$\times10^{%d}$' % int(np.log10(loss_scale))
            delta_fit /= loss_scale
        else:
            loss_str = ''

        # d_err += QHP_err / QHP**2
        # ax_d.set_ylim([0, np.max(delta_fit)+np.max(d_err)-1./QHP])

        # Plot with error bars
        ax_fc.errorbar(p, fc, yerr=fc_err,
                ls='', ms=10, marker=markers[pidx],
                label=cd_labels[pidx], capsize=csize)
        ax_qc.errorbar(p, Qc, yerr=Qc_err,
                ls='', ms=10, marker=markers[pidx],
                label=cd_labels[pidx], capsize=csize)
        ax_qi.errorbar(pp, Qi, yerr=Qi_err,
                ls='', ms=10, marker=markers[pidx],
                label=cd_labels[pidx], capsize=csize)

        if sub_QHP:
            ax_d.plot(n, delta_fit-1./QHP, ls='-',
                    color=colors[pidx])
        else:
            ax_d.plot(n, delta_fit, ls='-',
                    color=colors[pidx])
        ax_d.errorbar(n, dd, yerr=dd_err,
                ls='', ms=10, marker=markers[pidx],
                label=cd_labels[pidx]+'\n---------------------\n'\
                        +delta_fit_str\
                        +'\n---------------------',
                capsize=csize, color=colors[pidx])
        ax_d.plot(n, delta_fit, ls='-',
                color=colors[pidx])

        ax_qiqc.errorbar(pp, Qi, yerr=Qi_err,
                ls='', ms=10, marker=mrk_qi[pidx%mlen],
                color=clr_qi[pidx%clen],
                label=r'$Q_i$ '+cd_labels[pidx], capsize=csize)
        ax_qiqc.errorbar(p, Qc, yerr=Qc_err,
                color=clr_qc[pidx%clen],
                ls='', ms=10, marker=mrk_qc[pidx%mlen],
                label=r'$Q_c$ '+cd_labels[pidx], capsize=csize)

    # Update the x, y labels accordingly
    ax_qc.set_xlabel('Power [dBm]', fontsize=fsize)
    ax_qi.set_xlabel('Power [dBm]', fontsize=fsize)
    # ax_d.set_xlabel('Power [dBm]', fontsize=fsize)
    ax_d.set_xlabel(r'$\left<n\right>$', fontsize=fsize)
    ax_d.set_xscale('log')
    # ax_d.set_ylim([-0.5, 6])

    ax_qiqc.set_xlabel('Power [dBm]', fontsize=fsize)
    ax_fc.set_xlabel('Power [dBm]', fontsize=fsize)
    ax_qc.set_ylabel(r'$Q_c$', fontsize=fsize)
    ax_qi.set_ylabel(r'$Q_i$', fontsize=fsize)

    if sub_QHP:
        ax_d.set_ylabel(r'$(Q_i^{-1}-Q_{HP}^{-1})$%s' % loss_str, fontsize=fsize)
        Qstr = 'subQHP_'
    else:
        ax_d.set_ylabel(r'$Q_i^{-1}$%s' % loss_str, fontsize=fsize)
        Qstr = ''
    ax_qiqc.set_ylabel(r'$Q_i$, $Q_c$', fontsize=fsize)
    ax_fc.set_ylabel('Resonance Frequency [GHz]', fontsize=fsize)

    ax_qiqc.set_title('Measured at Base Temperature ~16 mK', fontsize=fsize)

    # Set the legends
    fc_lbls, fc_hdls = ax_fc.get_legend_handles_labels()
    qc_lbls, qc_hdls = ax_qc.get_legend_handles_labels()
    qi_lbls, qi_hdls = ax_qi.get_legend_handles_labels()
    d_lbls, d_hdls = ax_d.get_legend_handles_labels()
    qiqc_lbls, qiqc_hdls = ax_qiqc.get_legend_handles_labels()
    ax_fc.legend(fc_lbls, fc_hdls, loc='lower right', fontsize=fsize)
    ax_qc.legend(qc_lbls, qc_hdls, loc='lower right', fontsize=fsize)
    ax_qi.legend(qi_lbls, qi_hdls, loc='lower right', fontsize=fsize)
    # ax_d.legend(d_lbls, d_hdls, loc='center', fontsize=lsize)
    ax_d.legend(d_lbls, d_hdls, loc='center right', fontsize=lsize)
    ax_qiqc.legend(qiqc_lbls, qiqc_hdls, loc=(0.1, 0.6), fontsize=fsize)

    ## Save all figures to file vs. power
    dstr = datetime.datetime.today().strftime('%y%m%d')
    fig_fc.savefig(f'fc_vs_power_cooldowns_{cd_str}_{dstr}.pdf',
            format='pdf')
    fig_qc.savefig(f'qc_vs_power_cooldowns_{cd_str}_{dstr}.pdf',
            format='pdf')
    fig_qi.savefig(f'qi_vs_power_cooldowns_{cd_str}_{dstr}.pdf',
            format='pdf')
    fig_d.savefig(f'loss_vs_power_cooldowns_{cd_str}_{Qstr}{dstr}.pdf',
            format='pdf')
    fig_qiqc.savefig(f'qiqc_vs_power_cooldowns_{cd_str}_{dstr}.pdf',
            format='pdf')


    plt.close('all')
