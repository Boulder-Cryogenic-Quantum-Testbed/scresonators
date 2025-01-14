import numpy as np
import pandas as pd
import os
import datetime, regex
import re
from pathlib import Path
from resonator import Resonator
import uncertainties
import scipy as sp
import matplotlib.pyplot as plt


class Sweeper():

    def __init__(self, directory, data_columns=["Frequency [Hz]", "Magnitude [dB]", "Phase [deg]"], 
                 preprocess_method='circle', fit_method='DCM'):

        self.directory = Path(directory)
        self.data_columns = data_columns

        self.fit_method = fit_method
        self.preprocess_method = preprocess_method

        self._load_resonators()

    def _load_resonators(self):

        """
        Load resonators from CSV files in the specified directory and store them 
        in a dictionary with relevant metadata.
        """

        data_files = sorted(self.directory.glob("*MQC_BOE*.csv"))

        self.resonators_data = {}  # Dictionary to store resonators and metadata

        for file in data_files:

            # Create a Resonator object
            resonator = Resonator(
                file_path=file, 
                data_columns=self.data_columns, 
                preprocess_method=self.preprocess_method, 
                fit_method_name=self.fit_method
            )
            
            # Extract power and filename from the file path
            power = int(re.search(r'_(\-?\d+)dBm', str(file)).group(1))
            file_name = re.search(r'([^/]+)\.csv$', str(file)).group(1)
            
            # Store data in the dictionary
            self.resonators_data[file_name] = {
                "resonator": resonator,
                "power": power,
                "file_name": file_name
                }
            
        # Extract frequency 
        identifier = re.search(r"_\dp\d{0,4}GHz_", str(data_files[0]))
        self.freq = float(identifier[0].replace("GHz","").replace("_","").replace("p","."))

    def get_resonator(self, key=None, index=None, power=None):

        """
        Generalized function to retrieve a Resonator object.
        
        Args:
            key (str): Identifier (e.g., file name) for the Resonator.
            index (int): Index of the Resonator in the list.
            power (int): Power level of the Resonator.
        
        Returns:
            Resonator: The corresponding Resonator object.
        """

        if key:
            if key not in self.resonators_data:
                raise KeyError(f"No resonator found for key: {key}")
            return self.resonators_data[key]["resonator"]
        
        if index is not None:
            keys = list(self.resonators_data.keys())
            if index < 0 or index >= len(keys):
                raise IndexError(f"Index {index} out of range.")
            return self.resonators_data[keys[index]]["resonator"]
        
        if power is not None:
            for data in self.resonators_data.values():
                if data.get("power") == power:
                    return data["resonator"]
            raise ValueError(f"No resonator found with power: {power}")
        
        raise ValueError("Must specify one of 'key', 'index', or 'power'.")
    
    def initialize_resonator_data_processor(self, normalize_pts, preprocess_method, key=None, index=None, power=None):

        """
        Access a specific Resonator and initialize its data processor.
        
        Args:
            normalize_pts (int): Normalization points for the data processor.
            preprocess_method (str): The preprocessing method to apply.
            key (str): Identifier (e.g., file name) for the Resonator (optional).
            index (int): Index of the Resonator in the list (optional).
            power (int): Power level of the Resonator (optional).

        To-be implemented: Initialize the same thing to a list of Resonator obj or "all"
        
        """

        # Use the generalized `get_resonator` method to find the target Resonator
        resonator = self.get_resonator(key=key, index=index, power=power)
        
        # Initialize and return the DataProcessor
        resonator.initialize_data_processor(normalize_pts=normalize_pts, preprocess_method=preprocess_method)

    def initialize_resonator_fit_method(self, 
                                        fit_name: str = None,
                                        MC_iteration: int = 5,
                                        MC_rounds: int = 100,
                                        MC_weight: str = 'no',
                                        MC_weightvalue: int = 2,
                                        MC_fix: list = [],
                                        MC_step_const: float = 0.6,
                                        manual_init: list = None,
                                        vary: list = None,
                                        key=None, 
                                        index=None, 
                                        power=None):
        """
        Access a specific Resonator and initialize its fit method.
        
        Args:
            MC_iteration (int): Number of Monte Carlo iterations.
            MC_rounds (int): Number of Monte Carlo rounds.
            MC_weight (str): Weighting method for the fit.
            MC_weightvalue (int): Weight value for the fit.
            MC_fix (list): List of parameters to fix during the fit.
            MC_step_const (float): Step constant for the fit.
            manual_init (list): Initial manual values for the fit (optional).
            vary (list): Boolean list indicating which parameters can vary (optional).
            key (str): Identifier (e.g., file name) for the Resonator (optional).
            index (int): Index of the Resonator in the list (optional).
            power (int): Power level of the Resonator (optional).
        """

        # Use the generalized `get_resonator` method to find the target Resonator
        resonator = self.get_resonator(key=key, index=index, power=power)
    
        # Initialize the fit method for the resonator
        resonator.initialize_fit_method(fit_name=fit_name,
                                        MC_iteration=MC_iteration,
                                        MC_rounds=MC_rounds,
                                        MC_weight=MC_weight,
                                        MC_weightvalue=MC_weightvalue,
                                        MC_fix=MC_fix,
                                        MC_step_const=MC_step_const,
                                        manual_init=manual_init,
                                        vary=vary)

    def fit_qiqcfc_vs_power(self):
        
        """
        Fits multiple resonances at different powers for a given power
        """

        phi0 = 0.

        # Iterate over the filenames, save the values of fc, Qc, fit error
        Npts = len(self.resonators_data)
        Q, Qc, Qi, fc, navg = np.zeros(Npts), np.zeros(Npts), np.zeros(Npts), np.zeros(Npts), np.zeros(Npts)
        Qc_err, Qi_err, fc_err, errs = np.zeros(Npts), np.zeros(Npts), np.zeros(Npts), np.zeros(Npts)
        self.powers = np.zeros(Npts)
        
        for idx, resonator_key in enumerate(self.resonators_data):

            resonator = self.resonators_data[resonator_key].get('resonator')
            power = self.resonators_data[resonator_key].get('power')

            self.powers[idx] = power

            params, conf_int, err, init, fig = resonator.fit()

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
            navg[idx] = self.power_to_navg(power, Qi[idx], Qc[0], fc[0])

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
        
        # Save the data to file
        df = pd.DataFrame(np.vstack((self.powers, navg, fc, Qi, Qc, Q,
                        errs, Qi_err, Qc_err, fc_err)).T,
                columns=['Power [dBm]', 'navg', 'fc [GHz]', 'Qi', 'Qc', 'Q',
                    'error', 'Qi error', 'Qc error', 'fc error'])
        
        # write to .csv file
                                        
        return df

    def power_to_navg(self, power_dBm, Qi, Qc, fc, Z0_o_Zr=1.):

        """
        Converts power to photon number following Eq. (1) of arXiv:1801.10204
        and Eq. (3) of arXiv:1912.09119
        """
        # Physical constants, Planck's constant J s
        h = 6.62607015e-34
        hbar = 1.0545718e-34

        print(type(power_dBm))

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
 
    def fit_delta_tls(self, Qi, T, fc, Qc, p, display_scales={'QHP' : 1e5,
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

        navg = np.abs(self.power_to_navg(p, Qi, Qc, fc))
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
            print("QHP: ", QHP)
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
 
    def power_sweep_fit_drv(self, atten=[0, -60], sample_name=None,
                        temperature=0.014,
                        use_error_bars=True,
                        loss_scale=None,
                        show_dbm=False,
                        ds = {'QHP' : 1e4, 'nc' : 1e6, 'Fdtls' : 1e-6},
                        plot_twinx=True, plot_fit=False, QHP_fix=False, show_plots=False,
                        save_fit_dirs="fits/"):
        
        """
        Driver for fitting the power sweep data for a given set of data
        """

        data_dir = str(self.directory)
        
        filter_points = [[0, 0] for _ in self.resonators_data]
        # filter_points[-1:] = fpts
        print(f'filter_points:\n{filter_points}')
        dstr = datetime.datetime.today().strftime('%y_%m_%d')
        fsize = 20
        csize = 5

        # Plot the results after gathering all of the fits
        df = self.fit_qiqcfc_vs_power()

        # Extract the powers, quality factors, resonance frequencies, and 95 %
        # confidence intervals
        Qi = np.asarray(df['Qi'])
        Qc = df['Qc']
        delta = 1. / Qi
        fc = df['fc [GHz]']
        Qi_err = np.asarray(df['Qi error'])
        Qc_err = df['Qc error']
        delta_err = Qi_err / Qi**2
        fc_err = df['fc error']

        # Add attenuation to powers
        powers = self.powers
        powers += sum(atten)

        def pdBm_to_navg_ticks(p):
            n = np.abs(self.power_to_navg(powers[0::2], Qi[0::2], Qc[0], fc[0]))
            labels = [r'$10^{%.2g}$' % x for x in np.log10(n)]
            print(f'labels:\n{labels}')
            return labels

        # Fit the TLS loss
        # tcmp = regex.compile('[0-9]+.[0-9]+')
        T = temperature # float(tcmp.match(temperature).group())
        doff = 0
        if plot_fit:
            if doff > 0:
                Fdtls, nc, QHP, Fdtls_err, nc_err, QHP_err, delta_fit, \
                        delta_fit_str \
                        = self.fit_delta_tls(Qi[0:-doff], T, fc[0], \
                        Qc[0], powers[0:-doff],\
                        display_scales=ds, QHP_fix=QHP_fix, Qierr=Qi_err)
            else:
                Fdtls, nc, QHP, Fdtls_err, nc_err, QHP_err, \
                        delta_fit, delta_fit_str \
                        = self.fit_delta_tls(Qi, T, fc[0], Qc[0], powers,\
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

        plot_kwargs = {
            "figsize" : (8,6),
        }
        
        ## Plot the internal and external quality factors separately
        fig_qc, ax_qc = plt.subplots(1, 1, tight_layout=True, **plot_kwargs)
        fig_qi, ax_qi = plt.subplots(1, 1, tight_layout=True, **plot_kwargs)
        fig_qiqc, ax_qiqc = plt.subplots(1, 1, tight_layout=True, **plot_kwargs)
        fig_d, ax_d = plt.subplots(1, 1, tight_layout=True, **plot_kwargs)

        if not plot_twinx:
            powers = np.abs(self.power_to_navg(powers, Qi, Qc[0], fc[0]))

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
            for x, y, text in zip(powers, delta, self.powers):
                ax_d.text(x, y, f"{text} dBm", size=12,  rotation=45, rotation_mode="anchor",
                        horizontalalignment="left", verticalalignment="bottom")

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

            self.set_xaxis_rot(ax_d_top, 45)
            self.set_xaxis_rot(ax_d, 45)

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

        fc_val = self.freq
        fc_str = f"{fc_val:1.3f}".replace(".","p")
        fsuffix = f"_{fc_str}GHz_{temperature}mK_{dstr}.png"
        
        
        
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
            
            
            
        ## Save all figures to file
        for fit_dir in save_fit_dirs:
            
            if data_dir is None:
                fprefix = sample_name + '_' if sample_name else ''  # saved in script directory
            else:
                fprefix = data_dir + "/all_fit_plots/"
                self.check_and_make_dir(fprefix)  # save a copy in data_dir under "<sample_name_freq>/fits"
                
                fprefix_2 = f"reports/{sample_name}_{fc_str}GHz/"
                self.check_and_make_dir(fprefix_2)  # save a copy in main directory under "fit_reports/<sample_name_freq>/"
            
            print("fprefix: ", fprefix)
            
            fig_fc.savefig(f"{fprefix}/{fig_fc_title}", format='png')
            fig_qc.savefig(f"{fprefix}/{fig_qc_title}", format='png')
            fig_qiqc.savefig(f"{fprefix}/{fig_qiqc_title}", format='png')
            fig_qi.savefig(f"{fprefix}/{fig_qi_title}", format='png')
            fig_d.savefig(f"{fprefix}/{fig_d_title}", format='png')
                
            fig_fc.savefig(f"{fprefix_2}/{fig_fc_title}", format='png')
            fig_qc.savefig(f"{fprefix_2}/{fig_qc_title}", format='png')
            fig_qiqc.savefig(f"{fprefix_2}/{fig_qiqc_title}", format='png')
            fig_qi.savefig(f"{fprefix_2}/{fig_qi_title}", format='png')
            fig_d.savefig(f"{fprefix_2}/{fig_d_title}", format='png')
        
        if show_plots is False:
            plt.close('all')

    def set_xaxis_rot(self, ax, angle=45.):
        """
        Rotate x-axis labels
        """
        for tick in ax.get_xticklabels():
            tick.set_rotation(angle)

    def check_and_make_dir(self, directory_name):
        directory_name = directory_name.replace('.csv', '').replace('.pdf', '')  # sanitize input
        if not os.path.exists(directory_name):
            print(f'      target directory: {directory_name}')
            print(f'      directory does not exist. now creating...')
            os.makedirs(directory_name)
        else:
            print(f"      directory already exists.")
        print(f'      absolute path: {os.path.abspath(directory_name)}\n')
        return
                        
