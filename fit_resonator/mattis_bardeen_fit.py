"""
Functions to fit Mattis-Bardeen temperature dependence of Qi, fc
using the asymptotic expressions for the surface impedance and superconducting
BCS gap as a function of temperature.
"""

import numpy as np
import scipy.constants as sc
from scipy.special import i0, k0
from scipy.optimize import curve_fit
import uncertainties

from fit_resonator.plot_mb import MPLPlotWrapper
import fit_resonator.resonator as scr

from scipy.optimize import least_squares
from scipy.optimize import curve_fit
import os

class MBFitTemperatureSweep(object):
    """
    Class to fit the Qi and fc vs. temperature data
    """
    def __init__(self, temperatures : list, s21_files : list, *args, **kwargs):
        """
        Constructor for the Mattis-Bardeen temperature sweep fitting class

        Arguments:
        ---------

        temperatures    :list:  list of temperatures 
        s21_files       :list:  list of filenames with the transmission (S21)
                                data at each of the temperatures, ordered in the
                                same way as `temperatures`

        """
        # London penetration depth -- can determine separately by using alpha
        # from simulations in HFSS and fitting for Xs = mu0 wc lambda_{L0}
        self.lambda0 = 16e-9 # pure Al
        self.d = 5e-3
        self.output_fit_figures = None

        # Quantities relevant to Al
        self.init_fit_guess = {'Tc' : 1.2, 'alpha' : 1e-5,
                               'lambda' : self.lambda0}
        self.use_jordans_rule = False
        self.alpha_sim = 1e-4
        self.alpha_sim_err = 1e-5

        # Set the default confidence level (95 %)
        self.confidence_level = 0.95
        self.fit_normalization = 'linear'

        # https://www.nde-ed.org/NDETechniques/EddyCurrent/
        # ET_Tables/standardsmethods.xhtml
        # Aluminum -- TODO: fit for this quantity
        self.sigma_n = 3.767e7

        self.__dict__.update(locals())
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Errors on the temperatures are 5 % of the temperatures
        self.Terr = 0.05 * temperatures

        # Setup the resonator fit objects
        self.fit_resonator_responses()

    def fit_resonator_responses(self):
        """
        Generates resonator fitting objects for each file 
        """
        # Default fit properties
        MC_iteration = 10
        MC_rounds = 1e3
        MC_fix = []
        self.fits = {}
        files = self.s21_files
        flen = len(files)
        self.Qi = np.zeros(flen)
        self.Qierr = np.zeros(flen)
        self.fc = np.zeros(flen)
        self.fcerr = np.zeros(flen)
        for idx, fname in enumerate(files):
            # Create a new instance of the resonator fitting class
            pwd = os.getcwd()
            res = scr.Resonator(preprocess_method = self.fit_normalization)
            res.from_file(pwd + '/' + fname)
            res.fit_method('DCM', MC_iteration, MC_rounds=MC_rounds,
                    MC_fix=MC_fix, manual_init=None, MC_step_const=0.3)

            # Perform the fit for each temperature
            params, conf_ints, err, init = res.fit(self.output_fit_figures) 

            # Extract the coupling quality factor, internal quality factor,
            # resonance frequency, and the confidence intervals
            Qc = params[1] * np.exp(1j*params[3])
            self.Qi[idx] = 1. / (1. / params[0] - np.real(1. / Qc))
            self.fc[idx] = params[2]
            self.Qierr[idx] = conf_ints[1]
            self.fcerr[idx] = conf_ints[5]
            print('\n-----------------------------')
            print(f'Fitting T = {self.temperatures[idx]} K ...')
            print('-----------------------------')
            print(f'Qi: {self.Qi[idx]:.3g} +/- {self.Qierr[idx]:.3g}')
            print(f'fc: {self.fc[idx]:.3g} +/- {self.fcerr[idx]:.3g}')
            print('-----------------------------\n')


    def surface_impedance(self, T : float, Tc : float, fc : float) -> float:
        """
        Computes the surface impedance of the superconductor as a function of
        temperature, to be used in the fitting of the Qi, fc
        """
        # BCS gap function (approximate)
        D = np.tanh(1.74 * np.sqrt(Tc / T - 1.))

        # surface impedance
        kB = sc.k / sc.e
        D0 = 1.762 * kB * Tc
        xsci = sc.h * fc / (2 * sc.k * T)
        s1 = ((4 * D0 / (sc.h * fc / sc.e)) * np.exp(-1.762 * Tc / T) *
                np.sinh(xsci) * k0(xsci))
        s2 = (np.pi * D0 / (sc.h * fc / sc.e)
                * (1 - np.sqrt((2 * kB * T) / D0) * np.exp(-1.762 * Tc / T)
                - 2 * np.exp(-1.762 * Tc / T) * np.exp(-xsci) * i0(xsci)))
        sigma = self.sigma_n * (s1  - 1j * s2)
        Zs = np.sqrt(1j * sc.mu_0 * fc * 2 * np.pi / sigma)


        return Zs

    def format_error_strings(self, param_str : str, val, val_err):
        """
        Formats the parameter with its uncertainty
        """
        # Uncertainty formatting
        ## Rounding to n significant figures
        round_sigfig = lambda x, n : round(x, n -
                int(np.floor(np.log10(abs(x)))) - 1)

        # Handle inf case
        if np.isclose(val_err, np.inf):
            val_err = np.inf
            out_str = r'$%s: %.2g\pm %g$' % (param_str, val, val_err)
        else:
            val_err = round_sigfig(val_err, 1)

            ## Uncertainty objects
            val_un = uncertainties.ufloat(val, val_err)

            # Build a string with the results
            latex_str = f'{val_un:L}'
            out_str = r'$%s: %s$' % (param_str, latex_str)

        return out_str

    def fit_qi_vs_temperature(self, use_alpha_sim : bool = False):
        """
        Fits the Qi vs T. to the model:

        Qi^{-1}(T) - Qi^{-1}(0) = alpha (Rs(T) - Rs(0)) / Xs(0)

        """
        # Read off the temperatures and lowest temperature, lowest temperature
        # frequency
        TT = self.temperatures
        T0 = TT.min()
        fc0 = self.fc[0]
        Qi = self.Qi
        ooQi = 1./Qi - 1./Qi[0]
        alpha_sim = self.alpha_sim

        def fitfunres(x, *args):
            """
            Fitting function for Qi
            """
            T = args[0]
            oQi = args[1]
            Tc, alpha = x
            if type(T) == list or type(T) == np.ndarray:
                Zs = np.array([self.surface_impedance(TT, Tc, fc0) for TT in T])
            else:
                Zs = self.surface_impedance(T, Tc, fc0)

            Zs0 = self.surface_impedance(T0, Tc, fc0)
            Rs = np.real(Zs)
            Rs0 = np.real(Zs0)
            Xs0 = np.imag(Zs0)

            return (alpha * (Rs - Rs0) / Xs0 - oQi)**2

        def fitfunreslambda(x, *args):
            """
            Fitting function for Qi
            """
            T = args[0]
            oQi = args[1]
            Tc, lambdaL = x
            if type(T) == list or type(T) == np.ndarray:
                Zs = np.array([self.surface_impedance(TT, Tc, fc0) for TT in T])
            else:
                Zs = self.surface_impedance(T, Tc, fc0)

            Zs0 = self.surface_impedance(T0, Tc, fc0)
            Rs = np.real(Zs)
            Rs0 = np.real(Zs0)
            Xs0 = 2 * np.pi * fc0 * sc.mu_0 * lambdaL

            return (alpha_sim * (Rs - Rs0) / Xs0  - oQi)**2

        def fitfun(T, Tc, alpha):
            """
            Fitting function for Qi
            """
            if type(T) == list or type(T) == np.ndarray:
                Zs = np.array([self.surface_impedance(TT, Tc, fc0) for TT in T])
            else:
                Zs = self.surface_impedance(T, Tc, fc0)
            Zs0 = self.surface_impedance(T0, Tc, fc0)
            Rs = np.real(Zs)
            Xs = np.imag(Zs)
            Rs0 = np.real(Zs0)
            Xs0 = np.imag(Zs0)

            return alpha * (Rs - Rs0) / Xs0

        def fitfunlambda(T, Tc, lambdaL):
            """
            Fitting function for Qi
            """
            if type(T) == list or type(T) == np.ndarray:
                Zs = np.array([self.surface_impedance(TT, Tc, fc0) for TT in T])
            else:
                Zs = self.surface_impedance(T, Tc, fc0)
            Zs0 = self.surface_impedance(T0, Tc, fc0)
            Rs = np.real(Zs)
            Xs = np.imag(Zs)
            Rs0 = np.real(Zs0)
            Xs0 = 2 * np.pi * fc0 * sc.mu_0 * lambdaL

            return alpha_sim * (Rs - Rs0) / Xs0

        if use_alpha_sim:
            TT_dense, oQipred, upper, lower, label = self.fit_generic(TT, ooQi,
                fitfunlambda, fitfunreslambda, use_alpha_sim=use_alpha_sim)
        else:
            TT_dense, oQipred, upper, lower, label = self.fit_generic(TT, ooQi,
                fitfun, fitfunres, use_alpha_sim=use_alpha_sim)

        return TT_dense, oQipred, upper, lower, label

    def fit_fc_vs_temperature(self, use_alpha_sim : bool = False):
        """
        Fits the fc vs T. to the model:

        (fc(T) - fc(0)) / fc(0) = -alpha/2 (Xs(T) - Xs(0)) / Xs(0)

        """
        # Read off the temperatures and lowest temperature, lowest temperature
        # frequency
        TT = self.temperatures
        T0 = TT.min()
        fc = self.fc
        fc0 = fc[0]
        oofc = (fc - fc0) / fc0
        alpha_sim = self.alpha_sim

        def fitfunres(x, *args):
            """
            Fitting function for fc
            """
            T = args[0]
            ofc = args[1]
            Tc, alpha = x
            if type(T) == list or type(T) == np.ndarray:
                Zs = np.array([self.surface_impedance(TT, Tc, fc0) for TT in T])
            else:
                Zs = self.surface_impedance(T, Tc, fc0)

            Zs0 = self.surface_impedance(T0, Tc, fc0)
            Xs = Zs.imag
            Xs0 = Zs0.imag

            return (-0.5 * alpha * (Xs - Xs0) / Xs0 - ofc)**2

        def fitfunreslambda(x, *args):
            """
            Fitting function for fc
            """
            T = args[0]
            ofc = args[1]
            Tc, lambdaL = x
            if type(T) == list or type(T) == np.ndarray:
                Zs = np.array([self.surface_impedance(TT, Tc, fc0) for TT in T])
            else:
                Zs = self.surface_impedance(T, Tc, fc0)

            Zs0 = self.surface_impedance(T0, Tc, fc0)
            Xs = Zs.imag
            Xs0 = Zs0.imag
            Xs0p =  2 * np.pi * fc0 * sc.mu_0 * lambdaL

            return (-0.5 * alpha_sim * (Xs - Xs0) / Xs0p - ofc)**2

        def fitfunlambda(T, Tc, lambdaL):
            """
            Fitting function for fc
            """
            if type(T) == list or type(T) == np.ndarray:
                Zs = np.array([self.surface_impedance(TT, Tc, fc0) for TT in T])
            else:
                Zs = self.surface_impedance(T, Tc, fc0)

            Zs0 = self.surface_impedance(T0, Tc, fc0)
            Xs = Zs.imag
            Xs0 = Zs0.imag
            Xs0p =  2 * np.pi * fc0 * sc.mu_0 * lambdaL

            return -0.5 * alpha_sim * (Xs - Xs0) / Xs0p

        def fitfun(T, Tc, alpha):
            """
            Fitting function for fc
            """
            if type(T) == list or type(T) == np.ndarray:
                Zs = np.array([self.surface_impedance(TT, Tc, fc0) for TT in T])
            else:
                Zs = self.surface_impedance(T, Tc, fc0)

            Zs0 = self.surface_impedance(T0, Tc, fc0)
            Xs = Zs.imag
            Xs0 = Zs0.imag

            return -0.5 * alpha * (Xs - Xs0) / Xs0

        if use_alpha_sim:
            TT_dense, ofcpred, upper, lower, label = self.fit_generic(TT, oofc,
                fitfunlambda, fitfunreslambda, use_alpha_sim=use_alpha_sim)
        else:
            TT_dense, ofcpred, upper, lower, label = self.fit_generic(TT, oofc,
                fitfun, fitfunres, use_alpha_sim=use_alpha_sim)

        return TT_dense, ofcpred, upper, lower, label 

    def fit_generic(self, TT : list, data : list, fitfun, fitfunres,
            use_alpha_sim : bool = False):
        """
        Generic code used by both the qi and fc fitting routines
        """
        # Compute the least squares solution
        if use_alpha_sim:
            x0 = (self.init_fit_guess['Tc'], self.init_fit_guess['lambda'])
        else:
            x0 = (self.init_fit_guess['Tc'], self.init_fit_guess['alpha'])

        res_lsq = least_squares(fitfunres, x0, args=(TT, data))
        popt, pcov = curve_fit(fitfun, TT, data, p0=x0)

        # Estimate the covariance matrix
        J = res_lsq.jac
        residuals = res_lsq.fun
        mse = np.sum(residuals) / (2 * len(TT) - len(x0))
        cov = mse * np.linalg.inv(J.T @ J + np.eye(J.shape[1]) * 1e-8)
        errors = np.sqrt(np.diag(cov))

        # Dense temperature list to evaluate the fit function on
        TT_dense = np.linspace(TT.min(), TT.max(), 1000)

        if use_alpha_sim:
            Tcfit, lambdafit = res_lsq.x[0], res_lsq.x[1]

            print(f'least_squares results:')
            print(f'lambdaL: {lambdafit}')
            print(f'Tc: {Tcfit}')

            errors = np.sqrt(np.diag(pcov))
            Tcfit, lambdafit = popt
            print(f'curve_fit results:')
            print(f'Tc: {popt[0]:.2g} +/- {errors[0]:.1g}')
            print(f'lambdaL: {popt[1]:.2g} +/- {errors[1]:.1g}')

            # Prediction intervals following this thread:
            # https://stats.stackexchange.com/questions/
            # 472032/confidence-and-prediction-intervals-for-nonlinear-models
            datapred0 = fitfun(TT, Tcfit, lambdafit)
            datapred = fitfun(TT_dense, Tcfit, lambdafit)

            # Generate the label for the plot
            Tc_err, lambda_err = errors
            lambda_str = self.format_error_strings(r'\lambda_L', lambdafit,
                                                   lambda_err)
            Tc_str = self.format_error_strings(r'T_c', Tcfit, Tc_err)
            alpha_sim_str = self.format_error_strings(r'\alpha_s',
                    self.alpha_sim, self.alpha_sim_err)
            label = lambda_str + '\n' + Tc_str + '\n' + alpha_sim_str

        else:
            Tcfit, alphafit = res_lsq.x[0], res_lsq.x[1]

            print(f'least_squares results:')
            print(f'alpha: {alphafit}')
            print(f'Tc: {Tcfit}')

            errors = np.sqrt(np.diag(pcov))
            Tcfit, alphafit = popt
            print(f'curve_fit results:')
            print(f'Tc: {popt[0]:.2g} +/- {errors[0]:.1g}')
            print(f'alpha: {popt[1]:.2g} +/- {errors[1]:.1g}')

            # Prediction intervals following this thread:
            # https://stats.stackexchange.com/questions/
            # 472032/confidence-and-prediction-intervals-for-nonlinear-models
            datapred0 = fitfun(TT, Tcfit, alphafit)
            datapred = fitfun(TT_dense, Tcfit, alphafit)

            # Generate the label for the plot
            Tc_err, alpha_err = errors
            alpha_str = self.format_error_strings(r'\alpha', alphafit, alpha_err)
            Tc_str = self.format_error_strings(r'T_c', Tcfit, Tc_err)
            label = alpha_str + '\n' + Tc_str

        # Confidence interval estimates
        # XXX: Does not work, need to address with bootstrapping, other
        #      approaches to accurately estimate confidence intervals
        noise = np.std(data - datapred0)
        predictions = np.array([np.random.normal(datapred, noise) for j in
            range(10_000)])
        
        a = self.confidence_level
        upper, lower = np.quantile(predictions, [1 - a, a], axis = 0)

        return TT_dense, datapred, upper, lower, label 

    def get_Rs_gseam(self, qitotinv : np.ndarray, pcond : np.ndarray,
            yseam : np.ndarray, lambdaL : float, fc : np.ndarray,
            plot_fit : bool = True, fnames : list = None,
            qierr : np.ndarray = None,
            constrain_fit : bool = False) -> list:
        """
        Fits the total loss of two cavity resonances to extract the surface
        resistance Rs and seam conductance gseam
        """
        # XXX: Depreciated two-point implementation
        # # Qitot,1^-1 = pcond,1 Rs / (mu0 wc1 lambdaL) + yseam,1 / gseam
        # # Qitot,2^-1 = pcond,2 Rs / (mu0 wc2 lambdaL) + yseam,2 / gseam
        # y1 = qitotinv[0] / yseam[0]
        # y2 = qitotinv[1] / yseam[1]
        # x1 = pcond[0] / (sc.mu_0 * 2 * np.pi * fc[0] * lambdaL * yseam[0])
        # x2 = pcond[1] / (sc.mu_0 * 2 * np.pi * fc[1] * lambdaL * yseam[1])
        # # b = 1 / gseam, a = Rs
        # b = (y2 - y1 * (x2 / x1)) / (1 - (x2 / x1))
        # a = (y1 - b) / x1
        # 
        # return a, 1. / b

        # Setup the linear problem
        y = qitotinv / yseam
        x = pcond / (sc.mu_0 * 2 * np.pi * fc * lambdaL * yseam)
        
        def fitfun(x, a, b):
            return a * x + b
        def fitfuncons(x, a, b):
            """
            Crediting stackoverflow for this idea, adding a penalty to the fit
            function to zero out the terms to be conserved:

            https://stackoverflow.com/questions/16541171/
            how-do-i-put-a-constraint-on-scipy-curve-fit

            """
            dwalls, dseam = self.get_dwalls_dseam(a, 1./b, pcond, yseam,
                    lambdaL, fc)
            penalty = np.sum([(dw + ds - qit) * 10000 for dw, ds, qit in
                zip(dwalls, dseam, qitotinv)])
            return a * x + b + penalty

        p0 = (1e-6, 1e3)
        if qierr is not None:
            qiinverr = qierr * qitotinv**2
        #     popt, pcov = curve_fit(fitfun, x, y, p0=p0, sigma=qiinverr,
        #             absolute_sigma=True)
        # else:
        #     popt, pcov = curve_fit(fitfun, x, y)
        bounds = ((0., 0.), (1e-2, 1e9))
        if constrain_fit:
            popt, pcov = curve_fit(fitfuncons, x, y, p0=p0, bounds=bounds)
        else:
            popt, pcov = curve_fit(fitfun, x, y, p0=p0, bounds=bounds)

        # Extract the parameters and errors
        Rs = popt[0]
        gseam = 1. / popt[1]

        err = np.sqrt(np.diag(pcov))
        Rs_err = err[0]
        gseam_err = err[1] * gseam**2

        if plot_fit and fnames:
            # Plot the Qi vs. yseam data
            self.plot_qi_vs_yseam(yseam, 1./qitotinv, fnames[0], qierr=qierr)

            # Plot the Qi^-1 vs. yseam data with the fit line
            yseams = np.linspace(yseam.min(), yseam.max(), 1001)
            xs = np.linspace(x.min(), x.max(), 1001)

            # Include the denser yseam data, the fit data, and the confidence
            # intervals from the fit covariance matrix
            yfit = yseams * fitfun(xs, *popt)
            yupper = yseams * fitfun(xs, *(popt + err))
            ylower = yseams * fitfun(xs, *(popt - err))
            Rs_str = self.format_error_strings(r'R_s', Rs, Rs_err)
            gseam_str = self.format_error_strings(r'g_{\mathrm{seam}}', gseam,
                                       gseam_err)
            fit_str = Rs_str + '\n' + gseam_str
            fitline = {'yseam' : yseams, 'yfit' : yfit, 'ylower' : ylower,
                    'yupper' : yupper, 'fit_str' : fit_str}

            self.plot_qinv_vs_yseam(yseam, qitotinv, fnames[1], 
                    qiinverr=qiinverr, fitline=fitline)

        return Rs, gseam, Rs_err, gseam_err

    def plot_qi_vs_yseam(self, yseam : np.ndarray, qi : np.ndarray, fname : str,
            qierr : np.ndarray = None):
        """
        Plot the total internal quality factor vs. yseam
        """
        # Plotting commands --- plot 1./qi - 1./qi[0] with error bars, the fit
        # function, and prediction intervals
        myplt = MPLPlotWrapper()
        colors = myplt.get_set_alpha_color_cycler(alpha=1.)
        color = colors[0]
        myplt.xlabel = r'$y_{\mathrm{seam}}$ [S/m]'
        myplt.ylabel = r'$Q_{i,\mathrm{tot}}$'
        if qierr is not None:
            myplt.ax.errorbar(yseam, qi, yerr=qierr, ls='', marker='o', ms=10,
                    color=colors[0], capsize=5)
        else:
            myplt.plot(yseam, qi, ls='', marker='o', ms=10, color=colors[0])
        myplt.xscale = 'log'
        myplt.yscale = 'log'
        myplt.set_leg_hdls_lbs()
        myplt.write_fig_to_file(fname)

    def plot_qinv_vs_yseam(self, yseam : np.ndarray, qinv : np.ndarray,
                            fname : str, qiinverr : np.ndarray = None,
                            fitline : dict = None):
        """
        Plot the total loss vs. yseam
        """
        # Plotting commands --- plot 1./qi - 1./qi[0] with error bars, the fit
        # function, and prediction intervals
        myplt = MPLPlotWrapper()
        colors = myplt.get_set_alpha_color_cycler(alpha=1.)
        color = colors[0]
        myplt.xlabel = r'$y_{\mathrm{seam}}$ [S/m]'
        myplt.ylabel = r'$Q^{-1}_{i,\mathrm{tot}}$'
        if qiinverr is not None:
            myplt.ax.errorbar(yseam, qinv, yerr=qiinverr, ls='', marker='o',
                    ms=10, color=colors[0], capsize=5)
            if fitline is not None:
                myplt.plot(fitline['yseam'], fitline['yfit'], ls='-',
                        color=colors[0], label=fitline['fit_str'])
                myplt.ax.fill_between(fitline['yseam'], fitline['ylower'], 
                        fitline['yupper'], alpha=0.25, color=colors[0])
        else:
            myplt.plot(yseam, qi, ls='', marker='o', ms=10, color=colors[0])
        myplt.xscale = 'linear'
        myplt.yscale = 'linear'
        myplt.set_leg_hdls_lbs()
        myplt.write_fig_to_file(fname)

    def get_dwalls_dseam(self, Rs, gseam, pcond, yseam, lambdaL, fc):
        """
        Computes the wall and seam losses of the resonance at fc
        """
        dwalls = pcond * Rs / (2 * np.pi * fc * sc.mu_0 * lambdaL)
        dseam = yseam / gseam

        return dwalls, dseam

    def plot_qi_vs_temperature(self, filename : str, plot_fit : bool = True,
            use_yerrs : bool = True, use_alpha_sim : bool = False):
        """
        Plot the quality factor as a function of temperature, with its fit
        """
        # Get the data from the class
        Qi = self.Qi
        oQi = 1. / Qi  - 1. / Qi[0]
        Qierr = self.Qierr
        oQierr = Qierr / Qi**2
        T_dense, oQipred, upper, lower, label = self.fit_qi_vs_temperature(
                use_alpha_sim=use_alpha_sim)
        print(label)

        # Plotting commands --- plot 1./Qi - 1./Qi[0] with error bars, the fit
        # function, and prediction intervals
        myplt = MPLPlotWrapper()
        colors = myplt.get_set_alpha_color_cycler(alpha=1.)
        color = colors[0]
        myplt.xlabel = 'Temperature (K)'
        myplt.ylabel = r'$\delta\frac{1}{Q_i}$'
        myplt.ax.set_ylim(0.9 * np.min(oQi), 1.1 * np.max(oQi))
        if use_yerrs:
            myplt.ax.errorbar(self.temperatures, oQi, xerr=self.Terr,
                    yerr=oQierr, marker='o', ms=10, capsize=5, color=color,
                    ls='')
        else:
            myplt.ax.errorbar(self.temperatures, oQi, xerr=self.Terr,
                marker='o', ms=10, capsize=5, color=color, ls='')
        if plot_fit:
            myplt.plot(T_dense, oQipred, ls='-', lw=2, color=color, label=label)
        # myplt.ax.fill_between(T_dense, oQipred-lower, y2=oQipred+upper,
        #         color=color, alpha=0.5)
        myplt.set_leg_hdls_lbs()
        myplt.write_fig_to_file(filename)

    def plot_fc_vs_temperature(self, filename : str, plot_fit : bool = True,
            use_yerrs : bool = True, use_alpha_sim : bool = False):
        """
        Plot the frequency as a function of temperature, with its fit
        """
        # Get the data from the class
        fc = self.fc
        ofc = (fc - fc[0]) / fc[0] 
        fcerr = self.fcerr
        ofcerr = fcerr / fc[0]
        T_dense, ofcpred, upper, lower, label = self.fit_fc_vs_temperature(
                use_alpha_sim=use_alpha_sim)
        print(label)

        # Plotting commands --- plot 1./Qi - 1./Qi[0] with error bars, the fit
        # function, and prediction intervals
        myplt = MPLPlotWrapper()
        colors = myplt.get_set_alpha_color_cycler(alpha=1.)
        color = colors[0]
        myplt.xlabel = 'Temperature (K)'
        myplt.ylabel = r'$\frac{\delta f}{f}$'
        if use_yerrs:
            myplt.ax.errorbar(self.temperatures, ofc, xerr=self.Terr, yerr=ofcerr,
                marker='o', ms=10, capsize=5, color=color, ls='')
        else:

            myplt.ax.errorbar(self.temperatures, ofc, xerr=self.Terr,
                marker='o', ms=10, capsize=5, color=color, ls='')
        if plot_fit:
            myplt.plot(T_dense, ofcpred, ls='-', lw=2, color=color, label=label)
        # myplt.ax.fill_between(T_dense, ofcpred-lower, y2=ofcpred+upper,
        #         color=color, alpha=0.5)
        myplt.set_leg_hdls_lbs()
        myplt.write_fig_to_file(filename)

    def plot_zs_vs_temperature(self, filename : str):
        """
        Plot Zs = Rs + j Xs, as a debugging option
        """
        # Get the data from the class
        fc = self.fc
        T = self.temperatures

        # Plotting commands --- plot 1./Qi - 1./Qi[0] with error bars, the fit
        # function, and prediction intervals
        myplt = MPLPlotWrapper()
        colors = myplt.get_set_alpha_color_cycler(alpha=1.)
        color = colors[0]
        Tc = 1.2
        Zs = np.array([self.surface_impedance(TT, Tc, fc[0]) for TT in T])
        myplt.xlabel = 'Temperature (K)'
        myplt.plot(T,(Zs.real - Zs[0].real) / Zs[0].imag, ls='-', lw=2,
                color=colors[0])
        myplt.ylabel = r'$\frac{\delta R_s}{X_s}$'
        ax2 = myplt.ax.twinx()
        ax2.plot(T, -(Zs.imag - Zs[0].imag) / Zs[0].imag, ls='-', lw=2,
                color=colors[1])
        ax2.set_ylabel(r'$-\frac{\delta X_s}{X_s}$',
                fontsize=myplt.fsize)
        myplt.set_leg_hdls_lbs()
        myplt.write_fig_to_file(filename)
