import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import logging

class Plotter:
    # TODO separate some of the functionalities in 'plot()' to other methods
    # TODO write docstrings for all methods
    # TODO implement utility for different fitting methods (since each might have different parameters)
    # TODO use only one legend, not three
    def __init__(self, freqs: np.ndarray, cmplx_data: np.ndarray, fit_params=None, fit_method=None):
        """
        Initializes the Plotter instance attributes with experimental data and fit data (coming from an instance of Fitter). 
        Ensures data is input and is of the correct type. 
        Args:
            freqs (np.ndarray): Frequency data (Hz), an array of floats

            cmplx_data (np.ndarray): Experimental complex S21 data, an array of complex numbers

            cmplx_fit (np.ndarray): Fitted complex S21 data, an array of complex numbers

            fit_params (instance of 'lmfit.parameter.Parameters' class): contains fitted parameter data such as 'Q' (internal quality factor),
                        'Qc' (coupling quality factor), 'f0' (resonant frequency), 'phi' (asymmetry angle) (can be different for other fit methods)

            fit_method (instance of fit method class specified in user file): contains methods useful for the fitting function utility
        """
        
        # Check if any input data is missing
        # Add parameter name to a list 'missing_data' if it is missing
        missing_data = []
        if freqs is None:
            missing_data.append("freqs")
        if cmplx_data is None:
            missing_data.append("cmplx_data")
        ## TODO Implement fit_method in __init__
        ## The issue here is that if one wants to 'plot_before_fit', the fit hasn't been done yet, so 'fit_method' hasn't been defined yet.
        # if fit_method is None: 
        #     missing_data.append("fit_method")
        # Check if 'missing_data' is 'True'
        
        if missing_data: 
            # if any one of the inputs is missing, display which input(s) are missing in the error
            raise logging.warning(f"Missing data: {', '.join(missing_data)}")
        else:
            # Initialize attributes needed/helpful to plot
            self.freqs_Hz = freqs ## Do we want this here or in plot_dcm?
            self.cmplx_data = cmplx_data ## Do we want this here or in plot_dcm?
            self.amps_linear = np.abs(self.cmplx_data) ## Do we want this here or in plot_dcm?
            self.amps_dB = 20*np.log10(self.amps_linear) ## Do we want this here or in plot_dcm?
            self.phases = np.angle(self.cmplx_data) ## Do we want this here or in plot_dcm?

        # self.freq_factor = self._find_freq_order() TODO consider this idea
        
        # Check if each argument is a NumPy array
        if not isinstance(freqs, np.ndarray):
            raise TypeError("freqs must be a NumPy array")
        if not isinstance(cmplx_data, np.ndarray):
            raise TypeError("cmplx_data must be a NumPy array")
        
        # TODO Implement the ability to read fit_method and call whichever plot method is accordingly needed 
    
    def plot_preprocessing_steps(self, fit_method):
        pass
            

    def plot_before_fit(self, fig, ax_dict, **kwargs):
        # Keyword arguments
        figure_title = kwargs.get('figure_title', 'S21 Experimental Data')
        horiz_line = kwargs.get('horiz_line', True)
        vert_line = kwargs.get('vert_line', False)
        linear = kwargs.get('linear', False)
        
        # Complex circle plot
        ax = ax_dict["main"]
        ax = self._plot_complex_circle(ax, horiz_line=horiz_line, vert_line=vert_line)
        ax.axvline(x=0, color='black', linewidth=1) 
        ax.legend()

        freq_factor = 1e9 ## TODO code this more robust, and consider making freq_factor an attribute of the class
        rotation = 280 
        freqs_ticks, freqs_ticks_labels = self._xticks_setup(5, freq_factor)
        
        # Magnitude plot
        ax_mag = ax_dict["mag"]
        ax_mag = self._plot_magnitudes(ax_mag, freq_factor)
        ax_mag.set_xticks(freqs_ticks/freq_factor, labels=freqs_ticks_labels, rotation=rotation)
        # ax_mag.xaxis.set_major_formatter(ticker.FuncFormatter(self._formatter_func(freq_factor)))
        
        # Phase plot
        ax_ang = ax_dict["ang"]
        ax_ang = self._plot_phases(ax_ang, freq_factor)
        ax_ang.set_xticks(freqs_ticks/freq_factor, labels=freqs_ticks_labels, rotation=rotation)
        # ax_ang.xaxis.set_major_formatter(ticker.FuncFormatter(self._formatter_func(freq_factor)))
        
        # Adjust the bottom, left, right, and top margins
        plt.tight_layout(rect=[0, 0, 1, 0.95]) 
        
        fig.suptitle(figure_title, fontsize=16)

        return fig, ax_dict
        
    def plot(self, normalized_cmplx_data, fit_params, fig, ax_dict, fit_method=None, **kwargs):
        # TODO Make cmplx_fit within the Plotter class such that it has more points
        # (so it actually looks like a circle when the resonance doesn't have a ton of points)   
        """
        This method will plot fitted experimental data on a complex plane, frequency vs magnitude, frequency vs phase, 
        and will show fitted parameters with associated standard errors.
        Args:
            normalized_cmplx_data (np.ndarray of complex numbers): preprocessed experimental S21 data
            fit_params (instance of a class in lmfit): contains dcm fit parameters 
                            (internal Q, coupling Q, asymmetry phi, resonant frequency)
            dcm_method (instance of DCM class): Should be instantiated in a user file. 
                            Contains method 'generate_highres_fit' used in this method.
            keyword arguments:
                linear (boolean): a flag used to indicate whether or not plots 
                    should be displayed in linear scale or not. 
                    Should be set to True if one wants to see a linear plot. 
                    Defaults to false (logarithmic scale).
                num_fit_points: int
                    Number of points for high resolution evaluation of fit function 'dcm_method.generate_highres_fit'. 
                    Defaults to 1000.
                figure_title: str
                    Title of the whole figure, placed on upper middle of figure. Defaults to 'Fitted $S_{21}$ Data'.
                
        """
        if fit_method is None: 
            raise ValueError("Insert instance of fit method class")
        if normalized_cmplx_data is None:
            raise ValueError("Insert normalized complex data from fitting results")
        if fit_params is None:
            raise ValueError("Insert fit parameters from fitting results")
        
        # Keyword arguments
        linear = kwargs.get('linear', False)

        # Use existing 'dcm_method' instance to get higher resolution fitted data
        num_fit_points = kwargs.get('num_fit_points', 1000)
        high_res_freqs, high_res_cmplx_fit = fit_method.generate_highres_fit(self.freqs_Hz, fit_params, num_fit_points)

        # Subtract resonant frequency from all freqs data
        resonant_freq_Hz = fit_params['f0'].value
        normalized_freqs = self.freqs_Hz - resonant_freq_Hz
        normalized_highres_freqs = high_res_freqs - resonant_freq_Hz

        # Get the complex value at the resonant frequency for the resonance star
        resonant_complex_value = np.interp(resonant_freq_Hz, high_res_freqs, high_res_cmplx_fit)
        resonant_magnitude_value = np.abs(resonant_complex_value)
        resonant_phase_value = np.angle(resonant_complex_value)

        ## TODO fit_params may be same format as params in dcm.py. Look into this
        
        # Complex circle plot
        ax = ax_dict["main"]
        ax = self._plot_complex_circle(ax, horiz_line=True, vert_line=True)
        # Plot fitted complex S21 data with higher resolution than experimental data
        ax.plot(high_res_cmplx_fit.real, high_res_cmplx_fit.imag, label="fit function")
        # Plot a star at the resonance
        ax.plot(resonant_complex_value.real, resonant_complex_value.imag, 'r*', markersize=15, label="resonant frequency")
        ax.legend()

        freq_factor = 1e9
        rotation = 280
        freqs_ticks, freqs_ticks_labels = self._xticks_setup(5, freq_factor)

        # Linear/Logarithmic magnitude plot
        ax_mag = ax_dict["mag"]
        ax_mag = self._plot_magnitudes(ax_mag, freq_factor, linear=False)
        # Plot fitted magnitude data with higher resolution than experimental data
        ax_mag.plot(normalized_highres_freqs, (np.abs(high_res_cmplx_fit) if linear else 20 * np.log10(np.abs(high_res_cmplx_fit))), label="fit function")
        # Plot a star at the resonance
        ax_mag.plot(0, (resonant_magnitude_value if linear else 20 * np.log10(resonant_magnitude_value)), 'r*', markersize=15, label="resonant frequency")
        ax_mag.set_xlabel("$(f - f_0)$ [kHz]")
        ax_mag.set_xticks(freqs_ticks/freq_factor, labels=freqs_ticks_labels, rotation=rotation)
        ax_mag.xaxis.set_major_formatter(ticker.FuncFormatter(self._formatter_func()))

        # Phase plot
        ax_ang = ax_dict["ang"]
        ax_ang = self._plot_phases(ax_ang, freq_factor)
        # Plot fitted phase data with higher resolution than experimental data
        ax_ang.plot(normalized_highres_freqs, np.angle(high_res_cmplx_fit), label="fit function")
        # Plot a star at the resonance
        ax_ang.plot(0, resonant_phase_value, 'r*', markersize=15, label="resonant frequency")
        ax_ang.set_xlabel("$(f - f_0)$ [kHz]")
        # ax_ang.set_xticks(freqs_ticks/freq_factor, labels=freqs_ticks_labels, rotation=rotation)
        ax_ang.xaxis.set_major_formatter(ticker.FuncFormatter(self._formatter_func()))

        # Calculate the range of data with padding, set x and y limits for complex circle
        real_min, real_max = min(normalized_cmplx_data.real), max(normalized_cmplx_data.real)
        imag_min, imag_max = min(normalized_cmplx_data.imag), max(normalized_cmplx_data.imag)
        real_range = real_max - real_min
        imag_range = imag_max - imag_min
        padding_percentage = 0.05  # 5% padding
        real_padding = real_range * padding_percentage
        imag_padding = imag_range * padding_percentage
        ax.set_xlim([real_min - real_padding, real_max + real_padding])
        ax.set_ylim([imag_min - imag_padding, imag_max + imag_padding])

        # Set x limits to freqs vs magnitude and vs phase (quadrature) plots
        ax_mag.set_xlim([min(normalized_freqs), max(normalized_freqs)])
        ax_ang.set_xlim([min(normalized_freqs), max(normalized_freqs)])

        # Adjust the bottom, left, right, and top margins
        plt.tight_layout(rect=[0, 0, 1, 0.95]) 

        # Set whole figure title with keyword argument 'figure_title' if wanted different
        figure_title = kwargs.get('figure_title', 'Fitted $S_{21}$ Data')
        fig.suptitle(figure_title, fontsize=16)
        
        # Define internal quality factor value and propagated standard error
        Qi_value, Qi_stderr = self.calculate_Qi(fit_params['Q'].value, fit_params['Q'].stderr, fit_params['Qc'].value, fit_params['Qc'].stderr)

        # Text box with fitted parameters
        ax_text = ax_dict["text"]
        ax_text.axis("off")
        ## TODO generalize the fitted parameters text
        # text_list = []
        # for i in range(len(fit_params)):
            # if fit_params[i]
            # text_list[i] = 
        textstr_text = '\n'.join((
            f'$Q_l: {self._scientific_notation(fit_params["Q"].value, fit_params["Q"].stderr)}$',
            f'$Q_i: {self._scientific_notation(Qi_value, Qi_stderr)}$',
            f'$Q_c: {self._scientific_notation(fit_params["Qc"].value, fit_params["Qc"].stderr)}$',
            f'$\phi: {self._scientific_notation(fit_params["phi"].value, fit_params["phi"].stderr)}$ radians',
            f'$f_0: {self._scientific_notation(fit_params["f0"].value/1e9, fit_params["f0"].stderr/1e9, 9)}$ GHz'
        ))
        # assign units from dcm method

        ax_text.text(0.5, 0.5, textstr_text, fontsize=12, verticalalignment='center', horizontalalignment='center', transform=ax_text.transAxes)

        # TODO: return figure object(s) so that users can edit the plot as they see fit

        return fig, ax_dict

    def _plot_complex_circle(self, ax, **kwargs):
        horiz_line = kwargs.get('horiz_line', True)
        vert_line = kwargs.get('vert_line', False)

        ax.plot(self.cmplx_data.real, self.cmplx_data.imag, '.', label="normalized data")
        
        ax.set_xlabel("Re[$S_{21}$]")
        ax.set_ylabel("Im[$S_{21}$]")
        ax.axhline(y=0, color='black', linewidth=1) if horiz_line else None
        ax.axvline(x=1, color='black', linewidth=1) if vert_line else None
        ax.set_title("Complex Circle")

        return ax

    def _plot_magnitudes(self, ax_mag, freq_factor, **kwargs):
        linear = kwargs.get('linear', False)

        ax_mag.plot(self.freqs_Hz/freq_factor, (np.abs(self.cmplx_data) if linear else 20 * np.log10(np.abs(self.cmplx_data))), '.', label="normalized data")
    
        ax_mag.set_xlabel("Frequency [GHz]")
        ax_mag.set_ylabel("Mag[$S_{21}$]" if linear else "Mag[$S_{21}$] (dB)")
        ax_mag.set_title("Linear Magnitude Plot" if linear else "Logarithmic Magnitude Plot")

        return ax_mag

    def _plot_phases(self, ax_ang, freq_factor):

        ax_ang.plot(self.freqs_Hz/freq_factor, np.angle(self.cmplx_data), '.', label="normalized data")

        ax_ang.set_xlabel("Frequency [GHz]")
        ax_ang.set_ylabel("Ang[$S_{21}$] (rad)")
        ax_ang.set_title("Phase Plot")

        return ax_ang

    def _xticks_setup(self, n, freq_factor):
        # Calculate the indices for `n` evenly spaced frequency markers (evenly spaced in array-space, not frequency-space)
        indices = np.linspace(0, len(self.freqs_Hz) - 1, n, dtype=int)
        freqs_ticks = self.freqs_Hz[indices]
        freqs_ticks_labels = [f'{freq/freq_factor:.4f}' for freq in freqs_ticks]

        return freqs_ticks, freqs_ticks_labels

    @staticmethod
    def _formatter_func(factor=1e6):
        # converting to kHz unless otherwise specified
        def formatter(x, pos):
            return f'{x / factor:.2f}'
        return formatter
    
    @staticmethod
    def calculate_Qi(Q_l, sigma_Ql, Q_c, sigma_Qc):
        # Calculate Qi
        Q_i = (1 / Q_l - 1 / Q_c) ** -1

        # Calculate partial derivatives
        dQi_dQl = Q_i**2 / Q_l**2
        dQi_dQc = -Q_i**2 / Q_c**2

        # Propagate errors
        sigma_Qi = np.sqrt((dQi_dQl * sigma_Ql)**2 + (dQi_dQc * sigma_Qc)**2)
        return Q_i, sigma_Qi
        
    # TODO verify this method's uses with other fit_methods
    @staticmethod
    def _scientific_notation(v, err, rnd=2):
            if v == 0:
                return f'({0:.{rnd}f} ± {0:.{rnd}f})'
            
            power_v = int(np.floor(np.log10(abs(v))))
            power_err = int(np.floor(np.log10(abs(err))))
            power = power_v if power_v >= power_err else power_err
            v_base = v / (10 ** power)
            err_base = err / (10 ** power)

            # Ensure at least one non-zero digit for the uncertainty
            err_base_str = f"{err_base:.{rnd}e}"
            err_decimal_places = len(err_base_str.split('.')[1].split('e')[0].lstrip('0'))  # Get the number of significant decimal places
            
            # Adjust rounding to ensure at least one non-zero digit in uncertainty
            while round(err_base, rnd) == 0 and rnd < 10:
                rnd += 1

            # Ensure the value has the same number of decimal places as the adjusted uncertainty
            v_base_str = f"{v_base:.{rnd}f}"
            err_base_str = f"{err_base:.{rnd}f}"
            if power == 0:
                return f'({v_base_str} ± {err_base_str})'
            else:
                return f'({v_base_str} ± {err_base_str}) \\times 10^{{{power}}}'