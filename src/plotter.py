import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import logging
from scresonators.src.fit_methods import DCM

class Plotter:
    # TODO separate some of the functionalities in 'plot()' to other methods
    # TODO write docstrings for all methods
    # TODO implement utility for different fitting methods (since each might have different parameters)
    # TODO use only one legend, not three
    def __init__(self, freqs: np.ndarray, cmplx_data: np.ndarray, cmplx_fit=None, fit_params=None):
        """
        Initializes the Plotter instance attributes with experimental data and fit data (coming from an instance of Fitter). 
        Ensures data is input and is of the correct type. 

        Args:
            freqs (np.ndarray): Frequency data (Hz), an array of floats
            cmplx_data (np.ndarray): Experimental complex S21 data, an array of complex numbers
            cmplx_fit (np.ndarray): Fitted complex S21 data, an array of complex numbers
            fit_params (instance of 'lmfit.parameter.Parameters' class): contains fitted parameter data such as 'Q' (internal quality factor),
                        'Qc' (coupling quality factor), 'w1' (resonant frequency), 'phi' (asymmetry angle)  
        """

        # TODO Implement cmplx_data as normalized experimental data (how much of the preprocessing and/or fitting processes does this consider?)
        # At the moment this file was built with the idea that cmplx_data is precisely from the experimental measurement, which is not what we want.

        # Check if any input data is missing
        # Add parameter name to a list 'missing_data' if it is missing
        missing_data = []
        if freqs is None:
            missing_data.append("freqs")
        if cmplx_data is None:
            missing_data.append("cmplx_data")

        # Check if 'missing_data' is 'True'
        if missing_data: 
            # if any one of the inputs is missing, display which input(s) are missing in the error
            raise logging.warning(f"Missing data: {', '.join(missing_data)}")
        else:
            # Initialize attributes needed/helpful to plot
            self.freqs = freqs
            self.cmplx_data = cmplx_data
            self.amps_linear = np.abs(self.cmplx_data)
            self.amps_dB = 20*np.log10(self.amps_linear)
            self.phases = np.angle(self.cmplx_data)

            self.cmplx_fit = cmplx_fit

        # Check if each argument is a NumPy array
        if not isinstance(freqs, np.ndarray):
            raise TypeError("freqs must be a NumPy array")
        if not isinstance(cmplx_data, np.ndarray):
            raise TypeError("cmplx_data must be a NumPy array")


    def plot_before_fit(self, **kwargs):
        # Keyword arguments
        figure_title = kwargs.get('figure_title', 'S21 data')

        layout = [
        ["main", "main", "mag"],
        ["main", "main", "ang"],
        ]

        fig, ax_dict = plt.subplot_mosaic(layout, figsize=(12, 8))
        
        # Complex circle Plot
        ax = ax_dict["main"]
        ax.plot(self.cmplx_data.real, self.cmplx_data.imag, '.', label="normalized data")
        ax.set_xlabel("Re[$S_{21}$]")
        ax.set_ylabel("Im[$S_{21}$]")
        ax.axhline(y=0, color='black', linewidth=1)
        # ax.axvline(x=1, color='black', linewidth=1) ## add functionality to enable/disable either vertical or horizontal lines?
        # ax.set_xlim(-0.05, 1.05)
        # ax.set_ylim(-0.55, 0.55)
        ax.set_title("Complex Plane")
        ax.legend()

        # Magnitude plot
        ax_mag = ax_dict["mag"]
        ax_mag.plot(self.freqs, self.amps_dB, '.', label="normalized data")
        ax_mag.set_xlabel("$(f - f_c)$ [kHz]")
        ax_mag.set_ylabel("Mag[$S_{21}$] (dB)")
        ax_mag.set_title("Log Magnitude Plot")
        ax_mag.xaxis.set_major_formatter(ticker.FuncFormatter(self._formatter_func()))
        ax_mag.legend()

        # Phase plot
        ax_ang = ax_dict["ang"]
        ax_ang.plot(self.freqs, self.phases, '.', label="normalized data")
        ax_ang.set_xlabel("$(f - f_c)$ [kHz]")
        ax_ang.set_ylabel("Ang[$S_{21}$] (rad)")
        ax_ang.set_title("Phase Plot")
        ax_ang.xaxis.set_major_formatter(ticker.FuncFormatter(self._formatter_func()))
        ax_ang.legend()

        # Adjust the bottom, left, right, and top margins
        plt.tight_layout(rect=[0, 0, 1, 0.95]) 
        fig.suptitle(figure_title, fontsize=16)
        

    def plot_dcm(self, normalized_cmplx_data, fit_params, linear=False, **kwargs):
        # TODO Make cmplx_fit within the Plotter class such that it has more points
        # (so it actually looks like a circle when the resonance doesn't have a ton of points)   
        """
        This method will plot fitted experimental data on a complex plane, frequency vs magnitude, frequency vs phase, 
        and will show fitted parameters with associated standard errors.

        Args:
            cmplx_fit (np.ndarray of complex numbers): fitted complex S21 data 

            fit_params (instance of a class): 
                            contains dcm fit parameters 
                            (internal Q, coupling Q, asymmetry phi, resonant frequency)

            linear (boolean): a flag used to indicate whether or not plots 
                            should be displayed in linear scale or not. 
                            Should be set to True if one wants to see a linear plot. 
                            Defaults to false (logarithmic scale).
        """
        # TODO Fix star symbol on resonance: on all 3 plots (complex plane, magnitudes, phases)

        if normalized_cmplx_data is None:
            raise ValueError("Insert normalized complex data from fitting results")
        if fit_params is None:
            raise ValueError("Insert fit parameters from fitting results")

        num_fit_points = kwargs.get('num_fit_points', 1000)
        high_res_freqs, high_res_cmplx_fit = self._generate_cmplx_fit(self.freqs, fit_params, num_fit_points)

        # Subtract resonant frequency from all freqs data
        resonant_freq = fit_params['w1'].value
        self.normalized_freqs = self.freqs - resonant_freq
        normalized_highres_freqs = high_res_freqs - resonant_freq

        # Define loaded quality factor value and propagated standard error
        Ql_value, Ql_stderr = self.calculate_Ql(fit_params['Q'].value, fit_params['Q'].stderr, fit_params['Qc'].value, fit_params['Qc'].stderr)
        
        # Get the complex value at the resonant frequency for the resonance star
        resonant_complex_value = np.interp(resonant_freq, high_res_freqs, high_res_cmplx_fit)
        resonant_magnitude_value = np.abs(resonant_complex_value)
        resonant_phase_value = np.angle(resonant_complex_value)

        layout = [
        ["main", "main", "mag"],
        ["main", "main", "ang"],
        ["main", "main", "text"]
        ]

        fig, ax_dict = plt.subplot_mosaic(layout, figsize=(12, 8))
        
        # Complex circle Plot
        ax = ax_dict["main"]
        # Plot experimental data after preprocessing
        ax.plot(normalized_cmplx_data.real, normalized_cmplx_data.imag, '.', label="normalized data")
        # Plot complex S21 data with higher resolution than experimental data
        ax.plot(high_res_cmplx_fit.real, high_res_cmplx_fit.imag, label="fit function")
        # Plot a star at the resonance
        ax.plot(resonant_complex_value.real, resonant_complex_value.imag, 'r*', markersize=15, label="resonant frequency")
        ax.set_xlabel("Re[$S_{21}$]")
        ax.set_ylabel("Im[$S_{21}$]")
        ax.axhline(y=0, color='black', linewidth=1)
        ax.axvline(x=1, color='black', linewidth=1)
        ax.set_title("Complex Plane")

        if linear: # TODO test this if...else block
            # Linear magnitude plot
            ax_mag = ax_dict["mag"]
            ax_mag.plot(self.normalized_freqs, np.abs(normalized_cmplx_data), '.', label="normalized data")
            ax_mag.plot(normalized_highres_freqs, np.abs(high_res_cmplx_fit), label="fit function")
            ax_mag.plot(0, 20 * np.log10(resonant_magnitude_value), 'r*', markersize=15, label="resonant frequency")
            ax.plot(fit_params['w1'].value, )
            ax_mag.set_xlabel("$(f - f_c)$ [kHz]")
            ax_mag.set_ylabel("Mag[$S_{21}$]")
            ax_mag.set_title("Linear Magnitude Plot")
            ax_mag.xaxis.set_major_formatter(ticker.FuncFormatter(self._formatter_func()))
        else:
            # Logarithmic magnitude plot 
            ax_mag = ax_dict["mag"]
            ax_mag.plot(self.normalized_freqs, 20 * np.log10(np.abs(normalized_cmplx_data)), '.', label="normalized data")
            ax_mag.plot(normalized_highres_freqs, 20 * np.log10(np.abs(high_res_cmplx_fit)), label="fit function")
            ax_mag.plot(0, 20 * np.log10(resonant_magnitude_value), 'r*', markersize=15, label="resonant frequency")
            ax.plot(fit_params['w1'].value, )
            ax_mag.set_xlabel("$(f - f_c)$ [kHz]")
            ax_mag.set_ylabel("Mag[$S_{21}$] (dB)")
            ax_mag.set_title("Log Magnitude Plot")
            ax_mag.xaxis.set_major_formatter(ticker.FuncFormatter(self._formatter_func()))

        # Phase plot
        ax_ang = ax_dict["ang"]
        ax_ang.plot(self.normalized_freqs, np.angle(normalized_cmplx_data), '.', label="normalized data")
        ax_ang.plot(normalized_highres_freqs, np.angle(high_res_cmplx_fit), label="fit function")
        ax_ang.plot(0, resonant_phase_value, 'r*', markersize=15, label="resonant frequency")
        ax_ang.set_xlabel("$(f - f_c)$ [kHz]")
        ax_ang.set_ylabel("Ang[$S_{21}$] (rad)")
        ax_ang.set_title("Phase Plot")
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
        ax_mag.set_xlim([min(self.normalized_freqs), max(self.normalized_freqs)])
        ax_ang.set_xlim([min(self.normalized_freqs), max(self.normalized_freqs)])

        # Adjust the bottom, left, right, and top margins
        plt.tight_layout(rect=[0, 0, 1, 0.95]) 

        # Set whole figure title with keyword argument 'figure_title' if wanted different
        figure_title = kwargs.get('figure_title', 'Fitted $S_{21}$ Data')
        fig.suptitle(figure_title, fontsize=16)

        # TODO Separate this as a different method
        # TODO implement functionality with other fit methods
        def scientific_notation(v, err, rnd=2):
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

        # Text box
        ax_text = ax_dict["text"]
        ax_text.axis("off")

        textstr = '\n'.join((
            f'$Q_l: {scientific_notation(Ql_value, Ql_stderr)}$',
            f'$Q_i: {scientific_notation(fit_params["Q"].value, fit_params["Q"].stderr)}$',
            f'$Q_c: {scientific_notation(fit_params["Qc"].value, fit_params["Qc"].stderr)}$',
            f'$\phi: {scientific_notation(fit_params["phi"].value, fit_params["phi"].stderr)}$ radians',
            f'$f_c: {scientific_notation(fit_params["w1"].value/1e9, fit_params["w1"].stderr/1e9, 9)}$ GHz'
        ))

        ax_text.text(0.5, 0.5, textstr, fontsize=12, verticalalignment='center', horizontalalignment='center', transform=ax_text.transAxes)


         # Collect all handles and labels from each subplot
        handles, labels = [], []
        for ax in [ax_dict["main"], ax_dict["mag"], ax_dict["ang"]]:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                if label not in labels:  # Avoid duplicate labels
                    handles.append(handle)
                    labels.append(label)

        # Create a single legend
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.75, 0.95), ncol=3)


    def _formatter_func(self):
        factor = 1e6  # converting to kHz

        def formatter(x, pos):
            return f'{x / factor:.2f}'

        return formatter
    
    def calculate_Ql(self, Q_i, sigma_Qi, Q_c, sigma_Qc):
        # TODO Decide if we need this to be a separate method in Plotter class. What is most efficient?

        # Calculate Ql
        Q_l = (1 / Q_i + 1 / Q_c) ** -1

        # Calculate partial derivatives
        dQl_dQi = -Q_l**2 / Q_i**2
        dQl_dQc = -Q_l**2 / Q_c**2

        # Propagate errors
        sigma_Ql = np.sqrt((dQl_dQi * sigma_Qi)**2 + (dQl_dQc * sigma_Qc)**2) 
        ## The above line of code works if I don't use 'preprocessing_guesses' in 'Fitter.fit'
        ## The line of code does not work if I provide my own 'preprocessing_guesses' in 'Fitter.fit'

        return Q_l, sigma_Ql
    

    def _generate_cmplx_fit(self, freqs, fit_params, num_fit_points):
        dcm_method = DCM()
        
        # Generate a higher-resolution frequency array
        high_res_freqs = np.linspace(min(freqs), max(freqs), num_fit_points)
        # Use the fitted parameters to evaluate the model function at the new high-resolution frequencies
        high_res_fit_data = dcm_method.func(high_res_freqs, fit_params['Q'], fit_params['Qc'], fit_params['w1'], fit_params['phi'])

        return high_res_freqs, high_res_fit_data
