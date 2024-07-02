import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class Plotter:
    # TODO separate some of the functionalities in 'plot()' to other methods
    # TODO write docstrings for all methods
    # TODO implement utility for different fitting methods (since each might have different parameters)
    def __init__(self, freqs: np.ndarray, cmplx_data: np.ndarray, cmplx_fit: np.ndarray, fit_params):
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
        if cmplx_fit is None:
            missing_data.append("cmplx_fit")
        if fit_params is None:
            missing_data.append("fit_params")

        # Check if 'missing_data' is 'True'
        if missing_data: 
            # if any one of the inputs is missing, display which input(s) are missing in the error
            raise ValueError(f"Missing data: {', '.join(missing_data)}")
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
        if not isinstance(cmplx_fit, np.ndarray):
            raise TypeError("cmplx_fit must be a NumPy array")

        # Call the plot method within this class, passing along the 'fit_params' instance
        self.plot(fit_params)

    def plot(self, fit_params):   
        # TODO Implement star symbol on resonance: on all 3 plots (complex plane, magnitudes, phases)

        # Subtract resonant frequency from all freqs data
        self.normalized_freqs = self.freqs - fit_params['w1'].value

        # Define loaded quality factor value and propagated standard error
        Ql_value, Ql_stderr = self.calculate_Ql(fit_params['Q'].value, fit_params['Q'].stderr, fit_params['Qc'].value, fit_params['Qc'].stderr)
        
        layout = [
        ["main", "main", "mag"],
        ["main", "main", "ang"],
        ["main", "main", "text"]
        ]

        fig, ax_dict = plt.subplot_mosaic(layout, figsize=(12, 8))
        
        # Complex circle Plot
        ax = ax_dict["main"]
        ax.plot(self.cmplx_data.real, self.cmplx_data.imag, 'o', label="normalized data")
        ax.plot(self.cmplx_fit.real, self.cmplx_fit.imag, label="fit function")
        ax.set_xlabel("Re[$S_{21}$]")
        ax.set_ylabel("Im[$S_{21}$]")
        ax.axhline(y=0, color='black', linewidth=1)
        ax.axvline(x=1, color='black', linewidth=1)
        # ax.set_xlim(-0.05, 1.05)
        # ax.set_ylim(-0.55, 0.55)
        ax.set_title("Complex Plane")
        ax.legend()

        # Magnitude plot
        ax_mag = ax_dict["mag"]
        ax_mag.plot(self.normalized_freqs, self.amps_dB, 'o', label="normalized data")
        ax_mag.plot(self.normalized_freqs, np.abs(self.cmplx_fit), label="fit function")
        ax_mag.set_xlabel("$(f - f_c)$ [kHz]")
        ax_mag.set_ylabel("Mag[$S_{21}$] (dB)")
        ax_mag.set_title("Log Magnitude Plot")
        ax_mag.xaxis.set_major_formatter(ticker.FuncFormatter(self._formatter_func()))
        ax_mag.legend()

        # Phase plot
        ax_ang = ax_dict["ang"]
        ax_ang.plot(self.normalized_freqs, self.phases, 'o', label="normalized data")
        ax_ang.plot(self.normalized_freqs, np.angle(self.cmplx_fit), label="fit function")
        ax_ang.set_xlabel("$(f - f_c)$ [kHz]")
        ax_ang.set_ylabel("Ang[$S_{21}$] (rad)")
        ax_ang.set_title("Phase Plot")
        ax_ang.xaxis.set_major_formatter(ticker.FuncFormatter(self._formatter_func()))
        ax_ang.legend()


        # TODO Implement the text box below with scientific notation  
        # TODO Separate this as a different method
        # Text box
        # ax_text = ax_dict["text"]
        # ax_text.axis("off")
        # textstr = '\n'.join((
        #     f'$Q_l: ({Ql_value:.2e} \pm {Ql_stderr:.2e})$',
        #     f'$Q_i: ({fit_params["Q"].value:.2e} \pm {fit_params["Q"].stderr:.2e})$',
        #     f'$Q_c: ({fit_params["Qc"].value:.2e} \pm {fit_params["Qc"].stderr:.2e})$',
        #     f'$\phi: ({fit_params["phi"].value:.2e} \pm {fit_params["phi"].stderr:.2e})$ radians',
        #     f'$f_c: {fit_params["w1"].value/1e9:.2e} \pm {fit_params["w1"].stderr/1e9:.2e}$ GHz'
        # ))

        # ax_text.text(0.5, 0.5, textstr, fontsize=12, verticalalignment='center', horizontalalignment='center', transform=ax_text.transAxes)

        plt.tight_layout()
        # plt.show()


        # # Helper function to format value and stderr with scientific notation
        # def format_scientific(value, stderr):
        #     base_value = f"{value:.2e}".split('e')[0]
        #     base_stderr = f"{stderr:.2e}".split('e')[0]
        #     exponent = f"{value:.2e}".split('e')[1]
        #     if exponent == 0:
        #         return f'({base_value} \pm {base_stderr})'
        #     else:
        #         return f'({base_value} \pm {base_stderr}) \\times 10^{{{int(exponent)}}}'

        # # Text box
        # ax_text = ax_dict["text"]
        # ax_text.axis("off")

        # textstr = '\n'.join((
        #     f'$Q_l: {format_scientific(Ql_value, Ql_stderr)}$',
        #     f'$Q_i: {format_scientific(fit_params["Q"].value, fit_params["Q"].stderr)}$',
        #     f'$Q_c: {format_scientific(fit_params["Qc"].value, fit_params["Qc"].stderr)}$',
        #     f'$\phi: {format_scientific(fit_params["phi"].value, fit_params["phi"].stderr)}$ radians',
        #     f'$f_c: {format_scientific(fit_params["w1"].value, fit_params["w1"].stderr)}$ GHz'
        # ))

        # ax_text.text(0.5, 0.5, textstr, fontsize=12, verticalalignment='center', horizontalalignment='center', transform=ax_text.transAxes)
        # TODO implement functionality with other fit methods
        # TODO figure out why the frequency is rounding to one decimal place when we want two
        def sci_not(v, err, rnd=2):
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
            f'$Q_l: {sci_not(Ql_value, Ql_stderr)}$',
            f'$Q_i: {sci_not(fit_params["Q"].value, fit_params["Q"].stderr)}$',
            f'$Q_c: {sci_not(fit_params["Qc"].value, fit_params["Qc"].stderr)}$',
            f'$\phi: {sci_not(fit_params["phi"].value, fit_params["phi"].stderr)}$ radians',
            f'$f_c: {sci_not(fit_params["w1"].value/1e9, fit_params["w1"].stderr/1e9, 9)}$ GHz'
        ))

        ax_text.text(0.5, 0.5, textstr, fontsize=12, verticalalignment='center', horizontalalignment='center', transform=ax_text.transAxes)




    # def custom_formatter(self, freqs):
    #     # TODO Figure out if we want to plot in kHz or MHz with a factor
    #     # plot in kHz for now:
    #     factor = 1e6 
        
    #     # Figure out how many decimal points are needed
    #     difference = freqs[1]-freqs[0]
    #     str_difference = str(difference)
    #     if '.' in str_difference:
    #         # Split the string into the integer part and the decimal part
    #         decimal_part = str_difference.split('.')[1]
    #         # Count the number of digits in the decimal part
    #         decimal_places = len(decimal_part)
    #     else:
    #         # No decimal point means 0 decimal places
    #         decimal_places = 0

    #     def format_freqs(freqs, factor, decimal_places):
    #         return [f'{(freq / factor):.{decimal_places}f}' for freq in freqs]
            
    #     formatted_freqs = format_freqs(freqs, factor, decimal_places)

    #     return formatted_freqs


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

        return Q_l, sigma_Ql

















