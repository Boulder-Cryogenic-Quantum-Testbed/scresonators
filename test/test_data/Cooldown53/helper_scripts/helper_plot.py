# %%

'''
    helper_plot.py
'''

# print("    loading helper_plot.py")

# %%

import glob, os, sys, time

import matplotlib.pyplot as plt
import regex as re
import numpy as np
import scipy as sp

import helper_misc as hm
import helper_fit as hf


# %%
def plot_S21_data(freq, plot_config_dict, debug=False, **kwargs):

    fig, axes = None, None
    
    keys = kwargs.keys()
    if debug: 
        print("    ~~ printing all method kwargs")
        for key, value in kwargs.items():
            if len(value) < 10:
                print(key, value)
            else:
                print(key, type(value))

    ############ check method kwargs for data loading
    if "real" in keys: # real & imag -> cmpl
        if debug: print("    ~~ Received real & imag dataset")
        real = kwargs["real"]
        imag = kwargs["imag"]
        cmpl = real + 1j*imag
        del kwargs["imag"]
        del kwargs["real"]
    if "complex" in keys: # cmpl -> real & imag
        if debug: print("    ~~ Received complex dataset")
        cmpl = kwargs["complex"]
        real = np.real(cmpl)
        imag = np.imag(cmpl)
        del kwargs["complex"]
    if "magn" in keys: # cmpl -> real & imag
        if debug: print("    ~~ Received magn and phase dataset")
        magn = kwargs["magn"]
        phase = kwargs["phase"]
        # check if phase is in degrees
        if any(phase > 2.1*np.pi) or any(phase < -2.1*np.pi):
            if debug: print("hf.quick_plot_data:  Converting input phase from degrees to radians.")
            phase_deg = kwargs["phase"]
            phase = np.rad2deg(phase_deg)
        cmpl = magn * np.exp(1j * phase)
        real = np.real(cmpl)
        imag = np.imag(cmpl)

    ############ check plot config dictionary
    config_keys = plot_config_dict.keys()
    if debug: 
        print("    ~~ printing all plot_config_dict items ")
        for key, value in plot_config_dict.items():
            if len(value) < 10:
                print(key, value)
            else:
                print(key, type(value))
                
    if "plot_title" in config_keys:
        fig_title = plot_config_dict["plot_title"]     
        
    if "figsize" in config_keys:
        if debug: print("    ~~ Received figsize")
        figsize = plot_config_dict["figsize"] 
    else:
        figsize = (10, 6)
        
    if "mosaic" in config_keys:
        if debug: print("    ~~ Received mosaic")
        mosaic = plot_config_dict["mosaic"]    
    else:
        if "plot_complex" in config_keys:
            mosaic = "AACCC\n BBCCC"
        else:
            mosaic = "AAA\n BBB"
        
    if "markersize" in config_keys:
        markersize = plot_config_dict["markersize"]    
    else:
        markersize = 4
        
    magn = -np.abs(cmpl)
    phase = np.unwrap(np.angle(cmpl))

    fig, axes_dict = plt.subplot_mosaic(mosaic, figsize=(10,6), tight_layout=True)
    ax1, ax2 = axes_dict["A"], axes_dict["B"]
    axes = list(axes_dict.values())
    tick_label_size = 12
    
    ax1.plot(freq, magn, 'ko', markersize=markersize)
    ax1.set_xlabel("Frequency [GHz]", size=14)
    ax1.set_ylabel("S21 [a.u.]", size=14)
    ax1.set_title("Magnitude Data", size=16)
    
    ax2.plot(freq, phase, 'ro', markersize=markersize)
    ax2.set_xlabel("Frequency [GHz]", size=14)
    ax2.set_ylabel("Phase [rad]", size=14)
    ax2.set_title("Phase Data", size=16)
    
    if "plot_complex" in config_keys:
        ax3 = axes_dict["C"]
        ax3.plot(real, imag, 'bo')
        ax3.set_xlabel("Real [a.u.]", size=14)
        ax3.set_ylabel("Imag [a.u.]", size=14)
        ax3.set_title("Complex Data", size=16)
        ax3.set_aspect('equal') 
      
    fig.suptitle(fig_title, size=20)
    
    for ax in axes:
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        
    if "add_zero_lines" in config_keys:
        val = plot_config_dict["add_zero_lines"]
        if debug: print(f"    ~~ Received add_zero_lines: = {val}")
        if val and "plot_complex" in config_keys:
            ax3.axhline(0, linestyle=':', color='k')  
            ax3.axvline(0, linestyle=':', color='k')
    
    if "plot_filename" in config_keys:
        if plot_config_dict["plot_filename"] is not None:
            fig.savefig(plot_config_dict["plot_filename"], format='png')
            
        
    return fig, axes
        
        
  
def visualize_data(filenames, data_dirs, fcs, spans, powers, add_zero_lines=True,
                   fscale=1e9, filetype='pdf', show_plots=True, plot_dir=None):
    
    sparam = 'S21'
    # Determine number of frequency segments
    # if fcs is None:
    #     Nf = int(round((f2 - f1) / freq_step))
    #     center_freqs = [freq_band[0] + (1 + 2*j) * freq_step / 2 for j in range(Nf)]
    # else:
    #     center_freqs = fcs
        
    # Read the data and concatenate
    # freqs  = np.array([])
    # S21mag = np.array([])
    # S21ph  = np.array([])
    
    for data_dir in data_dirs:
        for fname in filenames:
            try:
                data = np.genfromtxt(data_dir + fname, delimiter=',').T
            except FileNotFoundError: 
                print(f"debug: {fname} not in {data_dir}")
                continue
            
            freqs = data[0]
            S21mag = 10**(data[1]/20)  # convert dBmV to mV
            S21ph = np.deg2rad(data[2])  # convert degrees to rad
            S21 = S21mag * np.exp(1j * S21ph)  # combine into complex number
        
            # Read the frequency band and powers used, use f strings to format
            f1 = f'{freqs[0]:1.2f}'.replace('.','p') 
            f2 = f'{freqs[-1]:1.2f}'.replace('.','p')
            
            # Plot the results
            # fig, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True, figsize=(10,6))
            mosaic = "AACCC\n BBCCC"
            fig, axes = plt.subplot_mosaic(mosaic, figsize=(10,6), tight_layout=True)
            ax1, ax2, ax3 = axes["A"], axes["B"], axes["C"]
            fsize = 16
            
            ax1.plot(freqs / fscale, S21mag, 'k.', markersize=4)
            ax2.plot(freqs / fscale, np.unwrap(S21ph), 'r.', markersize=4)
            ax1.set_ylabel(r'$|S_{%s}|$' % sparam[1:], fontsize=fsize)
            ax2.set_xlabel('Frequency [GHz]', fontsize=fsize)
            ax2.set_ylabel(r'$\left< S_{%s} \right.$' % sparam[1:], fontsize=fsize)
            fig.suptitle(fname, fontsize=20)
            
            ax3.plot(np.real(S21), np.imag(S21), 'bo', markersize=6)
            
            if add_zero_lines is True:
                ax3.axhline(0, color='k', linestyle=':', linewidth=2)
                ax3.axvline(0, color='k', linestyle=':', linewidth=2)
                
            ax3.set_aspect('equal')
            ax3.set_ylabel("Imaginary")
            ax3.set_xlabel("Real")
            ax3.set_title("Complex Plane")
            
            # create plot directory
            if plot_dir is not None:
                hm.check_and_make_dir(plot_dir+data_dir)
                savepath = plot_dir + data_dir + f'{fname}_check.{filetype}'.replace('.csv','')
                fig.savefig(savepath, format=filetype)
                print("saved check plot at: ", savepath)
            
            if show_plots is not True:
                plt.close('all')


 
def plot_dataframe(list_of_df_dicts):
    ## Plot the internal and external quality factors separately
    fig_d, ax_d = plt.subplots(1, 1, tight_layout=True)
    
    for df in list_of_df_dicts:
        for idx, (name, qiqc_df) in enumerate(df.items()):
            # #debug plots
            # plt.plot(qiqc_df["navg"], qiqc_df["Qi"])
            # plt.plot(qiqc_df["Power [dBm]"], qiqc_df["Qi"])
            # print(qiqc_df["Power [dBm]"], qiqc_df["Qi"])
            
            fig_d, ax_d = plt.subplots(1, 1, tight_layout=True)
            label = os.path.basename(os.path.basename(os.path.dirname(name)))
            try:
                if label != prev_label:
                    # fig_d, ax_d = plt.subplots(1, 1, tight_layout=True)
                    print("label != prev_label")
                    print(label, prev_label)
            except:
                pass
            prev_label = label
            
            markers = ['o', 'd', '>', 's', '<', 'h', '^', 'p', 'v']
            colors  = plt.rcParams['axes.prop_cycle'].by_key()['color']
            powers = qiqc_df["navg"]
            Qi = qiqc_df["Qi"]
            Qi_err = np.asarray(qiqc_df['Qi error'])
            delta = 1 / Qi
            delta_err = Qi_err / Qi**2
            csize = 5
            
            ax_d.errorbar(powers, delta, yerr=delta_err, marker='d', ls='', color=colors[idx], 
                          ms=10, capsize=csize, label=label)
            ax_d.set_xscale("log")
            ax_d.legend()
            
            
    # return fig_d, ax_d 
  
  
  
def plot_data_file_dict(data_file_dict, plot_config_dict):
    for filename, filepath in data_file_dict.items():
        print(f"\n~~~> Loading {filename} from '{filepath}' ")
        fn = filepath + filename
        data = np.genfromtxt(fn, delimiter=',').T
        freqs = data[0] / 1e9
        magn = data[1]
        phase = data[2]
        cmpl = magn * np.exp(1j * phase)
        
        # if freqs[0] <= 4:  # crop the data arrays to get the peak hiding at 4.7
        #     peak_value = 4.783  # GHz
        #     peak_span = 0.0015  # GHz
        #     peak_idx = np.abs(freqs - peak_value).argmin()
            
        #     # calculate how many indices we need to have to capture peak_span
        #     full_span = freqs[-1] - freqs[0]
        #     peak_span_idx = (peak_span/2) // (full_span/len(freqs))  
        #     lower_idx = int(peak_idx - peak_span_idx)
        #     upper_idx = int(peak_idx + peak_span_idx)
            
        #     freqs = freqs[lower_idx:upper_idx]
        #     magn = magn[lower_idx:upper_idx]
        #     phase = phase[lower_idx:upper_idx]
        #     cmpl = magn * np.exp(1j * phase)
            
             
        
        plot_config_dict["plot_title"] = filename
        # plot_config_dict["mosaic"] = "AAACC\nBBBCC"
        plot_config_dict["figsize"] = (12,8)
        plot_config_dict["markersize"] = 2
        
        fig, axes = hf.quick_plot_S21_data(freqs, plot_config_dict, complex=cmpl) 

        span = (freqs[-1]-freqs[0])
        pks_idx, _ = sp.find_peaks(-1*magn, distance=len(freqs)*0.2, prominence=2)
        pk_freqs = [freqs[idx] for idx in pks_idx]
        
        # TODO: maybe rainbow instead of red peaks? :)
        if len(pk_freqs) <= 5:
            for pk in pk_freqs:
                print(f"     > Resonance at:  {pk:1.9} GHz")
                axes[0].plot(freqs[pks_idx], magn[pks_idx], 'ro', label=f"{pk:1.6f} GHz",
                    markerfacecolor='none', markersize=16, markeredgewidth=2)
            axes[0].legend()
        else:
            axes[0].text(0.2, 0.2, "Too many \npeaks found", color='red', fontsize=16,
                         horizontalalignment='center', verticalalignment='center', transform = axes[0].transAxes)
        
    # return fig, axes


  