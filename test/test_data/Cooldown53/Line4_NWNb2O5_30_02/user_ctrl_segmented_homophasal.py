# %% -*- encoding: utf-8 -*-

#### press CTRL+K CTRL+2 to close all methods
#### press CTRL+K CTRL+J to open all
# %%
"""
User file for controlling the Janis and PNA instruments

    Make sure you login to the JetWay session on PuTTY
    with user: 'bco' and password: 'aish8Hu8'

"""
import sys
import time
import os
# Change this path
sys.path.append(r'C:\Users\Lehnert Lab\GitHub\bcqt-ctrl\temperature_control')
sys.path.append(r'C:\Users\Lehnert Lab\GitHub\bcqt-ctrl\pna_control')
sys.path.append(r'C:\Users\Lehnert Lab\GitHub\bcqt-ctrl\instrument_control')
from janis_ctrl import measure_multiple_resonators
import numpy as np
import matplotlib.pyplot as plt

def visualize_data(filenames, data_dirs, fcs, spans, powers, add_zero_lines=True,
                   fscale=1e9, filetype='pdf', show_plots=True, plot_dir=None):
    
    sparam = 'S21'
    # Determine number of frequency segmentsf
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
                check_and_make_dir(plot_dir+data_dir)
                savepath = plot_dir + data_dir + f'{fname}_check.{filetype}'.replace('.csv','')
                fig.savefig(savepath, format=filetype)
                print("saved check plot at: ", savepath)
            
            if show_plots is not True:
                plt.close('all')
                

def check_and_make_dir(directory_name):
    if not os.path.exists(directory_name):
        print(f'      checking directory: {directory_name}')
        print(f'      Does not exist. Making new directory.')
        os.makedirs(directory_name)
    else: print(f"      directory: {directory_name} \n      already exists.")
    
    return
          
          
def make_filenames(powers, freq_strs, temperatures, sample_name, seg_str, tag=""):
    fnames = []
    for freq in freq_strs:
        fnames.append([f'{tag}{sample_name}_{freq}_{int(p):02d}dB_{tstr*1e3:.0f}mK_{seg_str}.csv' 
              for p, tstr in zip(powers, temperatures)])
    return fnames


def print_text_block(tStart, num_avgs, IFBW_kHz, num_pts, num_powers=1):
    tElapsed = (time.time() - tStart)/60
    tTraceTime = tElapsed/num_powers
    print("\n\n")
    print("======================================================")
    print(f"    Elapsed time = {tElapsed:1.2f} mins ({tElapsed*60:1.0f} seconds)")
    print(f"      secs/power = {tTraceTime:1.2f} mins/power ({tTraceTime*60:1.0f} seconds)")
    print(f"                     for:  {num_powers} power(s)")
    print(f"                           {num_avgs} averages")
    print(f"                           {IFBW_kHz:1.1f} KHz IF bandwidth")
    print(f"                           {num_pts} pre-segment points")
    print("======================================================")
    print("\n\n")
    return

####################################################################
####################################################################
####################################################################

# TODO: create metadata file that shares these parameters with all user files in the folder
# XXX: Set the center frequencies (GHz), spans (MHz), delays(ns), and temperature
# fcs = [ 5.805, 5.850, 5.905, 5.940, 
        #  6.359, 6.396, 6.444, 6.499]
fcs = [ 5.75247 ]
# 
# 
        
#############################################################################
####################   30 DB ATTENUATOR ADDED 5/23 5pm  #####################
#############################################################################

spans = [15]*len(fcs)
delays = [77.905]*len(fcs)  
temperature = [11e-3]    

cur_dir = os.getcwd()
base_dir = os.path.basename(cur_dir)
line_num = base_dir[:5]
sample_name = base_dir[6:]  # get rid of "LineX_ from the beginning of folder name"

print("\n====================================================================================")
print( "                     Verify sample_name & directory:")
print(f"                     directory = {base_dir}")
print(f"                     sample_name = {sample_name}")
print(f"                     on line number = {line_num}")
print("====================================================================================\n")
time.sleep(1)

# XXX: pick type of segments, if any
is_segmented = True

# segment_option = None  #'hybrid', 'homophasal', or None for regular [5, 41, 5] distribution
segment_option = 'homophasal'
seg_str = segment_option if segment_option != 'None' else ""
    
# create strings for the filename frequencies and data directories 
freq_strs = [f'{fc:.3f}GHz'.replace('.', 'p') for fc in fcs]
data_dirs = [f'{sample_name}_{freq_str}\\' for freq_str in freq_strs]


# %% high power scan
high_powers_floats = np.arange(-15, -30, -2)  # about 70 secs/power
high_powers = [int(x) for x in high_powers_floats]
temperatures = temperature*len(high_powers)
HPow_num_avgs = 300
HPow_IFBW_kHz = 1.0
HPow_num_pts = 51
HPow_fnames = make_filenames(high_powers, freq_strs, temperatures, sample_name, seg_str, tag="")
HPow_fnames = np.array(HPow_fnames).flatten()
HPow_elapsed_time = []

tStart_high = time.time()
measure_multiple_resonators(fcs, spans, delays, high_powers,
                             ifbw = HPow_IFBW_kHz,  npts = HPow_num_pts,
                             Navg_init = HPow_num_avgs, file_names = HPow_fnames, 
                             data_dirs = data_dirs, sparam='S21',
                             adaptive_averaging = False, sample_name = sample_name,
                             runtime = 1., cal_set = None, start_delay = 0.,
                             Noffres = 20, Nf = 100, offresfraction = 0.8, 
                             is_segmented = is_segmented,  bypass_janis = True,
                             segment_option = segment_option, seg_str=seg_str,)
HPow_elapsed_time.append(time.time() - tStart_high)

print_text_block(tStart_high, HPow_num_avgs, HPow_num_pts, HPow_IFBW_kHz, num_powers=len(high_powers))
visualize_data(HPow_fnames, data_dirs, fcs, spans, high_powers, add_zero_lines=False, filetype='png', show_plots=True, plot_dir='.\\plots\\')


# %% mid power scan
med_powers_floats = np.arange(-30, -45, -4)  # 500 secs per power
med_powers = [int(x) for x in med_powers_floats]
temperatures = temperature*len(med_powers)
MPow_num_avgs = 600
MPow_IFBW_kHz = 1.0
MPow_num_pts = 51
MPow_fnames = make_filenames(med_powers, freq_strs, temperatures, sample_name, seg_str, tag="")
MPow_fnames = np.array(MPow_fnames).flatten()
MPow_elapsed_time = []

tStart_med = time.time()
measure_multiple_resonators(fcs, spans, delays, med_powers,
                            ifbw = MPow_IFBW_kHz,  npts = MPow_num_pts,
                            Navg_init = MPow_num_avgs, file_names = MPow_fnames, 
                            data_dirs = data_dirs, sparam='S21',
                            adaptive_averaging = False, sample_name = sample_name,
                            runtime = 1., cal_set = None, start_delay = 0.,
                            Noffres = 20, Nf = 100, offresfraction = 0.8, 
                            is_segmented = is_segmented,  bypass_janis = True, 
                            segment_option = segment_option, seg_str=seg_str,)
MPow_elapsed_time.append(time.time() - tStart_med)

print_text_block(tStart_med, MPow_num_avgs, MPow_num_pts, MPow_IFBW_kHz, num_powers=len(med_powers))
visualize_data(MPow_fnames, data_dirs, fcs, spans, med_powers, add_zero_lines=False, filetype='png', show_plots=True, plot_dir='.\\plots\\')

print(time.strftime("%a, %b %d, %H:%M %p"))
# %% low power scan
#############################################################################
####################   VNA DOES NOT GO BELOW -90 DBM  #######################
#############################################################################

low_powers_floats = np.arange(-50, -60, -5)  # about 8 mins/power
low_powers = [int(x) for x in low_powers_floats]
temperatures = temperature*len(low_powers)
LPow_num_avgs = 600
LPow_IFBW_kHz = 0.05
LPow_num_pts = 51
LPow_fnames = make_filenames(low_powers, freq_strs, temperatures, sample_name, seg_str, tag="")
LPow_fnames = np.array(LPow_fnames).flatten()
LPow_elapsed_time = []
 
tStart_low = time.time()
measure_multiple_resonators(fcs, spans, delays, low_powers,
                            ifbw = LPow_IFBW_kHz,  npts = LPow_num_pts,
                            Navg_init = LPow_num_avgs, file_names = LPow_fnames, 
                            data_dirs = data_dirs, sparam='S21',
                            adaptive_averaging = False, sample_name = sample_name,
                            runtime = 1, cal_set = None, start_delay = 0,
                            Noffres = 30, Nf = 100, offresfraction = 0.8, 
                            is_segmented = is_segmented,  bypass_janis = True, 
                            segment_option = segment_option, seg_str=seg_str,)
LPow_elapsed_time.append(time.time() - tStart_low)

print_text_block(tStart_low, LPow_num_avgs, LPow_num_pts, LPow_IFBW_kHz, num_powers=len(low_powers))
visualize_data(LPow_fnames, data_dirs, fcs, spans, low_powers, add_zero_lines=False, filetype='png', show_plots=True, plot_dir='.\\plots\\')

print(time.strftime("%a, %b %d, %H:%M %p"))
# %% time analysis
N_res = 7 #len(fcs)

T_A = 60 #HPow_elapsed_time[0]/N_res
T_B = 100
T_C = 10*T_B #LPow_elapsed_time[0]/N_res  

N_A = len(high_powers);  N_B = len(med_powers);  N_C = len(low_powers)
A = T_A/N_A;             B = T_B/N_B;            C = T_C/N_C

T_mins = N_res*(T_A+T_B+T_C)/60

total_elapsed_time = (time.time() - tStart_high)/60

print(f"Measurement time analysis")
print(f" T_res =   ({N_A} high powers * {A:1.2f} secs/ea)")
print(f"            + ({N_B} med powers * {B:1.2f} secs/ea)")
print(f"            + ({N_C} low powers * {C:1.2f} secs/ea) \n")
print(f" T_res  = ({T_A/60:1.2f} mins + {T_B/60:1.2f} mins + {T_C/60:1.2f} mins) = {T_mins:1.2f} mins/res")
print(f" T_res * {N_res} resonators = {T_mins*N_res:1.2f} mins ({T_mins*N_res/60:1.2f} hrs)")

print(time.strftime("%a, %b %d, %H:%M %p"))
print(f"\ntime.time() reported elapsed time: {total_elapsed_time:1.2f} mins ({total_elapsed_time/60:1.2f} hrs)")
# %%
