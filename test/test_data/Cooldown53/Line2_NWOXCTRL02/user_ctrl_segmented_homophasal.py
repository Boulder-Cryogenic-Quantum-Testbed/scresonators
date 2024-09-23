# %% -*- encoding: utf-8 -*-

#### press CTRL+K CTRL+2 to close all methods
#### press CTRL+K CTRL+J to open all
# %%
"""
User file for controlling the Janis and PNA instruments

    Make sure you login to the JetWay session on PuTTY
    with user: 'bco' and password: 'aish8Hu8'

"""
import sys, time, os, datetime

# Change this path
sys.path.append(r'C:\Users\Lehnert Lab\GitHub\bcqt-ctrl\temperature_control')
sys.path.append(r'C:\Users\Lehnert Lab\GitHub\bcqt-ctrl\pna_control')
sys.path.append(r'C:\Users\Lehnert Lab\GitHub\bcqt-ctrl\instrument_control')
from janis_ctrl import measure_multiple_resonators
from janis_ctrl import JanisCtrl

import numpy as np
import matplotlib.pyplot as plt

pathToParent = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pathToParent + "\\scripts\\")
import helper_misc as hm
import helper_user_fit as hf


# %%
def visualize_data(filenames, data_dirs, fcs, spans, powers, add_zero_lines=True,
                   fscale=1e9, filetype='pdf', show_plots=True, plot_dir=None, plot_complex=True):
    
    sparam = 'S21'
    
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
        
            # Plot the results
            if plot_complex is True:
                mosaic = "AACCC\n BBCCC"
            else:
                mosaic = "AAA\n BBB"
                
            fsize = 16
            fig, axes = plt.subplot_mosaic(mosaic, figsize=(10,6), tight_layout=True)
            ax1, ax2 = axes["A"], axes["B"]
            ax1.plot(freqs / fscale, S21mag, 'k.', markersize=4)
            ax2.plot(freqs / fscale, np.unwrap(S21ph), 'r.', markersize=4)
            ax1.set_ylabel(r'$|S_{%s}|$' % sparam[1:], fontsize=fsize)
            ax2.set_xlabel('Frequency [GHz]', fontsize=fsize)
            ax2.set_ylabel(r'$\left< S_{%s} \right.$' % sparam[1:], fontsize=fsize)
            fig.suptitle(fname, fontsize=20)
            
            if plot_complex is True:
                ax3 = axes["C"]            
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
                
def print_text_block(tStart, tEnd, num_avgs, num_pts, IFBW_kHz, num_powers=1, num_resonators=None):
    tElapsed = (tEnd - tStart)/60
    tTraceTime = tElapsed/num_powers
    print("\n\n")
    print("======================================================")
    print(f"    Elapsed time = {tElapsed:1.2f} mins ({tElapsed*60:1.0f} seconds)")
    print(f"      secs/power = {tTraceTime:1.2f} mins/power ({tTraceTime*60:1.0f} seconds)")
    print(f"                     for:  {num_powers} power(s)")
    print(f"                           {num_avgs} averages")
    print(f"                           {IFBW_kHz:1.1f} KHz IF bandwidth")
    print(f"                           {num_pts} pre-segment points")
    if num_resonators is not None:
        print(f"                           {num_resonators} resonators")
    print("======================================================")
    print("\n\n")
    return


def make_filenames(powers, freq_strs, temperatures, sample_name, prefix="", suffix=""):
    fnames = []
    # add underscores to prefix and suffix if they aren't there already
    if len(prefix) >= 1:
        prefix = f"{suffix}_" if prefix[-1] != "_" else prefix
    if len(suffix) >= 1:
        suffix = f"_{suffix}" if suffix[0] != "_" else suffix
   
    for freq in freq_strs:
        fnames.append([f'{prefix}{sample_name}_{freq}_{int(p):02d}dB_{tstr*1e3:.0f}mK{suffix}.csv' 
              for p, tstr in zip(powers, temperatures)])
    return fnames


def read_temp_JCtrl():
    # random bs values, it wont actually do a temp sweep
    Tstart = 0.03; Tstop = 0.315; dT = 0.015
    sample_time = 15; T_eps = 0.0025 # -- 255 mK and up
    therm_time  = 300. # wait an extra 5 minutes to thermalize

    JCtrl = JanisCtrl(Tstart, Tstop, dT,
            sample_time=sample_time, T_eps=T_eps,
            therm_time=therm_time, Nf=32,
            init_socket=True, bypass_janis=False,
            adaptive_averaging=False, output_file=None,
            data_dir=None)
    try:
        output = JCtrl.read_cmn()
        # output = JCtrl.read_temp(channel='Still')
    except Exception as e:
        print("Failed to read_temp, error:  ", e)
        return None
    finally:
        del JCtrl
    
    if any(elem is None for elem in output):
        print("Failed to read_temp, read_temp() returned None's")
        return None
    else:
        return output
    
####################################################################
####################################################################
####################################################################
# TODO: create metadata file that shares these parameters with all user files in the folder
# XXX: Set the center frequencies (GHz), spans (MHz), delays(ns), and temperature
# fcs = [ 4.7830, 5.206077, 5.604985, 5.986677, 6.424400 , 6.854508, 7.271462, 7.721354]
fcs = [4.7830]

dstr = datetime.datetime.today().strftime(r'%m%d%y_%I%M%p')
print(f"Current timestamp: {dstr}")

spans = [1000]*len(fcs)
delays = [76.14]*len(fcs)

temp_reading = read_temp_JCtrl()[1]  # convert to mK

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

    
# create strings for the filename frequencies and data directories 
freq_strs = [f'{fc:.3f}GHz'.replace('.', 'p') for fc in fcs]
data_dirs = [f'{sample_name}_{freq_str}\\' for freq_str in freq_strs]


# %% high power scan
high_powers_floats = np.arange(-15, -50, -3)  # about 2 secs
high_powers = [int(x) for x in high_powers_floats]
HPow_templist = [temp_reading]*len(high_powers)
HPow_segment_option = 'homophasal'
HPow_num_avgs = 3
HPow_IFBW_kHz = 1.0
HPow_num_pts = 201
HPow_fnames = make_filenames(high_powers, freq_strs, HPow_templist, sample_name, prefix="", suffix=HPow_segment_option)
HPow_fnames = np.array(HPow_fnames).flatten()

tStart_high = time.time()
measure_multiple_resonators(fcs, spans, delays, high_powers,
                             ifbw = HPow_IFBW_kHz,  npts = HPow_num_pts,
                             Navg_init = HPow_num_avgs, file_names = HPow_fnames, 
                             data_dirs = data_dirs, sparam='S21',
                             adaptive_averaging = False, sample_name = sample_name,
                             runtime = 1., cal_set = None, start_delay = 0.,
                             Noffres = int(HPow_num_pts*0.2), Nf = int(HPow_num_pts*0.8), offresfraction = 0.8, 
                             is_segmented = HPow_segment_option,  bypass_janis = False,
                             segment_option = HPow_segment_option, seg_str=HPow_segment_option,)
tEnd_high = time.time()
HPow_elapsed_time = tEnd_high - tStart_high

print_text_block(tStart_high, tEnd_high, HPow_num_avgs, HPow_num_pts, HPow_IFBW_kHz, num_powers=len(high_powers), num_resonators=len(fcs))
visualize_data(HPow_fnames, data_dirs, fcs, spans, high_powers, add_zero_lines=False, filetype='png', 
               show_plots=True, plot_complex=True, plot_dir='.\\plots\\')


# %% mid power scan
med_powers_floats = np.arange(-50, -85, -3)    # 7 seconds
med_powers = [int(x) for x in med_powers_floats]
MPow_templist = [temp_reading]*len(med_powers)
MPow_segment_option = 'homophasal'
MPow_num_avgs = 20
MPow_IFBW_kHz = 0.5
MPow_num_pts = 201
MPow_fnames = make_filenames(med_powers, freq_strs, MPow_templist, sample_name, prefix="", suffix=MPow_segment_option)
MPow_fnames = np.array(MPow_fnames).flatten()

tStart_med = time.time()
measure_multiple_resonators(fcs, spans, delays, med_powers,
                            ifbw = MPow_IFBW_kHz,  npts = MPow_num_pts,
                            Navg_init = MPow_num_avgs, file_names = MPow_fnames, 
                            data_dirs = data_dirs, sparam='S21',
                            adaptive_averaging = False, sample_name = sample_name,
                            runtime = 1., cal_set = None, start_delay = 0.,
                            Noffres = int(MPow_num_pts*0.2), Nf = int(MPow_num_pts*0.8), offresfraction = 0.8, 
                            is_segmented = MPow_segment_option,  bypass_janis = False, 
                            segment_option = MPow_segment_option, seg_str=MPow_segment_option,)
tEnd_med = time.time()
MPow_elapsed_time = tEnd_med - tStart_med

print_text_block(tStart_med, tEnd_med, MPow_num_avgs, MPow_num_pts, MPow_IFBW_kHz, num_powers=len(med_powers))
visualize_data(MPow_fnames, data_dirs, fcs, spans, med_powers, add_zero_lines=False, filetype='png', show_plots=True, plot_dir='.\\plots\\')

print(time.strftime("%a, %b %d, %H:%M %p"))
# %% low power scan
#############################################################################
####################   VNA DOES NOT GO BELOW -90 DBM  #######################
#############################################################################

low_powers_floats = np.arange(-84, -91, -2)  # 150? seconds
low_powers = [int(x) for x in low_powers_floats]
LPow_templist = [temp_reading] * len(low_powers)
LPow_segment_option = 'homophasal'
LPow_num_avgs = 75
LPow_IFBW_kHz = 0.05
LPow_num_pts = 201
LPow_fnames = make_filenames(low_powers, freq_strs, LPow_templist, sample_name, prefix="", suffix=LPow_segment_option)
LPow_fnames = np.array(LPow_fnames).flatten()

tStart_low = time.time()
measure_multiple_resonators(fcs, spans, delays, low_powers,
                            ifbw = LPow_IFBW_kHz,  npts = LPow_num_pts,
                            Navg_init = LPow_num_avgs, file_names = LPow_fnames, 
                            data_dirs = data_dirs, sparam='S21',
                            adaptive_averaging = False, sample_name = sample_name,
                            runtime = 1, cal_set = None, start_delay = 0,
                            Noffres = int(LPow_num_pts*0.2), Nf = int(LPow_num_pts*0.8), offresfraction = 0.8, 
                            is_segmented = LPow_segment_option,  bypass_janis = False, 
                            segment_option = LPow_segment_option, seg_str=LPow_segment_option,)
tEnd_low = time.time()
LPow_elapsed_time = tEnd_low - tStart_low

print_text_block(tStart_low, tEnd_low, LPow_num_avgs, LPow_num_pts, LPow_IFBW_kHz, num_powers=len(low_powers))
visualize_data(LPow_fnames, data_dirs, fcs, spans, low_powers, add_zero_lines=False, filetype='png', 
                    plot_complex=True, show_plots=True, plot_dir='.\\plots\\')

print(time.strftime("%a, %b %d, %H:%M %p"))


# %% repeat low power scan
#############################################################################
####################   VNA DOES NOT GO BELOW -90 DBM  #######################
#############################################################################

low_powers_floats = np.arange(-85, -91, -2)  # 150? seconds
low_powers = [int(x) for x in low_powers_floats]
LPow_templist = [temp_reading] * len(low_powers)
LPow_segment_option = 'homophasal'
LPow_num_avgs = 75
LPow_IFBW_kHz = 0.05
LPow_num_pts = 201
LPow_fnames = make_filenames(low_powers, freq_strs, LPow_templist, sample_name, prefix="", suffix=LPow_segment_option)
LPow_fnames = np.array(LPow_fnames).flatten()

tStart_low = time.time()
measure_multiple_resonators(fcs, spans, delays, low_powers,
                            ifbw = LPow_IFBW_kHz,  npts = LPow_num_pts,
                            Navg_init = LPow_num_avgs, file_names = LPow_fnames, 
                            data_dirs = data_dirs, sparam='S21',
                            adaptive_averaging = False, sample_name = sample_name,
                            runtime = 1, cal_set = None, start_delay = 0,
                            Noffres = int(LPow_num_pts*0.2), Nf = int(LPow_num_pts*0.8), offresfraction = 0.8, 
                            is_segmented = LPow_segment_option,  bypass_janis = False, 
                            segment_option = LPow_segment_option, seg_str=LPow_segment_option,)
tEnd_low = time.time()
LPow_elapsed_time = tEnd_low - tStart_low

print_text_block(tStart_low, tEnd_low, LPow_num_avgs, LPow_num_pts, LPow_IFBW_kHz, num_powers=len(low_powers))
visualize_data(LPow_fnames, data_dirs, fcs, spans, low_powers, add_zero_lines=False, filetype='png', 
                    plot_complex=True, show_plots=True, plot_dir='.\\plots\\')

print(time.strftime("%a, %b %d, %H:%M %p"))

# %% time analysis
N_res = 8 #len(fcs)

T_A = 2 #HPow_elapsed_time[0]/N_res
T_B = 7
T_C = 110*4 #LPow_elapsed_time[0]/N_res  

N_A = len(high_powers);  N_B = len(med_powers);  N_C = len(low_powers)
A = T_A*N_A;             B = T_B*N_B;            C = T_C*N_C

T_res = A+B+C
T_mins = N_res*T_res/60


print(f"Measurement time analysis")
print(f" T_res =   ({N_A} high powers * {T_A:1.2f} secs/ea)")
print(f"            + ({N_B} med powers * {T_B:1.2f} secs/ea)")
print(f"            + ({N_C} low powers * {T_C:1.2f} secs/ea) \n")
print(f" T_res  = ({A/60:1.2f} mins + {B/60:1.2f} mins + {C/60:1.2f} mins) = {T_res/60:1.2f} mins/res")
print(f" T_res * {N_res} resonators = {T_mins:1.2f} mins ({T_mins/60:1.2f} hrs)")

total_elapsed_time = (time.time() - tStart_high)/60
print(time.strftime("%a, %b %d, %H:%M %p"))
print(f"\ntime.time() reported elapsed time: {total_elapsed_time:1.2f} mins ({total_elapsed_time/60:1.2f} hrs)")



# %%
