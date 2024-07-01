# %% -*- encoding: utf-8 -*-

#### press CTRL+K CTRL+2 to close all methods
#### press CTRL+K CTRL+J to open all
# %%
"""
User file for controlling the Janis and PNA instruments

    Make sure you login to the JetWay session on PuTTY
    with user: 'bco' and password: 'aish8Hu8'

"""
import sys, time, os, datetime, glob
import numpy as np
import matplotlib.pyplot as plt

# Change this path
sys.path.append(r'C:\Users\Lehnert Lab\GitHub\bcqt-ctrl\temperature_control')
sys.path.append(r'C:\Users\Lehnert Lab\GitHub\bcqt-ctrl\pna_control')
sys.path.append(r'C:\Users\Lehnert Lab\GitHub\bcqt-ctrl\instrument_control')
from janis_ctrl import measure_multiple_resonators
from janis_ctrl import JanisCtrl

pathToParent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    
    
# %%
####################################################################
####################################################################
####################################################################
# TODO: create metadata file that shares these parameters with all user files in the folder
# XXX: Set the center frequencies (GHz), spans (MHz), delays(ns), and temperature
fcs = [ 4.5, 5.5, 6.5, 7.5]
# fcs = [4.500]

dstr = datetime.datetime.today().strftime(r'%m%d%y_%I%M%p')
print(f"Current timestamp: {dstr}")

spans = [1000]*len(fcs)
delays = [76.14]*len(fcs)

temp_reading = read_temp_JCtrl()[1]  # convert to mK

cur_dir = os.path.dirname(os.getcwd())
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
powers_floats = [-30]
powers = [int(x) for x in powers_floats]
pow_templist = [temp_reading]*len(powers)
num_avgs = 3
IFBW_kHz = 1.0
num_pts = 60001
fnames = make_filenames(powers, freq_strs, pow_templist, sample_name, prefix="", suffix="")
fnames = np.array(fnames).flatten()
segment_option = None

tStart = time.time()
measure_multiple_resonators(fcs, spans, delays, powers,
                             ifbw = IFBW_kHz,  npts = num_pts,
                             Navg_init = num_avgs, file_names = fnames, 
                             data_dirs = data_dirs, sparam='S21',
                             adaptive_averaging = False, sample_name = sample_name,
                             runtime = 1., cal_set = None, start_delay = 0.,
                             Noffres = int(num_pts*0.2), Nf = int(num_pts*0.8), offresfraction = 0.8, 
                             is_segmented = segment_option,  bypass_janis = False,
                             segment_option = segment_option, seg_str=segment_option,)
tEnd = time.time()
elapsed_time = tEnd - tStart

print_text_block(tStart, tEnd, num_avgs, num_pts, IFBW_kHz, num_powers=len(powers), num_resonators=len(fcs))
# visualize_data(fnames, data_dirs, fcs, spans, powers, add_zero_lines=False, filetype='png', 
#                show_plots=True, plot_complex=True, plot_dir='.\\plots\\')

# %% stitch together

def stitch_broadband(prefix, freq_band, freq_step, dstr, powers,
                    Tmxc=13, fscale=1e9, sparam='S21', show_plots=True,
                    manual_fnames=None, manual_dirs=None):
    """
    Stitch together multiple high power sweeps and plot as a single file
    """
    # Read the frequency band and powers used
    f1, f2 = freq_band
    p1 = powers

    if manual_fnames is None:
        # Determine number of frequency segments
        Nf = int(round((f2 - f1) / freq_step))
        center_freqs = [freq_band[0] + (1 + 2*j) * freq_step / 2 
                        for j in range(Nf)]
        center_freqs_strs = [f'{cf:.3f}'.replace('.', 'p') for cf in center_freqs]
        print(f'center_freqs: {center_freqs:.3f}')
        print(f'center_freqs_strs: {center_freqs_strs:.3f}')

        # Generate directories and paths
        sdirs = [f'{prefix}_{cf}GHz*'
                for cf in center_freqs_strs]
        print(f'sdirs:\n{sdirs}')
        dirs = [glob.glob(f'{prefix}_{cf}GHz*')[0]
                for cf in center_freqs_strs]
        print(f'sdirs:\n{sdirs}')
        fnames = [f'{d}/{prefix}_{cff}GHz_{p1:.0f}dB_{Tmxc:.0f}mK.csv'
                    for cff, d in zip(center_freqs_strs, dirs)]

    else:
        fnames = [ data_dir + "\\" + fname for fname, data_dir in zip(manual_fnames, manual_dirs)]
        
    # Read the data and concatenate
    freqs  = np.array([])
    magn = np.array([])
    phase  = np.array([])
    for fn in fnames:
        data = np.genfromtxt(fn, delimiter=',').T
        freqs = np.hstack((freqs, data[0]))
        magn = np.hstack((magn, data[1]))
        phase = np.hstack((phase, data[2]))

    # Plot the results
    if f1 == f2:
        fig_title = f'{prefix}_{f1:.3f}GHz'.replace('.','p')
    else:
        fig_title = f'{prefix}_{f1:.3f}GHz_{f2:.2f}GHz'.replace('.','p')
    
    markersize = 4

    fig, axes_dict = plt.subplot_mosaic("AAA\n BBB", figsize=(10,6), tight_layout=True)
    ax1, ax2 = axes_dict["A"], axes_dict["B"]
    axes = list(axes_dict.values())
    
    ax1.plot(freqs / fscale, magn, 'ko', markersize=markersize)
    ax1.set_xlabel("Frequency [GHz]", size=14)
    ax1.set_ylabel("S21 [a.u.]", size=14)
    ax1.set_title("Magnitude Data", size=16)
    
    ax2.plot(freqs / fscale, phase, 'ro', markersize=markersize)
    ax2.set_xlabel("Frequency [GHz]", size=14)
    ax2.set_ylabel("Phase [rad]", size=14)
    ax2.set_title("Phase Data", size=16)
    
    fig.suptitle(fig_title, size=20)
    
    for ax in axes:
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        
    fig.savefig(fig_title+".png", format='png')
    if show_plots is False:
        plt.close('all')
    

# %% Concatenate broadband sweeps
prefix = sample_name

freq_band = [fcs[0], fcs[-1]]
freq_step = spans[0]*1e-3 

# first grab every directory name that has the correct freq and sample name
manual_dirs = []
for fc in fcs:
    fc_string = f"{fc:1.3f}".replace(".",'p')
    list_compr = [folder for folder in glob.glob("*GHz") if sample_name in folder and fc_string in folder]
    manual_dirs.append(*list_compr)

# or just overwrite directly
# manual_dirs=["NWOXCTRL02_4p783GHz"]

# then go through each directory and grab each file
manual_fnames = []
for directory in manual_dirs:
    manual_fnames.append(glob.glob(directory + "\\*.csv")[0].replace(directory+"\\",""))



stitch_broadband(prefix, freq_band, freq_step, dstr, powers,
                Tmxc=24., fscale=1e9, sparam='S12', 
                manual_fnames=manual_fnames, manual_dirs=manual_dirs)


# stitch_broadband(prefix, [fcs[0]-spans[0]/2*1e-3,fcs[0]+spans[0]/2*1e-3], freq_step, dstr, powers,
#                 Tmxc=24., fscale=1e9, sparam='S12', 
#                 manual_fnames=[manual_fnames[0]], manual_dirs=[manual_dirs[0]])


# stitch_broadband(prefix, [fcs[1]-spans[1]/2*1e-3,fcs[1]+spans[1]/2*1e-3], freq_step, dstr, powers,
#                 Tmxc=24., fscale=1e9, sparam='S12', 
#                 manual_fnames=[manual_fnames[1]], manual_dirs=[manual_dirs[1]])

# %%
