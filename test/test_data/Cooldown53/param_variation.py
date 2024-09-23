# %%
"""
User file for doing parameter variation

    Make sure you login to the JetWay session on PuTTY
    with user: 'bco' and password: 'aish8Hu8'

"""
# %%
%run global_init
 
import os, time
import numpy as np

import helper_misc as hm
import helper_plotting as hp

from janis_ctrl import measure_multiple_resonators


# %%
####################################################################

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
    print(f"                        total: {num_pts} resonators")
    print("======================================================")
    print("\n\n")
    return

# %%
####################################################################
####################################################################
####################################################################

# XXX: Set the center frequencies (GHz), spans (MHz), delays(ns), and temperature
fcs = [5.805240]#, 5.849863, 5.905400, 5.939900]
        # 6.358500, 6.396100, 6.443700, 6.498900]
        
fc = fcs[0]
fc_idx = fcs.index(fc)

spans = [5]
delays = [75.905]
temperature = [12e-3]
powers = [-40]
num_pts = 51

# XXX: Change the sample name
cur_dir = os.getcwd()
base_dir = os.path.basename(cur_dir)
line_num = base_dir[:5]
sample_name = base_dir[6:]  # get rid of "LineX_ from the beginning of folder name"

# XXX: pick type of segments, if any
is_segmented = True
# segment_option = None  #'hybrid', 'homophasal', or None for regular [5, 41, 5] distribution
segment_option = 'homophasal'

# set seg_str to match the segment option
seg_str = segment_option if segment_option != 'None' else ""
    
# create strings for the filename frequencies and data directories 
# freq_strs = [f'{fc:.3f}GHz'.replace('.', 'p') for fc in fcs]
# data_dirs = [f'{sample_name}_{freq_str}\\' for freq_str in freq_strs]
# data_dir = data_dirs[fc_idx]

elapsed_times = []

# %% IF BW scan

all_IF_BWs = np.geomspace(100, 1000, 2)  # Hz
n_avgs = 1

fnames = []
for idx, IF_BW in enumerate(all_IF_BWs):
    ifbw_tag = f"{IF_BW:.0f}Hz_IFBW"
    freq_str = f'{fc:.3f}GHz'.replace('.', 'p') 
    data_dir = [f'param_variation\\IF_BW\\{ifbw_tag}_{sample_name}_{freq_str}']*len(all_IF_BWs)
    hm.check_and_make_dir(data_dir[idx])

    fname = hm.make_filenames(powers, freq_str, temperature, sample_name, seg_str, tag=f"{IF_BW:.0f}Hz_IFBW")[0]
    
    tStart = time.time()
    print("Measuring and saving as: ", fname)
    measure_multiple_resonators(fcs, spans, delays, powers,
                                ifbw = IF_BW,  npts = num_pts,
                                Navg_init = n_avgs, file_names = fname, 
                                data_dirs = data_dir, sparam='S21',
                                adaptive_averaging = False, sample_name = sample_name,
                                runtime = 1., cal_set = None, start_delay = 0.,
                                Noffres = 20, Nf = 100, offresfraction = 0.8, 
                                is_segmented = is_segmented,  bypass_janis = False, 
                                segment_option = segment_option, seg_str=seg_str)
    elapsed_times.append(time.time() - tStart)

    print_text_block(tStart, n_avgs, num_pts, IF_BW, num_powers=len(powers))
    hp.visualize_data(fname, data_dir, fc, spans, powers, add_zero_lines=True, filetype='png', show_plots=True, plot_dir='.\\plots\\')

# %% n avg scan
IF_BW = 1e3
all_n_avgs= np.geomspace(5, 100, 2)

fnames = []
for idx, n_avgs in enumerate(all_n_avgs):
    n_avg_tag = f"{IF_BW:.0f}_avgs"
    freq_str = f'{fc:.3f}GHz'.replace('.', 'p') 
    data_dir = [f'param_variation\\n_avgs\\{n_avg_tag}_{sample_name}_{freq_str}']*len(all_n_avgs)
    hm.check_and_make_dir(data_dir[idx])

    fname = hm.make_filenames(powers, freq_str, temperature, sample_name, seg_str, tag=n_avg_tag)[0]
    
    tStart = time.time()
    print("Measuring and saving as: ", fname)
    measure_multiple_resonators(fcs, spans, delays, powers,
                                ifbw = IF_BW,  npts = num_pts,
                                Navg_init = n_avgs, file_names = fname, 
                                data_dirs = data_dir, sparam='S21',
                                adaptive_averaging = False, sample_name = sample_name,
                                runtime = 1., cal_set = None, start_delay = 0.,
                                Noffres = 20, Nf = 100, offresfraction = 0.8, 
                                is_segmented = is_segmented,  bypass_janis = False, 
                                segment_option = segment_option, seg_str=seg_str,)
    elapsed_times.append(time.time() - tStart)

    print_text_block(tStart, n_avgs, num_pts, IF_BW, num_powers=len(powers))
    hp.visualize_data(fname, data_dir, fc, spans, powers, add_zero_lines=True, filetype='png', show_plots=True, plot_dir='.\\plots\\')



# %%
