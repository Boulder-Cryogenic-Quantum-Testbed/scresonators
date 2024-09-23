# %% if running interactive notebook w/ vscode jupyter

# try:
#     %load_ext autoreload
#     %autoreload 2
# except:
#     pass

# %%
import numpy as np
import pandas as pd #speadsheet commands
import regex as re
import sys #update paths
import os #import os in order to find relative path
import glob
import time

pathToParent = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(pathToParent)


sys.path.append(pathToParent + "\\scripts\\")
import helper_misc as hm
import helper_user_fit as hf

np.set_printoptions(precision=3,suppress=True)# display numbers with 3 sig. 

# %%

# TODO: create metadata file that shares these parameters with all user files in the folder
# XXX: Set the center frequencies (GHz), spans (MHz), delays(ns), and temperature

# XXX: Change the sample name
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

# fcs = [5.727375, 5.775313, 5.815000, 5.863812,
        #  6.254625, 6.303750, 6.349625, 6.418062]

        
data_directory_idx = 0  # 0 thru 7

load_key = "*"
# check if script was run by user_fit_automated.py
if "user_fit.py" in sys.argv:
    print([str(x) for x in sys.argv])
    data_dir = sys.argv[1] + "\\" 
    print(data_dir, type(data_dir), os.getcwd())
else:
    all_folder_paths = glob.glob(cur_dir + "\\*GHz")
    all_folder_names = [os.path.basename(x) + "\\" for x in all_folder_paths]
    data_dir = all_folder_names[data_directory_idx]
    print(f"Manually choosing '{data_dir}' as directory for fitting")
 
all_names, all_paths = hm.load_files_in_dir(data_dir, key="*") # remove the \\  
all_names = [x for x in all_names if ('.csv' in x and 'GHz' in x and 'qiqc' not in x)]
all_paths = [y for y in all_paths if ('.csv' in y and 'GHz' in y and 'qiqc' not in y)]

powers_in = [hm.get_power_from_filename(x) for x in all_names if 'dB' in x] 
# temperature =  [get_temperature_from_filename(x) for x in all_names if 'mK' in x]
temperature =  int(hm.get_temperature_from_filename(all_names[0]))

# remove these powers from the dataset
remove_powers = []
# remove_powers = np.arange(-75, -95, -5)

init_conds = []
# remove any files that are not in "powers_in"
print(f"Removing {remove_powers} from dataset:")
for file in all_names: # start with each file
    found_file = False 
    for power in remove_powers:  # loop through powers until we find a match
        if str(power)+"dB" in file or "bad" in file:
            print(f" !!! Removing {file}")
            all_names.remove(file)
            powers_in.remove(power)
            all_paths = [filepath for filepath in all_paths if str(power)+"dB" not in filepath]
            found_file = True 
            # break # stop the loop once we find it
        
    if found_file is False:
        print(f" ~~> Keeping {file}:")
    
    # now get the resonant frequency and create initial conditions    
    identifier = re.search(r"_\dp\d{0,4}GHz_", file)
    res_freq = float(identifier[0].replace("GHz","").replace("_","").replace("p","."))
    init = [1e5, 1.3e5, res_freq*1e9, np.pi/2]
    #####################################################
    ########### init conds are broken for now  ##########
    #####################################################
    init = None
    init_conds.append(init)
    
print(powers_in)

# %% show all data first

for filepath in all_paths:
    filename = os.path.basename(filepath)
    header = ["Frequency", "Magnitude", "Phase_Deg"]
    df = pd.read_csv(filepath, names=header)
    freq = df["Frequency"]
    magn = df["Magnitude"]
    phase_deg = df["Phase_Deg"]
    phase = np.deg2rad(phase_deg)
    df["Phase_Rad"] = phase
    plot_title =  filename.replace(".csv","")
    
    if any(freq > 1e9):  # scale to GHz
        freq = freq/1e9
    
    plot_filepath = os.path.dirname(filepath) + "\\plots\\" 
    plot_filename = plot_filepath + plot_title + ".png"
    
    plot_config_dict = {
        "add_zero_lines" : True,
        "plot_title" : plot_title,
        # "plot_filename" : None,
        "plot_filename" : plot_filename,
        "plot_complex" : True,
    }
    
    if plot_config_dict["plot_filename"] is not None:
        hm.check_and_make_dir(plot_filepath)
        print(f"saving plot at: {plot_filename}")
    
    fig, axes = hf.plot_S21_data(freq, plot_config_dict, magn=magn, phase=phase, debug=False,  )

# %%
# # Run the power sweep WITHOUT TLS 

try:
    hf.power_sweep_fit_drv(sample_name=sample_name,
                            atten=[0, -70], temperature=temperature,
                            powers_in=powers_in, all_paths=all_paths, 
                            plot_from_file=False,
                            use_error_bars=True, temp_correction='', phi0=0.,
                            use_gauss_filt=False, use_matched_filt=False,
                            use_elliptic_filt=False, use_mov_avg_filt=False,
                            loss_scale=1e-6, preprocess_method='circle',
                            ds = {'QHP' : 1e5, 'nc' : 1e1, 'Fdtls' : 1e-6},
                            plot_twinx=False, plot_fit=False, QHP_fix=True, show_plots=True,
                            data_dir=data_dir, save_dcm_plot=True, manual_init_list=init_conds,
                            show_dbm=True)
except Exception as e:
    print( "====================================================================================")
    print( "====================================================================================")
    print(f"=======================    Failed to fit:  {sample_name}    ========================")
    print( "====================================================================================")
    print( "====================================================================================")
    print("Error Message:  \n")
    print(e)   # TODO: add stack trace (trace stack?)
    print("\n\n")
    time.sleep(1.5)


# %%
