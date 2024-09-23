# %%

'''
    helper_load.py
'''

# print("    loading helper_load.py")

# %%

import glob, os, sys, time

import pandas as pd
import regex as re
import numpy as np

# %%
def load_csv_as_dataframe(search_dir, search_str, debug=False):
    data_dict = {}
    for item in glob.glob(search_dir + search_str):
        if ".csv" in item:
            item_name = item.replace(search_dir, "")
            resonator_name = os.path.dirname(item_name)
            if debug: print(resonator_name, "    ",  item_name)
            df = pd.read_csv(item)
            data_dict[item_name] = df
    return data_dict


def gen_dataframe_from_csv(filepath, **kwargs):
    filename = os.path.basename(filepath)
    df = pd.read_csv(filepath, names=["Frequency", "Magnitude", "Phase_Deg"])
    freq = df["Frequency"]
    
    if any(freq > 1e9):  # scale to GHz
        freq = freq/1e9
        
    phase_deg = df["Phase_Deg"]
    phase = np.deg2rad(phase_deg)
    df["Frequency"] = freq
    df["Phase_Rad"] = phase
    
    index = df.index
    index.name = filename
    
    return df


def load_files_in_dir(directory, key="*", debug=False):
    if directory[-1] != "\\":  #??? 
        directory = directory + "\\"
    if debug: print(directory + key)
    
    filepaths = [x for x in glob.glob(directory + key)]
    filenames = [os.path.basename(x) for x in glob.glob(directory + key)]
    
    if debug:
        print(f"Chosen directory: {directory}")
        for file, path in zip(filenames, filepaths):
            print(f"   {path}")
            
    return filenames, filepaths




# %%