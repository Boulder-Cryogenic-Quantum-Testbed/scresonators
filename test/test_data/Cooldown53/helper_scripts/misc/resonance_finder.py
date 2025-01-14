# %%
# import sys
import time, os, glob
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
import helper_functions as hf

# %% 

# get sample name from parent directory
cur_dir = os.getcwd()
# base_dir = os.path.basename(cur_dir)
line_num = '2'
sample_name = 'NWOXCTRL02'  # remove "LineX_ from folder name"

# data_directory = "broadband\\5.55 GHz to 6.55 GHz\\"
data_directory = r"Line2_NWOXCTRL02\\broadband\\"

# %%
def gen_data_file_dict(data_directory):
    data_filepaths = glob.glob(data_directory + "*.csv")
    data_file_dict = {}
    for filepath in data_filepaths:
        # use the filename as the key, because then every data set will
        # have a unique key. they all (may) have the same directory then.
        filename = os.path.basename(filepath)
        data_file_dict[filename] = filepath.replace(filename,"")
        
    return data_file_dict
    
# %%
# all_folders = glob.glob(data_directory+".\\NW*GHz")
# for data_folder in all_folders:
#     data_file_dict = gen_data_file_dict(data_directory)
#     plot_data_file_dict(data_file_dict)
#     plt.show()
#     time.sleep(1)




plot_config_dict = {}
data_file_dict = {"NWOXCTRL02_4p500GHz_-35dB_11mK_.csv" : "Line2_NWOXCTRL02\\broadband\\NWOXCTRL02_4p500GHz\\",
                  "NWOXCTRL02_5p500GHz_-35dB_11mK_.csv" : "Line2_NWOXCTRL02\\broadband\\NWOXCTRL02_5p500GHz\\",
                  "NWOXCTRL02_6p500GHz_-35dB_11mK_.csv" : "Line2_NWOXCTRL02\\broadband\\NWOXCTRL02_6p500GHz\\",
                  "NWOXCTRL02_7p500GHz_-35dB_11mK_.csv" : "Line2_NWOXCTRL02\\broadband\\NWOXCTRL02_7p500GHz\\",
                  }
plot_data_file_dict(data_file_dict, plot_config_dict)



# %%
