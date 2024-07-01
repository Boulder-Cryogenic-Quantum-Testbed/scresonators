# %%
try:
    %load_ext autoreload
    %autoreload 2
except:
    pass

# %% interactive notebook

import glob, os, sys, time

pathToParent = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pathToParent + "\\scripts\\")
import helper_misc as hm
import helper_user_fit as hf


# %%

# change this to select what devices to run the user_fit.py for
# all_line_nums = [4, 5, 6]  
all_line_nums = [4]  
all_line_nums = [str(x) for x in all_line_nums]

current_directory = os.getcwd()
for line_num in all_line_nums:
    # should only need to change the number at the end of device_folder_key
    device_folder_key = "\\Line" + line_num
    data_folder_key = "\\*GHz"

    # get all device folders, e.g. "line2_xxx"
    all_device_paths = glob.glob(current_directory+device_folder_key+"*")

    # get all folders within that device folder
    all_folders_in_device_paths = [glob.glob(x + "\\*") for x in all_device_paths if device_folder_key in x][0]

    # filter the previous list to only the folders with data in them
    try:
        chosen_data_folders = [x for x in all_folders_in_device_paths if 'GHz' in x]
    except:
        chosen_data_folders = [x for x in all_folders_in_device_paths if 'data' in x]
        

    device_folder_path = os.path.dirname(chosen_data_folders[0])
    device_folder_name = os.path.basename(device_folder_path)
    
    display(f"Running user_fit.py in: '{device_folder_name}'")
    print(f"Full path: {device_folder_path}")
    
    for path in chosen_data_folders:
        try:
            os.chdir(device_folder_path)
            single_resonator_folder_name = os.path.basename(path)
            data_dir = single_resonator_folder_name
            print(f"====================================================================================")
            print(f"=========== user_fit_automated.py is now running user_fit.py for:  =================")
            print(f"======================     {single_resonator_folder_name}        ============================")
            print(f"====================================================================================")
            display(data_dir)
            time.sleep(2)  # give time for kernel to catch up
            %run user_fit.py {data_dir}
        except Exception as e:
            print("Error occurred in running user_fit.py: \n", e)
            
        os.chdir(current_directory)  # go back to start

# %% 










