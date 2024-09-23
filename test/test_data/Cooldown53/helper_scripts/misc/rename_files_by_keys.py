
# quick script to rename files that are given the '-1000mK' tag
# from bypassing the janis while measuring

# %%
import glob, os
import regex as re

if "scripts" in os.getcwd():
    os.chdir(os.path.dirname(os.getcwd()))

# %%
subdir_depth = list(range(8))
search_str = "\\*" 
search_dir = "Line4_NWNb2O5_30_02\\100mK dataset\\*"

key_to_replace = "11mK"
replace_with = "100mK"

old_new_filename_pairs = []
for n in subdir_depth:
    file_list = glob.glob(search_dir + "*" + search_str*n)  # stack *\\*\\*\\*... to search all sub-dirs
    if len(file_list) == 0 or n == 4:
        print(f"Ran out of subdirs! (n={n})")
        break
    
    old_filenames = [x for x in file_list if key_to_replace in x]
    new_filenames = [x.replace(key_to_replace, replace_with) for x in old_filenames]
    print(f"Files with -1000mK that are {n} directories deep: {len(old_filenames)}")
    print(len(old_filenames), len(new_filenames))
    
    try:  # will throw an exception if num of files found is zero
        old_new_filename_pairs += list(zip(old_filenames, new_filenames))
    except:
        pass
    
print(f"\nTotal # of files: {len(old_new_filename_pairs)}")

# %% now rename files
# for filepath in 

for (old_filepath, new_filepath) in old_new_filename_pairs:  # list of 2-tuples
    print(os.path.basename(old_filepath), "--->", os.path.basename(new_filepath))
    
    ############################################################################
    #########   quadruple check everything, in case you fucked up :)   #########
    ############################################################################
    # os.rename(old_filepath, new_filepath) 

# %% replace temp and power at the same time

subdir_depth = list(range(8))
search_str = "\\*" 
search_dir = "Line3_NWNb2O5_15_01\\NWNb2O5_15_01_6p447GHz"

power_change = 30

file_list = []
for n in subdir_depth:
    file_list += glob.glob(search_dir + "*" + search_str*n)  # stack *\\*\\*\\*... to search all sub-dirs
    if len(file_list) == 0 or n == 4:
        print(f"Ran out of subdirs! (n={n})")
        break
    
for file in reversed(file_list):
    if ".csv" not in file and ".png" not in file:
        print(f"~{os.path.basename(file)} is not a .csv or .png")
        continue
    filename = os.path.basename(file)
    filepath = os.path.dirname(file)
    regex_expr_1 = r"-\d{1,2}dB"
    # regex_expr_2 = "-1000mK"
    
    try:
        power_string = os.path.basename(re.search(regex_expr_1, filename).group(0))
        replacement_power = f"{(int(power_string.replace("dB","")))+power_change:03d}dB"
        print("    ", power_string, "---->", replacement_power)
        first_replacement = re.sub(regex_expr_1, replacement_power, filename)
    except Exception as e:
        print("Nothing to replace in ", filename)
        first_replacement = filename
    try:
        temp_string = os.path.basename(re.search(regex_expr_2, filename).group(0))
        replacement_temp = "11mK"
        print("    ", temp_string, "---->", replacement_temp)
        second_replacement = re.sub(regex_expr_2, replacement_temp, first_replacement)
    except Exception as e:
        print("No negative temperature to replace in ", filename)
        second_replacement = first_replacement
        
    # print(second_replacement)
    # print(os.path.basename(second_replacement))
    print(os.path.basename(filename), "--->", os.path.basename(second_replacement), "\n")
    try:
        # os.rename(file, filepath+"\\"+second_replacement) 
        pass
    except Exception as e:
        display(e)

 # %%