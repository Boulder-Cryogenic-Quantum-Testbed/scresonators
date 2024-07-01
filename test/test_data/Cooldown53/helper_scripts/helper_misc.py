# %%

'''
    helper_misc.py
'''

# print("    loading helper_misc.py")

# %%

import glob, os, sys, time

import regex as re
import numpy as np

# %%

def check_and_make_dir(directory_name):
    directory_name = directory_name.replace('.csv', '').replace('.pdf', '')  # sanitize input
    if not os.path.exists(directory_name):
        print(f'      plot directory: {directory_name}')
        print(f'      Does not exist. Making new directory.')
        os.makedirs(directory_name)
    return
     
     

def get_power_from_filename(filename):
    # don't you just love regex?
    # this regex extracts any substring in 'filename'
    # that starts with a '-', has 1-3 digits, and ends with dB
    power_reg = re.search("[0-9]{1,3}dB", filename)
    power_dB = power_reg.captures()[0]  # there should only be one match
    power = int(power_dB[:-2]) # strip off the 'dB', add the negative sign
    return power*-1



def get_temperature_from_filename(filename):
    # this regex extracts any substring in 'filename'
    # that has 1-3 digits and ends with mK
    temp_reg = re.search("[0-9]{1,3}mK", filename)
    temp_mK = temp_reg.captures()[0]  # there should only be one match
    temp = temp_mK[:-2]  # strip off the 'dB'
    return temp



def get_frequency_from_filename(filename): 
    # more regex
    identifier = re.search(r"_\dp\d{0,4}GHz_", filename)
    res_freq = float(identifier[0].replace("GHz","").replace("_","").replace("p","."))
    return res_freq

    
    
def clean_directory(search_dir, regex_expr=".*", search_str="\\*", dry_run=True, exclude_list=None):
    subdir_depth = list(range(8))

    file_list = []
    for n in subdir_depth:
        file_list += glob.glob(search_dir + "*" + search_str*n)  # stack *\\*\\*\\*... to search all sub-dirs
        if len(file_list) == 0 or n == 4:
            print(f"Ran out of subdirs! (n={n})")
            break
        
    delete_list = []
    for file in reversed(file_list):  
        # if ".csv" not in file and ".png" not in file:
        #     print(f"~{os.path.basename(file)} is not a .csv or .png")
        #     continue
        
        # only delete files with these tags in the filename
        if "qiqc" not in file and ".png" not in file and "scres" not in file and "all_res" not in file and "plots" not in file and "fits" not in file:
            print(f"~{os.path.basename(file)} is not going to be deleted")
            continue
        
        if exclude_list is not None:
            for exclude_string in exclude_list:
                if file in exclude_string:
                    print(f"{file} manually excluded")
                    continue
        
        filename = os.path.basename(file)
        filepath = os.path.dirname(file)
        try:
            found_filename = re.search(regex_expr, filename).group(0)
            print("    deleting:", found_filename) #, "---->", found_file)
            final_name = f"{filepath}\\{found_filename}" if filepath != found_filename else filepath
            delete_list.append(final_name)
        except Exception as e:
            print("Skipping ", filename, e)
    
    print("\n")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("~~~~~~~~~Files To Be Deleted~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    for idx, name in enumerate(delete_list):
        print(f"    [{idx}] xxx -", name )
        
    time.sleep(1) # to let print statements catch up to input()
    
    print(f"Prompting user if they'd are okay with deleting {len(delete_list)} files:  [Y/n]")
    confirmation = input(f"Are you sure you want to delete {len(delete_list)} files?  [Y/n]")
    print(f"Response: '{confirmation}'")
    
    
    if dry_run == True:
        print("Dry run finished, ending without deleting data")
        
    if confirmation == "Y" and dry_run == False:
        trashbin_path = "trash_bin\\"
        for file_to_delete in delete_list:
            try:
                # check_and_make_dir(trashbin_path + file_to_delete)
                if '.' not in file_to_delete[-5:]:
                    os.rmdir(file_to_delete)
                else:
                    os.remove(file_to_delete)
                
            except Exception as e:
                print(f"failed to delete {file_to_delete}. \nError : {e}")
    else:
        print("Cancelled.")
    
    print("Finished!")
    return



def average_files(filenames):
    """
    Average two files and produce a new file
    """
    print(f'Averaging filenames:\n{filenames}')
    data = np.genfromtxt(filenames[0], delimiter=',').T
    mag = 10**(data[1, :]/20)
    phase = data[2, :] * np.pi / 180
    S21 = mag * np.exp(1j * phase)
    real = np.real(S21)
    imag = np.imag(S21)

    for fname in filenames[1:]:
        data = np.genfromtxt(fname, delimiter=',').T
        mag = 10**(data[1, :]/20)
        phase = data[2, :] * np.pi / 180
        S21 = mag * np.exp(1j * phase)
        real += np.real(S21)
        imag += np.imag(S21)

    # Average real, imag, convert back to mag/phase
    real /= len(filenames)
    imag /= len(filenames)
    S21 = real + 1j * imag
    mag = 20 * np.log10(np.abs(S21))
    phase = 180 * np.arctan(imag / real) / np.pi

    fname_out = filenames[0]

    with open(fname_out, 'w') as fid:
        fid.write('\n'.join([f'{f}, {m}, {p}' \
        for f, m, p in zip(data[0, :], mag, phase)]))



def read_temp_JCtrl(jctrl_path=None):
    if jctrl_path is None:
        jctrl_path = r'C:\Users\Lehnert Lab\GitHub\bcqt-ctrl\temperature_control'
    
    sys.path.append(jctrl_path)
    from janis_ctrl import JanisCtrl
    
    # random values 
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
        output_cmn = JCtrl.read_cmn()
        output_temp, tstamp = JCtrl.read_temp('all')
        
    except Exception as e:
        print("Failed to read_temp, error:  ", e)
        return None
    
    finally:
        del JCtrl
    
    return output_cmn, output_temp, tstamp


