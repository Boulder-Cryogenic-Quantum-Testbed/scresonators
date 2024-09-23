# %% global init

"""
    homemade package to organize all the methods that have 
    spawned in these disorganized hives of villainy

"""

# %% 
import glob, os, sys

print("Make sure you login to the JetWay session on PuTTY with user: 'bco' and password: 'aish8Hu8'")

print("Running __init__.py for helper_scripts!'")

# %% 
path_to_bcqt_ctrl = 'C:\\Users\\Lehnert Lab\\GitHub\\bcqt-ctrl\\'
path_to_resfit = 'C:\\Users\\Lehnert Lab\\Github\\scresonators\\'
path_to_parent = os.path.dirname(os.path.abspath(__file__))
path_to_parent_parent = os.path.dirname(path_to_parent)

# %%

item_list = [os.path.basename(item) for item in glob.glob(path_to_parent_parent + "\\*")]

assert 'helper_scripts' in item_list, f"helper_scripts' folder not found! {path_to_parent_parent}"
assert len(glob.glob(path_to_resfit)), f'Path: {path_to_resfit} does not exist'
assert len(glob.glob(path_to_parent)), f'Path: {path_to_parent} does not exist'
sys.path.append(path_to_resfit)
sys.path.append(path_to_parent)

assert len(glob.glob(path_to_bcqt_ctrl)), f'Path: {path_to_bcqt_ctrl} does not exist'
sys.path.append(path_to_bcqt_ctrl + r'temperature_control')
sys.path.append(path_to_bcqt_ctrl + r'pna_control')
sys.path.append(path_to_bcqt_ctrl + r'instrument_control')

sys.path.append(path_to_parent + "\\helper_scripts\\")


# %%

# import fit_resonator.resonator as res
# import fit_resonator.fit as fsd