# %%
import glob, os
import regex as re

directory = "Line3_NWNb2O5_15_01"

cooldown_dir = os.path.dirname(os.getcwd())
data_dir = cooldown_dir + "\\" + directory + "\\"
samplename = directory[6:]

folder_paths = data_dir + "*GHz"
file_dirs = glob.glob(folder_paths)

# %%
for folder in file_dirs:
    # item_key = f"{item}\\{item}*.csv"
    item_key = "\\*homophasal.csv"
    # print(item, "   ", item_key)
    try:
        for file in glob.glob(folder + item_key):
            # file_fixed = file.replace(f"{item}\\{item}", f"{item}\\")
            # print(os.path.basename(file))
            print("  ", file)
            # print("  ", file_fixed)
            # os.rename(file, file_fixed)
    except:
        # print("\n", glob.glob(item+"\\*"), "\n")
        pass
    # break
    # continue
# %%
