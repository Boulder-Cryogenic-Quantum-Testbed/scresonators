# %%
from resonator import Resonator
from sweeper import Sweeper
from file_io import FileIO
import glob, os, sys, time
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
from pathlib import Path
import re
    
# %%
current_dir = Path('.')
test_csvs = current_dir  / ".." / "test" / "Res0_5863MHz"
print(test_csvs.exists())

csv_filenames = list(test_csvs.glob("*.csv"))
print(len(csv_filenames))
print(csv_filenames[0])


# %%

csv_filepath = csv_filenames[5] # provide a path to .csv file here (in str format)
file_io = FileIO(csv_filepath, verbose=True)
df = file_io.load_csv(csv_filepath)


# %% load data from arrays

resonator_1 = Resonator(fileIO_obj=file_io, preprocess_method='circle', fit_method_name='DCM')

# Optional: Change default parameters for preprocessing and fit_method. For example: 
# resonator_1.initialize_data_processor(normalize_pts=30, preprocess_method='linear') 
# resonator_1.initialize_fit_method(MC_iteration=10, MC_fix=2)
# %%

output_dict = resonator_1.fit()
display(output_dict)

# %% repeat but making several FileIO objects and combining them

all_dfs = {}
for csv_filepath in csv_filenames:
    file_io = FileIO(csv_filepath, verbose=False)
    csv_filename = csv_filepath.stem
    df = file_io.load_csv(csv_filepath)
    
    all_dfs[csv_filename] = df
    
    
# %%

list_of_dfs = list(all_dfs.values())
df1 = list_of_dfs[1]
df2 = list_of_dfs[2]

test = pd.concat(list_of_dfs)

by_row_index = test.groupby(test.index)
df_means = by_row_index.mean()

# %%

resonator_avgd = Resonator(df=df_means, preprocess_method='circle', fit_method_name='DCM', filename="TestResonator")
output_dict = resonator_avgd.fit()




# %%
