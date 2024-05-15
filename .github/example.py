"""
This code is designed to be a quick example of the scresonators package. It will use data cloned 
with the scresonators github to perform a fit and output the results to a local file. Just it using

i.e. `python scresonators/example.py`

Feel free to modify to help you understand how to use the package.
"""

import sys
import os 
import glob

try:
    path_to_resfit = os.path.abspath(os.path.dirname( __file__ ))
except:
    print('Could not find relative path to scresonators. Please update manually.')
    path_to_resfit = './scresonators'

pglob = glob.glob(path_to_resfit)
assert len(pglob), f'Path: {path_to_resfit} does not exist'
sys.path.append(path_to_resfit)

import fit_resonator.resonator as scres

data = path_to_resfit + '/fit_resonator/Resources/sample_data.csv'

# Create resonator object from the sample data
my_resonator = scres.Resonator()
my_resonator.from_file(filepath=data)

# Set fit parameters
fit_type = 'DCM'
MC_iteration = 10
MC_rounds = 1e3
MC_fix = []
manual_init = None
my_resonator.preprocess_method = 'circle'
my_resonator.filepath = './' # Path to fit output

# Perform a fit on the data with given parameters
my_resonator.fit_method(fit_type, MC_iteration, MC_rounds=MC_rounds, MC_fix=MC_fix, 
                        manual_init=manual_init, MC_step_const=0.3)
my_resonator.fit('png')