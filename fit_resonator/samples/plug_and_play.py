import sys  # update paths
import os  # import os in order to find relative path
import glob
import requests

path_to_resfit = './scresonators'
pglob = glob.glob(path_to_resfit)
assert len(pglob), f'Path: {path_to_resfit} does not exist'
sys.path.append(path_to_resfit)

import fit_resonator.resonator as scres

# Load sample resonator data into a local file
directory = os.getcwd()
filename = directory + '/' + 'scres_example.csv'
# data BOE 230325 4 4p513GHz -15dB 13mK
url = 'https://raw.githubusercontent.com/Boulder-Cryogenic-Quantum-Testbed/scresonators/master/fit_resonator/Resources/sample_data.csv'
r = requests.get(url).content.decode('utf-8')
f = open(filename,'w')
f.write(r)
f.close()

# Create resonator object from the sample data
my_resonator = scres.Resonator()
# my_resonator.from_file(filepath=filename, measurement='S21')
my_resonator.from_file(filepath='scres_example.csv')

# Set fit parameters
fit_type = 'DCM'
MC_iteration = 10
MC_rounds = 1e3
MC_fix = []
manual_init = None
my_resonator.preprocess_method = 'circle' # Preprocess method: default = linear
my_resonator.filepath = './' # Path to fit output

# Perform a fit on the data with given parameters
my_resonator.fit_method(fit_type, MC_iteration, MC_rounds=MC_rounds, MC_fix=MC_fix, manual_init=manual_init, MC_step_const=0.3)
my_resonator.fit('png')

# Remove sample data
os.remove(filename) 