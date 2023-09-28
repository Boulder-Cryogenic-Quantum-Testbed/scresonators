import sys
import os 
import glob
import numpy as np

try:
    path_to_resfit = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
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

def test_values():
    # Perform a fit on the data with given parameters
    my_resonator.fit_method(fit_type, MC_iteration, MC_rounds=MC_rounds, MC_fix=MC_fix, 
                            manual_init=manual_init, MC_step_const=0.3)
    output_params, conf_array, error, init = my_resonator.fit(None)
    if not np.allclose(output_params, [306209.29520741466, 928387.8353093242, 4512506964.96403, -0.15431421139237836]):
        raise Exception("Different output parameters than expected in user_test.py")
    if not np.allclose(conf_array, [588.1821370752295, 1685.6238136050524, 1634.941474343068, 1654.6028376782779, 0.0017515102820822026, 14.244794845581055]):
        raise Exception("Different confidence array than expected in user_test.py")
    if not np.allclose(error, [0.00012610664079341314]):
        raise Exception("Different error than expected in user_test.py")