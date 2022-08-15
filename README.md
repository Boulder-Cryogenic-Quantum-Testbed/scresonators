# scresonators
Welcome to the scresonators repository of the Boulder Cryogenic Quantum Testbed! This is a library for measuring the loss in superconducting resonators. 


## Importing the Library
1. Install the library and its relevant dependencies with `pip install scresonators-fit`
2. Lead your python code with `import fit_resonator.resonator as resonator`

## Contributing/Modifying
1. clone the repository into a folder of your choice with `git clone https://github.com/Boulder-Cryogenic-Quantum-Testbed/scresonators.git`
2. Install the dependencies, we ***strongly*** recommend using [virtual environments](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) for managing your dependences. To install dependencies run:
  `pip install -r requirements.txt`
3. If you are running on Windows, install Microsoft Visual Studio before using the library
 
## Using the library

Fitting resonator data will revolve around the resonator class.

Here's an example using some of the data hosted on this repository. Hosted
datasets from groups around the world can be found [here](/cryores/test_data).

```python
import numpy as np
import fit_resonator.resonator as scres

# The object all following code will be called from
my_resonator = scres.Resonator()

# Load the raw data
url = 'https://raw.githubusercontent.com/Boulder-Cryogenic-Quantum-Testbed/scresonators/master/cryores/test_data/AWR/AWR_Data.csv'
raw = np.loadtxt(url, delimiter=',')
# Can also use our file input system of my_resonator.from_file(url)

# Test with file load into class
my_resonator.from_columns(raw)

# Assign your desired fit method variables
fit_type = 'DCM'
MC_iteration = 10
MC_rounds = 1e3
MC_fix = ['w1']
manual_init = None

# Pass these to your resonator object
my_resonator.fit_method(fit_type, MC_iteration, MC_rounds=MC_rounds, MC_fix=MC_fix, manual_init=manual_init,
                 MC_step_const=0.3)

# Fit!
my_resonator.fit()
```

Ane in depth description is given in the fit_resonators folder.


## Code Organization

For fitting code collaboration, all code should live in the `fit_resonator` namespace. This ensures easy integration
with other Python packages, and avoids name collisions; everything is referred
to as e.g. `fit_resonator.experiments` rather than just `experiments`.
