# measurement
Welcome to the measurement repostository of the Boulder Cryogenic Quantum Testbed! This is a library for measuring the loss in superconducting resonators. 

## Installation
1. clone the repository into a folder of your choice with `git clone https://github.com/Boulder-Cryogenic-Quantum-Testbed/measurement.git`
1. Install the dependencies, we ***strongly*** recommend using [virtual environments](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) for managing your dependences. To install dependencies run:
  `pip install -r requirements.txt`
  
## Using the library

Here's an example using some of the data hosted on this repository. Hosted
datasets from groups around the world can be found [here](/cryores/test_data).

This particular example code is meant to be run in the measurement directory.

```python
import numpy as np
import resfit.fit_resonator.fit_functions as ff
import resfit.fit_resonator.fit_S_data as fsd
import resfit.fit_resonator.resonator as res

url = 'https://raw.githubusercontent.com/Boulder-Cryogenic-Quantum-Testbed/measurement/master/cryores/test_data/AWR/AWR_Data.csv'

# Load the raw data:
raw = np.loadtxt(url, delimiter=',')

# Choose a fitting method:
fit_type = 'DCM'
MC_iteration = 10
MC_rounds = 1e3
MC_fix = ['w1']
manual_init = None
method = res.FitMethod(fit_type, MC_iteration, MC_rounds=MC_rounds,\
            MC_fix=MC_fix, manual_init=manual_init, MC_step_const=0.3)

# Fit the data:
fsd.fit_resonator("output test", method, normalize = 10, data_array = raw)
```

A more in depth example is given in the resfit folder.


## Code Organization

All code should live in the `resfit` namespace. This ensures easy integration
with other Python packages, and avoids name collisions; everything is referred
to as e.g. `resfit.experiments` rather than just `experiments`.
