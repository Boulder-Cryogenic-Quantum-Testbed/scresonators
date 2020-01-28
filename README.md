# measurement
Welcome to the measurement repostository of the Boulder Cryogenic Quantum Testbed! This is a library for measuring the loss in superconducting resonators. 

## Installation
1. clone the repository into a folder of your choice with `git clone https://github.com/Boulder-Cryogenic-Quantum-Testbed/measurement.git`
1. Install the dependencies, we ***strongly*** recommend using [virtual environments](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) for managing your dependences. To install dependencies run:
  `pip install -r requirements.txt`
  
## Using the library

Here's an example using some of the data hosted on this repository. Hosted
datasets from groups around the world can be found [here](/cryores/test_data).

```python
import numpy as np
import resfit.fit_resonator.fit_functions as ff
import resfit.fit_resonator.fit_S_data as fsd

url = 'https://raw.githubusercontent.com/Boulder-Cryogenic-Quantum-Testbed/measurement/master/cryores/test_data/AWR/AWR_Data.csv'

# Load the raw data:
raw = np.loadtxt(url, delimiter=',')

# Create an initial guess at parameters:
guess = ff.ModelParams.from_params(Qi = 1E6, Qc=3E5, f_res=5.801,phi=-0.1)

# Choose a fitting method:
method = ff.FittingMethod.DCM

# Fit the data:
fsd.fit_resonator(raw, guess, method, 10, None)
>>>{'Q': 273178.7431584903,
 'Qc': 242764.2007247585,
 'w1': 5.78529287158879,
 'phi': 0.06757726597956264}
```


## Code Organization

All code should live in the `resfit` namespace. This ensures easy integration
with other Python packages, and avoids name collisions; everything is referred
to as e.g. `resfit.experiments` rather than just `experiments`.
