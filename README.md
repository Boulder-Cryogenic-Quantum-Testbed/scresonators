# scresonators
Welcome to the scresonators repository of the Boulder Cryogenic Quantum Testbed! This code is made to fit complex S21 data for hangar type resonators.

## Functionality
The scresonators library supports fitting using the:
* Diameter Correction Method (DCM)
* Inverse S21 method (INV)
* Closest Pole and Zero Method (CPZM)
* Phi Rotation Method (PHI)

Additionally, the code is able to fit reflection type geometry resonators with an altered version of the Diameter Correction Method (DCM REFLECTION)


## Directory Structure
scresonators
* fit_resonator
* * Python Code to Fit Microwave Resonator Data

* temp_control
* * Code to be used in iterative fitting functionality (under development)
* pna_control
* * Code to be used in iterative fitting functionality (under development)
* filler_control
* * Code to be used in iterative fitting functionality (under development)

# How to use the Library:

## Importing the Library
1. Install the library and its relevant dependencies with `pip install scresonators-fit`
2. Lead your python code with `import fit_resonator.resonator as resonator`
 
## Using the library

Fitting resonator data will revolve around the resonator class. The resonator class includes many helpful functions for working with resonator data.

#### Setting Fit Variables:

###### User must initialize a Fit_Method class instance with arguments: 
* fit_type
* MC_iteration
* MC_rounds
* MC_fix
* manual_init
* MC_step_const

fit_type: 'DCM','DCM REFLECTION','PHI','INV' or 'CPZM' for the method user wishes to run

MC_iteration: The number of times the user wants the Monte Carlo fit to run

MC_rounds: The number of iterations the Monte Carlo fit will do per run. Default is 100 if not defined

MC_fix: An array of which variables the MC fit will not change during iteration.
   >An example of MC_fix is as follows:
   `MC_fix = ['w1','Qi']`
   >The strings user can use for MC_fix are as follows: 'Q','Qi','Qc','w1','phi','Qa'

manual_init: Used to define initial guess variables
   >If the user wants to have the program auto guess parameters, they can set manual_init equal to None
   
   >If the user wants to define their own initial guess parameters, they must define it in the following format: 
   `manual_init = [1,2,3,4]`
    1 = Qi
    2 = Qc
    3 = resonance frequency (GHz)
    4 = phi (radians) or Qa (Qa only used for CPZM)

> Note that if using CPZM, 4 needs to be Qa not phi. If using any other method, 4 needs to be phi.

MC_step_const: Range for the random parameter values chosen in MC fit. This scaling is exponential. The larger this number, the higher and lower the random values

#### Fitting Data:

Initialize resonator object with:
`my_resonator = scres.Resonator()`

Initialize raw data or file into resonator object with:
`my_resonator.from_columns(raw_data)`
`my_resonator.from_file(filename)`

You can also pass the data as a filename or data object when initializing your resonator object with:
`my_resonator = scres.Resonator(filepath='PATH/TO/FILE')`
`my_resonator = scres.Resonator(data=raw_data)`

If you're using a snp file with more than three columns please specify to the resonator object which value to use with:
`my_resonator.from_file('PATH/TO/FILE.snp', "S12")`
`my_resonator = scres.Resonator(filepath='PATH/TO/FILE.snp', measurement="S12")`
or with the index value(s)
`my_resonator.from_file('PATH/TO/FILE.snp', [2,3])`
`my_resonator = scres.Resonator(filepath='PATH/TO/FILE.snp', measurement=[2,3]`


Define the parameters of your fitting method with:
my_resonator.fit_method(method: str,
MC_iteration=None,
MC_rounds=100,
MC_weight='no',
MC_weightvalue=2,
MC_fix=[],
MC_step_const=0.6,
manual_init=None):)

Call the fitting function once resonator class hold all relevant information with:
`params1,fig1,chi1,init1 = my_resonator.fit()`

If the user wants to have the code remove their background, they need to include the path to their background removal file in resonator object initialization with:
`my_resonator = scres.Resonator(background = background_file)`
For scaling file data and background file data
`my_resonator.from_file(filepath, fscale)`
`my_resonator.init_background(filepath, fscale)`

normalize: The number of points from the start/end of S21 data the user wants to use in the linear fit of S21 data for magnitude and phase for normalization, set with:
`my_resonator = scres.Resonator(normalize = integer value)`

Remember for these special initializations you can include as many as you need, just don't forget the keyword arguments!
`my_resonator = scres.Resonator(background = background_file, normalize = 5, filepath = "PATH/TO/FILE, measurement = "S21")`


#### Here's an example using some of the data hosted on this repository. (needs to be updated now that cryores is deleted) (could put an example snp or csv in Resources to fit)

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

You can find more examples of user code [right here](https://raw.githubusercontent.com/Boulder-Cryogenic-Quantum-Testbed/measurement/master/fit_resonator/user_files).



### INPUT:
The code takes in a file (currently accepts .snp .csv and .txt files). If the file is a .snp file it should follow the .snp format. If the file is a .csv or .txt it should contain 3 columns separated by commas where each line represents one point of data.
   >Headers are not accepted in the file .csv and .txt files, the code only accepts the data with a header for .snp

Format for .csv and .txt files:

1. The first column is monotonically increasing frequency in GHz
1. The second column is magnitude of S21 in dB (log mag)
1. The third column is phase of S21 in degrees
   >More information regarding standard data format can be found here https://github.com/Boulder-Cryogenic-Resonator-Testbed/measurement/issues/19

The user has the option of including a background removal file in order to have a more accurate fit. This is recommended if the background is not linear.
   >Background file needs to be of the same format as the main data file as described with the three columns above.

User needs to ensure that the data has at least 20 points otherwise the code will fail

If user does not include points near off resonance in addition to points near resonance, fitting will not be accurate and could fail entirely
   >In simple terms: The fitting needs a full circle (in complex plane) to work optimally

### OUTPUT:

All output will be put in a new folder titled with a timestamp in the folder with the user's data.

1. If the user has a background removal file, the code will output graphs displaying the main data and the background for both the magnitude and phase
1. The code will output four figures showing the steps it is taking to normalize the data titled Normalize_1 through Normalize_4
1. If the user opted to have the code guess initial parameters, the code will output three figures showing the steps taken to find resonance and phi guesses
1. The code outputs a figure displaying a variety of information regarding how the fit was completed. This includes a plot of both the raw and final fit of data
in the complex plane, plots of the linear fits that normalize the data for both magnitude and phase, plots of the normalized magnitude and phase with their final
fits, manually input guess parameters, and the final parameters found from the fit with their 95% confidence intervals.
1. The code will also print a .csv file displaying the information that the fit has gathered with each term on a new line:
    * DCM/DCM REFLECTION/PHI: Q, Qi, Qc, 1/Re[1/Qc], phi, f_c
    * INV: Qi, Qc*, phi, f_c
    * CPZM: Qi, Qc, Qa, f_c

>User has the option of disabling extra graphs such as the normalization process graphs when calling the fit_resonator function

#### Check Data:

A simple script to detect if the user's data file is of the correct format.

To use, simply change the path variables dir to directory with file, and filename to the name of the file to be checked.

To use, import the module:

`import fit_resonator.check_data as cd`

Then run the method corresponding to your data form as a file, or raw data as arrays/array-likes:

`cd.file(path_to_data_file)`

`cd.raw(frequency, magnitude, phase)`

This code will check data files for:
* Header (not currently part of standard format)
* Correct number of columns
* Correct delimiter (currently set to ',')

Files and raw data for:
* Containing more than 1 line of data
* Frequency in GHz (determined by checking if frequency is above 10^8 in magnitude)
* Phase in radians (determined by checking if phase less/greater than 2pi)

>Code does not check if magnitude is in dB

The code will prompt the user to check if they would like to make a new file with an edited version of their data using the correct format for each individual change.


# Help us improve

Please alert us to any issues you encounter so we may improve our existing tools as we work to expand the library's functionality.

## Contributing/Modifying
1. Fork the repository and create a new branch to commit your changes to.
2. clone the forked repository into a folder of your choice; we ***strongly*** recommend using [virtual environments](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) for managing your dependences.
3. Install the dependencies by running:
  `pip install -r requirements.txt`
4. Submit a pull request with changes made to your branch.

## Code Organization

For fitting code collaboration, all code should live in the `fit_resonator` namespace. This ensures easy integration
with other Python packages, and avoids name collisions; everything is referred
to as e.g. `fit_resonator.experiments` rather than just `experiments`.

# References

## Fitting Functions:
![alt text](https://raw.githubusercontent.com/Boulder-Cryogenic-Quantum-Testbed/measurement/master/fit_resonator/Resources/Fit_Equations.PNG)

## Publications
1. DCM: M. S. Khalil, M. J. A. Stoutimore, F. C. Wellstood, and K. D. Osborn    Journal of Applied Physics 111, 054510 (2012); doi: 10.1063/1.3692073
2. INV: A. Megrant et al.     APPLIED PHYSICS LETTERS 100, 113510 (2012)
3. CPZM: Chunqing Deng, Martin Otto, and Adrian Lupascu University of Waterloo, Waterloo, Ontario N2L 3G1, Canada (9 August 2013)
4. PHI: J. Gao, "The Physics of Superconducting Microwave Resonators" Cal-tech Ph.D. thesis, May 2008
