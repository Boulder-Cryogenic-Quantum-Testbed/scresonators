# Python Code to Fit Microwave Resonator Data
>Keegan Mullins (CU Boulder NIST Affiliate)
Corey Rae McRae (NIST)
Haozhi Wang (NIST)

#### This code is a modified version of the original code written by:

>Kevin Osborn (University of Maryland, LPS)
Chih-Chiao Hung (University of Maryland, LPS)

## Starting Notes

This code is made to fit complex S21 data for hangar type resonators using the Diameter Correction Method (DCM), Inverse S21 method (INV), Closest Pole and Zero Method (CPZM), and Phi Rotation Method (PHI) fittings

Additionally, the code is able to fit reflection type geometry resonators with an altered version of the Diameter Correction Method (DCM REFLECTION)

1. DCM: M. S. Khalil, M. J. A. Stoutimore, F. C. Wellstood, and K. D. Osborn    Journal of Applied Physics 111, 054510 (2012); doi: 10.1063/1.3692073
1. INV: A. Megrant et al.     APPLIED PHYSICS LETTERS 100, 113510 (2012)
1. CPZM: Chunqing Deng, Martin Otto, and Adrian Lupascu University of Waterloo, Waterloo, Ontario N2L 3G1, Canada (9 August 2013)
1. PHI: J. Gao, "The Physics of Superconducting Microwave Resonators" Cal-tech Ph.D. thesis, May 2008

## Fitting Functions:
![alt text](https://raw.githubusercontent.com/Boulder-Cryogenic-Quantum-Testbed/measurement/master/Resfit/Fit_Equations.PNG)

## INPUT:
The code takes in a .csv file (accepts .txt as well), containing 3 columns separated by commas where each line represents one point of data
   >Note that there cannot be a header in this file, the code only accepts the data with no header

1. The first column is monotonically increasing frequency in GHz
1. The second column is magnitude of S21 in dB (log mag)
1. The third column is phase of S21 in radians
   >More information regarding standard data format can be found here https://github.com/Boulder-Cryogenic-Resonator-Testbed/measurement/issues/19

The user has the option of including a background removal file in order to have a more accurate fit. This is recommended if the background is not linear.
   >Background file needs to be of the same format as the main data file as described with the three columns above.

User needs to ensure that the data has at least 20 points otherwise the code will fail

If user does not include points near off resonance in addition to points near resonance, fitting will not be accurate and could fail entirely
   >In simple terms: The fitting needs a full circle (in complex plane) to work optimally

## OUTPUT:

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

-----------------------------------------------------------------------------------

## Code Overview

1. From user python file, code takes in user data file name and passes it to Fit_Resonator function along with user preferences and an optional background removal file name
1. If user has a background file, the code will use it to remove the background for a more accurate fit
1. Data is pre-processed using a linear fit for both magnitude and phase of S21 so that start and end points of S21 data are at (1,0i) in the complex plane
1. If the user does not manually initialize a guess, the code will attempt to find a guess for fit parameters using Find_initial_guess function
1. Once the code has a guess for fit parameters, it will crop the data to points near resonance
1. After cropping, the code will minimize the guess parameters based on data points using a least squares fit then compare parameters to a Monte Carlo fit
   >This step will be repeated until the Monte Carlo fit does not give better results than the minimization.
   >Monte Carlo fit is meant to check if fitting parameters are trapped in a local minimum.
   >If Monte Carlo fit got better results, the parameters obtained from Monte Carlo fit will be minimized.

1. At this point the final parameter values have been found and fitting is complete. Final fitting is plotted and fit parameters are written to a .csv file

-------------------------------------------------------------------------------------------------------------------------------------------------------------

## Installation

User will need the following python modules:
* numpy
* matplotlib
* pandas
* lmfit
* sympy
* scipy
* inflect

> Install a module with "pip install ______" on the command line for python 3.
Must install pip module before installing pips for python 2. Once pip is installed for python 2, install modules the same way as python 3.

-------------------------------------------------------------------------------------------------------------------------------------------------------------

## USER INPUT FILE

###### An example of the way the following is done is included in the example file (User_Example.py)

#### Section 1: Initial Setup

###### The user will need to set the "dir" variable equal to the directory of folder containing user file. This can be done with the command:

`dir = "your path to data folder here"`

###### User will need to set the "filename" variable to the name of their file with:
`filename = 'your name here.csv'`
   >Note that code accepts both .txt and .csv file formats
   
###### The user will then have to set the "filepath" variable to be equal to their directory plus the filename:
`filepath = dir+'/'+filename`

#### Section 2: Setting Fit Variables

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

#### Section 3: Fitting Data

The user calls the Fit_Resonator function with:

`params1,fig1,chi1,init1 = Fit_Resonator(filename,filepath,Method,normalize,dir)`

If the user wants to have the code remove their background, they need to include the path to their background removal file as well with:

`params1,fig1,chi1,init1 = Fit_Resonator(filename,filepath,Method,normalize,dir,path_to_background)`

normalize: The number of points from the start/end of S21 data the user wants to use in the linear fit of S21 data for magnitude and phase for normalization, set with:

`normalize = 10`

User sets the path to their background removal file with:

`path_to_background = dir+'/'+background_file_name`
   >Here it is assumed that dir is the same directory for both the user's main data file and the background and can thus be used for both

-------------------------------------------------------------------------------------------------------------------------------------------------------------
# Fit Function Code

>From here on, the Readme will cover the code that is actually fitting the user data. Reading through this portion is not necessary for the user to fit their data, however, it is listed here for the user to read through to be able to see how the code works.

## Fit_Resonator Function

* Load data from user file
* Create folder for output files
* Remove background if applicable
* Initialize Resonator class with user data

* Normalize data
    * Output plot "Normalize_1" to show data before normalization
    * First remove cable delay by subtracting a linear fit of phase using the first 10 and last 10 points of data (10 by default, changed with variable normalize)
    * Output plot "Normalize_2" to show data after cable delay has been removed (subtract slope of linear fit for phase)
    * Output plot "Normalize_3" to show data after first and last points of data have been rotated to the real axis (subtract intercept of linear fit for phase)
    * Normalize magnitude in dB by subtracting a linear fit of magnitude using the first 10 and last 10 points of data (also changed with variable normalize)
    * Output plot "Normalize_4" to show data after subtraction of linear fit of magnitude (at this point, normalization is complete)
* Find initial guess parameters and put them into init variable
* kappa: Defined to be f_0/Q (DCM,DCM REFLECTION,PHI,CPZM) or f_0/Qi(INV), the bandwidth of frequencies at which the circle is at 90 degrees on either side of resonance
* Set xdata and ydata to contain the raw data for the correct frequency and transmission datapoints within the bandwidth determined by kappa and extract_factor
   >Note that all data points not within the bandwidth will not be used for the fit.
   >extract_factor currently set to 1 to correlate with a single 3dB bandwidth to fit data with
* If the method is INV, inverse the ydata values
* Define 4 parameters "params"[0:3] based on initial guess.
    >params[0] = Q for DCM,DCM REFLECTION,PHI and params[0] = Qi for INV,CPZM
    
    >params[3] =Qa for CPZM and params[3] = phi for everything else
* Call to min_fit() function to minimize parameters. Outputs minimized parameters according to the fit function chosen and their 95% confidence intervals
* Set up variables for the while loop of the minimize and Monte Carlo fit functions. Loop runs at most 5 times by default (defined by user) and at least once
* Call Monte Carlo fit to find if the parameters are trapped in a local minimum using xdata, ydata, fit_params and the method
* If Monte Carlo fit does not have better accuracy than the minimize function, terminate the while loop and update the parameters into variable "output_params"
    * Else, continue minimize function and Monte Carlo fit for another iteration
* If Monte Carlo got better fit, minimize those parameters with call to min_fit(), function outputs minimized parameters and 95% confidence interval error.
* Check that bandwidth of extracted data is not equal to zero
* Plot the fit and save values found to output file, both with title according to the method used
   >Plot has the following information:
   1. Graph of magnitude of S21 before normalization (blue) with linear fit shown (orange) shown in upper right
   1. Graph of magnitude of S21 after normalization (blue) and final fit (green) below #1
   1. Graph of phase of S21 before normalization (blue) with linear fit shown (orange) below #2
   1. Graph of phase of S21 after normalization (blue) and final fit (green) below #3
   1. Final fitting parameters with their 95% confidence intervals below #4
   1. Graph of S21 in complex plane shown in bottom left

-------------------------------------------------------------------------------------------------------------------------------------------------------------

## Find_initial_guess Function

* Reassemble S21 from real and imaginary parts y1 and y2 into variable y

* Inverse y if method is INV
* Reset variables y1 and y2 in case y was inversed
* Find approximate circle fit with call to Find_Circle(): x_c = real coordinate of center of circle, y_c = imaginary coordinate of center of circe, r = radius
   >Given a set of x,y data return a circle that fits data using LeastSquares Circle Fit 
Randy Bullock (2017)
* Store location of center of circle in complex variable z_c
* Plot the data to show the circle fit in plot "circle.png"

* Subtract 1 from the complex data such that off resonant point P is at the origin and determine phi to be the angle from point P to the center of the circle
* Determine resonance point to be the point with largest magnitude (this means point furthest from off resonance point P)
* Plot the data to show the determined resonant point in plot "resonance.png" (resonant point shown as red star)
* Rotate data by phi, then plot to show data after phi rotation in plot "phi.png"

The code here is different depending on the method used due to how each method goes about guessing its respective initial guess parameters:

DCM/PHI{
* Q_Qc set to be the diameter of the circle (found by getting distance from off resonant point P at the origin to point with max amplitude, aka. resonance)
* y_temp set equal to magnitude of S21 - S21/sqrt(2) such that y_temp will be equal to zero at approximately the 3dB data points
* Set idx1 and idx2 to be the indices of the 3dB data points
* Set kappa to be the bandwidth (absolute value of one 3dB data point minus the other)
* Set Q to be resonance frequency divided by kappa. This is the definition of Q for DCM
* Set Qc to be Q divided by the diameter (variable Q_Qc) because the diameter is Q/Qc
* Fit variables Q and Qc based on ideal resonator behavior: curve One_Cavity_peak_abs
* Set the variable init_guess to the initial guess parameters found

}

DCM REFLECTION{
* Q_Qc set to be the diameter of the circle divided by 2 (found by getting distance from off resonant point P at the origin to point with max amplitude)
* y_temp set equal to magnitude of S21 - S21/sqrt(2) such that y_temp will be equal to zero at approximately the 3dB data points
* Set idx1 and idx2 to be the indices of the 3dB data points
* Set kappa to be the bandwidth (absolute value of one 3dB data point minus the other)
* Set Q to be resonance frequency divided by kappa. This is the definition of Q for DCM
* Set Qc to be Q divided by the half diameter (variable Q_Qc) because the diameter is 2Q/Qc
* Fit variables Q and Qc based on ideal resonator behavior: curve One_Cavity_peak_abs_REFLECTION
* Set the variable init_guess to the initial guess parameters found

}

INV{
* Qi_Qc set to be the diameter of the circle (found by getting distance from off resonant point P at the origin to max amplitude, aka. resonance)
* y_temp set equal to magnitude of S21 - S21/sqrt(2) such that y_temp will be equal to zero at approximately the 3dB data points
* Set idx1 and idx2 to be the indices of the 3dB data points
* Set kappa to be the bandwidth (absolute value of one 3dB data point minus the other)
* Set Qi to be resonance frequency divided by kappa. This is the definition of Qi for INV
* Set Qc to be Qi divided by the diameter (variable Qi_Qc) because the diameter is Qi/Qc
* Fit variables Qi and Qc based on ideal resonator behavior: curve One_Cavity_peak_abs
* Set the variable init_guess to the initial guess parameters found

}

CPZM{
* Q_Qc set to be the diameter of the circle (found by getting distance from off resonant point P at the origin to point with max amplitude, aka. resonance)
* y_temp set equal to magnitude of S21 - S21/sqrt(2) such that y_temp will be equal to zero at approximately the 3dB data points
* Set idx1 and idx2 to be the indices of the 3dB data points
* Set kappa to be the bandwidth (absolute value of one 3dB data point minus the other)
* Set Q to be resonance frequency divided by kappa. This is the definition of Q for DCM, but it can approximate Q for CPZM
* Set Qc to be Q divided by the diameter (variable Q_Qc) because the diameter is Q/Qc for DCM which can approximate Qc for CPZM
* Fit variables Q and Qc based on ideal resonator behavior: curve One_Cavity_peak_abs
* Set Qa to be the negative inverse of the imaginary part of e^(i*phi)/Qc
* Set Qc to be the inverse of the real part of e^(i*phi)/Qc
   >New Qc is 1/Re[1/Q_c] as described in DCM, using raw Qc from diameter is phi rotation method which is not correct
* Set Qic = Qi/Qc
* Set Qia = Qi/Qa
* Set the variable init_guess to the initial guess parameters found [Qi,Qic,f_c,Qia]

}

* Return init_guess, x_c, y_c and r. Here x_c, y_c and r represent the circle when it was first found, before any transformations

-------------------------------------------------------------------------------------------------------------------------------------------------------------

## MonteCarloFit Function

* Set ydata_1stfit equal to S21 data by plugging in values for parameters into chosen method's fit equation
* If weight_array is 'yes': make array weight_array equal to inverse of ydata, else fill entire array with 1s
* Define weighted_ydata to be equal to weight_array times ydata. End result filled with 1s if MC_weight='yes' and exact same array as ydata otherwise
* Define weighted_ydata_1stfit to be equal to weight_array times ydata_1stfit. Equals ydata_1stfit/ydata if MC_weight='yes' else exact same array as ydata_1stfit
* Set error equal to the average vector magnitude (sqrt(n1^2 + n2^2 + ... + nlast^2)/# of terms) where each term is weighted_ydata - weighted_ydata_1stfit
   >In other words, error is the least squares error
* Set error_0 equal to error. This is so the initial value of error is preserved to check against error of Monte Carlo fit

* Run 100,000 iterations of the while loop for Monte Carlo fit by default:

while (counts<100,000){
* Increase counts by 1 each iteration
* Generate an array 'random' of 4 random numbers where each has the value of the initial parameters times the step constant (found in method class)
* If parameter set to be fixed (in user input file), set it's value in array 'random' to be 0
* For all fit functions with phi, divide random value for phi by 10 (slot [3] in random array)
* Set all items in array random as e to the power of their old values
    >Note that the random values are raised as an exponent of e such that the distribution will still remain positive
    By multiplying by the new exponentiated random values by the old parameters, the distribution still remains somewhat linear while ensuring the correct sign
* Multiply the parameters by these random values to get variable new_parameter
* For all fit functions with phi, modulus new phi term by 2pi such that it is between 0 and 2pi in range
* Make new set of S21 data called ydata_MC based on these new random parameters
* Check error the same way as before and put it in new_error variable
* If new_error < error: set parameters to be new parameters as they are better that before
* Set error as new_error to match error of new parameters

}

* If while loop got better results than minimize function parameters, run another iteration of minimize and Monte Carlo, else stop iterations of Monte Carlo fit

* Return values for parameter, stop_MC (determines if it should do another iteration), and error (array of all trials of Monte Carlo)

-------------------------------------------------------------------------------------------------------------------------------------------------------------


