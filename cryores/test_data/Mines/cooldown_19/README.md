# Cooldown 19
* Includes data for the Mines Al 6061 3D cavity measurements

## Directories
* `rt_pin_microscope_tests_210427`
     - Room temperature tests of the coupling quality factor vs. pin insertion
       depth by measuring the pin insertion into the cavity
     - Initial test ended at the cavity wall with too few data points to fit an
       exponential, i.e. Qc ~ exp(-B * z) when the pin and dielectric were lost
       in Ken Smith's office
     - Plan is to repeat the experiment on 210430 following a similar procedure

## Fitting Approach
* Uses the DCM method from the `user_example.py` code in the `measurment` code
  in the test bed github respository (here)
  - Each directory uses a file that calculates the fits, manually changing the
    file names for each trace and manually removing the file headers
  - Data in the csv files has rows with rows entries comma separated as
    f [Hz], |S21| [dB], arg(S21) [deg.]
