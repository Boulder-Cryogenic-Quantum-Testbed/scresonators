## Extract Factor Test:

This test shows how the accuracy of the fit changes depending on how much of the data away from resonance is used.

## Number of 3dB Bandwidths Used:

1: A single 3dB bandwidth correlating to Q (DCM, DCM REFLECTION, PHI) or Qi (INV, CPZM)

4: The recommended number of bandwidths to use for fitting according to:

>S. Probst, F. B. Song, P. A. Bushev, A. V. Ustinov, and M. Weides      Review of Scientific Instruments 86, 024706 (2015); doi: 10.1063/1.4907935

100: An extreme case correlating to an excessive amount of data used

## Data:

It is apparent from the following graphs that using the full data from the output of AWR will not give as accurate of a result as when the data is cropped to an
appropriate number of 3dB bandwidths. This is likely because the fit equations become less accurate the further the point is from resonance due to approximations
built into them.

#### Qi=118:

![alt text](https://raw.githubusercontent.com/Boulder-Cryogenic-Resonator-Testbed/measurement/master/BCRTresfit/circuit_simulation_results/extract_factor/DCM_extract_Qi%3D118.png)

![alt text](https://raw.githubusercontent.com/Boulder-Cryogenic-Resonator-Testbed/measurement/master/BCRTresfit/circuit_simulation_results/extract_factor/INV_extract_Qi%3D118.png)

#### Qi=11135:

![alt text](https://raw.githubusercontent.com/Boulder-Cryogenic-Resonator-Testbed/measurement/master/BCRTresfit/circuit_simulation_results/extract_factor/DCM_extract_Qi%3D11135.png)

![alt text](https://raw.githubusercontent.com/Boulder-Cryogenic-Resonator-Testbed/measurement/master/BCRTresfit/circuit_simulation_results/extract_factor/INV_extract_Qi%3D11135.png)

While the difference in percent error between 1 bandwidth used and 100 bandwidths used is considerable in the case of the small Qi, the difference between using 1
bandwidth and using 4 does not appear to be very large. This indicates that as long as a reasonable number of bandwidths are used, the data should have a low percent
error for the order of Qi=100.

Additionally, the percent error across various impedance mismatches does not appear to change as much for INV as it seems to for DCM, however this is misleading.
The difference is due to the fact that the definitions for 3dB bandwidths change depending on which fit is being used.
Originally the 3dB bandwidth is defined with resect to Q, however, once the data has been inversed the 3dB bandwidth is defined with respect to Qi.
It is of importance to note that the act of inversing the S21 data will cause the number of points being fit in INV and DCM to differ, with INV having less points
used for fitting in almost every case. This is equivalant to using a small extract factor in DCM.
Although the graphs may indicate that INV has better results for given 3dB bandwidths used
(due to the change in definition of 3dB bandwidth), if a similar number of points are used in both fits they will get similar results.

Ideally, the smallest possible bandwidth would be used. However, in a practical measurement with noise and other contributing factors this is not always possible.
Therefore, in the case of this program an extract factor of 1 (corresponding to data withing 1 3dB bandwidth centered around resonance) will be used.
This ensures that the program will have a reasonable number of points to fit with provided that the data is taken around resonance, while still being low enough
that the accuracy of the fitting will not be compromised to a considerable extent.

The user must also keep in mind that the INV and CPZM methods will require more data near resonance due their use of Qi for the 3dB bandwidth calculation.
