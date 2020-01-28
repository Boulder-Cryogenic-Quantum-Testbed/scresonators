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

## Qi order 100:

#### Impedance Mismatch = 0 Ohms
![alt text](https://raw.githubusercontent.com/Boulder-Cryogenic-Quantum-Testbed/measurement/master/Resfit/circuit_simulation_results/extract_factor/2_L%3D0.png)

#### Impedance Mismatch = 0.32 Ohms

![alt text](https://raw.githubusercontent.com/Boulder-Cryogenic-Quantum-Testbed/measurement/master/Resfit/circuit_simulation_results/extract_factor/2_L%3D1.png)

#### Impedance Mismatch = 7.6 Ohms

![alt text](https://raw.githubusercontent.com/Boulder-Cryogenic-Quantum-Testbed/measurement/master/Resfit/circuit_simulation_results/extract_factor/2_L%3D5.png)

The results of the tests for Qi at order 100 indicate a few things. The first is that the fit function appears to be the most accurate at an extract factor of 1 3dB bandwidth.
The second thing visible is that past a Qi/Qc ratio of 10 (the limit where Qi is dominating), the fit function loses much of its accuracy.
The third thing to notice is that impedance mismatch appears to have very little effect on the results in this regime.
The fourth take away from this section is that DCM, INV and CPZM fits all appear to give similar accuracy across all parameters tested here with the exception that INV does better at larger Qi/Qc ratios.

## Qi order 10,000:

![alt text](https://raw.githubusercontent.com/Boulder-Cryogenic-Quantum-Testbed/measurement/master/Resfit/circuit_simulation_results/extract_factor/4_L%3D0.png)

#### Impedance Mismatch = 0.32 Ohms

![alt text](https://raw.githubusercontent.com/Boulder-Cryogenic-Quantum-Testbed/measurement/master/Resfit/circuit_simulation_results/extract_factor/4_L%3D1.png)

#### Impedance Mismatch = 7.6 Ohms

![alt text](https://raw.githubusercontent.com/Boulder-Cryogenic-Quantum-Testbed/measurement/master/Resfit/circuit_simulation_results/extract_factor/4_L%3D5.png)

The results of the tests for Qi at order 10,000 indicate a few things. The first is that the fit function appears to be the most accurate at an extract factor of 1 3dB bandwidth just like for Qi of order 100.
The second thing visible is that past a Qi/Qc ratio of 10 (the limit where Qi is dominating), the fit function loses much of its accuracy.
The third take away is that the impedance mismatch when combined with inaccuracy from a high Qi/Qc regime, appears to create more error than without an impedance mismatch.
The fourth take away from this section is that DCM, INV and CPZM fits all appear to give similar accuracy across all parameters tested here with the exception that INV does better at larger Qi/Qc ratios.
As shown, the only noticeable source of error appears to be the high Qi/Qc range for this order of Qi.

## Qi order 1,000,000

![alt text](https://raw.githubusercontent.com/Boulder-Cryogenic-Quantum-Testbed/measurement/master/Resfit/circuit_simulation_results/extract_factor/6_L%3D0.png)

#### Impedance Mismatch = 0.32 Ohms

![alt text](https://raw.githubusercontent.com/Boulder-Cryogenic-Quantum-Testbed/measurement/master/Resfit/circuit_simulation_results/extract_factor/6_L%3D1.png)

#### Impedance Mismatch = 7.6 Ohms

![alt text](https://raw.githubusercontent.com/Boulder-Cryogenic-Quantum-Testbed/measurement/master/Resfit/circuit_simulation_results/extract_factor/6_L%3D5.png)

The results of the tests for Qi at order 10,000 indicate a few things. The first is that the fit function appears to be the most accurate at an extract factor of 1 3dB bandwidth just like for the two previous Qi orders of magnitude.
The second thing visible is that, once again, past a Qi/Qc ratio of 10 (the limit where Qi is dominating), the fit function loses much of its accuracy.
The third take away, shown in a more extreme case at Qi order 1,000,000 than at order 10,000 is that impedance mismatch in the realm of high Qi/Qc ratio creates even more error than without the mismatch.
The fourth take away from this section is that DCM, INV and CPZM fits all appear to give similar accuracy across all parameters tested here with the exception that INV does better at larger Qi/Qc ratios.

## Overall Conclusions

The difference in percent error between 1 bandwidth and 100 bandwidths becomes very apparent in certain areas of the parameter space shown here. Particularly when
Qi is small, when Qi/Qc is large and when there is a large impedance mismatch to go with a high Qi/QC ratio. All orders of Qi display this behavior. These graphs
clearly indicate that having a low value for the extract factor will give the most accurate results for this set of data. However, this needs to be tested when the
data is noisy to simulate a more realistic data collection environment. It is believed that the reason for this result is that the normalization process becomes less
and less accurate as points approach off resonance.

The results collected from these trials clearly indicate that a Qi/Qc ratio larger than 10 will cause a large percent error to become apparent. However, this is
believed to be due to the normalization process. As Qi/Qc approaches the range where Qi dominates the ratio, it becomes difficult to have many points near resonance
at the same time as having many points off resonance to establish a good linear fit to normalize the data unless a very large number of points are taken
(greater than 10,000).

Additionally, the percent error across various impedance mismatches does not appear to change as much for INV as it seems to for DCM, however this is misleading.
The difference is due to the fact that the definitions for 3dB bandwidths change depending on which fit is being used.
Originally the 3dB bandwidth is defined with resect to Q, however, once the data has been inversed the 3dB bandwidth is defined with respect to Qi.
It is of importance to note that the act of inversing the S21 data will cause the number of points being fit in INV and DCM to differ, with INV having less points
used for fitting in almost every case. This is equivalent to using a small extract factor in DCM.
Although the graphs may indicate that INV has better results for given 3dB bandwidths used (due to the change in definition of 3dB bandwidth), if a similar number
of points are used in both fits they will get similar results.
It is believed that the reason for this increased accuracy near resonance is that the error from a non-perfect normalization becomes more apparent as points approach
off resonance. This error is likely a consequence from the way the data is pre-processed. Thus it is recommended that the user take their data in a way that this
normalization will be as accurate as possible. Namely, the user should take as much data as they can off resonance while keeping a good number of points near resonance
to fit their data with.

Ideally, the smallest possible bandwidth would be used. However, in a practical measurement with noise and other contributing factors this is not always possible.
Therefore, in the case of this program an extract factor of 1 (corresponding to data within 1 3dB bandwidth centered around resonance) will be used.
This ensures that the program will have a reasonable number of points to fit with provided that the data is taken around resonance, while still being low enough
that the accuracy of the fitting will not be compromised to a considerable extent due to the assumptions built into the normalization.

The user must also keep in mind that the INV method will require more data near resonance due its use of Qi instead of Q for the 3dB bandwidth calculation.
