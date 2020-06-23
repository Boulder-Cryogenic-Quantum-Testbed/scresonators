## Normalize Test:

This test shows the accuracy of the fits as a function of the number of 3dB bandwidths used to normalize data. The normalization used here is a linear fit of both
magnitude and phase of the points furthest from resonance. 10 points are taken on both ends of the data and a linear fit is found using the 20 points.
The linear fit is then subtracted such that the endpoints of user data are at (1,0) in complex plane.

## Data:

#### Qi/Qc = 0, Impedance Mismatch = 0

![alt text](https://raw.githubusercontent.com/Boulder-Cryogenic-Quantum-Testbed/measurement/master/resfit/circuit_simulation_results/normalize/L%3D0_Qc0.png)

Here, it is clearly shown that for the orders of Qi tested and for all fit types, the percent error of Qi is only considerable for an extract factor of 1.

#### Qi/Qc = 10, Impedance Mismatch = 0

![alt text](https://raw.githubusercontent.com/Boulder-Cryogenic-Quantum-Testbed/measurement/master/resfit/circuit_simulation_results/normalize/L%3D0_Qc1.png)

Here it can be seen that DCM and CPZM for high Qi carry the highest error in their respective fits. The error of all fit functions for all Qi values appears to
converge within a 1-2% error range at some point close to a value of 10 for the number of 3dB bandwidths used for normalization.

#### Qi/Qc = 1/10, Impedance Mismatch = 0

![alt text](https://raw.githubusercontent.com/Boulder-Cryogenic-Quantum-Testbed/measurement/master/resfit/circuit_simulation_results/normalize/L%3D0_Qc-1.png)

Here, it can be seen that nearly all of the functions are close to each other in error right from the beginning with a percent error of less than 5 for all even at
only one 3dB bandwidth being used to normalize. All fit functions for all Qi values tested fall below 1% error at 3 3dB bandwidths used to normalize.

#### Qi/Qc = 0, Impedance Mismatch = 0.32

![alt text](https://raw.githubusercontent.com/Boulder-Cryogenic-Quantum-Testbed/measurement/master/resfit/circuit_simulation_results/normalize/L%3D1_Qc0.png)

Right away, it can be seen that all the fit functions for all Qi values tested seem to converge at 2 3dB bandwidths used to normalize. It can also be seen that
error is very small for values of 2 and beyond for bandwidth used.

![alt text](https://raw.githubusercontent.com/Boulder-Cryogenic-Quantum-Testbed/measurement/master/resfit/circuit_simulation_results/normalize/L%3D1_Qc1.png)

Here it can be seen that CPZM carries the highest error in its fit. The error of all fit functions for all Qi values appears to converge below 1% error at around
a value of 5 for the number of 3dB bandwidths used for normalization.

## Overall Conclusions

It can be gathered from these graphs that the best fitting is achieved for the largest number of 3dB bandwidths used. This may seem intuitive, but it is important to note.
More importantly, the necessary number of 3dB bandwidths for normalization depends on the regime of Qi/Qc being used in addition to the impedance mismatch. This means that
a larger number of 3dB bandwidths may be necessary for a better normalization when the data is taken for a resonator with a high impedance mismatch and/or a Qi/Qc ratio
nearing an order of magnitude different from 1. The graphs would indicate that it is best for the user to take data in a range of up to even 10 3dB bandwidths as it may be
necessary for some of the worst case scenarios. While a bandwidth of only 2 may be all that is necessary for some of the better cases (specifically when Qi and Qc are of the
same order and there is no impedance mismatch).
