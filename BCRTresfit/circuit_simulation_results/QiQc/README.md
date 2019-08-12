## Qi/Qc Test:

This test shows the accuracy of the fits as a function of changing Qc values for three different values of Qi.

## Data:

![alt text](https://raw.githubusercontent.com/Boulder-Cryogenic-Resonator-Testbed/measurement/master/BCRTresfit/circuit_simulation_results/QiQc/DCM.png)

This data illustrates that there appears to be a range of Qi/Qc where the accuracy of Qi is high for the DCM fit. This range appears to be between 10^-2 and 10 for Qi
on the order of 100, having an upper bound of about 10 for Qi on the order of 10^4, and an upper bound of around 10^2 for Qi on the order of 10^6. In short, it appears
that as long as Qc and Qi are within two orders of magnitude from each other, the result will be quite accurate for determining Qi.

![alt text](https://raw.githubusercontent.com/Boulder-Cryogenic-Resonator-Testbed/measurement/master/BCRTresfit/circuit_simulation_results/QiQc/INV.png)

For INV.png, unlike the DCM fit, there does not appear to be a range of Qi/Qc where the accuracy of Qi is high. There appears to be an upper bound of around 10 for
Qi on the order of 100, however no other line appears to be bounded in terms of accuracy for the same data run as DCM.

These graphs demonstrate that the INV fitting method appears to be more robust across high and low Qi/Qc ratios and that each Qi appears to have an ideal Qi/Qc ratio
range that gives accurate fitting results.
