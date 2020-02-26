## Purpose:

The purpose of these simulations is to show the user how the accuracy of the fitting functions changes depending on various factors.
By reading through this data, the user should be able to see the best data taking and processing practices to get a good fit.

## Model:

![alt text](https://raw.githubusercontent.com/Boulder-Cryogenic-Quantum-Testbed/measurement/master/resfit/circuit_simulation_results/circuit.PNG)

The model used here is a simple lumped element resonator RCL circuit capacitively coupled to a transmission line with data generated in AWR.
Asymmetry is added by increasing inductance on one side of the transmission line and Qc is changed by varying the value of the coupling capacitor.
The value for Qi is changed here by varying the value of R in the RCL circuit and Qi_SIM is calculated using the known simulation parameters.
Knowing Qi_SIM allows for the calculation of percent error once Qi is found from the fit code.

>Schematic of circuit used in AWR included as Schematic.sch

## Tests:

* extract_factor: Test to see how big of a difference using a certain number of 3dB bandwidths to fit data makes for accuracy. Tested across varying Qi, Qc and impedance mismatch values.
* normalize: Test to see how percent error changes when using different numbers of 3dB bandwidths to normalize data.
* phi_test: Find out how big of a difference in accuracy there is between Phi Rotation Method and DCM.
