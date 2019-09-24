## Phi vs. DCM Test:

The Diameter Correction Method and the Phi Rotation Method are both graphed with their respective percent error for three Qi values across changing impedance
mismatches.

## Data:

Note that while Qi changes due to the impedance mismatch, the change is very small (less than 0.01% across the 40 Ohm span) and can be neglected for the
calculation of percent error for these results considering the scale of percent error being graphed.

![alt text](https://raw.githubusercontent.com/Boulder-Cryogenic-Quantum-Testbed/measurement/master/Resfit/circuit_simulation_results/phi_test/DCM_vs_PHI_Qi%3D118.png)

![alt text](https://raw.githubusercontent.com/Boulder-Cryogenic-Quantum-Testbed/measurement/master/Resfit/circuit_simulation_results/phi_test/DCM_vs_PHI_Qi%3D11135.png)

![alt text](https://raw.githubusercontent.com/Boulder-Cryogenic-Quantum-Testbed/measurement/master/Resfit/circuit_simulation_results/phi_test/DCM_vs_PHI_Qi%3D1096357.png)

It is clear from the results that the Phi Rotation Method has a large percent error when exposed to an impedance mismatch environment, above even 30 percent
for the 11135 Qi resonator. This matches with results we would expect as the DCM is the corrected version of the Phi Rotation Method to account for impedance
mismatches. It is thus recommended that the user not use the Phi Rotation Method, and instead use the Diameter Correction Method.
