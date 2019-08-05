## Normalize Test:

This test shows the accuracy of the fits as a function of the number of 3dB bandwidths used to normalize data.

## Data:

![alt text](https://raw.githubusercontent.com/mullinska/measurement/master/BCRTresfit/circuit_simulation_results/normalize/R=1e6.png)

Here, the percent error in Qi versus number of 3dB bandwidths used to normalize data at a Qi on the order of 10^4 shows that percent error
becomes effectively negligible past 4 3dB bandwidths used. This is indicated by the rapid change in percent error between 3 and 4 bandwidths used. For this specific 
Qi, the reason for the rapid change is that the fitting parameters are getting stuck in a local minimum for DCM. This causes the Monte Carlo fit function to get 
better results and leads to a high percent error due to the nature of Monte Carlo fits.

![alt text](https://raw.githubusercontent.com/mullinska/measurement/master/BCRTresfit/circuit_simulation_results/normalize/R=1e8.png)

Here, these results indicate once again that at 4 3dB bandwidths used, the percent error becomes effectively negligible. Additionally the data indicates that DCM is 
not as reliable at low number of bandwidths used. This is due to the fact that the INV fitting uses less points, the ones near resonance,
which grants the fit better accuracy. Using the same points would give a similar accuracy for DCM, however, because the definitions of 3dB bandwidth 
change between fits DCM ends up using more points (the ones further from resonance). Note that in a real measurement, it is possible that noise could affect the 
accuracy quite heavily in the case of INV because it uses a fewer number of points as a result of the definition of it's 3dB bandwidth.
