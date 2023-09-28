scresonators-fit: fitting superconducting resonator data using Python
===========================================

[![CodeFactor](https://www.codefactor.io/repository/github/boulder-cryogenic-quantum-testbed/scresonators/badge/master)](https://www.codefactor.io/repository/github/boulder-cryogenic-quantum-testbed/scresonators/overview/master)
![Tests](https://github.com/boulder-cryogenic-quantum-testbed/scresonators/actions/workflows/tests.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

scresonators code is made to fit complex S21 data for hangar mode resonators. The scresonators library supports fitting using the Diameter Correction Method (DCM), Inverse S21 method (INV), Closest Pole and Zero Method (CPZM), and Phi Rotation Method (PHI). Additionally, the code is able to fit reflection type geometry resonators with an altered version of the Diameter Correction Method (DCM REFLECTION).



Code Installation
-------------------------

We ***strongly*** recommend using [virtual environments](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) for managing your dependences.

scresonators can be installed via pip by running
```
pip install scresonators-fit
```
To use the module lead your Python code with
```
import fit_resonator.resonator as resonator
```
For minimum working example with sample data, clone the repository and run
```
python scresonators/example.py
```

Documentation
-------------

For scresonators documentation visit our [readthedocs](https://boulder-cryogenic-quantum-testbed.github.io/scresonators-docs/) (Doccumentation has not yet been checked for outdated information)

The documentation source code is located at https://github.com/Boulder-Cryogenic-Quantum-Testbed/scresonators-docs



Citation
--------
If you use scresonators to fit data or generate plots used in a publication you can cite it as follows:

```latex
@misc{scresonators,
  title        = {{boulder-cryogenic-quantum-testbed/scresonators}},
  year         = {2023},
  howpublished = {\url{https://github.com/Boulder-Cryogenic-Quantum-Testbed/scresonators}}
}
```



Contribution
----------
We invite you to contribute to the continued development of scresonators by forking this repository and sending pull requests.

Even if you do not contribute to the code yourself, please support us by alerting us to problems you encounter or improvements we can make by making a new issue [here](https://github.com/Boulder-Cryogenic-Quantum-Testbed/scresonators/issues).


If you make a contribution, feel free to add yourself to our [list of contributors](https://github.com/Boulder-Cryogenic-Quantum-Testbed/scresonators/blob/master/contrib/HELLO.md).

All contributions are expected to be consistent with [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/).



License
-------
scresonators uses the MIT license. Details are described in [the LICENSE file](https://github.com/Boulder-Cryogenic-Quantum-Testbed/scresonators/blob/master/LICENSE), but restrictions are very loose so feel free to use and modify our code to meet your requirements.



References
-------
Fitting Functions:

![alt text](https://raw.githubusercontent.com/Boulder-Cryogenic-Quantum-Testbed/measurement/master/fit_resonator/Resources/Fit_Equations.PNG)

Publications:
1. DCM: M. S. Khalil, M. J. A. Stoutimore, F. C. Wellstood, and K. D. Osborn    Journal of Applied Physics 111, 054510 (2012); doi: 10.1063/1.3692073
2. INV: A. Megrant et al.     APPLIED PHYSICS LETTERS 100, 113510 (2012)
3. CPZM: Chunqing Deng, Martin Otto, and Adrian Lupascu University of Waterloo, Waterloo, Ontario N2L 3G1, Canada (9 August 2013)
4. PHI: J. Gao, "The Physics of Superconducting Microwave Resonators" Cal-tech Ph.D. thesis, May 2008
