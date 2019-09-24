# measurement
Welcome to the measurement repostository of the Boulder Cryogenic Quantum Testbed! This is a library for measuring the loss in superconducting resonators. 

## Installation
1. clone the repository into a folder of your choice with `git clone https://github.com/Boulder-Cryogenic-Quantum-Testbed/measurement.git
1. Install the dependencies, we ***strongly*** recommend using [virtual environments](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) for managing your dependences. To install dependencies run:
  `pip install -r requirements.txt`
  
## Using the library



## Code Organization

All code should live in the `cryores` namespace. This ensures easy integration
with other Python packages, and avoids name collisions; everything is referred
to as e.g. `cryores.experiments` rather than just `experiments`.
