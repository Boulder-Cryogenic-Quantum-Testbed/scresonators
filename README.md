# measurement
Instrument automation, sweeps and analysis code for cryogenic resonator measurement


## Code Organization

All code should live in the `cryores` namespace. This ensures easy integration
with other Python packages, and avoids name collisions; everything is referred
to as e.g. `cryores.experiments` rather than just `experiments`.
