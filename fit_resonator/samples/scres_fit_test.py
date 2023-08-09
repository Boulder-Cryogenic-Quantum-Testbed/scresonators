import numpy as np
import fit_resonator.resonator as scres

# NOTE: lmfit and inflect modules produced errors on initial run and had to be installed to conda.
# Might just be anaconda not using the locally installed modules

my_resonator = scres.Resonator(filepath='./schematic 1.s2p', measurement="S21")

# Assign your desired fit method variables
# Using the example from the README to be tweaked later
fit_type = 'DCM'
MC_iteration = 10
MC_rounds = 1e3
MC_fix = []
manual_init = None

# Pass these to your resonator object
my_resonator.fit_method(fit_type, MC_iteration, MC_rounds=MC_rounds, MC_fix=MC_fix, manual_init=manual_init,
                 MC_step_const=0.3)

my_resonator.fit()
