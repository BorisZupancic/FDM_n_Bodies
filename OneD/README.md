# OneD
This sub-package contains the modules that make the main program (`Program.py`) work and do physics.

`Global.py` is a module that contains functions which are shared across the entire framework, such as the FFT-Poisson Solver, as well as the main loop for the simulation.

`FDM.py` contains functions relevant to simulating fuzzy dark matter. 

`NBody.py` contains functions only used for the particle simulations.

`Init_C.py` is for setting up initial conditions, as well as other initial setup that's used in the main program.

The sub-package `Analysis` contains modules used for post-simulation analysis.
