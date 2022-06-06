# OneD
This is a package of modules.

GlobalFuncs.py is a module that contains functions which are shared across the entire framework, such as the FFT-Poisson Solver (\texttt{fourer_potential}),
as well as functions which run a full calculation for either FDM, N-Bodies, or both.

WaveDim.py and WaveNonDim.py are modules containing functions relevant to the numerical schemes for the FDM simulations.
"Dim" refers to the fact that the parameters are dimensional, while "NonDim" refers to parameters being non-dimensional.
WaveNonDim.py is the module used in GlobalFuncs.py, while WaveDim.py contains old functions from early development. 

NBody.py similarly contains functions only used for the particle simulations.
