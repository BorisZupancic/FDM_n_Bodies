# FDM_n_Bodies
This is the code that was used to generate the data in the paper [Zupancic & Widrow (2024)](https://doi.org/10.1093/mnras/stad3620), 
where we simulated Fuzzy Dark Matter and Particles in a self-consistent gravitational field in 1 spatial dimension.

> [!WARNING]
> This repository is not actively maintained and the code is largely undocumented.
> To understand the physics, please refer to the paper [Zupancic & Widrow (2024)](https://doi.org/10.1093/mnras/stad3620).

## Usage
`Program.py` is the main script that can run a simulation. It asks for input to setup the initial conditions of the simulation,
then calls on modules and functions in `OneD` which actually perform the relevant numerical computations. The output is simulation data
saved in .csv files.

> [!NOTE]
> To generate the data used in [Zupancic & Widrow (2024)](https://doi.org/10.1093/mnras/stad3620), I used a bash script to automate the initialization for 
> the many simulations that we ran.

## Citation
If you use this code, please cite the following paper:

##### Plain-Text:
Zupancic, B., & Widrow, L. M. (2024). Fuzzy dark matter dynamics and the 
quasi-particle hypothesis. Monthly Notices of the Royal Astronomical Society, 
527(3), 6189–6197. https://doi.org/10.1093/mnras/stad3620

##### BibTeX:
```bibtex
@article{zupancic2024,
  title={Fuzzy dark matter dynamics and the quasi-particle hypothesis},
  author={Zupancic, Boris and Widrow, Lawrence M},
  journal={Monthly Notices of the Royal Astronomical Society},
  volume={527},
  number={3},
  pages={6189--6197},
  year={2024},
  month={01},
  doi={10.1093/mnras/stad3620},
  url={https://doi.org/10.1093/mnras/stad3620}
}
```
