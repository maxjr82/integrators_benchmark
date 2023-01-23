# Benchmark of numerical integrators for classical dynamics

![python version](https://img.shields.io/badge/python-3.8-blue?logo=python) 

This repository contains a Python program developed to test the performance of several numerical integrators
used to propagate the time evolution of a particles system within the classical approximation. As model systems 
used in the dynamics simulations, the program support the analytical Spin-Boson hamiltonian (through the sbh 
keyword) or fitted potential energy surface (PES) of molecules available in the [POLIB](https://comp.chem.umn.edu/potlib/index.html) library.

## Currently implemented integrators

 |  Method                 |  Symplectic |
 |-------------------------|-----|
 |  Euler                  | :heavy_check_mark: |
 |  Velocity-Verlet        | :heavy_check_mark: |
 |  Ruth (3th order)       | :heavy_check_mark: |
 |  Ruth (4th order)       | :heavy_check_mark: |
 |  Runge-Kutta            |                    |
 |  Yoshida (6th order)    | :heavy_check_mark: |
 |  Yoshida (8th order)    | :heavy_check_mark: |

## How to use

To run the program locally, one needs first to download the application from this GitHub repository, and
prepare the input data at the same folder as the Python program. The general settings for the simulation
should be provided as a YAML file with name `config.yml` for any of the available model systems (SBH or 
molecular PES). 

In the case of the SBH model, two additional files are required:

- a text file with the initial conditions, positions and velocities, in a two-columns format.
- a file containing the SBH model parameters distributed in three columns (freq, reduced mass, and g).

To run the dynamics with a fitted PES, there are a few preliminary steps to follow. First, one needs to
download the source code of the POTLIB program from its website and compile it. This program will be 
responsible for providing the molecular energies and energy-gradients at every step of the dynamics to 
propagate the trajectory for the specified molecule. Then, the initial molecular geometry and corresponding
velocities must be provided as XYZ files with the names `init_geom.xyz` and `init_veloc.xyz`. Finally, a 
single-column file called coeff.dat containing the coefficients for the PES fitting procedure should be 
created in the working directory. This file can be downloaded from the [POLIB](https://comp.chem.umn.edu/potlib/index.html) website.

Once the input configuration is finished, the dynamics simulation can be initiated by simply excuting the 
following command at a bash terminal:

```sh
$ python run_dynamics.py > dynamics.log
```

