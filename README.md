# MAGreeTe
Materials Analysis through Green's Tensor


Ceci n’est pas un algorithme FDTD

This is not a Finite Difference Time Domain algorithm

## Installation - conda

Create a new conda environment named `magreete` from `MAGreeTe.yml`, which will install all packages using the channels `pytorch` and `conda-forge`. Among other things, this will install pytorch (cpu-only version) and Julia. GPU parallelization is a work in progress.

```bash
conda env create --name magreete --file=MAGreeTe.yml
```

Upon executing `magreete.py` for the first time, julia will take some time to set itself up. You can take the time to view possible command-line arguments with:

```bash
python magreete.py -h
```

In order to utilize Julia and its hmatrices, you will need to install a few Julia packages. Open a `julia` runtime in the terminal and run the following:

```bash
using Pkg;Pkg.activate("Transmission2D")
Pkg.add("SpecialFunctions")
```
You can exit the Julia runtime and now use `--method hmatrices` on your calls to MAGreeTe! This is a slightly slower variant that saves on memory.

## Usage

This branch creates or imports a point pattern in 2d or 3d, then solves the associated coupled dipoles problem for a given choice of source parameters, assuming a scalar wave.

A typical run command looks like

```bash
python magreete_scalar.py 2 -n 2.4-0.4j -N 646 --lattice square --compute_transmission --plot_transmission -o my/beautiful/folder/that/I/love
```
which computes the 2d transmission plots for a system of 646 particles on a square lattice (before cutting the system to a disk), each with index 2.4-0.4j assuming that the host medium has index 1.0.

The `--compute_transmission` option solves the linear system. That is, for the given source field (default laser), it will solve for the steady-state field at each scatterer position.

The `--plot_transmission` option propagates the solution to measurement points. It will use the results from `--compute_transmission` if called concurrently or search for a file in which the solutions were dumped.

Outputs are sent to the designated output directory.

The wave-vectors are in units of 2pi/L with L the diameter of the disk of points, and can be fed in via `-k kstart kend step` e.g. `-k 1 100 0.5` spans 1-99.5 in increments of 0.5.

By default, the system is solved for 360 angles spaced by 1 degree on the circle.
