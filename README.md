# MAGreeTe
Materials Analysis through Green's Tensor


Ceci n’est pas un algorithme FDTD

This is not a Finite Difference Time Domain algorithm


Installation:
Create a conda environment using MAGreeTe.yml, then install PyTorch using the commented command in the yml

Usage:
This branch creates or imports a point pattern in 2d or 3d, then solves the associated coupled dipoles problem for a given choice of source parameters, assuming a scalar wave.
A typical run command looks like
python magreete_scalar.py 2 -n 2.4-0.4j -N 646 --lattice square --compute_transmission --plot_transmission -o my/beautiful/folder/that/I/love
which computes the 2d transmission plots for a system of 646 particles on a square lattice (before cutting the system to a disk), each with index 2.4-0.4j assuming that the host medium has index 1.0.
The --compute_transmission option solves the linear system, the plot_transmission propagates the solution to measurement points.
Outputs are sent to the designated output directory.
The wave-vectors are in units of 2pi/L with L the diameter of the disk of points, and can be fed in via -k kstart kend krange.
By default, the system is solved for 360 angles spaced by 1 degree on the circle.
