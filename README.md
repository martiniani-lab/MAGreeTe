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
