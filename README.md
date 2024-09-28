# MAGreeTe
Materials Analysis through Green's Tensor


Ceci n’est pas un algorithme FDTD

This is not a Finite Difference Time Domain algorithm


## Installation - conda

Create a new conda environment named `magreete` from `MAGreeTe.yml`, which will install all packages using the channels `pytorch` and `conda-forge`. Among other things, this will install pytorch (cpu-only version). GPU parallelization is a work in progress.

```bash
conda env create --name magreete --file=MAGreeTe.yml
```

You can take the time to view possible command-line arguments with:

```bash
python magreete.py -h
```
