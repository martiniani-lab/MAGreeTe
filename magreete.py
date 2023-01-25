import numpy as onp
import torch as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import colorsys
import hickle as hkl
import sys
import os
from utils import alpha_cold_atoms_2d, alpha_small_dielectric_object, plot_transmission_angularbeam, generate_square_lattice_chunk
from Transmission2D import Transmission2D

import argparse

w = 2.1e-5 #beam waist m
L = 100e-6 #box side length m

def main(head_directory, n_cpus=1, lattice=None):
    phi = 0.1
    N = 4096
    size_ratio = 1.0
    a = 0.0
    k = 32
    ndim = 2

    if ndim == 2:
        volume = L*L*phi/N
        radius = onp.sqrt(volume/onp.pi )
    else:
        volume = L*L*L*phi/N
        radius = onp.cbrt(volume * 3.0 / (4.0 * onp.pi))

    k0range = onp.arange(40,81)*64/128*2*onp.pi/L
    cold_atoms = False
    if cold_atoms:
        alpharange = alpha_cold_atoms_2d(k0range)
    else:
        refractive_n = 1.6
        alpharange = onp.ones(len(k0range)) * alpha_small_dielectric_object(refractive_n,volume)

    if lattice == None:
        dname = head_directory+'HPY'+str(ndim)+'D/phi'+str(phi)+'/a'+str(a)+'/'
        file_name = 'HPY'+str(ndim)+'D_phi'+str(phi)+'_a'+str(a)+'_N'+str(N)+'_K'+str(k)
        file_name += '_points'

        i=0
        points = hkl.load(dname+file_name+'_'+str(i)+'.hkl')
        points = np.tensor(points[:,0:ndim]-0.5,dtype=np.double)
        idx = np.nonzero(np.linalg.norm(points,axis=-1)<=0.5)
        points = np.squeeze(points[idx])
        points *= L

    elif lattice == 'square':
        file_name = 'square'

        i=0

        Nside = 65
        points = generate_square_lattice_chunk(Nside)
        N = points.shape[0]
        points *= L

    Ntheta = 360
    thetas = onp.arange(Ntheta)/Ntheta*2*np.pi
    meas_points = 2*L*onp.vstack([onp.cos(thetas),onp.sin(thetas)]).T


    
    solver = Transmission2D(points)
    ETEall = []
    ETMall = []
    for k0, alpha in zip(k0range,alpharange):
        EjTE, EjTM = solver.run_EM(k0, alpha, thetas, radius, n_cpus=n_cpus, self_interaction=not cold_atoms)
        k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
        params = [alpha, k0]
        hkl.dump([onp.array(EjTE), onp.array(EjTM), onp.array(params),onp.array(points), onp.array(thetas)],file_name+'_Ek_k0_'+str(k0_)+'_'+str(i)+'.hkl')

        EkTE, EkTM = solver.calc_EM(meas_points, EjTE, EjTM, k0, alpha, thetas, n_cpus=n_cpus)
        EkTE = np.linalg.norm(EkTE,axis=1)
        ETEall.append(EkTE.numpy())
        ETMall.append(EkTM.numpy())
    ETEall = onp.array(ETEall)
    ETMall = onp.array(ETMall)

    TEtotal = onp.absolute(ETEall)**2
    TMtotal = onp.absolute(ETMall)**2

    plot_transmission_angularbeam(k0range, L, thetas, TMtotal, file_name, appended_string='TM')
    plot_transmission_angularbeam(k0range, L, thetas, TEtotal, file_name, appended_string='TE') 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run a full solving and plotting routine of MAGreeTe")
    parser.add_argument("head_directory", type=str, help="parent directory containing all configurations")

    args = parser.parse_args()

    head_directory = args.head_directory

    n_cpus = 32
    np.set_num_threads(n_cpus)
    np.device("cpu")
    main(head_directory, n_cpus, lattice='square')
    sys.exit()


