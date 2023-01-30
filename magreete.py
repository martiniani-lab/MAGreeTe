import numpy as onp
import torch as np
import scipy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import colorsys
import hickle as hkl
import sys
import os
from utils import alpha_cold_atoms_2d, alpha_small_dielectric_object, plot_transmission_angularbeam, plot_transmission_flat, uniform_unit_disk_picking
from Transmission2D import Transmission2D
import lattices


import argparse

L = 100e-6 #box side length m

def main(head_directory, n_cpus=1, lattice=None, just_plot = False):
    '''
    Simple front-end for MAGreeTe
    '''

    #Todo: clean up this main to have explicit 2d or 3d cases, just make main callable from the outside
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
    else:
        if ndim==2:
            if lattice == 'square':
                file_name = 'square'

                i=0

                points = lattices.square()
                N = points.shape[0]
                points *= L

            elif lattice == 'triangular':
                file_name = 'triangular'

                i = 0

                points = lattices.triangular()
                N = points.shape[0]
                points *= L

            elif lattice == 'quasidual':
                file_name = 'quasidual'

                i = 0

                points = lattices.quasicrystal(mode='quasidual')
                N = points.shape[0]
                points *= L 
            
            else:
                print("Not a valid lattice!")
                exit()
        elif ndim == 3:
            if lattice == 'cubic':
                file_name = 'cubic'
                i=0

                points = lattices.cubic()
                N = points.shape[0]
                points *= L
            else: 
                print("Not a valid lattice!")
                exit()


    Ntheta = 360
    thetas = onp.arange(Ntheta)/Ntheta*2*np.pi
    meas_points = 2*L*onp.vstack([onp.cos(thetas),onp.sin(thetas)]).T

    if just_plot:

        ETEall = []
        ETMall = []
        for k0, alpha in zip(k0range,alpharange):
            
            k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
            EjTE, EjTM, params, points, thetas = hkl.load(file_name+'_Ek_k0_'+str(k0_)+'_'+str(i)+'.hkl')
            EjTE = np.tensor(EjTE, dtype=np.complex128)
            EjTM = np.tensor(EjTM, dtype=np.complex128)
            points = np.tensor(points, dtype=np.complex128)
            thetas = onp.float64(thetas)
            alpha, k0 = params
            k0 = onp.float64(k0)
            alpha = onp.complex128(alpha)
            solver = Transmission2D(points)

            EkTE, EkTM = solver.calc_EM(meas_points, EjTE, EjTM, k0, alpha, thetas, n_cpus=n_cpus)
            EkTE = np.linalg.norm(EkTE,axis=1)
            ETEall.append(EkTE.numpy())
            ETMall.append(EkTM.numpy())
        ETEall = onp.array(ETEall)
        ETMall = onp.array(ETMall)
        TEtotal = onp.absolute(ETEall)**2
        TMtotal = onp.absolute(ETMall)**2

    else:
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
    plot_transmission_flat(k0range, L, thetas, TMtotal, file_name, appended_string='TM')
    plot_transmission_flat(k0range, L, thetas, TEtotal, file_name, appended_string='TE')

    compute_DOS = False

    if compute_DOS:
        DOSall_TE = []
        DOSall_TM = []
        k0_range = []

        M = 2 * N
        measurement_points = uniform_unit_disk_picking(M)
        measurement_points *= L/2

        for k0, alpha in zip(k0range,alpharange):
            dos_TE, dos_TM = solver.mean_DOS_measurements(measurement_points, k0, alpha, radius, n_cpus=n_cpus)
            DOSall_TE.append(dos_TE.numpy())
            DOSall_TM.append(dos_TM.numpy())

            k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
            k0_range.append(k0_)

            onp.savetxt(file_name+'_temp_dos_TE.csv',[k0_range,DOSall_TE])
            onp.savetxt(file_name+'_temp_dos_TM.csv',[k0_range,DOSall_TM])

        onp.savetxt(file_name+'_dos_TE.csv',[k0_range,DOSall_TE])
        onp.savetxt(file_name+'_dos_TM.csv',[k0_range,DOSall_TM])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run a full solving and plotting routine of MAGreeTe")
    parser.add_argument("head_directory", type=str, help="parent directory containing all configurations")

    args = parser.parse_args()

    head_directory = args.head_directory

    n_cpus = 28
    np.set_num_threads(n_cpus)
    np.device("cpu")
    main(head_directory, n_cpus, lattice='square', just_plot=True)
    sys.exit()


