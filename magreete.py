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
from utils import alpha_cold_atoms_2d, alpha_small_dielectric_object, plot_transmission_angularbeam, plot_transmission_flat, uniform_unit_disk_picking, plot_3d_points
from Transmission2D import Transmission2D
from Transmission3D import Transmission3D
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
    k = 16#32
    ndim = 3
    w = 0.2*L

    #k0range = onp.arange(40,81)*64/128*2*onp.pi/L
    k0range = onp.arange(10,41)*64/128*2*onp.pi/L
    #k0range = onp.arange(5,61)*2*2*onp.pi/L

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
                points = lattices.square()
            elif lattice == 'triangular':
                points = lattices.triangular()
            elif lattice == 'quasidual':
                points = lattices.quasicrystal(mode='quasidual')
            else:
                print("Not a valid lattice!")
                exit()

        elif ndim == 3:
            if lattice == 'cubic':
                points = lattices.cubic()
            elif lattice == 'bcc':
                points = lattices.bcc()
            elif lattice == 'fcc':
                points = lattices.fcc()
            elif lattice == 'diamond':
                points = lattices.diamond(9)
            else: 
                print("Not a valid lattice!")
                exit()
        file_name = lattice
        i=0
        N = points.shape[0]
        points *= L
    assert ndim == points.shape[1]
    Ntheta = 360
    thetas = onp.arange(Ntheta)/Ntheta*2*np.pi
    if ndim == 2:
        volume = L*L*phi/N
        radius = onp.sqrt(volume/onp.pi )
        meas_points = 2*L*onp.vstack([onp.cos(thetas),onp.sin(thetas)]).T
    else:
        volume = L*L*L*phi/N
        radius = onp.cbrt(volume * 3.0 / (4.0 * onp.pi))
        meas_points = 2*L*onp.vstack([onp.cos(thetas),onp.sin(thetas),onp.zeros(len(thetas))]).T
        plot_3d_points(points,file_name)

    cold_atoms = False
    if cold_atoms:
        alpharange = alpha_cold_atoms_2d(k0range)
    else:
        refractive_n = 1.6
        alpharange = onp.ones(len(k0range)) * alpha_small_dielectric_object(refractive_n,volume)


    if just_plot and ndim==2:

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

            EkTE, EkTM = solver.calc_EM(meas_points, EjTE, EjTM, k0, alpha, thetas, w, n_cpus=n_cpus)
            EkTE = np.linalg.norm(EkTE,axis=1)
            ETEall.append(EkTE.numpy())
            ETMall.append(EkTM.numpy())
        ETEall = onp.array(ETEall)
        ETMall = onp.array(ETMall)
        TEtotal = onp.absolute(ETEall)**2
        TMtotal = onp.absolute(ETMall)**2
    elif just_plot and ndim==3:
        Eall = []
        u = onp.stack([onp.cos(thetas),onp.sin(thetas),onp.zeros(len(thetas))]).T
        u = np.tensor(u)
        print(points.shape)
        p = np.zeros(u.shape)
        p[:,2] = 1
        for k0, alpha in zip(k0range,alpharange):
            
            k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
            Ej, params, points, thetas = hkl.load(file_name+'_Ek_k0_'+str(k0_)+'_'+str(i)+'.hkl')
            Ej = np.tensor(Ej, dtype=np.complex128)
            points = np.tensor(points, dtype=np.complex128)
            thetas = onp.float64(thetas)
            alpha, k0 = params
            k0 = onp.float64(k0)
            alpha = onp.complex128(alpha)
            solver = Transmission3D(points)

            Ek = solver.calc(meas_points, Ej, k0, alpha, u, p, w, n_cpus=n_cpus)
            Ek = np.linalg.norm(Ek,axis=1)
            Eall.append(Ek.numpy())
        Eall = onp.array(Eall)
        Etotal = onp.absolute(Eall)**2

    elif ndim == 2:
        solver = Transmission2D(points)
        ETEall = []
        ETMall = []
        for k0, alpha in zip(k0range,alpharange):
            EjTE, EjTM = solver.run_EM(k0, alpha, thetas, radius, w, n_cpus=n_cpus, self_interaction=not cold_atoms)
            k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
            params = [alpha, k0]
            hkl.dump([onp.array(EjTE), onp.array(EjTM), onp.array(params),onp.array(points), onp.array(thetas)],file_name+'_Ek_k0_'+str(k0_)+'_'+str(i)+'.hkl')

            EkTE, EkTM = solver.calc_EM(meas_points, EjTE, EjTM, k0, alpha, thetas, w, n_cpus=n_cpus)
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

    elif ndim == 3:
        solver = Transmission3D(points)
        u = onp.stack([onp.cos(thetas),onp.sin(thetas),onp.zeros(len(thetas))]).T
        u = np.tensor(u)
        print(points.shape)
        p = np.zeros(u.shape)
        p[:,2] = 1
        Eall = []
        for k0, alpha in zip(k0range,alpharange):
            Ej = solver.run(k0, alpha, u, p, radius, w, n_cpus=n_cpus)
            k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
            params = [alpha, k0]
            hkl.dump([onp.array(Ej), onp.array(params),onp.array(points), onp.array(thetas)],file_name+'_Ek_k0_'+str(k0_)+'_'+str(i)+'.hkl')

            Ek = solver.calc(meas_points, Ej, k0, alpha, u, p, w, n_cpus=n_cpus)
            Ek = np.linalg.norm(Ek,axis=1)
            Eall.append(Ek.numpy())
        Eall = onp.array(Eall)
        Etotal = onp.absolute(Eall)**2
    
        plot_transmission_angularbeam(k0range, L, thetas, Etotal, file_name) 
        plot_transmission_flat(k0range, L, thetas, Etotal, file_name) 
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run a full solving and plotting routine of MAGreeTe")
    parser.add_argument("head_directory", type=str, help="parent directory containing all configurations")

    args = parser.parse_args()

    head_directory = args.head_directory

    n_cpus = 28
    np.set_num_threads(n_cpus)
    np.device("cpu")
    main(head_directory, n_cpus, lattice='cubic', just_plot=False)
    sys.exit()


