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
from utils import alpha_cold_atoms_2d, alpha_cold_atoms_3d, alpha_small_dielectric_object, plot_transmission_angularbeam, plot_transmission_flat, uniform_unit_disk_picking, uniform_unit_ball_picking, plot_3d_points
from Transmission2D import Transmission2D
from Transmission3D import Transmission3D
import lattices


import argparse


def main(head_directory, ndim,  lattice=None, just_plot = False, compute_DOS=False, dospoints=1, cold_atoms=False, L = 1):
    '''
    Simple front-end for MAGreeTe
    '''

    #Todo: clean up this main to have explicit 2d or 3d cases, just make main callable from the outside
    phi = 0.1
    N = 4096
    size_ratio = 1.0
    a = 0.0
    k = 16#32
    w = 0.2*L


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

    if ndim==2:

        k0range = onp.arange(40,81)*64/128*2*onp.pi/L
        volume = L*L*phi/N
        radius = onp.sqrt(volume/onp.pi )
        meas_points = 2*L*onp.vstack([onp.cos(thetas),onp.sin(thetas)]).T

        if cold_atoms:
            alpharange = alpha_cold_atoms_2d(k0range)
        else:
            refractive_n = 1.6
            alpharange = onp.ones(len(k0range)) * alpha_small_dielectric_object(refractive_n,volume)

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

                EkTE, EkTM = solver.calc_EM(meas_points, EjTE, EjTM, k0, alpha, thetas, w)
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
                EjTE, EjTM = solver.run_EM(k0, alpha, thetas, radius, w, self_interaction=not cold_atoms)
                k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                params = [alpha, k0]
                hkl.dump([onp.array(EjTE), onp.array(EjTM), onp.array(params),onp.array(points), onp.array(thetas)],file_name+'_Ek_k0_'+str(k0_)+'_'+str(i)+'.hkl')

                EkTE, EkTM = solver.calc_EM(meas_points, EjTE, EjTM, k0, alpha, thetas, w)
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

        if compute_DOS:
            DOSall_TE = []
            DOSall_TM = []
            k0_range = []

            M = dospoints
            measurement_points = uniform_unit_disk_picking(M)
            measurement_points *= L/2

            for k0, alpha in zip(k0range,alpharange):
                dos_TE, dos_TM = solver.mean_DOS_measurements(measurement_points, k0, alpha, radius)
                DOSall_TE.append(dos_TE.numpy())
                DOSall_TM.append(dos_TM.numpy())

                k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                k0_range.append(k0_)

                onp.savetxt(file_name+'_temp_dos_TE.csv',[k0_range,DOSall_TE])
                onp.savetxt(file_name+'_temp_dos_TM.csv',[k0_range,DOSall_TM])

            onp.savetxt(file_name+'_dos_TE.csv',[k0_range,DOSall_TE])
            onp.savetxt(file_name+'_dos_TM.csv',[k0_range,DOSall_TM])

    elif ndim==3:

        k0range = onp.arange(10,41)*64/128*2*onp.pi/L
        volume = L*L*L*phi/N
        radius = onp.cbrt(volume * 3.0 / (4.0 * onp.pi))
        meas_points = 2*L*onp.vstack([onp.cos(thetas),onp.sin(thetas),onp.zeros(len(thetas))]).T
        plot_3d_points(points,file_name)

        if cold_atoms:
            alpharange = alpha_cold_atoms_3d(k0range)
        else:
            refractive_n = 1.6
            alpharange = onp.ones(len(k0range)) * alpha_small_dielectric_object(refractive_n,volume)

        if just_plot:
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

                Ek = solver.calc(meas_points, Ej, k0, alpha, u, p, w)
                Ek = np.linalg.norm(Ek,axis=1)
                Eall.append(Ek.numpy())
            Eall = onp.array(Eall)
            Etotal = onp.absolute(Eall)**2
        
        else: 

            solver = Transmission3D(points)
            u = onp.stack([onp.cos(thetas),onp.sin(thetas),onp.zeros(len(thetas))]).T
            u = np.tensor(u)
            print(points.shape)
            p = np.zeros(u.shape)
            p[:,2] = 1
            Eall = []
            for k0, alpha in zip(k0range,alpharange):
                Ej = solver.run(k0, alpha, u, p, radius, w)
                k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                params = [alpha, k0]
                hkl.dump([onp.array(Ej), onp.array(params),onp.array(points), onp.array(thetas)],file_name+'_Ek_k0_'+str(k0_)+'_'+str(i)+'.hkl')

                Ek = solver.calc(meas_points, Ej, k0, alpha, u, p, w)
                Ek = np.linalg.norm(Ek,axis=1)
                Eall.append(Ek.numpy())
            Eall = onp.array(Eall)
            Etotal = onp.absolute(Eall)**2
    
        plot_transmission_angularbeam(k0range, L, thetas, Etotal, file_name) 
        plot_transmission_flat(k0range, L, thetas, Etotal, file_name) 

        if compute_DOS:
            DOSall = []
            k0_range = []

            # Expensive computation in 3d
            M = dospoints
            measurement_points = uniform_unit_ball_picking(M, ndim)
            measurement_points *= L/2

            for k0, alpha in zip(k0range,alpharange):
                dos = solver.mean_DOS_measurements(measurement_points, k0, alpha, radius)
                DOSall.append(dos.numpy())

                k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                k0_range.append(k0_)

                onp.savetxt(file_name+'_temp_dos.csv',[k0_range,DOSall])

            onp.savetxt(file_name+'_dos.csv',[k0_range,DOSall])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run a full solving and plotting routine of MAGreeTe")
    parser.add_argument("head_directory", type=str, help="parent directory containing all configurations")
    parser.add_argument("ndim", type=int, help="Dimensionality of the problem at hand")
    parser.add_argument("-n","--n_cpus", type=int, help="Number of cpus to use for computation\
        default = os.cpu_count", default=os.cpu_count())
    parser.add_argument("-l", "--lattice", type=str, help="Use a simple lattice in lieu of datapoints as entry. \
        Options are 'square', 'triangular', 'honeycomb', 'quasicrystal', 'quasidual', 'quasivoro' in 2d, and 'cubic', 'fcc', 'bcc', 'diamond' in 3d. \
        default=None", default=None)
    parser.add_argument("-p","--just_plot", action='store_true', help="Bypass the calculation and just produce plots\
        default=False", default=False)
    parser.add_argument("-dos","--compute_DOS", action='store_true', help="Compute the mean DOS of the medium  \
        default=False", default=False)
    parser.add_argument("--dospoints",type=int, help="Number of points to use for the mean DOS computation \
        default = 1000", default=1000)
    parser.add_argument("--boxsize", type=float, help="Set physical units for the box size: the results are dimensionless so that default=1m", default = 1)

    args = parser.parse_args()

    head_directory = args.head_directory
    ndim = args.ndim
    n_cpus=args.n_cpus
    just_plot=args.just_plot
    lattice = args.lattice
    compute_DOS=args.compute_DOS
    dospoints=args.dospoints
    boxsize=args.boxsize

    np.set_num_threads(n_cpus)
    np.device("cpu")
    main(head_directory, ndim, lattice=lattice, just_plot=just_plot, compute_DOS=compute_DOS, dospoints=dospoints, L=boxsize)
    sys.exit()


