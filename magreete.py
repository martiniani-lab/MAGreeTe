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
import utils
from Transmission2D import Transmission2D
from Transmission3D import Transmission3D
import lattices


import argparse


def main(head_directory, ndim, refractive_n = 1.65 - 0.025j, phi = 0.1,
        k0range_args = None, lattice=None, just_plot = False, regularize = True,
        compute_DOS=False, compute_interDOS=False, dospoints=1, compute_SDOS=False, write_eigenvalues=True, compute_LDOS=False, gridsize=(301,301), window_width=1.2, batch_size = 101*101,
        intensity_fields = False, amplitude_fields = False, phase_fields = False,
        cold_atoms=False, L = 1, output_directory="",
        donut = False):
    '''
    Simple front-end for MAGreeTe
    '''

    N = 4096
    a = 0.0
    k = 16#32so
    w = 0.2*L
    phi_ = phi

    #Todo: finish sending these to options with more uh, transparent names
    if donut:
        N = 4096
        a = -1.0
        k = 80
        phi_ = 0.6

    # Name the output directory in a human-readable way containing the two physical parameters, volume fraction and refractive index
    output_directory = output_directory+"phi_"+str(phi)+"/"
    if cold_atoms:
        output_directory = output_directory+"cold_atoms"
    else:
        output_directory = output_directory+"refractive_n_"+str(refractive_n)
    if regularize:
        output_directory = output_directory+"_reg"
    utils.trymakedir(output_directory)

    if lattice == None:
        dname = head_directory+'HPY'+str(ndim)+'D/phi'+str(phi_)+'/a'+str(a)+'/'
        file_name = 'HPY'+str(ndim)+'D_phi'+str(phi_)+'_a'+str(a)+'_N'+str(N)+'_K'+str(k)
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
            elif lattice == 'honeycomb':
                points = lattices.honeycomb()
            elif lattice == 'quasicrystal':
                points = lattices.quasicrystal(mode='quasicrystal')
            elif lattice == 'quasidual':
                points = lattices.quasicrystal(mode='quasidual')
            elif lattice == 'quasivoro':
                points = lattices.quasicrystal(mode='quasivoro')
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
        else:
            print("Not a valid dimensionality!")
            exit()
        file_name = lattice
        i=0
        points = lattices.cut_circle(points)
        N = points.shape[0]
        points *= L
    assert ndim == points.shape[1]

    Ntheta = 360
    thetas = onp.arange(Ntheta)/Ntheta*2*np.pi

    file_name = output_directory+"/"+file_name

    if ndim==2:
        if k0range_args == None:
            k0range = onp.arange(40,81)*64/128*2*onp.pi/L
        else:
            if len(k0range_args)==1:
                k0range = onp.array([k0range_args[0]])* 2*onp.pi/L
            elif len(k0range_args)==2:
                k0range = onp.arange(k0range_args[0],k0range_args[1]+1,1)* 2*onp.pi/L
            else:
                k0range = onp.arange(k0range_args[0],k0range_args[1]+k0range_args[2],k0range_args[2])* 2*onp.pi/L
        volume = L*L*phi/N
        radius = onp.sqrt(volume/onp.pi )
        meas_points = 2*L*onp.vstack([onp.cos(thetas),onp.sin(thetas)]).T

        if cold_atoms:
            alpharange = utils.alpha_cold_atoms_2d(k0range)
        else:
            alpharange = onp.ones(len(k0range)) * utils.alpha_small_dielectric_object(refractive_n,volume)

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

                EkTE, EkTM = solver.calc_EM(meas_points, EjTE, EjTM, k0, alpha, thetas, w, regularize=regularize, radius=radius)
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

                EkTE, EkTM = solver.calc_EM(meas_points, EjTE, EjTM, k0, alpha, thetas, w, regularize = regularize, radius=radius)
                EkTE = np.linalg.norm(EkTE,axis=1)
                ETEall.append(EkTE.numpy())
                ETMall.append(EkTM.numpy())
            ETEall = onp.array(ETEall)
            ETMall = onp.array(ETMall)

            TEtotal = onp.absolute(ETEall)**2
            TMtotal = onp.absolute(ETMall)**2

        utils.plot_transmission_angularbeam(k0range, L, thetas, TMtotal, file_name, appended_string='TM')
        utils.plot_transmission_angularbeam(k0range, L, thetas, TEtotal, file_name, appended_string='TE')
        utils.plot_transmission_flat(k0range, L, thetas, TMtotal, file_name, appended_string='TM')
        utils.plot_transmission_flat(k0range, L, thetas, TEtotal, file_name, appended_string='TE')
        utils.plot_angular_averaged_transmission(k0range, L, TMtotal, file_name, appended_string='TM')
        utils.plot_angular_averaged_transmission(k0range, L, TEtotal, file_name, appended_string='TE')

        # Compute full fields
        # Pretty expensive!
        some_fields = intensity_fields+amplitude_fields+phase_fields
        if some_fields:
            # Expensive computation
            ngridx = gridsize[0]
            ngridy = gridsize[1]
            xyratio = ngridx/ngridy
            x,y = onp.meshgrid(onp.linspace(0,xyratio,ngridx)  - xyratio/2.0,onp.linspace(0,1,ngridy) - 0.5)
            meas_points = np.tensor((onp.vstack([x.ravel(),y.ravel()]).T)*L*window_width)

            batches = np.split(meas_points, batch_size)
            n_batches = len(batches)

            extra_string=""
            if n_batches > 1:
                extra_string = extra_string+"es"
            print("Computing the full fields at "+str(gridsize)+" points in "+str(n_batches)+" batch"+extra_string+" of "+str(onp.min([batch_size, ngridx*ngridy])))

            for k0, alpha in zip(k0range,alpharange):
                k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                print("k0L/2pi = "+str(k0_))

                for index, angle in enumerate(thetas):
                    angle_ = onp.round(angle*180/onp.pi)
                    print("angle = "+str(angle_)+"degrees")

                    ETEall = []
                    ETMall = []

                    for batch in range(0, n_batches):
                        print("Batch "+str(batch+1))
                        batch_points = batches[batch]

                        EkTE, EkTM = solver.calc_EM(batch_points, EjTE[:,index], EjTM[:,index].unsqueeze(-1), k0, alpha, [angle], w, regularize=regularize, radius=radius)

                        ETEall.append(EkTE)
                        ETMall.append(EkTM)


                    ETEall = np.cat(ETEall, dim=0).squeeze(-1)
                    ETMall = np.cat(ETMall, dim=0)

                    ETEall_amplitude         = np.sqrt(ETEall[:,0]**2 + ETEall[:,1]**2)
                    ETEall_longitudinal      = ETEall[:,0]*onp.cos(angle) - ETEall[:,1]*onp.sin(angle)
                    ETEall_transverse        = ETEall[:,0]*onp.sin(angle) + ETEall[:,1]*onp.cos(angle)

                    ETEall_amplitude    = ETEall_amplitude.reshape(ngridy, ngridx)
                    ETEall_longitudinal = ETEall_longitudinal.reshape(ngridy, ngridx)
                    ETEall_transverse   = ETEall_transverse.reshape(ngridy, ngridx)
                    ETMall = ETMall.reshape(ngridy, ngridx)

                    utils.plot_full_fields(ETEall_amplitude, ngridx, ngridy, k0_, angle_, intensity_fields, amplitude_fields, phase_fields, file_name, appended_string='_width_'+str(window_width)+'_grid_'+str(ngridx)+'x'+str(ngridy)+'_TE', my_dpi = 300)
                    utils.plot_full_fields(ETEall_longitudinal, ngridx, ngridy, k0_, angle_, intensity_fields, amplitude_fields, phase_fields, file_name, appended_string='_width_'+str(window_width)+'_grid_'+str(ngridx)+'x'+str(ngridy)+'_TE_long', my_dpi = 300)
                    utils.plot_full_fields(ETEall_transverse, ngridx, ngridy, k0_, angle_, intensity_fields, amplitude_fields, phase_fields, file_name, appended_string='_width_'+str(window_width)+'_grid_'+str(ngridx)+'x'+str(ngridy)+'_TE_trans', my_dpi = 300)
                    utils.plot_full_fields(ETMall, ngridx, ngridy, k0_, angle_, intensity_fields, amplitude_fields, phase_fields, file_name, appended_string='_width_'+str(window_width)+'_grid_'+str(ngridx)+'x'+str(ngridy)+'_TM', my_dpi = 300)

        if compute_SDOS:
            DOSall_TE = []
            DOSall_TM = []
            k0_range = []

            for k0, alpha in zip(k0range,alpharange):
                dos_TE, dos_TM = solver.compute_eigenvalues_and_scatterer_LDOS( k0, alpha, radius, file_name, write_eigenvalues=write_eigenvalues)
                DOSall_TE.append(dos_TE.numpy())
                DOSall_TM.append(dos_TM.numpy())

                k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                k0_range.append(k0_)

                onp.savetxt(file_name+'_temp_sdos_TE.csv',onp.stack([k0_range,DOSall_TE]).T)
                onp.savetxt(file_name+'_temp_sdos_TM.csv',onp.stack([k0_range,DOSall_TM]).T)

            onp.savetxt(file_name+'_sdos_TE.csv',onp.stack([k0_range,DOSall_TE]).T)
            onp.savetxt(file_name+'_sdos_TM.csv',onp.stack([k0_range,DOSall_TM]).T)

            utils.plot_averaged_DOS(k0range, L, DOSall_TE, file_name, 'sdos', appended_string='TE')
            utils.plot_averaged_DOS(k0range, L, DOSall_TM, file_name, 'sdos', appended_string='TM')

        if compute_DOS:
            DOSall_TE = []
            DOSall_TM = []
            k0_range = []

            M = dospoints
            measurement_points = utils.uniform_unit_disk_picking(M)
            measurement_points *= L/2

            utils.plot_2d_points(measurement_points, file_name+'_measurement')

            for k0, alpha in zip(k0range,alpharange):
                dos_TE, dos_TM = solver.mean_DOS_measurements(measurement_points, k0, alpha, radius, file_name, regularize=regularize)
                DOSall_TE.append(dos_TE.numpy())
                DOSall_TM.append(dos_TM.numpy())

                k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                k0_range.append(k0_)

                onp.savetxt(file_name+'_temp_dos_TE.csv',onp.stack([k0_range,DOSall_TE]).T)
                onp.savetxt(file_name+'_temp_dos_TM.csv',onp.stack([k0_range,DOSall_TM]).T)

            onp.savetxt(file_name+'_dos_TE.csv',onp.stack([k0_range,DOSall_TE]).T)
            onp.savetxt(file_name+'_dos_TM.csv',onp.stack([k0_range,DOSall_TM]).T)

            utils.plot_averaged_DOS(k0range, L, DOSall_TE, file_name, 'dos', appended_string='TE')
            utils.plot_averaged_DOS(k0range, L, DOSall_TM, file_name, 'dos', appended_string='TM')

        if compute_interDOS:
            DOSall_TE = []
            DOSall_TM = []
            k0_range = []

            M = dospoints
            measurement_points = utils.uniform_unit_disk_picking(M)
            measurement_points *= L/2

            # Find all overlaps and redraw while you have some
            # Following Pierrat et al., I use 1 diameter as the spacing there
            spacing = 2.0*radius
            overlaps = np.nonzero(np.sum(np.cdist(measurement_points.to(np.double), points.to(np.double), p=2) <= spacing, axis = -1)).squeeze()
            count = overlaps.shape[0]
            while count > 0:
                print("Removing "+str(count)+" overlaps...")
                measurement_points[overlaps] = L/2 * utils.uniform_unit_disk_picking(count)
                overlaps = np.nonzero(np.sum(np.cdist(measurement_points.to(np.double), points.to(np.double), p=2) <= spacing, axis = -1)).squeeze()
                if len(overlaps.shape) == 0:
                    count = 0
                else:
                    count = overlaps.shape[0]

            utils.plot_2d_points(measurement_points, file_name+'_measurement')

            for k0, alpha in zip(k0range,alpharange):
                dos_TE, dos_TM = solver.mean_DOS_measurements(measurement_points, k0, alpha, radius, file_name, regularize=regularize)
                DOSall_TE.append(dos_TE.numpy())
                DOSall_TM.append(dos_TM.numpy())

                k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                k0_range.append(k0_)

                onp.savetxt(file_name+'_temp_idos_TE.csv',onp.stack([k0_range,DOSall_TE]).T)
                onp.savetxt(file_name+'_temp_idos_TM.csv',onp.stack([k0_range,DOSall_TM]).T)

            onp.savetxt(file_name+'_idos_TE.csv',onp.stack([k0_range,DOSall_TE]).T)
            onp.savetxt(file_name+'_idos_TM.csv',onp.stack([k0_range,DOSall_TM]).T)

            utils.plot_averaged_DOS(k0range, L, DOSall_TE, file_name, 'idos', appended_string='TE')
            utils.plot_averaged_DOS(k0range, L, DOSall_TM, file_name, 'idos', appended_string='TM')

        if compute_LDOS:
            # Expensive computation
            ngridx = gridsize[0]
            ngridy = gridsize[1]
            xyratio = ngridx/ngridy
            x,y = onp.meshgrid(onp.linspace(0,xyratio,ngridx),onp.linspace(0,1,ngridy))
            measurement_points = np.tensor((onp.vstack([x.ravel(),y.ravel()]).T-0.5)*L*window_width)

            batches = np.split(measurement_points, batch_size)
            n_batches = len(batches)

            extra_string=""
            if n_batches > 1:
                extra_string = extra_string+"es"
            print("Computing the LDOS at "+str(gridsize)+" points in "+str(n_batches)+" batch"+extra_string+" of "+str(batch_size))

            for k0, alpha in zip(k0range,alpharange):

                outputs_TE = []
                outputs_TM = []
                k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)

                for batch in range(0, n_batches):
                    print("Batch "+str(batch+1))
                    batch_points = batches[batch]
                    ldos_TE, ldos_TM = solver.LDOS_measurements(batch_points, k0, alpha, radius, regularize=regularize)

                    outputs_TE.append(ldos_TE)
                    outputs_TM.append(ldos_TM)

                #    onp.savetxt(file_name+'_temp_ldos_'+str(k0_)+'_TE.csv',np.cat(outputs_TE).numpy())
                #    onp.savetxt(file_name+'_temp_ldos_'+str(k0_)+'_TM.csv',np.cat(outputs_TM).numpy())

                ldos_TE = np.cat(outputs_TE)
                ldos_TM = np.cat(outputs_TM)

                utils.plot_LDOS_2D(ldos_TE,k0_,ngridx,ngridy,file_name, appended_string='TE', my_dpi = 300)
                utils.plot_LDOS_2D(ldos_TM,k0_,ngridx,ngridy,file_name, appended_string='TM', my_dpi = 300)


    elif ndim==3:
        if k0range_args == None:
            k0range = onp.arange(10,41)*64/128*2*onp.pi/L
        else: 
            if len(k0range_args)==1:
                k0range = onp.array([k0range_args[0]])* 2*onp.pi/L
            elif len(k0range_args)==2:
                k0range = onp.arange(k0range_args[0],k0range_args[1]+1,1)* 2*onp.pi/L
            else:
                k0range = onp.arange(k0range_args[0],k0range_args[1]+k0range_args[2],k0range_args[2])* 2*onp.pi/L
        volume = L*L*L*phi/N
        radius = onp.cbrt(volume * 3.0 / (4.0 * onp.pi))
        meas_points = 2*L*onp.vstack([onp.cos(thetas),onp.sin(thetas),onp.zeros(len(thetas))]).T
        print(meas_points.shape)
        utils.plot_3d_points(points,file_name)

        if cold_atoms:
            alpharange = utils.alpha_cold_atoms_3d(k0range)
        else:
            alpharange = onp.ones(len(k0range)) * utils.alpha_small_dielectric_object(refractive_n,volume)

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

                Ek = solver.calc(meas_points, Ej, k0, alpha, u, p, w, regularize=regularize, radius = radius)
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

                Ek = solver.calc(meas_points, Ej, k0, alpha, u, p, w, regularize = regularize, radius=radius)
                Ek = np.linalg.norm(Ek,axis=1)
                Eall.append(Ek.numpy())
            Eall = onp.array(Eall)
            Etotal = onp.absolute(Eall)**2
    
        utils.plot_transmission_angularbeam(k0range, L, thetas, Etotal, file_name) 
        utils.plot_transmission_flat(k0range, L, thetas, Etotal, file_name) 
        utils.plot_angular_averaged_transmission(k0range, L, Etotal, file_name)

        # Compute full fields
        # Pretty expensive!
        some_fields = intensity_fields+amplitude_fields+phase_fields
        if some_fields:
            # Expensive computation
            ngridx = gridsize[0]
            ngridy = gridsize[1]
            xyratio = ngridx/ngridy
            x,y,z = onp.meshgrid(onp.linspace(0,xyratio,ngridx)  - xyratio/2.0,onp.linspace(0,1,ngridy) - 0.5, [0.0])
            meas_points = np.tensor((onp.vstack([x.ravel(),y.ravel(), z.ravel()]).T)*L*window_width)

            batches = np.split(meas_points, batch_size)
            n_batches = len(batches)

            extra_string=""
            if n_batches > 1:
                extra_string = extra_string+"es"
            print("Computing the full fields at "+str(gridsize)+" points in "+str(n_batches)+" batch"+extra_string+" of "+str(onp.min([batch_size, ngridx*ngridy])))

            for k0, alpha in zip(k0range,alpharange):
                k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                print("k0L/2pi = "+str(k0_))

                for index, angle in enumerate(thetas):
                    angle_ = onp.round(angle*180/onp.pi)
                    print("angle = "+str(angle_)+"degrees")

                    Eall = []

                    for batch in range(0, n_batches):
                        print("Batch "+str(batch+1))
                        batch_points = batches[batch]

                        E = solver.calc_EM(batch_points, Ej[:,index], k0, alpha, u[index], p[index], w, regularize=regularize, radius=radius)

                        Eall.append(E)


                    Eall = np.cat(Eall, dim=0)

                    Eall_amplitude         = np.sqrt(Eall[:,0]**2 + Eall[:,1]**2 + Eall[:,2]**2)
                    Eall_longitudinal      = Eall[:,0]*onp.cos(angle) - Eall[:,1]*onp.sin(angle)
                    Eall_transverse        = Eall[:,0]*onp.sin(angle) + Eall[:,1]*onp.cos(angle)
                    Eall_vertical          = Eall[:,2]

                    

                    utils.plot_full_fields(Eall_amplitude, ngridx, ngridy, k0_, angle_, intensity_fields, amplitude_fields, phase_fields, file_name, appended_string='_width_'+str(window_width)+'_grid_'+str(ngridx)+'x'+str(ngridy), my_dpi = 300)
                    utils.plot_full_fields(Eall_longitudinal, ngridx, ngridy, k0_, angle_, intensity_fields, amplitude_fields, phase_fields, file_name, appended_string='_width_'+str(window_width)+'_grid_'+str(ngridx)+'x'+str(ngridy)+'_long', my_dpi = 300)
                    utils.plot_full_fields(Eall_transverse, ngridx, ngridy, k0_, angle_, intensity_fields, amplitude_fields, phase_fields, file_name, appended_string='_width_'+str(window_width)+'_grid_'+str(ngridx)+'x'+str(ngridy)+'_trans', my_dpi = 300)
                    utils.plot_full_fields(Eall_vertical, ngridx, ngridy, k0_, angle_, intensity_fields, amplitude_fields, phase_fields, file_name, appended_string='_width_'+str(window_width)+'_grid_'+str(ngridx)+'x'+str(ngridy)+'_trans', my_dpi = 300)


        if compute_SDOS:
            DOSall = []
            k0_range = []

            for k0, alpha in zip(k0range,alpharange):
                dos = solver.compute_eigenvalues_and_scatterer_LDOS( k0, alpha, radius, file_name, write_eigenvalues=write_eigenvalues)
                DOSall.append(dos.numpy())

                k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                k0_range.append(k0_)

                onp.savetxt(file_name+'_temp_sdos.csv',onp.stack(([k0_range,DOSall])).T)

            onp.savetxt(file_name+'_sdos.csv', onp.stack([k0_range,DOSall]).T)
            utils.plot_averaged_DOS(k0range, L, DOSall, file_name, 'sdos')

        if compute_DOS:
            DOSall = []
            k0_range = []

            # Expensive computation in 3d
            M = dospoints
            measurement_points = utils.uniform_unit_ball_picking(M, ndim)
            measurement_points *= L/2

            utils.plot_3d_points(measurement_points, file_name+'_measurement')

            for k0, alpha in zip(k0range,alpharange):
                dos = solver.mean_DOS_measurements(measurement_points, k0, alpha, radius, file_name, regularize=regularize)
                DOSall.append(dos.numpy())

                k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                k0_range.append(k0_)

                onp.savetxt(file_name+'_temp_dos.csv',onp.stack([k0_range,DOSall]).T)

            onp.savetxt(file_name+'_dos.csv',onp.stack([k0_range,DOSall]).T)
            utils.plot_averaged_DOS(k0range, L, DOSall, file_name, 'dos')

        if compute_interDOS:
            DOSall = []
            k0_range = []

            # Expensive computation in 3d
            M = dospoints
            measurement_points = utils.uniform_unit_ball_picking(M, ndim)
            measurement_points *= L/2

            # Find all overlaps and redraw while you have some
            # Following Pierrat et al., I use 1 diameter as the spacing there
            spacing = 2.0*radius
            overlaps = np.nonzero(np.sum(np.cdist(measurement_points, points, p=2) <= spacing)).squeeze()
            count = overlaps.shape[0]
            while count > 0:
                print("Removing "+str(count)+" overlaps...")
                measurement_points[overlaps] = L/2 * utils.uniform_unit_ball_picking(count, ndim).squeeze()
                overlaps = np.nonzero(np.sum(np.cdist(measurement_points, points, p=2) <= spacing))
                if len(overlaps.shape) == 0:
                    count = 0
                else:
                    count = overlaps.shape[0]


            utils.plot_3d_points(measurement_points, file_name+'_measurement')

            for k0, alpha in zip(k0range,alpharange):
                dos = solver.mean_DOS_measurements(measurement_points, k0, alpha, radius, file_name, regularize=regularize)
                DOSall.append(dos.numpy())

                k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                k0_range.append(k0_)

                onp.savetxt(file_name+'_temp_idos.csv',onp.stack([k0_range,DOSall]).T)

            onp.savetxt(file_name+'_idos.csv',onp.stack([k0_range,DOSall]).T)
            utils.plot_averaged_DOS(k0range, L, DOSall, file_name, 'idos')

        
        if compute_LDOS:
            # Expensive computation
            # For now, taking the central plane z = 0
            ngridx = gridsize[0]
            ngridy = gridsize[1]
            xyratio = ngridx/ngridy
            x,y,z = onp.meshgrid(onp.linspace(0,xyratio,ngridx),onp.linspace(0,1,ngridy), [0.5])
            measurement_points = np.tensor((onp.vstack([x.ravel(),y.ravel(), z.ravel()]).T-0.5)*L*window_width)

            batches = np.split(measurement_points, batch_size)
            n_batches = len(batches)

            extra_string=""
            if n_batches > 1:
                extra_string = extra_string+"es"
            n_points_LDOS = measurement_points.shape[0]
            print("Computing the LDOS at "+str(n_points_LDOS)+" points in "+str(n_batches)+" batch"+extra_string+" of "+str(batch_size))

            for k0, alpha in zip(k0range,alpharange):

                outputs = []
                k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)

                for batch in range(0, n_batches):
                    print("Batch "+str(batch+1))
                    batch_points = batches[batch]
                    ldos = solver.LDOS_measurements(batch_points, k0, alpha, radius, regularize=regularize)

                    outputs.append(ldos)

                #    onp.savetxt(file_name+'_temp_ldos_'+str(k0_)+'_TE.csv',np.cat(outputs_TE).numpy())
                #    onp.savetxt(file_name+'_temp_ldos_'+str(k0_)+'_TM.csv',np.cat(outputs_TM).numpy())

                ldos = np.cat(outputs)

                utils.plot_LDOS_2D(ldos,k0_,ngridx,ngridy,file_name, appended_string='z=0', my_dpi = 300)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run a full solving and plotting routine of MAGreeTe")
    parser.add_argument("head_directory", type=str, help="parent directory containing all configurations")
    parser.add_argument("ndim", type=int, help="Dimensionality of the problem at hand")
    parser.add_argument("--n_cpus", type=int, help="Number of cpus to use for computation\
        default = os.cpu_count", default=os.cpu_count())
    parser.add_argument("-n", "--refractive_n", type=complex, help="Complex refractive index of the dielectric material \
        default = 1.65 - 0.025j", default = 1.6 - 0.025j)
    parser.add_argument("--phi", type=float, help="Volume fraction of scatterers within the medium \
        default = 0.1", default = 0.1)  
    parser.add_argument("-k", "--k0range", nargs='+', type=float, help = "Values of k0 to span, in units of 2pi/L. Can be a single-value argument, a k_min and a k_max (with default step 1), or k_min, k_max, and step\
        default=(20,40,0.5) in 2d, (10,40,0.5) in 3d", default=None)
    parser.add_argument("-l", "--lattice", type=str, help="Use a simple lattice in lieu of datapoints as entry. \
        Options are 'square', 'triangular', 'honeycomb', 'quasicrystal', 'quasidual', 'quasivoro' in 2d, and 'cubic', 'fcc', 'bcc', 'diamond' in 3d. \
        default=None", default=None)
    parser.add_argument("-p","--just_plot", action='store_true', help="Bypass the calculation and just produce plots\
        default=False", default=False)
    parser.add_argument("-r", "--regularize", action='store_true', help="Regularize the fields and DOS inside of scatterers\
        default=False", default=False)
    parser.add_argument("-dos","--compute_DOS", action='store_true', help="Compute the mean DOS of the medium  \
        default=False", default=False)
    parser.add_argument("-idos","--compute_interDOS", action='store_true', help="Compute the mean DOS of the medium away from scatterers  \
        default=False", default=False)
    parser.add_argument("--dospoints",type=int, help="Number of points to use for the mean DOS computation \
        default = 1000", default=1000)
    parser.add_argument("-sdos","--compute_SDOS", action='store_true', help="Compute the spectrum of the Green's matrix, as well as the mean DOS at scatterers  \
        default=False", default=False)
    parser.add_argument("-ev","--write_eigenvalues", action='store_false', help="Write the eigenvalues of the Green's matrix at every frequency  \
        default=True", default=False)
    parser.add_argument("-ldos","--compute_LDOS", action='store_true', help="Compute an LDOS map  \
        default=False", default=False)
    parser.add_argument("--intensity_fields", action = "store_true", help="Output images of intensity fields for every beam used in the angular plot, in real space\
        default = False", default=False)
    parser.add_argument("--amplitude_fields", action = "store_true", help="Output images of amplitude fields for every beam used in the angular plot, in real space\
        default = False", default=False)
    parser.add_argument("--phase_fields", action = "store_true", help="Output images of phase fields for every beam used in the angular plot, in real space\
        default = False", default=False)
    parser.add_argument("-g","--gridsize",nargs=2,type=int, help="Number of pixels to use in the sidelength of output images \
        default = (301,301)", default=(301,301))
    parser.add_argument("-w","--window_width", type=float, help="Width of the viewfield for real-space plots, in units of system diameters, \
                        default = 1.2", default = 1.2)
    parser.add_argument("--boxsize", type=float, help="Set physical units for the box size: the results are dimensionless so that default=1m", default = 1)
    parser.add_argument("-o", "--output", type=str, help="Output directory\
        default = ./refractive_n_$Value/", default='')

    # DEBUG ARGUMENTS
    parser.add_argument("--donut", action="store_true", help="Do the donut thing! \
        default=False", default=False)

    args = parser.parse_args()

    head_directory      = args.head_directory
    ndim                = args.ndim
    n_cpus              = args.n_cpus
    refractive_n        = args.refractive_n
    phi                 = args.phi
    k0range_args        = args.k0range
    if k0range_args     != None:
        k0range_args    = tuple(k0range_args)
    just_plot           = args.just_plot
    regularize          = args.regularize
    lattice             = args.lattice
    compute_DOS         = args.compute_DOS
    compute_interDOS    = args.compute_interDOS
    dospoints           = args.dospoints
    compute_SDOS        = args.compute_SDOS
    write_eigenvalues   = args.write_eigenvalues
    compute_LDOS        = args.compute_LDOS
    intensity_fields    = args.intensity_fields
    amplitude_fields    = args.amplitude_fields
    phase_fields        = args.phase_fields
    gridsize            = tuple(args.gridsize)
    window_width        = args.window_width
    boxsize             = args.boxsize
    output_directory    = args.output

    # Debug arguments
    donut               = args.donut

    np.set_num_threads(n_cpus)
    np.device("cpu")
    main(head_directory, ndim, 
        refractive_n = refractive_n, phi=phi, k0range_args = k0range_args, lattice=lattice, regularize=regularize,
        just_plot=just_plot,
        compute_DOS=compute_DOS, compute_interDOS=compute_interDOS, dospoints=dospoints, compute_SDOS=compute_SDOS, write_eigenvalues=write_eigenvalues,  compute_LDOS=compute_LDOS,
        intensity_fields = intensity_fields, amplitude_fields=amplitude_fields, phase_fields=phase_fields,
        gridsize=gridsize, window_width=window_width,
        L=boxsize, output_directory=output_directory,
        donut = donut)
    sys.exit()


