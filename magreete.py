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
from Transmission2D import Transmission2D_vector, Transmission2D_scalar
from Transmission3D import Transmission3D_vector, Transmission3D_scalar
import lattices
import magreete_scalar

import argparse


def main(ndim, # Required arguments
        refractive_n = 1.65 + 0.025j, phi = 0.1, regularize = True, N_raw = 16384, beam_waist = 0.2, L = 1, size_subsample = 1.0, source = "beam", scalar = False, # Physical parameters
        lattice=None, cold_atoms=False, kresonant_ = None, annulus = 0, composite = False, kick = 0.0, shift = 0.0, input_files_args = None, # Special cases
        k0range_args = None, thetarange_args = None,# Range of values to use
        compute_transmission = False, plot_transmission = False, single_scattering_transmission = False, scattered_fields=False, transmission_radius = 2.0,
        compute_DOS=False, compute_cavityDOS = False, compute_interDOS=False, compute_SDOS=False, compute_LDOS=False, dos_sizes_args = None, dospoints=1, spacing_factor = 1.0, idos_radius = 1.0, N_fibo = 1000,
        compute_eigenmodes = False, number_eigenmodes = 1, plot_eigenmodes = False, sorting_type = 'IPR', adapt_z = True,
        intensity_fields = False, amplitude_fields = False, phase_fields = False, just_compute_averages = False,# Computations to perform
        write_eigenvalues=False, write_ldos= False,  gridsize=(301,301), window_width=1.2, angular_width = 0.0, plot_theta_index = 0, batch_size = 101*101, adapt_scale = False, output_directory="" # Parameters for outputs
        ):
    '''
    Simple front-end for MAGreeTe
    '''
    
    # Keep cut_radius as the internal here
    cut_radius = 0.5 * size_subsample
    beam_waist *= size_subsample
    transmission_radius *= size_subsample

    # The full option does not conserve energy but is interesting to have for pedagogy?
    self_interaction_type = "Rayleigh" # Rayleigh or full

    if onp.imag(refractive_n) < 0:
        print("Imaginary parts of indices should be positive!")
        sys.exit()

    # Name the output directory in a human-readable way containing the three physical parameters: raw number of particles, volume fraction and refractive index
    output_directory_suffix = "phi_"+str(phi)+"/"
    if cold_atoms:
        output_directory_suffix += "cold_atoms"
    else:
        output_directory_suffix += "refractive_n_"+str(refractive_n)
    if kick != 0.0:
        output_directory_suffix +="_kicked_"+str(kick)
    if regularize:
        output_directory_suffix += "_reg"
    if scalar:
        output_directory_suffix += "_scalar"
        
    # Angles to use for transmission and fields
    if thetarange_args == None:
        Ntheta = 360
        thetas = onp.arange(Ntheta)/Ntheta*2*np.pi
    else:
        if len(thetarange_args)==1:
            Ntheta = 1
            thetas = onp.array(thetarange_args)*np.pi / 180.0
        elif len(thetarange_args)==2:
            thetas = onp.arange(thetarange_args[0],thetarange_args[1]+1,1) * np.pi / 180.0
            Ntheta = len(thetas)
        else:
            thetas = onp.arange(thetarange_args[0],thetarange_args[1]+thetarange_args[2],thetarange_args[2]) * np.pi / 180.0
            Ntheta = len(thetas)
            
    # Sizes to use for DOS
    if dos_sizes_args == None:
        Ndos_sizes = 1
        dos_sizes  = onp.array([1.0])
    else:
        if len(dos_sizes_args)==1:
            Ndos_sizes = 1
            dos_sizes  = onp.array(dos_sizes_args)
        elif len(dos_sizes_args)==2:
            dos_sizes = onp.linspace(dos_sizes_args[0],dos_sizes_args[1],num=10) 
            Ndos_sizes = len(dos_sizes)
        else:
            dos_sizes = onp.arange(dos_sizes_args[0],dos_sizes_args[1]+dos_sizes_args[2],dos_sizes_args[2])
            Ndos_sizes = len(dos_sizes)
            
    # Keep a copy of the thetas used to plot if thetas get overwritten when loading files
    thetas_plot = thetas 
    # Figure out how many angles around the central one to use for the definition of transmission
    n_thetas_trans = int(onp.floor(angular_width * 0.5 * len(thetas)))

    # Beam waist
    w = beam_waist * L

    # Check number of configurations to go over
    if input_files_args != None:
        number_copies = len(input_files_args)
        file_index_list = onp.arange(number_copies)
    elif lattice != None:
        file_index_list = [0]
    else:
        print("Please provide a valid input either as an input file or as a lattice option")
        sys.exit()

    # Loop over copies
    for file_index in file_index_list:
        print("____________________________________________________\nCopy #"+str(file_index)+"\n____________________________________________________")

        # A custom file was provided
        if lattice == None:

            file_name = input_files_args[file_index]
            
            points = utils.loadpoints(file_name, ndim)
            points = np.tensor(points,dtype=np.double)
            
            if np.amax(points)>0.5:
                points -= np.mean(points)
                points /= points.amax()
                points /= 2.0
            shape_before = points.shape
            
            # Make output dir
            N_raw = shape_before[0]
            output_directory = os.path.abspath(output_directory)
            output_directory = os.path.join(output_directory, "N"+str(N_raw), output_directory_suffix)
            utils.trymakedir(output_directory)
            
            # Override filename so that output files are well-behaved
            file_name = file_name.split("/")[-1]
            
            # Adjust point pattern by removing overlap, cutting, kicking
            points = np.unique(points, dim=0)
            shape_after = points.shape
            if shape_before[0] != shape_after[0]:
                print("There were {} points overlapping with others! Removing.".format(shape_before[0]-shape_after[0]))
            points = lattices.cut_circle(points,cut_radius)
            #points *= 0.5/np.amax(points)
            # Add random kicks
            if kick != 0.0:
                points = lattices.add_displacement(points, dr=kick)
            if shift != 0.0:
                points += shift * lattices.uniform_unit_ball_picking(1,ndim)

        # A generative recipe was selected
        else:

            file_name = lattice
            output_directory = os.path.abspath(output_directory)
            output_directory = os.path.join(output_directory, "N"+str(N_raw), output_directory_suffix)
            utils.trymakedir(output_directory)
            points = make_lattice(lattice, N_raw, kick, ndim)
            if lattice == 'poisson':
                file_name += str(ndim)+'d'
            points = lattices.cut_circle(points, cut_radius)
            
            if shift != 0.0:
                points += shift * lattices.uniform_unit_ball_picking(1,ndim)

        # Cut configuration if needed
        if annulus > 0:
            points = lattices.exclude_circle(points,annulus)
            file_name += '_annulus_'+str(annulus)
        if composite:
            comp = lattices.square(128)
            comp = lattices.cut_circle(comp,annulus)
            points = np.vstack([points,comp])
            file_name += '_composite'

        # After all this, write down the actual N and make the system the right size
        N = points.shape[0]
        if N == 0:
            print("0 points remain after cutting")
            sys.exit()
        points *= L
        assert ndim == points.shape[1]
        print("\n\nLoaded a ("+str(file_name)+") system of N = "+str(N_raw)+" points in d = "+str(ndim))
        print("N = "+str(N)+" points remain after cutting to a disk and rescaling to L = "+str(L)+"\n\n")

        if source != "beam":
            source_suffix = source
        else:
            source_suffix = ""
            
        output_directory = os.path.join(output_directory,file_name+source_suffix)
        if size_subsample < 1.0:
            sss_subdir = "size_subsampling_"+str(size_subsample)
            output_directory = os.path.join(output_directory, sss_subdir)
        utils.trymakedir(output_directory)
        file_name = output_directory+"/"+file_name

        # Define wave-vector list here to avoid defining it again when averaging
        if ndim == 2:
            
            # Volume and radius of (circular cross-section) scatterers
            volume = L**2 * phi/N_raw
            radius = onp.sqrt(volume/onp.pi )
            
            if k0range_args == None:
                # Set the max to be the last one where the assumptions are still somewhat ok, 2pi / radius
                k_max = 0.25 * L /radius
                k0range = onp.arange(1.0, k_max, 0.5)*2*onp.pi/L
            else:
                if len(k0range_args)==1:
                    k0range = onp.array([k0range_args[0]])* 2*onp.pi/L
                elif len(k0range_args)==2:
                    k0range = onp.arange(k0range_args[0],k0range_args[1]+1,1)* 2*onp.pi/L
                else:
                    k0range = onp.arange(k0range_args[0],k0range_args[1]+k0range_args[2],k0range_args[2])* 2*onp.pi/L

            # Consistency check: plot set of scatterers
            utils.plot_2d_points(points,file_name)

            # Polarizability list
            if cold_atoms:
                if kresonant_ == None:
                    kresonant_ = 0.1 * L / radius
                kresonant = 2 * onp.pi * kresonant_
                static_deltaeps = refractive_n**2 - 1
                # as omega -> 0, Lorentz -> omegap**2 / omega0 ** 2 
                # Therefore kplasma = kresonant * sqrt(static_deltaeps)
                kplasma = kresonant * onp.sqrt(onp.real(static_deltaeps))
                damping = onp.imag(static_deltaeps) * kplasma # Just an ansatz
                alpharange = utils.alpha_Lorentz(k0range, volume, kresonant, kplasma, damping)
                self_interaction = True
                print("Effective indices:"+str(onp.sqrt(alpharange/volume + 1)))
            else:
                alpharange = onp.ones(len(k0range)) * utils.alpha_small_dielectric_object(refractive_n,volume)
                self_interaction = True

        elif ndim ==3:
            
            # Volume and radius of (spherical) scatterers
            volume = L**3 * phi/N_raw
            radius = onp.cbrt(volume * 3.0 / (4.0 * onp.pi))
            
            if k0range_args == None:
                # Set the max to be the last one where the assumptions are still somewhat ok, 2pi / radius
                k_max = 0.25 * L /radius
                k0range = onp.arange(1.0, k_max, 0.5)*2*onp.pi/L
            else: 
                if len(k0range_args)==1:
                    k0range = onp.array([k0range_args[0]])* 2*onp.pi/L
                elif len(k0range_args)==2:
                    k0range = onp.arange(k0range_args[0],k0range_args[1]+1,1)* 2*onp.pi/L
                else:
                    k0range = onp.arange(k0range_args[0],k0range_args[1]+k0range_args[2],k0range_args[2])* 2*onp.pi/L

            # Consistency check: plot set of scatterers
            utils.plot_3d_points(points,file_name)

            # Polarizability list
            if cold_atoms:
                if kresonant_ == None:
                    kresonant_ = 0.1 * L / radius
                kresonant = 2 * onp.pi * kresonant_
                static_deltaeps = refractive_n**2 - 1
                # as omega -> 0, Lorentz -> omegap**2 / omega0 ** 2 
                # Therefore kplasma = kresonant * sqrt(static_deltaeps)
                kplasma = kresonant * onp.sqrt(onp.real(static_deltaeps))
                damping = onp.imag(static_deltaeps) * kplasma # Just an ansatz
                alpharange = utils.alpha_Lorentz(k0range, volume, kresonant, kplasma, damping)
                self_interaction = True
                print("Effective indices:"+str(onp.sqrt(alpharange/volume + 1)))
            else:
                alpharange = onp.ones(len(k0range)) * utils.alpha_small_dielectric_object(refractive_n,volume)
                self_interaction = True
                
        # Generate the corresponding list of optical thicknesses, and plot them
        utils.plot_optical_thickness(k0range, L, alpharange, ndim, phi, volume, file_name)
        # Also plot the values of ka to check whether hypotheses are consistent
        utils.plot_k_times_radius(k0range, radius, L, file_name)
        # Finally, plot dressed polarizability of a single scatterer to pinpoint resonances
        utils.plot_dressed_polarizability(k0range, L, alpharange, ndim, radius, volume, self_interaction, file_name, self_interaction_type=self_interaction_type)

        # If the code is run solely to put together data already obtained for several copies, skip this
        if just_compute_averages:
            break

        ### ###############
        ### Solver choice: 2d or 3d, vector or scalar
        ### ###############
        
        if ndim==2:
            
            if scalar:
                solver = Transmission2D_scalar(points, source = source)
            else:
                solver = Transmission2D_vector(points, source = source)
                
        elif ndim == 3:
            
            if scalar:
                solver = Transmission3D_scalar(points, source = source)
            else:
                solver = Transmission3D_vector(points, source = source)
                
        else:
            raise NotImplementedError
            
        ### ###############
        ### Transmission plot computations
        ### ###############

        if compute_transmission or plot_transmission or single_scattering_transmission:

            # Define the list of measurement points for transmission plots
            if ndim ==2:
                # Use regularly spaced angles on the circle
                Ntheta_meas = 360
                thetas_measurement = onp.arange(Ntheta_meas)/Ntheta_meas*2*np.pi
                measurement_points = transmission_radius*L*onp.vstack([onp.cos(thetas_measurement),onp.sin(thetas_measurement)]).T
            else: 
                # Use Fibonacci sphere as samples on the sphere
                measurement_points = transmission_radius*L*utils.fibonacci_sphere(N_fibo)
                # Also define the unit vectors describing the source orientation and its polarization from here
                u = onp.stack([onp.cos(thetas),onp.sin(thetas),onp.zeros(len(thetas))]).T
                u = np.tensor(u)
                p = np.zeros(u.shape)
                p[:,2] = 1
                
            # A fresh computation is required
            if compute_transmission: 

                Eall = []
                E0all = []
                Eall_scat = []
                
                for k0, alpha in zip(k0range,alpharange):
                    
                    # Compute source value AT scatterers and measurement points
                    if ndim == 2:
                        # In 2d: no ambiguity to make thetas into k vectors and polarizations even if vector wave
                        E0_scat = solver.generate_source(np.tensor(points), k0, thetas, beam_waist, print_statement='Source at scatterers')
                        E0_meas = solver.generate_source(np.tensor(measurement_points), k0, thetas, beam_waist, print_statement='Source at measurement points')
                    else:
                        if scalar:
                            # In 3d scalar, no need to specify polarization vector
                            E0_scat = solver.generate_source(np.tensor(points), k0, u, beam_waist, print_statement='Source at scatterers')
                            E0_meas = solver.generate_source(np.tensor(measurement_points), k0, u, beam_waist, print_statement='Source at measurement points')
                        else:
                            # In 3d vector, need to specify polarization vector
                            E0_scat = solver.generate_source(np.tensor(points), k0, u, p, beam_waist, print_statement='Source at scatterers')
                            E0_meas = solver.generate_source(np.tensor(measurement_points), k0, u, p, beam_waist, print_statement='Source at measurement points')
                    
                    Ej = solver.solve(k0, alpha, radius, E0_scat, self_interaction=self_interaction, self_interaction_type=self_interaction_type)
                    k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                    params = [alpha, k0]
                    hkl.dump([onp.array(Ej), onp.array(params),onp.array(points), onp.array(thetas)],file_name+'_Ek_k0_'+str(k0_)+'_'+str(file_index)+'.hkl')

                    Ek = solver.propagate(measurement_points, Ej, k0, alpha, E0_meas, regularize = regularize, radius=radius)

                    if scattered_fields:
                        Ek_scat = Ek - E0_meas
                        Eall_scat.append(Ek_scat.numpy())

                    E0all.append(E0_meas.numpy())
                    Eall.append(Ek.numpy())

            # A computation has already been performed
            elif plot_transmission:

                Eall = []
                E0all = []
                Eall_scat = []

                for k0, alpha in zip(k0range,alpharange):
                    
                    k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                    Ej, params, points, thetas = hkl.load(file_name+'_Ek_k0_'+str(k0_)+'_'+str(file_index)+'.hkl')
                    Ej = np.tensor(Ej, dtype=np.complex128)
                    points = np.tensor(points, dtype=np.float64)
                    thetas = onp.float64(thetas)
                    alpha, k0 = params
                    k0 = onp.float64(k0)
                    alpha = onp.complex128(alpha)

                    # Compute source value AT scatterers and measurement points
                    if ndim == 2:
                        
                        # Need to initialize solver with points from loaded files here
                        # Otherwise random kicks and/or poisson point patterns would be different from hkl files
                        if scalar:
                            solver = Transmission2D_scalar(points, source = source)
                        else:
                            solver = Transmission2D_vector(points, source = source)
                        
                        # In 2d: no ambiguity to make thetas into k vectors and polarizations even if vector wave
                        E0_scat = solver.generate_source(np.tensor(points), k0, thetas, beam_waist, print_statement='Source at scatterers')
                        E0_meas = solver.generate_source(np.tensor(measurement_points), k0, thetas, beam_waist, print_statement='Source at measurement points')
                    else:
                        
                        # Need to initialize solver with points from loaded files here
                        # Otherwise random kicks and/or poisson point patterns would be different from hkl files
                        if scalar:
                            solver = Transmission3D_scalar(points, source = source)
                        else:
                            solver = Transmission3D_vector(points, source = source)
                        
                        # In 3d, need to specify polarization vector
                        E0_scat = solver.generate_source(np.tensor(points), k0, u, p, beam_waist, print_statement='Source at scatterers')
                        E0_meas = solver.generate_source(np.tensor(measurement_points), k0, u, p, beam_waist, print_statement='Source at measurement points')

                    Ek = solver.propagate(measurement_points, Ej, k0, alpha, E0_meas, regularize = regularize, radius = radius)
                    
                    
                    if scattered_fields:
                        Ek_scat = Ek - E0_meas
                        Eall_scat.append(Ek_scat.numpy())
                    

                    E0all.append(E0.numpy())
                    Eall.append(Ek.numpy())
            
            if compute_transmission or plot_transmission:
                # Save final transmission data for plotting purposes and/or averaging purposes
                hkl.dump([onp.array(Eall), onp.array(k0range), onp.array(thetas)],file_name+'_transmission_'+str(file_index)+'.hkl')

            # If required: plot results
            if plot_transmission:
                # Compute intensities at measurement points
                Eall = onp.array(Eall)
                total = onp.absolute(Eall)**2
                if not scalar:
                    total = onp.sum(total, axis=2)

                # Produce plots
                if ndim == 2:
                    utils.plot_transmission_angularbeam(k0range, L, thetas, total, file_name,  n_thetas_trans = n_thetas_trans, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index), adapt_scale = adapt_scale)
                else:
                    utils.plot_transmission_angularbeam_3d(k0range, L, thetas, total, measurement_points, file_name, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index), adapt_scale = adapt_scale)
                
                # Produce transmission normalized by total intensity of the INCIDENT FIELD on the sphere
                I0all = onp.absolute(E0all)**2
                if not scalar:
                    I0all = onp.sum(I0all, axis = 2)

                if ndim ==2:
                    utils.plot_transmission_angularbeam(k0range, L, thetas, total, file_name,  n_thetas_trans = n_thetas_trans, normalization = I0all, adapt_scale = adapt_scale, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_incnorm')
                    utils.plot_transmission_angularbeam(k0range, L, thetas, total, file_name,  n_thetas_trans = n_thetas_trans, normalization = total, adapt_scale = adapt_scale, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_norm')
                else:
                    utils.plot_transmission_angularbeam_3d(k0range, L, thetas, total, measurement_points, file_name, normalization = I0all, adapt_scale = adapt_scale, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_incnorm')
                    utils.plot_transmission_angularbeam_3d(k0range, L, thetas, total, measurement_points, file_name, normalization = total, adapt_scale = adapt_scale, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_norm')



                if scattered_fields:
                    # Compute scattered intensities at measurement points
                    Eall_scat = onp.array(Eall_scat)
                    total_scat = onp.absolute(Eall_scat)**2
                    if not scalar:
                        total_scat = onp.sum(total_scat, axis=2)
                    
                    
                    # Produce plots
                    if ndim == 2:
                        utils.plot_transmission_angularbeam(k0range, L, thetas, total_scat, file_name,  n_thetas_trans = n_thetas_trans, adapt_scale = adapt_scale, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_scat')
                        utils.plot_transmission_angularbeam(k0range, L, thetas, total_scat, file_name,  n_thetas_trans = n_thetas_trans, adapt_scale = adapt_scale, normalization = total_scat, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_scat_norm')
                    else:
                        utils.plot_transmission_angularbeam_3d(k0range, L, thetas, total_scat, measurement_points, file_name, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_scat', adapt_scale = adapt_scale)
                        utils.plot_transmission_angularbeam_3d(k0range, L, thetas, total_scat, measurement_points, file_name, normalization = total, adapt_scale = adapt_scale, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_scat_norm')
                            
            # Single-scattering transmission
            if single_scattering_transmission:

                Eall_ss = []
                Eall_scat_ss = []
                
                for k0, alpha in zip(k0range,alpharange):
                    
                    # Compute source value AT scatterers and measurement points
                    # XXX Merge into one function to call
                    if ndim == 2:
                        # In 2d: no ambiguity to make thetas into k vectors and polarizations even if vector wave
                        E0_scat = solver.generate_source(np.tensor(points), k0, thetas, beam_waist, print_statement='Source at scatterers')
                        E0_meas = solver.generate_source(np.tensor(measurement_points), k0, thetas, beam_waist, print_statement='Source at measurement points')
                    else:
                        if scalar:
                            # In 3d scalar, no need to specify polarization vector
                            E0_scat = solver.generate_source(np.tensor(points), k0, u, beam_waist, print_statement='Source at scatterers')
                            E0_meas = solver.generate_source(np.tensor(measurement_points), k0, u, beam_waist, print_statement='Source at measurement points')
                        else:
                            # In 3d vector, need to specify polarization vector
                            E0_scat = solver.generate_source(np.tensor(points), k0, u, p, beam_waist, print_statement='Source at scatterers')
                            E0_meas = solver.generate_source(np.tensor(measurement_points), k0, u, p, beam_waist, print_statement='Source at measurement points')
                    
                    Ek_ss = solver.propagate_ss(measurement_points, k0, alpha, E0_meas, E0_scat, regularize = regularize, radius = radius)
                    
                    if scattered_fields:
                        Ek_scat_ss = Ek_ss - E0_meas
                        Eall_scat_ss.append(Ek_scat_ss.numpy())
                    
                    Eall_ss.append(Ek_ss.numpy())
                    
                # Compute intensities at measurement points
                Eall_ss = onp.array(Eall_ss)
                total_ss = onp.absolute(Eall_ss)**2
                if not scalar:
                    total_ss = onp.sum(total_ss, axis=2)

                # Produce plots
                if ndim == 2:
                    utils.plot_transmission_angularbeam(k0range, L, thetas, total_ss, file_name,  n_thetas_trans = n_thetas_trans, adapt_scale = adapt_scale, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_ss')
                    theta_plot = onp.round(180 * thetas[plot_theta_index]/onp.pi)
                    utils.plot_singlebeam_angular_frequency_plot(k0range, L, thetas, total_ss, file_name, plot_theta_index = plot_theta_index, appended_string='_'+str(file_index)+'_angle_'+str(theta_plot)+'_ss')
                else:
                    utils.plot_transmission_angularbeam_3d(k0range, L, thetas, total_ss, measurement_points, file_name, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_ss', adapt_scale = adapt_scale)
                
                if plot_transmission:
                    # Also compute the intensity associated to the multiple-scattering contribution of the field, if the full field was computed
                    Eall_multiple = Eall - Eall_ss
                    total_multiple = onp.absolute(Eall_multiple)**2
                    if not scalar:
                        total_multiple = onp.sum(total_multiple, axis=2)

                    # Produce plots
                    if ndim == 2:
                        utils.plot_transmission_angularbeam(k0range, L, thetas, total_multiple, file_name,  n_thetas_trans = n_thetas_trans, adapt_scale = adapt_scale, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_multiple')
                        utils.plot_singlebeam_angular_frequency_plot(k0range, L, thetas, total_multiple, file_name, plot_theta_index = plot_theta_index, appended_string='_'+str(file_index)+'_angle_'+str(theta_plot)+'_multiple')
                    else:
                        utils.plot_transmission_angularbeam_3d(k0range, L, thetas, total_multiple, measurement_points, file_name, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_multiple', adapt_scale = adapt_scale)

                if scattered_fields:
                    # Compute scattered intensities at measurement points
                    Eall_scat_ss = onp.array(Eall_scat_ss)
                    total_scat_ss = onp.absolute(Eall_scat_ss)**2
                    if not scalar:
                        total_scat_ss = onp.sum(total_scat_ss, axis=2)
                    
                                    # Produce plots
                    if ndim == 2:
                        utils.plot_transmission_angularbeam(k0range, L, thetas, total_scat_ss, file_name,  n_thetas_trans = n_thetas_trans, adapt_scale = adapt_scale, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_scat_ss')
                        utils.plot_transmission_angularbeam(k0range, L, thetas, total_scat_ss, file_name,  n_thetas_trans = n_thetas_trans, adapt_scale = adapt_scale, normalization=total_scat_ss, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_scat_ss_norm')
                        utils.plot_singlebeam_angular_frequency_plot(k0range, L, thetas, total_scat_ss, file_name, plot_theta_index = plot_theta_index, appended_string='_'+str(file_index)+'_angle_'+str(theta_plot)+'_scat_ss')
                        utils.plot_singlebeam_angular_frequency_plot(k0range, L, thetas, total_scat_ss, file_name, plot_theta_index = plot_theta_index, normalization = total_scat_ss, appended_string='_'+str(file_index)+'_angle_'+str(theta_plot)+'_scat_ss_norm')
                    else:
                        utils.plot_transmission_angularbeam_3d(k0range, L, thetas, total_scat_ss, measurement_points, file_name, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_scat_ss', adapt_scale = adapt_scale)
                        utils.plot_transmission_angularbeam_3d(k0range, L, thetas, total_scat_ss, measurement_points, file_name, normalization=total_scat_ss, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_scat_ss_norm', adapt_scale = adapt_scale)


        sys.exit()
        ### ###############
        ### Intensity fields calculations
        ### ###############

        # Compute full fields
        # Pretty expensive!
        some_fields = intensity_fields+amplitude_fields+phase_fields
        if some_fields:
            
            ngridx = gridsize[0]
            ngridy = gridsize[1]
            xyratio = ngridx/ngridy
            
            if ndim == 2:
                    
                x,y = onp.meshgrid(onp.linspace(0,xyratio,ngridx)  - xyratio/2.0,onp.linspace(0,1,ngridy) - 0.5)
                measurement_points = np.tensor((onp.vstack([x.ravel(),y.ravel()]).T)*L*window_width)
                
            else:

                x,y,z = onp.meshgrid(onp.linspace(0,xyratio,ngridx)  - xyratio/2.0,onp.linspace(0,1,ngridy) - 0.5, [0.0])
                measurement_points = np.tensor((onp.vstack([x.ravel(),y.ravel(), z.ravel()]).T)*L*window_width)

            batches = np.split(measurement_points, batch_size)
            n_batches = len(batches)

            extra_string=""
            if n_batches > 1:
                extra_string = extra_string+"es"
            print("Computing the full fields at "+str(gridsize)+" points in "+str(n_batches)+" batch"+extra_string+" of "+str(onp.min([batch_size, ngridx*ngridy])))


            for k0, alpha in zip(k0range,alpharange):
                k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                print("k0L/2pi = "+str(k0_))

                # Check if file already exists or if computation is needed
                file = file_name+'_Ek_k0_'+str(k0_)+'_'+str(file_index)+'.hkl'
                # File is there: load data
                if os.path.isfile(file):
                    EjTE, EjTM, params, points, thetas = hkl.load(file_name+'_Ek_k0_'+str(k0_)+'_'+str(file_index)+'.hkl')
                    EjTE = np.tensor(EjTE, dtype=np.complex128)
                    EjTM = np.tensor(EjTM, dtype=np.complex128)
                    points = np.tensor(points, dtype=np.complex128)
                    thetas = onp.float64(thetas)
                    alpha, k0 = params
                    k0 = onp.float64(k0)
                    alpha = onp.complex128(alpha)
                else:
                    EjTE, EjTM = solver.solve_EM(k0, alpha, thetas, radius, w, self_interaction=self_interaction, self_interaction_type=self_interaction_type)
                    k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                    params = [alpha, k0]
                    hkl.dump([onp.array(EjTE), onp.array(EjTM), onp.array(params),onp.array(points), onp.array(thetas)],file_name+'_Ek_k0_'+str(k0_)+'_'+str(file_index)+'.hkl')

                for angle in thetas_plot:
                    angle_ = onp.round(angle*180/onp.pi)
                    index = onp.where(thetas == angle)[0][0]
                    print("angle = "+str(angle_)+"degrees")

                    ETEall = []
                    ETMall = []
                    if scattered_fields:
                        ETEall_scat = []
                        ETMall_scat = []

                    for batch in range(0, n_batches):
                        print("Batch "+str(batch+1))
                        batch_points = batches[batch]

                        EkTE, EkTM = solver.propagate_EM(batch_points, EjTE[:,index], EjTM[:,index].unsqueeze(-1), k0, alpha, [angle], w, regularize = regularize, radius = radius)

                        ETEall.append(EkTE)
                        ETMall.append(EkTM)
                        
                        if scattered_fields:
                            E0TE, E0TM = solver.generate_source(batch_points, k0, [angle], w, print_statement='scattered_fields')
                            ETMall_scat.append(EkTM - E0TM)
                            ETEall_scat.append(EkTE - E0TE)

                    ETEall = np.cat(ETEall, dim=0).squeeze(-1)
                    ETMall = np.cat(ETMall, dim=0)
                    
                    # The medium is centered at (0,0)
                    viewing_angle = np.arctan2(measurement_points[:,1], measurement_points[:,0]) #y,x

                    ETEall_amplitude          = np.sqrt(np.absolute(ETEall[:,0])**2 + np.absolute(ETEall[:,1])**2)
                    ETEall_longitudinal       = ETEall[:,0]*np.cos(viewing_angle) - ETEall[:,1]*np.sin(viewing_angle)
                    ETEall_transverse         = ETEall[:,0]*np.sin(viewing_angle) + ETEall[:,1]*np.cos(viewing_angle)

                    ETEall_amplitude    = ETEall_amplitude.reshape(ngridy, ngridx)
                    ETEall_longitudinal = ETEall_longitudinal.reshape(ngridy, ngridx)
                    ETEall_transverse   = ETEall_transverse.reshape(ngridy, ngridx)
                    ETMall = ETMall.reshape(ngridy, ngridx)

                    utils.plot_full_fields(ETEall_amplitude, ngridx, ngridy, k0_, angle_, intensity_fields, False, False, file_name, appended_string='_width_'+str(window_width)+'_grid_'+str(ngridx)+'x'+str(ngridy)+'_'+str(file_index)+'_TE', my_dpi = 300)
                    utils.plot_full_fields(ETEall_longitudinal, ngridx, ngridy, k0_, angle_, intensity_fields, amplitude_fields, phase_fields, file_name, appended_string='_width_'+str(window_width)+'_grid_'+str(ngridx)+'x'+str(ngridy)+'_'+str(file_index)+'_TE_long', my_dpi = 300)
                    utils.plot_full_fields(ETEall_transverse, ngridx, ngridy, k0_, angle_, intensity_fields, amplitude_fields, phase_fields, file_name, appended_string='_width_'+str(window_width)+'_grid_'+str(ngridx)+'x'+str(ngridy)+'_'+str(file_index)+'_TE_trans', my_dpi = 300)
                    utils.plot_full_fields(ETMall, ngridx, ngridy, k0_, angle_, intensity_fields, amplitude_fields, phase_fields, file_name, appended_string='_width_'+str(window_width)+'_grid_'+str(ngridx)+'x'+str(ngridy)+'_'+str(file_index)+'_TM', my_dpi = 300)

                    if scattered_fields:
                        ETEall = np.cat(ETEall_scat, dim=0).squeeze(-1)
                        ETMall = np.cat(ETMall_scat, dim=0)
                        
                        ETEall_amplitude          = np.sqrt(np.absolute(ETEall[:,0])**2 + np.absolute(ETEall[:,1])**2)
                        ETEall_longitudinal       = ETEall[:,0]*np.cos(viewing_angle) - ETEall[:,1]*np.sin(viewing_angle)
                        ETEall_transverse         = ETEall[:,0]*np.sin(viewing_angle) + ETEall[:,1]*np.cos(viewing_angle)

                        ETEall_amplitude    = ETEall_amplitude.reshape(ngridy, ngridx)
                        ETEall_longitudinal = ETEall_longitudinal.reshape(ngridy, ngridx)
                        ETEall_transverse   = ETEall_transverse.reshape(ngridy, ngridx)
                        ETMall = ETMall.reshape(ngridy, ngridx)

                        utils.plot_full_fields(ETEall_amplitude, ngridx, ngridy, k0_, angle_, intensity_fields, False, False, file_name, appended_string='_width_'+str(window_width)+'_grid_'+str(ngridx)+'x'+str(ngridy)+'_'+str(file_index)+'_TE_scat', my_dpi = 300)
                        utils.plot_full_fields(ETEall_longitudinal, ngridx, ngridy, k0_, angle_, intensity_fields, amplitude_fields, phase_fields, file_name, appended_string='_width_'+str(window_width)+'_grid_'+str(ngridx)+'x'+str(ngridy)+'_'+str(file_index)+'_TE_long_scat', my_dpi = 300)
                        utils.plot_full_fields(ETEall_transverse, ngridx, ngridy, k0_, angle_, intensity_fields, amplitude_fields, phase_fields, file_name, appended_string='_width_'+str(window_width)+'_grid_'+str(ngridx)+'x'+str(ngridy)+'_'+str(file_index)+'_TE_trans_scat', my_dpi = 300)
                        utils.plot_full_fields(ETMall, ngridx, ngridy, k0_, angle_, intensity_fields, amplitude_fields, phase_fields, file_name, appended_string='_width_'+str(window_width)+'_grid_'+str(ngridx)+'x'+str(ngridy)+'_'+str(file_index)+'_TM_scat', my_dpi = 300)


            if compute_SDOS:

                solver = Transmission2D_TETM(points, source = source)

                DOSall_TE = []
                DOSall_TM = []
                k0_range = []

                for k0, alpha in zip(k0range,alpharange):
                    dos_TE, dos_TM = solver.compute_eigenvalues_and_scatterer_LDOS( k0, alpha, radius, file_name, write_eigenvalues=write_eigenvalues, self_interaction = self_interaction, self_interaction_type = self_interaction_type)
                    DOSall_TE.append(dos_TE.numpy())
                    DOSall_TM.append(dos_TM.numpy())
                    k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                    k0_range.append(k0_)

                    onp.savetxt(file_name+'_temp_sdos_TE.csv',onp.stack([k0_range,DOSall_TE]).T)
                    onp.savetxt(file_name+'_temp_sdos_TM.csv',onp.stack([k0_range,DOSall_TM]).T)

                onp.savetxt(file_name+'_sdos_TE.csv',onp.stack([k0_range,DOSall_TE]).T)
                onp.savetxt(file_name+'_sdos_TM.csv',onp.stack([k0_range,DOSall_TM]).T)

                utils.plot_averaged_DOS(k0range, L, DOSall_TE, file_name, 'sdos', appended_string='_'+str(file_index)+'_TE')
                utils.plot_averaged_DOS(k0range, L, DOSall_TM, file_name, 'sdos', appended_string='_'+str(file_index)+'_TM')

            if compute_eigenmodes:
                
                # Expensive computation
                ngridx = gridsize[0]
                ngridy = gridsize[1]
                xyratio = ngridx/ngridy
                x,y = onp.meshgrid(onp.linspace(0,xyratio,ngridx)  - xyratio/2.0,onp.linspace(0,1,ngridy) - 0.5)
                measurement_points = np.tensor((onp.vstack([x.ravel(),y.ravel()]).T)*L*window_width)

                batches = np.split(measurement_points, batch_size)
                n_batches = len(batches)

                extra_string=""
                if n_batches > 1:
                    extra_string = extra_string+"es"
                print("Computing the eigenfields and plotting the "+str(number_eigenmodes)+" most localized at "+str(gridsize)+" points in "+str(n_batches)+" batch"+extra_string+" of "+str(onp.min([batch_size, ngridx*ngridy])))

                

                solver = Transmission2D_TETM(points, source = None)

                    
                k0_range = []

                for k0, alpha in zip(k0range,alpharange):
                    k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                    k0_range.append(k0_)

                    _, eigenmodes_TM,_ = solver.compute_eigenmodes_IPR( k0, alpha, radius, file_name, write_eigenvalues = True, number_eigenmodes = number_eigenmodes, self_interaction = self_interaction, self_interaction_type = self_interaction_type, sorting_type = sorting_type, scalar = True)
                    _, eigenmodes_TE,_ = solver.compute_eigenmodes_IPR( k0, alpha, radius, file_name, write_eigenvalues = True, number_eigenmodes = number_eigenmodes, self_interaction = self_interaction, self_interaction_type = self_interaction_type, sorting_type = sorting_type, scalar = False)

                    if plot_eigenmodes:
                        
                        for i in range(number_eigenmodes):
                            
                            ETEall = []
                            ETMall = []
                            
                            # By default, the eigenvectors are such that their modulus is 1
                            eigenmodes_TM[:,i] /= np.abs(eigenmodes_TM[:,i]).amax()
                            eigenmodes_TE[:,i] /= np.abs(eigenmodes_TE[:,i]).amax()

                            for batch in range(0, n_batches):
                                print("Batch "+str(batch+1))
                                batch_points = batches[batch]

                                eigenfield_TE, eigenfield_TM = solver.propagate_EM(batch_points, eigenmodes_TE[:,i], eigenmodes_TM[:,i].unsqueeze(-1), k0, alpha, [0.0], w, regularize = regularize, radius=radius)

                                ETEall.append(eigenfield_TE)
                                ETMall.append(eigenfield_TM)

                            ETEall = np.cat(ETEall, dim=0).squeeze(-1)
                            ETMall = np.cat(ETMall, dim=0)

                            ETEall_amplitude    = np.sqrt(np.absolute(ETEall[:,0])**2 + np.absolute(ETEall[:,1])**2)
                            ETEall_amplitude    = ETEall_amplitude.reshape(ngridy, ngridx)
                            ETMall = ETMall.reshape(ngridy, ngridx)
                            
                            plot_IPR_TM = np.sum(np.abs(ETMall**4)) / (np.sum(np.abs(ETMall**2)))**2
                            plot_IPR_TE = np.sum(np.abs(ETEall**4)) / (np.sum(np.abs(ETEall**2)))**2
                            
                            print(f"Effective TM IPR of the whole eigenfield: {plot_IPR_TM}")
                            print(f"Effective TE IPR of the whole eigenfield: {plot_IPR_TE}")

                            utils.plot_full_fields(ETEall_amplitude, ngridx, ngridy, k0_, 0, True, False, False, file_name, appended_string='_width_'+str(window_width)+'_grid_'+str(ngridx)+'x'+str(ngridy)+'_'+str(file_index)+'_TE_eigen_'+sorting_type+str(i), my_dpi = 300)
                            utils.plot_full_fields(ETMall, ngridx, ngridy, k0_, 0, True, False, False, file_name, appended_string='_width_'+str(window_width)+'_grid_'+str(ngridx)+'x'+str(ngridy)+'_'+str(file_index)+'_TM_eigen_'+sorting_type+str(i), my_dpi = 300)


            if compute_DOS:

                solver = Transmission2D_TETM(points, source = source)

                DOSall_TE = []
                DOSall_TM = []
                k0_range = []

                M = dospoints
                measurement_points = utils.uniform_unit_disk_picking(M)
                measurement_points *= L/2

                utils.plot_2d_points(measurement_points, file_name+'_measurement')

                for k0, alpha in zip(k0range,alpharange):
                    dos_TE, dos_TM = solver.mean_DOS_measurements(measurement_points, k0, alpha, radius, regularize = regularize, self_interaction = self_interaction, self_interaction_type = self_interaction_type)

                    DOSall_TE.append(dos_TE.numpy())
                    DOSall_TM.append(dos_TM.numpy())


                    k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                    k0_range.append(k0_)

                    onp.savetxt(file_name+'_temp_dos_TE.csv',onp.stack([k0_range,DOSall_TE]).T)
                    onp.savetxt(file_name+'_temp_dos_TM.csv',onp.stack([k0_range,DOSall_TM]).T)

                onp.savetxt(file_name+'_dos_TE.csv',onp.stack([k0_range,DOSall_TE]).T)
                onp.savetxt(file_name+'_dos_TM.csv',onp.stack([k0_range,DOSall_TM]).T)

                utils.plot_averaged_DOS(k0range, L, DOSall_TE, file_name, 'dos', appended_string='_'+str(file_index)+'_TE')
                utils.plot_averaged_DOS(k0range, L, DOSall_TM, file_name, 'dos', appended_string='_'+str(file_index)+'_TM')

            if compute_interDOS:

                for dos_size in dos_sizes[::-1]:

                    DOSall_TE = onp.array([])
                    DOSall_TM = onp.array([])
                    k0_range = onp.array([])

                    if os.path.exists(file_name+'_temp_idos_size'+str(dos_size)+'_irad'+str(idos_radius)+'_TM.csv'):
                        existing_TM = onp.loadtxt(file_name+'_temp_idos_size'+str(dos_size)+'_irad'+str(idos_radius)+'_TM.csv')
                        DOSall_TM = existing_TM[:,1]
                        k0_range = existing_TM[:,0]
                    if os.path.exists(file_name+'_temp_idos_size'+str(dos_size)+'_irad'+str(idos_radius)+'_TE.csv'):
                        existing_TE = onp.loadtxt(file_name+'_temp_idos_size'+str(dos_size)+'_irad'+str(idos_radius)+'_TE.csv')
                        DOSall_TE = existing_TE[:,1]
                        k0_range = existing_TE[:,0]
                        
                    M = dospoints
                    measurement_points = utils.uniform_unit_disk_picking(M)
                    measurement_points *= dos_size * L/2 * idos_radius
                    disk_points = lattices.cut_circle(points, rad = dos_size * 0.5)

                    solver = Transmission2D_TETM(disk_points, source = source)

                    # Find all overlaps and redraw while you have some
                    # Following Pierrat et al., I use 1 diameter as the spacing there
                    spacing = 2.0*radius
                    spacing *= spacing_factor
                    overlaps = np.nonzero(np.sum(np.cdist(measurement_points.to(np.double), disk_points.to(np.double), p=2) <= spacing, axis = -1)).squeeze()
                    if len(overlaps.shape) == 0:
                        count = 0
                    else:
                        count = overlaps.shape[0]
                    while count > 0:
                        print("Removing "+str(count)+" overlaps using an exclusion distance of "+str(spacing_factor)+" scatterer diameters...")
                        measurement_points[overlaps] = dos_size * L/2 * idos_radius * utils.uniform_unit_disk_picking(count)
                        overlaps = np.nonzero(np.sum(np.cdist(measurement_points.to(np.double), disk_points.to(np.double), p=2) <= spacing, axis = -1)).squeeze()
                        if len(overlaps.shape) == 0:
                            count = 0
                        else:
                            count = overlaps.shape[0]

                    utils.plot_2d_points(measurement_points, file_name+'_measurement_size'+str(dos_size))


                    for k0, alpha in zip(k0range,alpharange):
                        k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                        if k0_ not in k0_range:
                            k0_range = onp.append(k0_range,k0_)
                            dos_TE, dos_TM = solver.mean_DOS_measurements(measurement_points, k0, alpha, radius, regularize = regularize, self_interaction = self_interaction, self_interaction_type = self_interaction_type)

                            DOSall_TE = onp.append(DOSall_TE,dos_TE.numpy())
                            DOSall_TM = onp.append(DOSall_TM,dos_TM.numpy())

                            idx = onp.argsort(k0_range)
                            k0_range = k0_range[idx]
                            DOSall_TE = DOSall_TE[idx]
                            DOSall_TM = DOSall_TM[idx]
                            onp.savetxt(file_name+'_temp_idos_size'+str(dos_size)+'_irad'+str(idos_radius)+'_TE.csv',onp.stack([k0_range,DOSall_TE]).T)
                            onp.savetxt(file_name+'_temp_idos_size'+str(dos_size)+'_irad'+str(idos_radius)+'_TM.csv',onp.stack([k0_range,DOSall_TM]).T)
                    
                    
                    onp.savetxt(file_name+'_idos_size'+str(dos_size)+'_irad'+str(idos_radius)+'_TE.csv',onp.stack([k0_range,DOSall_TE]).T)
                    onp.savetxt(file_name+'_idos_size'+str(dos_size)+'_irad'+str(idos_radius)+'_TM.csv',onp.stack([k0_range,DOSall_TM]).T)

                    utils.plot_averaged_DOS(k0range, L, DOSall_TE, file_name, 'idos', appended_string='_'+str(file_index)+'_size'+str(dos_size)+'_irad'+str(idos_radius)+'_TE')
                    utils.plot_averaged_DOS(k0range, L, DOSall_TM, file_name, 'idos', appended_string='_'+str(file_index)+'_size'+str(dos_size)+'_irad'+str(idos_radius)+'_TM')
                    
            if compute_cavityDOS:

                for dos_size in dos_sizes[::-1]:

                    DOSall_TE = onp.array([])
                    DOSall_TM = onp.array([])
                    k0_range = onp.array([])

                    if os.path.exists(file_name+'_temp_cdos_size'+str(dos_size)+'_TM.csv'):
                        existing_TM = onp.loadtxt(file_name+'_temp_cdos_size'+str(dos_size)+'_TM.csv')
                        DOSall_TM = existing_TM[:,1]
                        k0_range = existing_TM[:,0]
                    if os.path.exists(file_name+'_temp_cdos_size'+str(dos_size)+'_TE.csv'):
                        existing_TE = onp.loadtxt(file_name+'_temp_cdos_size'+str(dos_size)+'_TE.csv')
                        DOSall_TE = existing_TE[:,1]
                        k0_range = existing_TE[:,0]
                        
                    measurement_points = np.zeros(ndim).reshape(1, ndim)
                    disk_points = lattices.cut_circle(points, rad = dos_size * 0.5)
                    
                    # Find all overlaps and remove from system
                    # Following Pierrat et al., I use 1 diameter as the spacing there
                    spacing = 2.0*radius
                    spacing *= spacing_factor
                    disk_points = lattices.exclude_circle(disk_points, spacing)
                    

                    solver = Transmission2D_TETM(disk_points, source = source)


                    utils.plot_2d_points(disk_points, file_name+'_kept_points_size'+str(dos_size))

                    for k0, alpha in zip(k0range,alpharange):
                        k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                        if k0_ not in k0_range:
                            k0_range = onp.append(k0_range,k0_)
                            dos_TE, dos_TM = solver.mean_DOS_measurements(measurement_points, k0, alpha, radius, regularize = regularize, self_interaction = self_interaction, self_interaction_type = self_interaction_type)

                            DOSall_TE = onp.append(DOSall_TE,dos_TE.numpy())
                            DOSall_TM = onp.append(DOSall_TM,dos_TM.numpy())

                            idx = onp.argsort(k0_range)
                            k0_range = k0_range[idx]
                            DOSall_TE = DOSall_TE[idx]
                            DOSall_TM = DOSall_TM[idx]
                            onp.savetxt(file_name+'_temp_cdos_size'+str(dos_size)+'_TE.csv',onp.stack([k0_range,DOSall_TE]).T)
                            onp.savetxt(file_name+'_temp_cdos_size'+str(dos_size)+'_TM.csv',onp.stack([k0_range,DOSall_TM]).T)
                    
                    
                    onp.savetxt(file_name+'_cdos_size'+str(dos_size)+'_TE.csv',onp.stack([k0_range,DOSall_TE]).T)
                    onp.savetxt(file_name+'_cdos_size'+str(dos_size)+'_TM.csv',onp.stack([k0_range,DOSall_TM]).T)

                    utils.plot_averaged_DOS(k0range, L, DOSall_TE, file_name, 'cdos', appended_string='_'+str(file_index)+'_size'+str(dos_size)+'_TE')
                    utils.plot_averaged_DOS(k0range, L, DOSall_TM, file_name, 'cdos', appended_string='_'+str(file_index)+'_size'+str(dos_size)+'_TM')

            if compute_LDOS:

                solver = Transmission2D_TETM(points, source = source)

                # Expensive computation
                ngridx = gridsize[0]
                ngridy = gridsize[1]
                xyratio = ngridx/ngridy
                x,y = onp.meshgrid(onp.linspace(0,xyratio,ngridx)  - xyratio/2.0,onp.linspace(0,1,ngridy) - 0.5)
                measurement_points = np.tensor((onp.vstack([x.ravel(),y.ravel()]).T)*L*window_width)
                
                # Determine which points are within the system
                idx_inside = np.nonzero(np.linalg.norm(measurement_points,axis=-1)<=L/2)

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
                        ldos_TE, ldos_TM = solver.LDOS_measurements(batch_points, k0, alpha, radius, regularize = regularize, self_interaction = self_interaction, self_interaction_type = self_interaction_type)

                        outputs_TE.append(ldos_TE)
                        outputs_TM.append(ldos_TM)

                    #    onp.savetxt(file_name+'_temp_ldos_'+str(k0_)+'_TE.csv',np.cat(outputs_TE).numpy())
                    #    onp.savetxt(file_name+'_temp_ldos_'+str(k0_)+'_TM.csv',np.cat(outputs_TM).numpy())

                    ldos_TE = np.cat(outputs_TE)
                    ldos_TM = np.cat(outputs_TM)
                    
                    ldos_TE = ldos_TE.reshape(ngridy, ngridx)
                    ldos_TM = ldos_TM.reshape(ngridy, ngridx)

                    utils.plot_LDOS_2D(ldos_TE,k0_,ngridx,ngridy,file_name, appended_string='_width_'+str(window_width)+'_grid_'+str(ngridx)+'x'+str(ngridy)+'_'+str(file_index)+'_TE', my_dpi = 300)
                    utils.plot_LDOS_2D(ldos_TM,k0_,ngridx,ngridy,file_name, appended_string='_width_'+str(window_width)+'_grid_'+str(ngridx)+'x'+str(ngridy)+'_'+str(file_index)+'_TM', my_dpi = 300)

                    if write_ldos:
                        onp.savetxt(file_name+'_ldos_'+str(k0_)+'_TE_'+str(index)+'.csv',ldos_TE.numpy())
                        onp.savetxt(file_name+'_ldos_'+str(k0_)+'_TM_'+str(index)+'.csv',ldos_TM.numpy())  


    ### Deal with averaging if several files provided
    n_copies = len(file_index_list)
    if n_copies > 1:
        print("Computing averages across "+str(n_copies)+" configurations")

        if ndim == 2:

            if compute_transmission: 

                # Accumulate data from calculations
                ETE_all = []
                ETM_all = []
                ITE_all = []
                ITM_all = []

                for file_index in file_index_list: 

                    ETE_onecopy, ETM_onecopy, k0range, thetas = hkl.load(file_name+'_transmission_'+str(file_index)+'.hkl')
                    ETE_onecopy = onp.complex128(ETE_onecopy)
                    ETM_onecopy = onp.complex128(ETM_onecopy)
                    thetas = onp.float64(thetas)
                    k0range = onp.float64(k0range)
                    ITE_onecopy = onp.absolute(ETE_onecopy)**2
                    ITE_onecopy = onp.sum(ITE_onecopy, axis=2)

                    ETE_all.append(ETE_onecopy)
                    ETM_all.append(ETM_onecopy)
                    ITE_all.append(ITE_onecopy)
                    ITM_all.append(onp.absolute(ETM_onecopy)**2)

                # Define averaged fields, both amplitude and intensity
                ETE_mean  = onp.mean(ETE_all, axis = 0)
                ETM_mean  = onp.mean(ETM_all, axis = 0)
                ITE_mean  = onp.mean(ITE_all, axis = 0)
                ITM_mean  = onp.mean(ITM_all, axis = 0)
                # Define the ballistic intensity
                ITE_ball = onp.absolute(ETE_mean)**2
                ITE_ball = onp.sum(ITE_ball, axis = 2)
                ITM_ball = onp.absolute(ETM_mean)**2
                # Also define the average fluctuating intensity field
                ITE_fluct = ITE_mean - ITE_ball
                ITM_fluct = ITM_mean - ITM_ball

                # Produce transmission normalized by total intensity of the INCIDENT FIELD on the sphere
                if just_compute_averages:

                    solver = Transmission2D_TETM(points, source = source)

                    # Define the list of measurement points for transmission plots
                    measurement_points = transmission_radius*L*onp.vstack([onp.cos(thetas),onp.sin(thetas)]).T
                    E0TEall = []
                    E0TMall = []
                    for k0 in k0range:
                        E0TE, E0TM = solver.generate_source(np.tensor(measurement_points), k0, thetas, beam_waist, print_statement='scattered_fields')
                        E0TEall.append(E0TE.numpy())
                        E0TMall.append(E0TM.numpy())

                    I0TMall = onp.absolute(E0TMall)**2
                    I0TEall = onp.absolute(E0TEall)**2
                    I0TEall = onp.sum(I0TEall, axis = 2)

                # If required: plot results
                if plot_transmission:
                    # Produce plots for average intensity
                    utils.plot_transmission_angularbeam(k0range, L, thetas, ITM_mean, file_name, n_thetas_trans = n_thetas_trans, appended_string='_averageintensity_'+str(n_copies)+'copies_TM', adapt_scale = adapt_scale)
                    utils.plot_transmission_angularbeam(k0range, L, thetas, ITE_mean, file_name, n_thetas_trans = n_thetas_trans, appended_string='_averageintensity_'+str(n_copies)+'copies_TE', adapt_scale = adapt_scale)
                    utils.plot_transmission_flat(k0range, L, thetas, ITM_mean, file_name, n_thetas_trans = n_thetas_trans, appended_string='_averageintensity_'+str(n_copies)+'copies_TM', adapt_scale = adapt_scale)
                    utils.plot_transmission_flat(k0range, L, thetas, ITE_mean, file_name, n_thetas_trans = n_thetas_trans, appended_string='_averageintensity_'+str(n_copies)+'copies_TE', adapt_scale = adapt_scale)

                    # Produce plots for normalized average intensity
                    utils.plot_transmission_angularbeam(k0range, L, thetas, ITM_mean, file_name, n_thetas_trans = n_thetas_trans, normalization=I0TMall, appended_string='_averageintensity_'+str(n_copies)+'copies_TM_incnorm', adapt_scale = adapt_scale)
                    utils.plot_transmission_angularbeam(k0range, L, thetas, ITE_mean, file_name, n_thetas_trans = n_thetas_trans, normalization=I0TEall, appended_string='_averageintensity_'+str(n_copies)+'copies_TE_incnorm', adapt_scale = adapt_scale)
                    utils.plot_transmission_flat(k0range, L, thetas, ITM_mean, file_name, n_thetas_trans = n_thetas_trans, normalization=I0TMall, appended_string='_averageintensity_'+str(n_copies)+'copies_TM_incnorm', adapt_scale = adapt_scale)
                    utils.plot_transmission_flat(k0range, L, thetas, ITE_mean, file_name, n_thetas_trans = n_thetas_trans, normalization=I0TEall, appended_string='_averageintensity_'+str(n_copies)+'copies_TE_incnorm', adapt_scale = adapt_scale)

                    # Produce plots for intensity of the average field = ballistic intensity
                    utils.plot_transmission_angularbeam(k0range, L, thetas, ITM_ball, file_name, n_thetas_trans = n_thetas_trans, appended_string='_ballisticintensity_'+str(n_copies)+'copies_TM', adapt_scale = adapt_scale)
                    utils.plot_transmission_angularbeam(k0range, L, thetas, ITE_ball, file_name, n_thetas_trans = n_thetas_trans, appended_string='_ballisticintensity_'+str(n_copies)+'copies_TE', adapt_scale = adapt_scale)
                    utils.plot_transmission_flat(k0range, L, thetas, ITM_ball, file_name, n_thetas_trans = n_thetas_trans, appended_string='_ballisticintensity_'+str(n_copies)+'copies_TM', adapt_scale = adapt_scale)
                    utils.plot_transmission_flat(k0range, L, thetas, ITE_ball, file_name, n_thetas_trans = n_thetas_trans, appended_string='_ballisticintensity_'+str(n_copies)+'copies_TE', adapt_scale = adapt_scale)

                    # Produce plots for NORMALIZED intensity of the average field = ballistic intensity
                    utils.plot_transmission_angularbeam(k0range, L, thetas, ITM_ball, file_name, n_thetas_trans = n_thetas_trans, normalization=I0TMall, appended_string='_ballisticintensity_'+str(n_copies)+'copies_TM_incnorm', adapt_scale = adapt_scale)
                    utils.plot_transmission_angularbeam(k0range, L, thetas, ITE_ball, file_name, n_thetas_trans = n_thetas_trans, normalization=I0TEall, appended_string='_ballisticintensity_'+str(n_copies)+'copies_TE_incnorm', adapt_scale = adapt_scale)
                    utils.plot_transmission_flat(k0range, L, thetas, ITM_ball, file_name, n_thetas_trans = n_thetas_trans, normalization=I0TMall, appended_string='_ballisticintensity_'+str(n_copies)+'copies_TM_incnorm', adapt_scale = adapt_scale)
                    utils.plot_transmission_flat(k0range, L, thetas, ITE_ball, file_name, n_thetas_trans = n_thetas_trans, normalization=I0TEall, appended_string='_ballisticintensity_'+str(n_copies)+'copies_TE_incnorm', adapt_scale = adapt_scale)

                    # Produce plots for intensity of the fluctuating field
                    utils.plot_transmission_angularbeam(k0range, L, thetas, ITM_fluct, file_name, n_thetas_trans = n_thetas_trans, appended_string='_fluctuatingintensity_'+str(n_copies)+'copies_TM', adapt_scale = adapt_scale)
                    utils.plot_transmission_angularbeam(k0range, L, thetas, ITE_fluct, file_name, n_thetas_trans = n_thetas_trans, appended_string='_fluctuatingintensity_'+str(n_copies)+'copies_TE', adapt_scale = adapt_scale)
                    utils.plot_transmission_flat(k0range, L, thetas, ITM_fluct, file_name, n_thetas_trans = n_thetas_trans, appended_string='_fluctuatingintensity_'+str(n_copies)+'copies_TM', adapt_scale = adapt_scale)
                    utils.plot_transmission_flat(k0range, L, thetas, ITE_fluct, file_name, n_thetas_trans = n_thetas_trans, appended_string='_fluctuatingintensity_'+str(n_copies)+'copies_TE', adapt_scale = adapt_scale)

        elif ndim == 3:
            
            if compute_transmission:
                # Accumulate data from calculations
                E_all = []
                I_all = []
                
                
                for file_index in file_index_list: 

                    E_onecopy, k0range, thetas = hkl.load(file_name+'_transmission_'+str(file_index)+'.hkl')
                    E_onecopy = onp.complex128(E_onecopy)
                    thetas = onp.float64(thetas)
                    k0range = onp.float64(k0range)

                    E_all.append(E_onecopy)
                    I_all.append(onp.absolute(E_onecopy)**2)

                # Define averaged fields, both amplitude and intensity
                E_mean  = onp.mean(E_all, axis = 0)
                I_mean  = onp.mean(I_all, axis = 0)
                # Define the ballistic intensity
                I_ball = onp.absolute(E_mean)**2
                # Also define the average fluctuating intensity field
                I_fluct = I_mean - I_ball

                # If required: plot results
                if plot_transmission:
                    # Produce plots for average intensity
                    utils.plot_transmission_angularbeam(k0range, L, thetas, I_mean, file_name, appended_string='_averageintensity_'+str(n_copies)+'copies', adapt_scale = adapt_scale)
                    utils.plot_transmission_flat(k0range, L, thetas, I_mean, file_name, appended_string='_averageintensity_'+str(n_copies)+'copies', adapt_scale = adapt_scale)

                    # Produce plots for intensity of the average field = ballistic intensity
                    utils.plot_transmission_angularbeam(k0range, L, thetas, I_ball, file_name, appended_string='_ballisticintensity_'+str(n_copies)+'copies', adapt_scale = adapt_scale)
                    utils.plot_transmission_flat(k0range, L, thetas, I_ball, file_name, appended_string='_ballisticintensity_'+str(n_copies)+'copies', adapt_scale = adapt_scale)

                    # Produce plots for intensity of the fluctuating field
                    utils.plot_transmission_angularbeam(k0range, L, thetas, I_fluct, file_name, appended_string='_fluctuatingintensity_'+str(n_copies)+'copies', adapt_scale = adapt_scale)
                    utils.plot_transmission_flat(k0range, L, thetas, I_fluct, file_name, appended_string='_fluctuatingintensity_'+str(n_copies)+'copies', adapt_scale = adapt_scale)

def make_lattice(lattice, N_raw, kick, ndim):

    if ndim==2:
        if lattice == 'square':
            Nside = int(onp.round(onp.sqrt(N_raw)))
            if Nside%2==0:
                Nside += 1
            points = lattices.square(Nside=Nside, disp=kick)
        elif lattice == 'triangular':
            Nx = int(onp.round(onp.sqrt(N_raw / onp.sqrt(3.0))))
            Ny = int(onp.round(onp.sqrt(3.0) * Nx))
            if Nx%2==0:
                Nx += 1
            if Ny%2 == 0:
                Ny += 1
            points = lattices.triangular(Nx=Nx, Ny=Ny, disp=kick)
        elif lattice == 'honeycomb':
            Nx = int(onp.round(onp.sqrt(N_raw / onp.sqrt(3.0))))
            Ny = int(onp.round(onp.sqrt(3.0) * Nx))
            if Nx%2==0:
                Nx += 1
            if Ny%2 == 0:
                Ny += 1
            points = lattices.honeycomb(Nx=Nx, Ny=Ny, disp=kick)
        elif lattice == 'vogel':
            points = lattices.vogel_spiral(N_raw)
        elif lattice.split("_")[0] == 'quasicrystal':
            if len(lattice.split("_")) == 1:
                qc_symmetry = 5
            else:
                qc_symmetry = int(lattice.split("_")[1])
            points = lattices.quasicrystal(N=N_raw, mode='quasicrystal', disp=kick, ndirs = qc_symmetry)
        elif lattice.split("_")[0] == 'quasivoro':
            if len(lattice.split("_")) == 1:
                qc_symmetry = 5
            else:
                qc_symmetry = int(lattice.split("_")[1])
            points = lattices.quasicrystal(N=N_raw, mode='quasivoro', disp=kick, ndirs = qc_symmetry)
        elif lattice.split("_")[0] == 'quasidual':
            if len(lattice.split("_")) == 1:
                qc_symmetry = 5
            else:
                qc_symmetry = int(lattice.split("_")[1])
            points = lattices.quasicrystal(N=N_raw, mode='quasidual', disp=kick, ndirs = qc_symmetry)
        elif lattice.split("_")[0] == 'quasideBruijn':
            if len(lattice.split("_")) == 1:
                qc_symmetry = 5
            else:
                qc_symmetry = int(lattice.split("_")[1])
            points = lattices.quasicrystal(N=N_raw, mode='deBruijndual', disp=kick, ndirs = qc_symmetry)
        elif lattice == 'poisson':
            points = lattices.poisson(N_raw, ndim)
        else:
            print("Not a valid lattice!")
            exit()

    elif ndim == 3:
        if lattice == 'cubic':
            Nside  = int(onp.round(onp.cbrt(N_raw)))
            points = lattices.cubic(Nside=Nside, disp=kick)
        elif lattice == 'bcc':
            # bcc has two atoms per unit cell
            Nside  = int(onp.round(onp.cbrt(N_raw/2)))
            points = lattices.bcc(Nside=Nside, disp=kick)
        elif lattice == 'fcc':
            # fcc has four atoms per unit cell
            Nside  = int(onp.round(onp.cbrt(N_raw/4)))
            points = lattices.fcc(Nside=Nside, disp=kick)
        elif lattice == 'diamond':
            # diamond has two atoms per unit cell
            Nside  = int(onp.round(onp.cbrt(N_raw/2)))
            points = lattices.diamond(Nside=Nside, disp=kick)
        elif lattice == 'simple_hexagonal':
            Nx = int(onp.round(onp.cbrt(N_raw / onp.sqrt(3.0))))
            Ny = int(onp.round(onp.sqrt(3.0) * Nx))
            if Nx%2==0:
                Nx += 1
            if Ny%2 == 0:
                Ny += 1
            points = lattices.simple_hexagonal(Nx=Nx, Ny=Ny,Nz=Ny, disp=kick)
        elif lattice == 'poisson':
            points = lattices.poisson(N_raw, ndim)
        else: 
            print("Not a valid lattice!")
            exit()
    else:
        print("Not a valid dimensionality!")
        exit()

    return points


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run a full solving and plotting routine of MAGreeTe")
    # Required arguments
    parser.add_argument("ndim", type=int, help="Dimensionality of space")
    parser.add_argument("--n_cpus", type=int, help="Number of cpus to use for computation\
        default = os.cpu_count", default=os.cpu_count())
    # Physical quantities
    parser.add_argument("-n", "--refractive_n", type=complex, help="Complex refractive index of the dielectric material \
        default = 1.65 + 0.025j", default = 1.65 + 0.025j)
    parser.add_argument("--cold_atoms", action='store_true', help="Use a Lorentz model of the electron as a polarizability \
        default = False", default = False)
    parser.add_argument("--kres", type = float, help = "Value of the bare resonance frequency in the Lorentz polarizability \
        default = 0.1 * L / radius", default = None)
    parser.add_argument("--phi", type=float, help="Volume fraction of scatterers within the medium \
        default = 0.1", default = 0.1)  
    parser.add_argument("-r", "--regularize", action='store_true', help="Regularize the fields and DOS inside of scatterers\
        default=False", default=False)
    parser.add_argument("-N", "--number_particles", type = int, help="Number of particles in the system, before cutting a circle\
        default = 16384", default=16384)
    parser.add_argument("-s", "--source", type = str, help="Type of source to use\
        Options: beam, plane, point. Default: beam", default = "beam")
    parser.add_argument("-bw", "--beam_waist", type = float, help="Waist of the beam used for transmission plots and full fields, in units of L\
        default = 0.2", default=0.2)
    parser.add_argument("--boxsize", type=float, help="Set physical units for the box size: the results are dimensionless so that default=1", default = 1)
    parser.add_argument("-sss", "--size_subsample", type = float, help = "Fraction of the initial system sidelength to keep if only a subsample is necessary,\
        default = 1.0 (largest inscribed disk)", default = 1.0)
    parser.add_argument("--scalar", action = 'store_true', help = "Use scalar waves\
        default = false", default = False)
    # Ranges of wave-vectors and beam orientations, index of copy to look at
    parser.add_argument("-k", "--k0range", nargs='+', type=float, help = "Values of k0 to span, in units of 2pi/L. Can be a single-value argument, a k_min and a k_max (with default step 1), or k_min, k_max, and step\
        default=(1,0.25 * L/scatterer_radius,0.5)*2pi/L ", default=None)
    parser.add_argument("-t","--thetas",  nargs = "+", type = float, help = "Angles to consider, in degrees. Can be a single-value argument, a theta_min and a theta_max (with default step 1), or theta_min, theta_max, and step\
        default=(0,359,1)", default = None)
    # Special systems
    parser.add_argument("-i", "--input_files", nargs='+', type=str, help="Name of hkl files containing points. May contain several, that will be averaged over. \
        default=None", default=None)
    parser.add_argument("-l", "--lattice", type=str, help="Use a simple lattice in lieu of datapoints as entry. \
        Options are 'square', 'triangular', 'honeycomb', 'vogel', 'quasicrystal', 'quasidual', 'quasivoro', 'quasideBruijn', 'poisson' in 2d, and 'cubic', 'fcc', 'bcc', 'diamond', 'poisson' in 3d. \
        default=None", default=None)
    parser.add_argument("-a", "--annulus", type=float, help="radius of circular removal of points \
        default=0", default=0)
    parser.add_argument("-c","--composite", action='store_true', help="Whether to fill annulus vacancy with square lattice\
        default=False", default=False)
    parser.add_argument("--kick", type=float, help="Value of max amplitude of randomly oriented, random uniform length small kicks to add to all positions, in units of L\
        default = 0", default = 0.0)
    parser.add_argument("--shift", type = float, help ="Shifts the positions of the whole system by one random vector of the specified modulus\
        default = 0", default = 0.0)
    # Computation type arguments
    parser.add_argument("--compute_transmission", action='store_true', help="Compute transmission for laser beams\
        default = False", default=False)
    parser.add_argument("--plot_transmission", action='store_true', help="Produce transmission plots\
        default=False", default=False)
    parser.add_argument("-ss","--single_scattering_transmission", action="store_true", help="Produce transmission plots using a single-scattering approximation of the Lippman-Schwinger equation\
        default=False", default=False)
    parser.add_argument("--scattered_fields", action="store_true", help="Plot scattered fields instead of total fields wherever applicable \
        default=False", default=False)
    parser.add_argument("-trad","--transmission_radius", type = float, help = "Radius of the sphere on which transmission measurements are performed, in units of L,\
        default=0.51", default = 0.51)
    parser.add_argument("-dos","--compute_DOS", action='store_true', help="Compute the mean DOS of the medium  \
        default=False", default=False)
    parser.add_argument("-cdos","--compute_cavityDOS", action='store_true', help="Compute the DOS at the center of the medium, removing nearby points if any \
        default=False", default=False)
    parser.add_argument("-idos","--compute_interDOS", action='store_true', help="Compute the mean DOS of the medium away from scatterers  \
        default=False", default=False)
    parser.add_argument("-sdos","--compute_SDOS", action='store_true', help="Compute the spectrum of the Green's matrix, as well as the mean DOS at scatterers  \
        default=False", default=False)
    parser.add_argument("-ldos","--compute_LDOS", action='store_true', help="Compute an LDOS map  \
        default=False", default=False)
    parser.add_argument("-ds", "--dos_sizes_args", nargs = "+", type = float, help = "System linear sizes to consider, as fractions of L\
        default=1", default = None)
    parser.add_argument("-em", "--compute_eigenmodes", action='store_true', help="Compute the eigenmodes of the linear system used to solve coupled dipoles, and saves eigenvalues, IPR, and some eigenfields\
        default = False", default=False)
    parser.add_argument("-nem","--number_eigenmodes", type = int, help = "Number of eigenmodes to save on both ends of the IPR extremes\
        default = 1", default = 1)
    parser.add_argument("-pem", "--plot_eigenmodes", action='store_true', help = "Whether to plot the eigenmodes that were computed\
        default = false", default = False)
    parser.add_argument("-sort", "--sorting_type", type = str, help = "Sorting used to choose plotted eigenmodes\
        Options: IPR (largest) or damping (smallest), Default = IPR", default = "IPR")
    parser.add_argument("--intensity_fields", action = "store_true", help="Output images of intensity fields for every beam used in the angular plot, in real space\
        default = False", default=False)
    parser.add_argument("--amplitude_fields", action = "store_true", help="Output images of amplitude fields for every beam used in the angular plot, in real space\
        default = False", default=False)
    parser.add_argument("--phase_fields", action = "store_true", help="Output images of phase fields for every beam used in the angular plot, in real space\
        default = False", default=False)
    parser.add_argument("--just_averages", action = "store_true", help="Only compute average quantities from existing outputs\
        default = False", default=False)
    # Parameters of outputs
    parser.add_argument("--dospoints",type=int, help="Number of points to use for the mean DOS computation \
        default = 1000", default=1000)
    parser.add_argument("-sf","--spacing_factor", type=float, help="Number of diameters to use as excluded volume around measurement points for idos\
        default = 1.0", default = 1.0)
    parser.add_argument("-irad", "--idos_radius", type=float, help="Fraction of system radius within which to sample idos measurement points\
        default = 1.0", default = 1.0)
    parser.add_argument("-ev","--write_eigenvalues", action='store_false', help="Write the eigenvalues of the Green's matrix at every frequency  \
        default=True", default=False)
    parser.add_argument("--write_ldos", action="store_true", help="Save all computed LDOS outputs. Warning: this can grow pretty big.\
        default = False", default = False)
    parser.add_argument("-g","--gridsize",nargs=2,type=int, help="Number of pixels to use in the sidelength of output images \
        default = (301,301)", default=(301,301))
    parser.add_argument("-w","--window_width", type=float, help="Width of the viewfield for real-space plots, in units of system diameters, \
        default = 1.2", default = 1.2)
    parser.add_argument("-b", "--batch_size", type = int, help = "Batch size (in number of pixels/batch) used to compute full field images, \
        default = 101x101", default = 101*101)
    parser.add_argument("-aw", "--angular_width", type = float, help="Angular width used in the definition of transmission, as a fraction of half the number of used angles: 0 is a single-point and 1 is the full half-space. Warning: this only uses angles defined in the list of computed angles!\
        Default = 1", default = 1.0)
    parser.add_argument("--plot_theta_index", type = int, help="Index of special theta to use for some plots\
        default = 0", default = 0)
    parser.add_argument("--adapt_scale", action='store_true', help="Whether to adapt intensity scales in transmission plots to actual value, otherwise snapped to 1e-3 to 1e0\
        default = false0", default = False)
    parser.add_argument("-o", "--output", type=str, help="Output directory\
        default = ./refractive_n_$Value/", default='')

    args = parser.parse_args()

    # Required arguments
    ndim                            = args.ndim
    n_cpus                          = args.n_cpus
    # Physical quantities
    refractive_n                    = args.refractive_n
    phi                             = args.phi
    regularize                      = args.regularize
    N                               = args.number_particles
    source                          = args.source
    beam_waist                      = args.beam_waist
    boxsize                         = args.boxsize
    size_subsample                  = args.size_subsample
    scalar                          = args.scalar
    # Ranges of wave-vectors and beam orientations, index of copy for source points
    k0range_args                    = args.k0range
    if k0range_args     != None:
        k0range_args                = tuple(k0range_args)
    thetarange_args = args.thetas
    if thetarange_args     != None:
        thetarange_args             = tuple(thetarange_args)
    input_files_args                = args.input_files
    if input_files_args     != None:
        input_files_args            = tuple(input_files_args)
    # Special cases
    cold_atoms                      = args.cold_atoms
    kresonant_                      = args.kres
    lattice                         = args.lattice
    annulus                         = args.annulus
    composite                       = args.composite
    kick                            = args.kick
    shift                           = args.shift
    # Outputs
    compute_transmission            = args.compute_transmission
    plot_transmission               = args.plot_transmission
    single_scattering_transmission  = args.single_scattering_transmission
    scattered_fields                = args.scattered_fields
    transmission_radius             = args.transmission_radius
    compute_DOS                     = args.compute_DOS
    compute_cavityDOS               = args.compute_cavityDOS
    compute_interDOS                = args.compute_interDOS
    compute_SDOS                    = args.compute_SDOS
    compute_LDOS                    = args.compute_LDOS
    dos_sizes_args                       = args.dos_sizes_args
    if dos_sizes_args     != None:
        dos_sizes_args                   = tuple(dos_sizes_args)
    compute_eigenmodes              = args.compute_eigenmodes
    number_eigenmodes               = args.number_eigenmodes
    plot_eigenmodes                 = args.plot_eigenmodes
    sorting_type                    = args.sorting_type
    intensity_fields                = args.intensity_fields
    amplitude_fields                = args.amplitude_fields
    phase_fields                    = args.phase_fields
    just_compute_averages           = args.just_averages
    # Options for outputs
    dospoints                       = args.dospoints
    spacing_factor                  = args.spacing_factor
    idos_radius                     = args.idos_radius
    write_eigenvalues               = args.write_eigenvalues
    write_ldos                      = args.write_ldos
    gridsize                        = tuple(args.gridsize)
    window_width                    = args.window_width
    batch_size                      = args.batch_size
    angular_width                   = args.angular_width
    plot_theta_index                = args.plot_theta_index
    adapt_scale                     = args.adapt_scale
    output_directory                = args.output

    np.set_num_threads(n_cpus)
    np.device("cpu")
    
        
    main(ndim,
        refractive_n = refractive_n,  phi=phi, regularize=regularize, N_raw=N, source = source, beam_waist=beam_waist, L=boxsize, size_subsample=size_subsample, scalar=scalar,
        k0range_args = k0range_args, thetarange_args=thetarange_args, input_files_args = input_files_args,
        cold_atoms=cold_atoms, kresonant_ = kresonant_, lattice=lattice, annulus = annulus, composite = composite, kick = kick, shift = shift,
        compute_transmission = compute_transmission, plot_transmission=plot_transmission, single_scattering_transmission=single_scattering_transmission, scattered_fields=scattered_fields, transmission_radius=transmission_radius,
        compute_DOS=compute_DOS, compute_cavityDOS = compute_cavityDOS, compute_interDOS=compute_interDOS, compute_SDOS=compute_SDOS, compute_LDOS=compute_LDOS, dos_sizes_args= dos_sizes_args, 
        compute_eigenmodes = compute_eigenmodes, number_eigenmodes = number_eigenmodes, plot_eigenmodes = plot_eigenmodes, sorting_type = sorting_type,
        intensity_fields = intensity_fields, amplitude_fields=amplitude_fields, phase_fields=phase_fields, just_compute_averages=just_compute_averages,
        dospoints=dospoints, spacing_factor=spacing_factor, idos_radius=idos_radius, write_eigenvalues=write_eigenvalues, write_ldos=write_ldos, gridsize=gridsize, window_width=window_width, batch_size = batch_size, angular_width=angular_width, plot_theta_index=plot_theta_index, adapt_scale = adapt_scale,
        output_directory=output_directory
        )
    sys.exit()


