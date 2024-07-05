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
from Transmission2D import Transmission2D_scalar, Transmission2D_scalar_hmatrices
from Transmission3D import Transmission3D_scalar, Transmission3D_scalar_hmatrices
import lattices
from magreete import make_lattice


import argparse


def main(ndim, # Required arguments
        refractive_n = 1.65 + 0.025j, phi = 0.1, regularize = True, N_raw = 16384, beam_waist = 0.2, L = 1, size_subsample = 1.0, source = "beam", # Physical parameters
        lattice=None, cold_atoms=False, kresonant_ = None, annulus = 0, composite = False, kick = 0.0, input_files_args = None, method = "torch", # Special cases
        k0range_args = None, thetarange_args = None,# Range of values to use
        compute_transmission = False, plot_transmission = False, single_scattering_transmission = False, scattered_fields=False, transmission_radius = 2.0,
        compute_DOS=False, compute_interDOS=False, compute_SDOS=False, compute_LDOS=False, dos_sizes_args = None, dospoints=1, spacing_factor = 1.0, idos_radius = 1.0, 
        compute_eigenmodes = False, number_eigenmodes = 1, plot_eigenmodes = False, sorting_type = 'IPR',
        intensity_fields = False, amplitude_fields = False, phase_fields = False, just_compute_averages = False,# Computations to perform
        write_eigenvalues=False, write_ldos= False,  gridsize=(301,301), window_width=1.2, angular_width = 0.0, plot_theta_index = 0, batch_size = 101*101, adapt_scale = False, output_directory="" # Parameters for outputs
        ):
    '''
    Simple front-end for MAGreeTe with scalar waves
    '''
    
    # Keep cut_radius as the internal here
    cut_radius = 0.5 * size_subsample
    
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
            points = hkl.load(file_name)
            points = np.tensor(points[:,0:ndim],dtype=np.double)
            shape_before = points.shape
            
            # Make output dir
            N_raw = shape_before[0]
            output_directory += "N"+str(N_raw)+"/"
            output_directory += output_directory_suffix
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

        # A generative recipe was selected
        else:

            file_name = lattice
            output_directory += "N"+str(N_raw)+"/"
            output_directory += output_directory_suffix
            utils.trymakedir(output_directory)
            points = make_lattice(lattice, N_raw, kick, ndim)
            points = lattices.cut_circle(points)

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

        output_directory = output_directory+"/"+file_name
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
        utils.plot_dressed_polarizability(k0range, L, alpharange, ndim, radius, volume, self_interaction, file_name, scalar = True, self_interaction_type = self_interaction_type)

        ### ###############
        ### 2d calculations
        ### ###############
        if ndim==2:
            # Transmission plot computations
            if compute_transmission or plot_transmission:

                # Define the list of measurement points for transmission plots
                measurement_points = transmission_radius*L*onp.vstack([onp.cos(thetas),onp.sin(thetas)]).T

                
                # A fresh computation is required
                if compute_transmission: 
                    if method == "torch":
                        solver = Transmission2D_scalar(points, source = source)
                    elif method == "hmatrices":
                        solver = Transmission2D_scalar_hmatrices(points, source = source)
                    else:
                        print("Choose a valid method")
                        sys.exit()
                    Eall = []
                    E0all = []
                    Eall_scat = []
                    
                    for k0, alpha in zip(k0range,alpharange):
                        Ej = solver.solve(k0, alpha, thetas, radius, w, self_interaction = self_interaction, self_interaction_type = self_interaction_type)
                        k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                        params = [alpha, k0]
                        hkl.dump([onp.array(Ej), onp.array(params),onp.array(points), onp.array(thetas)],file_name+'_Ek_k0_'+str(k0_)+'_'+str(file_index)+'.hkl')

                        Ek = solver.propagate(measurement_points, Ej, k0, alpha, thetas, w, regularize = regularize, radius = radius)
                        
                        E0 = solver.generate_source(np.tensor(measurement_points), k0, thetas, beam_waist, print_statement = 'scattered_fields')

                        if scattered_fields:
                            Ek_scat = Ek - E0
                            Eall_scat.append(Ek_scat.numpy())

                        E0all.append(E0.numpy())
                        
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
                        if method == "torch":
                            solver = Transmission2D_scalar(points, source = source)
                        elif method == "hmatrices":
                            solver = Transmission2D_scalar_hmatrices(points, source = source)
                        else:
                            print("Choose a valid method")
                            sys.exit()

                        Ek = solver.propagate(measurement_points, Ej, k0, alpha, thetas, w, regularize = regularize, radius = radius)
                        E0 = solver.generate_source(np.tensor(measurement_points), k0, thetas, beam_waist, print_statement = 'scattered_fields')
                        
                        if scattered_fields:
                            Ek_scat = Ek - E0
                            Eall_scat.append(Ek_scat.numpy())
                        
                        E0all.append(E0.numpy())
                        Eall.append(Ek.numpy())

                hkl.dump([onp.array(Eall), onp.array(k0range), onp.array(thetas)],file_name+'_transmission_'+str(file_index)+'.hkl')

                # If required: plot results
                if plot_transmission:
                    # Compute intensities at measurement points
                    Eall = onp.array(Eall)
                    total = onp.absolute(Eall)**2

                    # Produce plots
                    utils.plot_transmission_angularbeam(k0range, L, thetas, total, file_name,  n_thetas_trans = n_thetas_trans, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index))
                    utils.plot_transmission_flat(k0range, L, thetas, total, file_name,  n_thetas_trans = n_thetas_trans, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index))
                    utils.plot_angular_averaged_transmission(k0range, L, total, file_name, appended_string='_'+str(file_index))
                    plot_theta = onp.round(180 * thetas[plot_theta_index]/onp.pi)
                    utils.plot_singlebeam_angular_frequency_plot(k0range, L, thetas, total, file_name, plot_theta_index = plot_theta_index, appended_string='_'+str(file_index)+'_angle_'+str(plot_theta))
                    
                    # Produce transmission normalized by total intensity of the INCIDENT FIELD on the sphere
                    I0all = onp.absolute(E0all)**2

                    utils.plot_transmission_angularbeam(k0range, L, thetas, total, file_name,  n_thetas_trans = n_thetas_trans, normalization = I0all, adapt_scale = adapt_scale, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_incnorm')
                    utils.plot_transmission_flat(k0range, L, thetas, total, file_name,  n_thetas_trans = n_thetas_trans, normalization = I0all, adapt_scale = adapt_scale, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_incnorm')

                    # Same but with total field
                    utils.plot_transmission_angularbeam(k0range, L, thetas, total, file_name,  n_thetas_trans = n_thetas_trans, normalization = total, adapt_scale = adapt_scale, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_norm')
                    utils.plot_transmission_flat(k0range, L, thetas, total, file_name,  n_thetas_trans = n_thetas_trans, normalization = total, adapt_scale = adapt_scale, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_norm')



                    if scattered_fields:
                        # Compute scattered intensities at measurement points
                        Eall_scat = onp.array(Eall_scat)
                        total_scat = onp.absolute(Eall_scat)**2
                        
                        # Produce plots
                        utils.plot_transmission_angularbeam(k0range, L, thetas, total_scat, file_name,  n_thetas_trans = n_thetas_trans, adapt_scale = adapt_scale, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_scat')
                        utils.plot_transmission_flat(k0range, L, thetas, total_scat, file_name,  n_thetas_trans = n_thetas_trans, adapt_scale = adapt_scale, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_scat')
                        utils.plot_angular_averaged_transmission(k0range, L, total_scat, file_name, appended_string='_'+str(file_index)+'_scat')
                        plot_theta = onp.round(180 * thetas[plot_theta_index]/onp.pi)
                        utils.plot_singlebeam_angular_frequency_plot(k0range, L, thetas, total_scat, file_name, plot_theta_index = plot_theta_index, appended_string='_'+str(file_index)+'_angle_'+str(plot_theta)+'_scat')
                 
                        # Also produce scattered field normalised by total scattered intensity
                        utils.plot_transmission_angularbeam(k0range, L, thetas, total_scat, file_name,  n_thetas_trans = n_thetas_trans, adapt_scale = adapt_scale, normalization = total_scat, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_scat_norm')
                        

                            
            # Single-scattering transmission
            if single_scattering_transmission:
                # Define the list of measurement points for transmission plots
                measurement_points = transmission_radius*L*onp.vstack([onp.cos(thetas),onp.sin(thetas)]).T
                if method == "torch":
                    solver = Transmission2D_scalar(points, source = source)
                elif method == "hmatrices":
                    solver = Transmission2D_scalar_hmatrices(points, source = source)
                else:
                    print("Choose a valid method")
                    sys.exit()
                    
                Eall_ss = []
                Eall_scat_ss = []
                
                for k0, alpha in zip(k0range,alpharange):
                    
                    Ek_ss = solver.propagate_ss(measurement_points, k0, alpha, thetas, w, regularize = regularize, radius = radius)
                    
                    if scattered_fields:
                        E0 = solver.generate_source(np.tensor(measurement_points), k0, thetas, beam_waist, print_statement = 'scattered_fields')
                        Ek_scat_ss = Ek_ss - E0
                        Eall_scat_ss.append(Ek_scat_ss.numpy())
                    
                    Eall_ss.append(Ek_ss.numpy())
                    
                # Compute intensities at measurement points
                Eall_ss = onp.array(Eall_ss)
                total_ss = onp.absolute(Eall_ss)**2
                
                # Produce plots
                utils.plot_transmission_angularbeam(k0range, L, thetas, total_ss, file_name,  n_thetas_trans = n_thetas_trans, adapt_scale = adapt_scale, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_ss')
                utils.plot_transmission_flat(k0range, L, thetas, total_ss, file_name,  n_thetas_trans = n_thetas_trans, adapt_scale = adapt_scale, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_ss')
                utils.plot_angular_averaged_transmission(k0range, L, total_ss, file_name, appended_string='_'+str(file_index)+'_ss')
                theta_plot = onp.round(180 * thetas[plot_theta_index]/onp.pi)
                utils.plot_singlebeam_angular_frequency_plot(k0range, L, thetas, total_ss, file_name, plot_theta_index = plot_theta_index, appended_string='_'+str(file_index)+'_TM_angle_'+str(theta_plot)+'_ss')
                
                if plot_transmission:
                    # Also compute the intensity associated to the multiple-scattering contribution of the field, if the full field was computed
                    Eall_multiple = Eall - Eall_ss
                    total_multiple = onp.absolute(Eall_multiple)**2

                    # Produce plots
                    utils.plot_transmission_angularbeam(k0range, L, thetas, total_multiple, file_name,  n_thetas_trans = n_thetas_trans, adapt_scale = adapt_scale, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_multiple')
                    utils.plot_transmission_flat(k0range, L, thetas, total_multiple, file_name,  n_thetas_trans = n_thetas_trans, adapt_scale = adapt_scale, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_multiple')
                    utils.plot_angular_averaged_transmission(k0range, L, total_multiple, file_name, appended_string='_'+str(file_index)+'_multiple')
                    theta_plot = onp.round(180 * thetas[plot_theta_index]/onp.pi)
                    utils.plot_singlebeam_angular_frequency_plot(k0range, L, thetas, total_multiple, file_name, plot_theta_index = plot_theta_index, appended_string='_'+str(file_index)+'_angle_'+str(theta_plot)+'_multiple')


                if scattered_fields:
                    # Compute scattered intensities at measurement points
                    Eall_scat_ss = onp.array(Eall_scat_ss)
                    total_scat_ss = onp.absolute(Eall_scat_ss)**2
                    
                    # Produce plots
                    utils.plot_transmission_angularbeam(k0range, L, thetas, total_scat_ss, file_name,  n_thetas_trans = n_thetas_trans, adapt_scale = adapt_scale, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_scat_ss')
                    utils.plot_transmission_flat(k0range, L, thetas, total_scat_ss, file_name,  n_thetas_trans = n_thetas_trans, adapt_scale = adapt_scale, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_scat_ss')
                    utils.plot_angular_averaged_transmission(k0range, L, total_scat_ss, file_name, appended_string='_'+str(file_index)+'_scat_ss')
                    theta_plot = onp.round(180 * thetas[plot_theta_index]/onp.pi)
                    utils.plot_singlebeam_angular_frequency_plot(k0range, L, thetas, total_scat_ss, file_name, plot_theta_index = plot_theta_index, appended_string='_'+str(file_index)+'_angle_'+str(theta_plot)+'_scat_ss')

                    utils.plot_singlebeam_angular_frequency_plot(k0range, L, thetas, total_scat_ss, file_name, plot_theta_index = plot_theta_index, normalization=total_scat_ss, appended_string='_'+str(file_index)+'_angle_'+str(theta_plot)+'_scat_ss_norm')

                    # Also produce scattered field normalised by total scattered intensity
                    utils.plot_transmission_angularbeam(k0range, L, thetas, total_scat_ss, file_name,  n_thetas_trans = n_thetas_trans, adapt_scale = adapt_scale, normalization = total_scat_ss, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_scat_ss_norm')
                    utils.plot_transmission_flat(k0range, L, thetas, total_scat_ss, file_name,  n_thetas_trans = n_thetas_trans, adapt_scale = adapt_scale, normalization = total_scat_ss, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_scat_ss_norm')
                    


            # Compute full fields
            # Pretty expensive!
            some_fields = intensity_fields+amplitude_fields+phase_fields
            if some_fields:
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
                print("Computing the full fields at "+str(gridsize)+" points in "+str(n_batches)+" batch"+extra_string+" of "+str(onp.min([batch_size, ngridx*ngridy])))

                thetas_plot_indices = onp.searchsorted(thetas, thetas_plot)
                if method == "torch":
                    solver = Transmission2D_scalar(points, source = source)
                elif method == "hmatrices":
                    solver = Transmission2D_scalar_hmatrices(points, source = source)
                else:
                    print("Choose a valid method")
                    sys.exit()

                for k0, alpha in zip(k0range,alpharange):
                    k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                    print("k0L/2pi = "+str(k0_))

                    # Check if file already exists or if computation is needed
                    file = file_name+'_Ek_k0_'+str(k0_)+'_'+str(file_index)+'.hkl'
                    # File is there: load data
                    if os.path.isfile(file):
                        Ej, params, points, thetas = hkl.load(file_name+'_Ek_k0_'+str(k0_)+'_'+str(file_index)+'.hkl')
                        Ej = np.tensor(Ej, dtype=np.complex128)
                        points = np.tensor(points, dtype=np.complex128)
                        thetas = onp.float64(thetas)
                        alpha, k0 = params
                        k0 = onp.float64(k0)
                        alpha = onp.complex128(alpha)
                    else:
                        Ej = solver.solve(k0, alpha, thetas, radius, w, self_interaction = self_interaction, self_interaction_type = self_interaction_type)
                        k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                        params = [alpha, k0]
                        hkl.dump([onp.array(Ej), onp.array(params),onp.array(points), onp.array(thetas)],file_name+'_Ek_k0_'+str(k0_)+'_'+str(file_index)+'.hkl')

                    for index, angle in zip(thetas_plot_indices, thetas_plot):
                        angle_ = onp.round(angle*180/onp.pi)
                        print("angle = "+str(angle_)+"degrees")

                        Eall = []
                        
                        if scattered_fields:
                            Eall_scat = []

                        for batch in range(0, n_batches):
                            print("Batch "+str(batch+1))
                            batch_points = batches[batch]

                            Ek = solver.propagate(batch_points, Ej[:,index].unsqueeze(-1), k0, alpha, [angle], w, regularize = regularize, radius = radius)

                            Eall.append(Ek)
                            
                            if scattered_fields:
                                E0 = solver.generate_source(batch_points, k0, [angle], w, print_statement = 'scattered_fields')
                                Eall_scat.append(Ek - E0)


                        Eall = np.cat(Eall, dim=0)
                        Eall = Eall.reshape(ngridy, ngridx)

                        utils.plot_full_fields(Eall, ngridx, ngridy, k0_, angle_, intensity_fields, amplitude_fields, phase_fields, file_name, appended_string='_width_'+str(window_width)+'_grid_'+str(ngridx)+'x'+str(ngridy)+'_'+str(file_index), my_dpi = 300)
                        
                        if scattered_fields:
                            Eall = np.cat(Eall_scat, dim = 0)
                            Eall = Eall.reshape(ngridy, ngridx)

                            utils.plot_full_fields(Eall, ngridx, ngridy, k0_, angle_, intensity_fields, amplitude_fields, phase_fields, file_name, appended_string='_width_'+str(window_width)+'_grid_'+str(ngridx)+'x'+str(ngridy)+'_'+str(file_index)+'_scat', my_dpi = 300)

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

                
                if method == "torch":
                    solver = Transmission2D_scalar(points, source = None)
                elif method == "hmatrices":
                    solver = Transmission2D_scalar_hmatrices(points, source = None)
                else:
                    print("Choose a valid method")
                    sys.exit()
                    
                k0_range = []

                for k0, alpha in zip(k0range,alpharange):
                    k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                    k0_range.append(k0_)

                    _, eigenmodes,_ = solver.compute_eigenmodes_IPR( k0, alpha, radius, file_name, write_eigenvalues = True, number_eigenmodes = number_eigenmodes, self_interaction = self_interaction, self_interaction_type = self_interaction_type, sorting_type = sorting_type)

                    if plot_eigenmodes:
                        
                        for i in range(number_eigenmodes):
                            
                            Eall = []
                            
                            # By default, the eigenvectors are such that their modulus is 1
                            eigenmodes[:,i] /= np.abs(eigenmodes[:,i]).amax()

                            for batch in range(0, n_batches):
                                print("Batch "+str(batch+1))
                                batch_points = batches[batch]

                                eigenfield = solver.propagate(batch_points, eigenmodes[:,i].unsqueeze(-1), k0, alpha, [0.0], w, regularize = regularize, radius=radius)

                                Eall.append(eigenfield)

                            Eall = np.cat(Eall, dim=0)

                            Eall = Eall.reshape(ngridy, ngridx)
                            
                            plot_IPR = np.sum(np.abs(Eall**4)) / (np.sum(np.abs(Eall**2)))**2
                            
                            print(f"Effective IPR of the whole eigenfield: {plot_IPR}")

                            utils.plot_full_fields(Eall, ngridx, ngridy, k0_, 0, True, False, False, file_name, appended_string='_width_'+str(window_width)+'_grid_'+str(ngridx)+'x'+str(ngridy)+'_'+str(file_index)+'_eigen_'+sorting_type+str(i), my_dpi = 300)

            if compute_SDOS:
                if method == "torch":
                    solver = Transmission2D_scalar(points, source = source)
                elif method == "hmatrices":
                    solver = Transmission2D_scalar_hmatrices(points, source = source)
                else:
                    print("Choose a valid method")
                    sys.exit()
                DOSall = []
                k0_range = []

                for k0, alpha in zip(k0range,alpharange):
                    dos = solver.compute_eigenvalues_and_scatterer_LDOS( k0, alpha, radius, file_name, write_eigenvalues = write_eigenvalues, self_interaction = self_interaction, self_interaction_type = self_interaction_type)
                    DOSall.append(dos.numpy())
                    k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                    k0_range.append(k0_)

                    onp.savetxt(file_name+'_temp_sdos.csv',onp.stack([k0_range,DOSall]).T)

                onp.savetxt(file_name+'_sdos.csv',onp.stack([k0_range,DOSall]).T)

                utils.plot_averaged_DOS(k0range, L, DOSall, file_name, 'sdos', appended_string = '_'+str(file_index))

            if compute_DOS:
                if method == "torch":
                    solver = Transmission2D_scalar(points, source = source)
                elif method == "hmatrices":
                    solver = Transmission2D_scalar_hmatrices(points, source = source)
                else:
                    print("Choose a valid method")
                    sys.exit()
                DOSall = []
                k0_range = []

                M = dospoints
                measurement_points = utils.uniform_unit_disk_picking(M)
                measurement_points *= L/2

                utils.plot_2d_points(measurement_points, file_name+'_measurement')

                for k0, alpha in zip(k0range,alpharange):
                    dos = solver.mean_DOS_measurements(measurement_points, k0, alpha, radius, regularize = regularize, self_interaction = self_interaction, self_interaction_type = self_interaction_type)
                    if method == "torch":
                        DOSall.append(dos.numpy())
                    else:
                        DOSall.append(dos)

                    k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                    k0_range.append(k0_)

                    onp.savetxt(file_name+'_temp_dos.csv',onp.stack([k0_range,DOSall]).T)

                onp.savetxt(file_name+'_dos.csv',onp.stack([k0_range,DOSall]).T)

                utils.plot_averaged_DOS(k0range, L, DOSall, file_name, 'dos', appended_string='_'+str(file_index))

            if compute_interDOS:
                if method == "torch":
                    solver = Transmission2D_scalar(points, source = source)
                elif method == "hmatrices":
                    solver = Transmission2D_scalar_hmatrices(points, source = source)
                else:
                    print("Choose a valid method")
                    sys.exit()
                DOSall = []
                k0_range = []

                M = dospoints
                measurement_points = utils.uniform_unit_disk_picking(M)
                measurement_points *= L/2

                # Find all overlaps and redraw while you have some
                # Following Pierrat et al., I use 1 diameter as the spacing there
                spacing = 2.0*radius
                spacing *= spacing_factor
                # XXX TODO: use a random cavity or something similar instead!
                overlaps = np.nonzero(np.sum(np.cdist(measurement_points.to(np.double), points.to(np.double), p=2) <= spacing, axis = -1)).squeeze()
                count = overlaps.shape[0]
                while count > 0:
                    print("Removing "+str(count)+" overlaps using an exclusion distance of "+str(spacing_factor)+" scatterer diameters...")
                    measurement_points[overlaps] = L/2 * utils.uniform_unit_disk_picking(count)
                    overlaps = np.nonzero(np.sum(np.cdist(measurement_points.to(np.double), points.to(np.double), p=2) <= spacing, axis = -1)).squeeze()
                    if len(overlaps.shape) == 0:
                        count = 0
                    else:
                        count = overlaps.shape[0]

                utils.plot_2d_points(measurement_points, file_name+'_measurement')

                for k0, alpha in zip(k0range,alpharange):
                    dos = solver.mean_DOS_measurements(measurement_points, k0, alpha, radius, regularize = regularize, self_interaction = self_interaction, self_interaction_type = self_interaction_type)
                    if method == "torch":
                        DOSall.append(dos.numpy())
                    else:
                        DOSall.append(dos)

                    k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                    k0_range.append(k0_)

                    onp.savetxt(file_name+'_temp_idos.csv',onp.stack([k0_range,DOSall]).T)

                onp.savetxt(file_name+'_idos.csv',onp.stack([k0_range,DOSall]).T)

                utils.plot_averaged_DOS(k0range, L, DOSall, file_name, 'idos', appended_string='_'+str(file_index))

            if compute_LDOS:
                if method == "torch":
                    solver = Transmission2D_scalar(points, source = source)
                elif method == "hmatrices":
                    solver = Transmission2D_scalar_hmatrices(points, source = source)
                else:
                    print("Choose a valid method")
                    sys.exit()
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

                    outputs = []
                    k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)

                    for batch in range(0, n_batches):
                        print("Batch "+str(batch+1))
                        batch_points = batches[batch]
                        ldos = solver.LDOS_measurements(batch_points, k0, alpha, radius, regularize = regularize, self_interaction = self_interaction, self_interaction_type = self_interaction_type)

                        outputs.append(ldos)

                    #    onp.savetxt(file_name+'_temp_ldos_'+str(k0_)+'_TE.csv',np.cat(outputs_TE).numpy())
                    #    onp.savetxt(file_name+'_temp_ldos_'+str(k0_)+'_TM.csv',np.cat(outputs_TM).numpy())

                    ldos = np.cat(outputs)
                    
                    ldos = ldos.reshape(ngridy, ngridx)

                    utils.plot_LDOS_2D(ldos,k0_,ngridx,ngridy,file_name, appended_string='_width_'+str(window_width)+'_grid_'+str(ngridx)+'x'+str(ngridy)+'_'+str(file_index), my_dpi = 300)

                    if write_ldos:
                        onp.savetxt(file_name+'_ldos_'+str(k0_)+'_'+str(index)+'.csv',ldos.numpy())  
                    
        ### ###############
        ### 3d calculations
        ### ###############
        elif ndim==3:

            if compute_transmission or plot_transmission:
                
                # Define the list of measurement points for transmission plots
                measurement_points = transmission_radius*L*onp.vstack([onp.cos(thetas),onp.sin(thetas),onp.zeros(len(thetas))]).T

                # A fresh computation is required
                if compute_transmission:

                    if method == "torch":
                        solver = Transmission3D_scalar(points, source = source)
                    elif method == "hmatrices":
                        solver = Transmission3D_scalar_hmatrices(points, source = source)
                    else:
                        print("Choose a valid method")
                        sys.exit()
                    u = onp.stack([onp.cos(thetas),onp.sin(thetas),onp.zeros(len(thetas))]).T
                    u = np.tensor(u)
                    Eall  = []
                    E0all = []
                    Eall_scat = []
                    
                    for k0, alpha in zip(k0range,alpharange):
                        Ej = solver.solve(k0, alpha, u, radius, w, self_interaction = self_interaction, self_interaction_type = self_interaction_type)
                        k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                        params = [alpha, k0]
                        hkl.dump([onp.array(Ej), onp.array(params),onp.array(points), onp.array(thetas)],file_name+'_Ek_k0_'+str(k0_)+'_'+str(file_index)+'.hkl')

                        Ek = solver.propagate(measurement_points, Ej, k0, alpha, u, w, regularize = regularize, radius = radius)

                        E0meas = solver.generate_source(np.tensor(measurement_points), k0, u, beam_waist, print_statement = 'propagate') #(M,3,Ndirs)

                        if scattered_fields:
                            Ekscat = Ek - E0meas
                            Eall_scat.append(Ekscat.numpy())
                        
                        E0all.append(E0meas.numpy())

                        Eall.append(Ek.numpy())

                # A computation has already been performed
                elif plot_transmission:

                    Eall  = []
                    E0all = []
                    Eall_scat = []
                    u = onp.stack([onp.cos(thetas),onp.sin(thetas),onp.zeros(len(thetas))]).T
                    u = np.tensor(u)
                    print(points.shape)

                    for k0, alpha in zip(k0range,alpharange):
                        
                        k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                        Ej, params, points, thetas = hkl.load(file_name+'_Ek_k0_'+str(k0_)+'_'+str(file_index)+'.hkl')
                        Ej = np.tensor(Ej, dtype=np.complex128)
                        points = np.tensor(points, dtype=np.float64)
                        thetas = onp.float64(thetas)
                        alpha, k0 = params
                        k0 = onp.float64(k0)
                        alpha = onp.complex128(alpha)
                        if method == "torch":
                            solver = Transmission3D_scalar(points, source = source)
                        elif method == "hmatrices":
                            solver = Transmission3D_scalar_hmatrices(points, source = source)
                        else:
                            print("Choose a valid method")
                            sys.exit()

                        Ek = solver.propagate(measurement_points, Ej, k0, alpha, u, w, regularize = regularize, radius = radius)
                        
                        E0meas = solver.generate_source(np.tensor(measurement_points), k0, u, beam_waist, print_statement='propagate') #(M,3,Ndirs)

                        if scattered_fields:
                            Ekscat = Ek - E0meas
                            Eall_scat.append(Ekscat.numpy())
                        
                        E0all.append(E0meas.numpy())
                        Eall.append(Ek.numpy())

                hkl.dump([onp.array(Eall), onp.array(k0range), onp.array(thetas)],file_name+'_transmission_'+str(file_index)+'.hkl')
        
            # If required: plot results
            if plot_transmission:
                # Compute intensities at measurement points
                Eall = onp.array(Eall)
                Etotal = onp.absolute(Eall)**2

                # Produce the plots
                utils.plot_transmission_angularbeam(k0range, L, thetas, Etotal, file_name, n_thetas_trans = n_thetas_trans, appended_string ='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)) 
                utils.plot_transmission_flat(k0range, L, thetas, Etotal, file_name, n_thetas_trans = n_thetas_trans, appended_string ='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)) 
                utils.plot_angular_averaged_transmission(k0range, L, Etotal, file_name, appended_string = '_'+str(file_index))
                theta_plot = onp.round(180 * thetas[plot_theta_index]/onp.pi)
                utils.plot_singlebeam_angular_frequency_plot(k0range, L, thetas, Etotal, file_name, plot_theta_index = plot_theta_index,  appended_string='_'+str(file_index)+'_angle_'+str(theta_plot))
                
                # Produce transmission normalized by total intensity of the INCIDENT FIELD on the sphere
                E0all = onp.array(E0all)
                I0all = onp.absolute(E0all)**2

                utils.plot_transmission_angularbeam(k0range, L, thetas, Etotal, file_name,  n_thetas_trans = n_thetas_trans, normalization = I0all, adapt_scale = adapt_scale, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_incnorm')
                utils.plot_transmission_flat(k0range, L, thetas, Etotal, file_name,  n_thetas_trans = n_thetas_trans, normalization = I0all, adapt_scale = adapt_scale, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_incnorm')

                # Same but with total field
                utils.plot_transmission_angularbeam(k0range, L, thetas, Etotal, file_name,  n_thetas_trans = n_thetas_trans, normalization = Etotal, adapt_scale = adapt_scale, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_norm')
                utils.plot_transmission_flat(k0range, L, thetas, Etotal, file_name,  n_thetas_trans = n_thetas_trans, normalization = Etotal, adapt_scale = adapt_scale, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_norm')

                if scattered_fields:
                    Eall_scat = onp.array(Eall_scat)
                    Etotal_scat = onp.absolute(Eall_scat)**2

                    # Produce the plots
                    utils.plot_transmission_angularbeam(k0range, L, thetas, Etotal_scat, file_name, n_thetas_trans = n_thetas_trans, adapt_scale = adapt_scale, appended_string ='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+"_scat") 
                    utils.plot_transmission_flat(k0range, L, thetas, Etotal_scat, file_name, n_thetas_trans = n_thetas_trans, adapt_scale = adapt_scale, appended_string ='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+"_scat") 
                    utils.plot_angular_averaged_transmission(k0range, L, Etotal_scat, file_name, appended_string = '_'+str(file_index)+"_scat")
                    theta_plot = onp.round(180 * thetas[plot_theta_index]/onp.pi)
                    utils.plot_singlebeam_angular_frequency_plot(k0range, L, thetas, Etotal_scat, file_name, plot_theta_index = plot_theta_index,  appended_string='_'+str(file_index)+'_angle_'+str(theta_plot)+'_scat')

                    # Also produce scattered field normalised by total scattered intensity on the circle. XXX Should make it a sphere
                    utils.plot_transmission_angularbeam(k0range, L, thetas, Etotal_scat, file_name,  n_thetas_trans = n_thetas_trans, adapt_scale = adapt_scale, normalization = Etotal_scat, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_scat_norm')

                
            # Single-scattering transmission
            if single_scattering_transmission:
                # Define the list of measurement points for transmission plots
                measurement_points = transmission_radius*L*onp.vstack([onp.cos(thetas),onp.sin(thetas),onp.zeros(len(thetas))]).T
                if method == "torch":
                    solver = Transmission3D_scalar(points, source = source)
                elif method == "hmatrices":
                    solver = Transmission3D_scalar_hmatrices(points, source = source)
                else:
                    print("Choose a valid method")
                    sys.exit()
                u = onp.stack([onp.cos(thetas),onp.sin(thetas),onp.zeros(len(thetas))]).T
                u = np.tensor(u)
                Eall_ss = []
                Eall_scat_ss = []
                
                for k0, alpha in zip(k0range,alpharange):
                    Ek_ss = solver.propagate_ss(measurement_points, k0, alpha, u, w, regularize = regularize, radius = radius)
                    
                    if scattered_fields:
                        E0meas = solver.generate_source(np.tensor(measurement_points), k0, u, beam_waist, print_statement = 'propagate') #(M,3,Ndirs)
                        Ekscat_ss = Ek_ss - E0meas
                        Eall_scat_ss.append(Ekscat_ss.numpy())
                    
                    Eall_ss.append(Ek_ss.numpy())

                # Compute intensities at measurement points
                Eall_ss = onp.array(Eall_ss)
                Etotal_ss = onp.absolute(Eall_ss)**2
                
                # Produce plots
                utils.plot_transmission_angularbeam(k0range, L, thetas, Etotal_ss, file_name, n_thetas_trans = n_thetas_trans, appended_string ='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+"_ss") 
                utils.plot_transmission_flat(k0range, L, thetas, Etotal_ss, file_name, n_thetas_trans = n_thetas_trans, appended_string ='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+"_ss") 
                utils.plot_angular_averaged_transmission(k0range, L, Etotal_ss, file_name, appended_string = '_'+str(file_index)+"_ss")
                theta_plot = onp.round(180 * thetas[plot_theta_index]/onp.pi)
                utils.plot_singlebeam_angular_frequency_plot(k0range, L, thetas, Etotal_ss, file_name,  plot_theta_index = plot_theta_index, appended_string='_'+str(file_index)+'_angle_'+str(theta_plot)+'_ss')
                
                if plot_transmission:
                    # Also compute the intensity associated to the multiple-scattering contribution of the field, if the full field was computed
                    Eall_multiple = Eall - Eall_ss
                    Etotal_multiple = onp.absolute(Eall_multiple)**2

                     # Produce plots
                    utils.plot_transmission_angularbeam(k0range, L, thetas, Etotal_multiple, file_name, n_thetas_trans = n_thetas_trans, appended_string ='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+"_multiple") 
                    utils.plot_transmission_flat(k0range, L, thetas, Etotal_multiple, file_name, n_thetas_trans = n_thetas_trans, appended_string ='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+"_multiple") 
                    utils.plot_angular_averaged_transmission(k0range, L, Etotal_multiple, file_name, appended_string = '_'+str(file_index)+"_multiple")
                    theta_plot = onp.round(180 * thetas[plot_theta_index]/onp.pi)
                    utils.plot_singlebeam_angular_frequency_plot(k0range, L, thetas, Etotal_multiple, file_name,  plot_theta_index = plot_theta_index, appended_string='_'+str(file_index)+'_angle_'+str(theta_plot)+'_multiple')


                if scattered_fields:
                    Eall_scat_ss = onp.array(Eall_scat_ss)
                    Etotal_scat_ss = onp.absolute(Eall_scat_ss)**2
                    
                    # Produce plots
                    utils.plot_transmission_angularbeam(k0range, L, thetas, Etotal_scat_ss, file_name, n_thetas_trans = n_thetas_trans, adapt_scale = adapt_scale, appended_string ='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+"_scat_ss") 
                    utils.plot_transmission_flat(k0range, L, thetas, Etotal_scat_ss, file_name, n_thetas_trans = n_thetas_trans,  adapt_scale = adapt_scale, appended_string ='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+"_scat_ss") 
                    utils.plot_angular_averaged_transmission(k0range, L, Etotal_scat_ss, file_name, appended_string = '_'+str(file_index)+"_scat_ss")
                    theta_plot = onp.round(180 * thetas[plot_theta_index]/onp.pi)
                    utils.plot_singlebeam_angular_frequency_plot(k0range, L, thetas, Etotal_scat_ss, file_name, plot_theta_index = plot_theta_index, appended_string='_'+str(file_index)+'_angle_'+str(theta_plot)+'_scat_ss')
                
                    # Also produce scattered field normalised by total scattered intensity on the circle. XXX Should make it a sphere
                    utils.plot_transmission_angularbeam(k0range, L, thetas, Etotal_scat_ss, file_name,  n_thetas_trans = n_thetas_trans, adapt_scale = adapt_scale, normalization = Etotal_scat_ss, appended_string='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+'_scat_ss_norm')
                    utils.plot_transmission_flat(k0range, L, thetas, Etotal_scat_ss, file_name, n_thetas_trans = n_thetas_trans, normalization = Etotal_scat_ss,  adapt_scale = adapt_scale, appended_string ='_trad'+str(transmission_radius)+'_angwidth'+str(angular_width)+'_'+str(file_index)+"_scat_ss_norm") 



            # Compute full fields
            # Pretty expensive!
            some_fields = intensity_fields+amplitude_fields+phase_fields
            if some_fields:
                # Expensive computation
                ngridx = gridsize[0]
                ngridy = gridsize[1]
                xyratio = ngridx/ngridy
                x,y,z = onp.meshgrid(onp.linspace(0,xyratio,ngridx)  - xyratio/2.0,onp.linspace(0,1,ngridy) - 0.5, [0.0])
                measurement_points = np.tensor((onp.vstack([x.ravel(),y.ravel(), z.ravel()]).T)*L*window_width)

                batches = np.split(measurement_points, batch_size)
                n_batches = len(batches)

                extra_string=""
                if n_batches > 1:
                    extra_string = extra_string+"es"
                print("Computing the full fields at "+str(gridsize)+" points in "+str(n_batches)+" batch"+extra_string+" of "+str(onp.min([batch_size, ngridx*ngridy])))

                thetas_plot_indices = onp.searchsorted(thetas, thetas_plot)
                if method == "torch":
                    solver = Transmission3D_scalar(points, source = source)
                elif method == "hmatrices":
                    solver = Transmission3D_scalar_hmatrices(points, source = source)
                else:
                    print("Choose a valid method")
                    sys.exit()

                for k0, alpha in zip(k0range,alpharange):
                    k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                    print("k0L/2pi = "+str(k0_))

                    # Check if file already exists or if computation is needed
                    file = file_name+'_Ek_k0_'+str(k0_)+'_'+str(file_index)+'.hkl'
                    # File is there: load data
                    if os.path.isfile(file):
                        Ej, params, points, thetas = hkl.load(file_name+'_Ek_k0_'+str(k0_)+'_'+str(file_index)+'.hkl')
                        Ej = np.tensor(Ej, dtype=np.complex128)
                        points = np.tensor(points, dtype=np.complex128)
                        thetas = onp.float64(thetas)
                        alpha, k0 = params
                        k0 = onp.float64(k0)
                        alpha = onp.complex128(alpha)
                    # File is not there: compute
                    else:
                        u = onp.stack([onp.cos(thetas_plot),onp.sin(thetas_plot),onp.zeros(len(thetas_plot))]).T
                        u = np.tensor(u)
                        Eall = []
                        Ej = solver.solve(k0, alpha, u, radius, w, self_interaction = self_interaction, self_interaction_type = self_interaction_type)
                        k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                        params = [alpha, k0]
                        hkl.dump([onp.array(Ej), onp.array(params),onp.array(points), onp.array(thetas)],file_name+'_Ek_k0_'+str(k0_)+'_'+str(file_index)+'.hkl')   

                    for index, angle in zip(thetas_plot_indices,thetas_plot):
                        angle_ = onp.round(angle*180/onp.pi)
                        print("angle = "+str(angle_)+"degrees")

                        Eall = []
                        
                        if scattered_fields:
                            Eall_scat = []

                        for batch in range(0, n_batches):
                            print("Batch "+str(batch+1))
                            batch_points = batches[batch]

                            E = solver.propagate(batch_points, Ej[:,index], k0, alpha, u[index], w, regularize = regularize, radius = radius)

                            Eall.append(E)
                            
                            if scattered_fields:
                                E0 = solver.generate_source(batch_points, k0, u[index], w, print_statement = 'scattered_fields')
                                Eall_scat.append(E - E0)


                        Eall = np.cat(Eall, dim=0)
                        Eall_amplitude         = np.abs(Eall)**2

                        utils.plot_full_fields(Eall_amplitude, ngridx, ngridy, k0_, angle_, intensity_fields, amplitude_fields, phase_fields, file_name, appended_string='_width_'+str(window_width)+'_grid_'+str(ngridx)+'x'+str(ngridy)+'_'+str(file_index), my_dpi = 300)
                        
                        if scattered_fields:
                            Eall = np.cat(Eall_scat, dim=0)
                            Eall_amplitude         = np.abs(Eall)**2

                            utils.plot_full_fields(Eall_amplitude, ngridx, ngridy, k0_, angle_, intensity_fields, amplitude_fields, phase_fields, file_name, appended_string='_width_'+str(window_width)+'_grid_'+str(ngridx)+'x'+str(ngridy)+'_'+str(file_index)+'_scat', my_dpi = 300)
                        
            if compute_eigenmodes:
                
                # Expensive computation
                ngridx = gridsize[0]
                ngridy = gridsize[1]
                xyratio = ngridx/ngridy
                x,y,z = onp.meshgrid(onp.linspace(0,xyratio,ngridx)  - xyratio/2.0,onp.linspace(0,1,ngridy) - 0.5, [0.0])
                measurement_points = np.tensor((onp.vstack([x.ravel(),y.ravel(), z.ravel()]).T)*L*window_width)

                batches = np.split(measurement_points, batch_size)
                n_batches = len(batches)

                extra_string=""
                if n_batches > 1:
                    extra_string = extra_string+"es"
                print("Computing the eigenfields and plotting the "+str(number_eigenmodes)+" most localized at "+str(gridsize)+" points in "+str(n_batches)+" batch"+extra_string+" of "+str(onp.min([batch_size, ngridx*ngridy])))

                
                if method == "torch":
                    solver = Transmission3D_scalar(points, source = None)
                elif method == "hmatrices":
                    solver = Transmission3D_scalar_hmatrices(points, source = None)
                else:
                    print("Choose a valid method")
                    sys.exit()
                    
                k0_range = []

                for k0, alpha in zip(k0range,alpharange):
                    k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                    k0_range.append(k0_)

                    _, eigenmodes,_ = solver.compute_eigenmodes_IPR( k0, alpha, radius, file_name, write_eigenvalues = True, number_eigenmodes = number_eigenmodes, self_interaction = self_interaction, self_interaction_type = self_interaction_type, sorting_type = sorting_type)

                    if plot_eigenmodes:
                        
                        for i in range(number_eigenmodes):
                            
                            Eall = []
                            
                            # By default, the eigenvectors are such that their modulus is 1
                            eigenmodes[:,i] /= np.abs(eigenmodes[:,i]).amax()

                            for batch in range(0, n_batches):
                                print("Batch "+str(batch+1))
                                batch_points = batches[batch]

                                eigenfield = solver.propagate(batch_points, eigenmodes[:,i].unsqueeze(-1), k0, alpha, [0.0, 0.0, 0.0], w, regularize = regularize, radius=radius)

                                Eall.append(eigenfield)

                            Eall = np.cat(Eall, dim=0)

                            Eall = Eall.reshape(ngridy, ngridx)
                            
                            plot_IPR = np.sum(np.abs(Eall**4)) / (np.sum(np.abs(Eall**2)))**2
                            
                            print(f"Effective IPR of the whole eigenfield: {plot_IPR}")

                            utils.plot_full_fields(Eall, ngridx, ngridy, k0_, 0, True, False, False, file_name, appended_string='_width_'+str(window_width)+'_grid_'+str(ngridx)+'x'+str(ngridy)+'_'+str(file_index)+'_eigen_'+sorting_type+str(i), my_dpi = 300)

            if compute_SDOS:
                if method == "torch":
                    solver = Transmission3D_scalar(points, source = source)
                elif method == "hmatrices":
                    solver = Transmission3D_scalar_hmatrices(points, source = source)
                else:
                    print("Choose a valid method")
                    sys.exit()
                DOSall = []
                k0_range = []

                for k0, alpha in zip(k0range,alpharange):
                    dos = solver.compute_eigenvalues_and_scatterer_LDOS( k0, alpha, radius, file_name, write_eigenvalues = write_eigenvalues, self_interaction = self_interaction, self_interaction_type = self_interaction_type)
                    DOSall.append(dos.numpy())

                    k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                    k0_range.append(k0_)

                    onp.savetxt(file_name+'_temp_sdos.csv',onp.stack(([k0_range,DOSall])).T)

                onp.savetxt(file_name+'_sdos.csv', onp.stack([k0_range,DOSall]).T)
                utils.plot_averaged_DOS(k0range, L, DOSall, file_name, 'sdos', appended_string='_'+str(file_index))

            if compute_DOS:
                if method == "torch":
                    solver = Transmission3D_scalar(points, source = source)
                elif method == "hmatrices":
                    solver = Transmission3D_scalar_hmatrices(points, source = source)
                else:
                    print("Choose a valid method")
                    sys.exit()
                DOSall = []
                k0_range = []

                # Expensive computation in 3d
                M = dospoints
                measurement_points = utils.uniform_unit_ball_picking(M, ndim)
                measurement_points *= L/2

                utils.plot_3d_points(measurement_points, file_name+'_measurement')

                for k0, alpha in zip(k0range,alpharange):
                    dos = solver.mean_DOS_measurements(measurement_points, k0, alpha, radius, regularize = regularize, self_interaction = self_interaction, self_interaction_type = self_interaction_type)
                    if method == "torch":
                        DOSall.append(dos.numpy())
                    else:
                        DOSall.append(dos)

                    k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                    k0_range.append(k0_)

                    onp.savetxt(file_name+'_temp_dos.csv',onp.stack([k0_range,DOSall]).T)

                onp.savetxt(file_name+'_dos.csv',onp.stack([k0_range,DOSall]).T)
                utils.plot_averaged_DOS(k0range, L, DOSall, file_name, 'dos',appended_string='_'+str(file_index))

            if compute_interDOS:
                if method == "torch":
                    solver = Transmission3D_scalar(points, source = source)
                elif method == "hmatrices":
                    solver = Transmission3D_scalar_hmatrices(points, source = source)
                else:
                    print("Choose a valid method")
                    sys.exit()
                DOSall = []
                k0_range = []

                # Expensive computation in 3d
                M = dospoints
                measurement_points = utils.uniform_unit_ball_picking(M, ndim)
                measurement_points *= L/2

                # Find all overlaps and redraw while you have some
                # Following Pierrat et al., I use 1 diameter as the spacing there
                spacing = 2.0*radius
                spacing *= spacing_factor
                overlaps = np.nonzero(np.sum(np.cdist(measurement_points, points, p=2) <= spacing)).squeeze()
                count = overlaps.shape[0]
                while count > 0:
                    print("Removing "+str(count)+" overlaps using an exclusion distance of "+str(spacing_factor)+" scatterer diameters...")
                    measurement_points[overlaps] = L/2 * utils.uniform_unit_ball_picking(count, ndim).squeeze()
                    overlaps = np.nonzero(np.sum(np.cdist(measurement_points, points, p=2) <= spacing))
                    if len(overlaps.shape) == 0:
                        count = 0
                    else:
                        count = overlaps.shape[0]


                utils.plot_3d_points(measurement_points, file_name+'_measurement')

                for k0, alpha in zip(k0range,alpharange):
                    dos = solver.mean_DOS_measurements(measurement_points, k0, alpha, radius, regularize = regularize, self_interaction = self_interaction, self_interaction_type = self_interaction_type)
                    if method == "torch":
                        DOSall.append(dos.numpy())
                    else:
                        DOSall.append(dos)

                    k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
                    k0_range.append(k0_)

                    onp.savetxt(file_name+'_temp_idos.csv',onp.stack([k0_range,DOSall]).T)

                onp.savetxt(file_name+'_idos.csv',onp.stack([k0_range,DOSall]).T)
                utils.plot_averaged_DOS(k0range, L, DOSall, file_name, 'idos', appended_string = '_'+str(file_index))

            
            if compute_LDOS:
                if method == "torch":
                    solver = Transmission3D_scalar(points, source = source)
                elif method == "hmatrices":
                    solver = Transmission3D_scalar_hmatrices(points, source = source)
                else:
                    print("Choose a valid method")
                    sys.exit()
                # Expensive computation
                # For now, taking the central plane z = 0
                ngridx = gridsize[0]
                ngridy = gridsize[1]
                xyratio = ngridx/ngridy
                x,y,z = onp.meshgrid(onp.linspace(0,xyratio,ngridx)  - xyratio/2.0,onp.linspace(0,1,ngridy) - 0.5, [0.0])
                measurement_points = np.tensor((onp.vstack([x.ravel(),y.ravel(), z.ravel()]).T)*L*window_width)

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
                        ldos = solver.LDOS_measurements(batch_points, k0, alpha, radius, regularize = regularize, self_interaction = self_interaction, self_interaction_type = self_interaction_type)

                        outputs.append(ldos)


                    ldos = np.cat(outputs)
                    
                    ldos = ldos.reshape(ngridy, ngridx)

                    utils.plot_LDOS_2D(ldos,k0_,ngridx,ngridy,file_name, appended_string='_width_'+str(window_width)+'_grid_'+str(ngridx)+'x'+str(ngridy)+'_z=0'+'_'+str(file_index), my_dpi = 300)

                    if write_ldos:
                        onp.savetxt(file_name+'_ldos_'+str(k0_)+'_'+str(index)+'.csv',ldos.numpy())


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
    # Ranges of wave-vectors and beam orientations, index of copy to look at
    parser.add_argument("-k", "--k0range", nargs='+', type=float, help = "Values of k0 to span, in units of 2pi/L. Can be a single-value argument, a k_min and a k_max (with default step 1), or k_min, k_max, and step\
        default=(1,0.25 * L/scatterer_radius,0.5)*2pi/L ", default=None)
    parser.add_argument("-t","--thetas",  nargs = "+", type = float, help = "Angles to consider, in degrees. Can be a single-value argument, a theta_min and a theta_max (with default step 1), or theta_min, theta_max, and step\
        default=(0,359,1)", default = None)
    # Special systems
    parser.add_argument("-i", "--input_files", nargs='+', type=str, help="Name of hkl files containing points. May contain several, that will be averaged over. \
        default=None", default=None)
    parser.add_argument("-l", "--lattice", type=str, help="Use a simple lattice in lieu of datapoints as entry. \
        Options are 'square', 'triangular', 'honeycomb', 'quasicrystal', 'quasidual', 'quasivoro', 'quasideBruijn', 'poisson' in 2d, and 'cubic', 'fcc', 'bcc', 'diamond', 'poisson' in 3d. \
        default=None", default=None)
    parser.add_argument("-a", "--annulus", type=float, help="radius of circular removal of points \
        default=0", default=0)
    parser.add_argument("-c","--composite", action='store_true', help="Whether to fill annulus vacancy with square lattice\
        default=False", default=False)
    parser.add_argument("--kick", type=float, help="Value of max amplitude of randomly oriented, random uniform length small kicks to add to all positions, in units of L\
        default = 0", default = 0.0)
    parser.add_argument("--method", "-m", type=str, help="Method used to solve the linear system.\
        Options = torch, hmatrices \
        default = torch", default = "torch")
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
        default=2.0", default = 2.0)
    parser.add_argument("-dos","--compute_DOS", action='store_true', help="Compute the mean DOS of the medium  \
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
    parser.add_argument("--spacing_factor", type=float, help="Number of diameters to use as excluded volume around measurement points for idos\
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
        Default = 0", default = 0.0)
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
    method                          = args.method
    # Outputs
    compute_transmission            = args.compute_transmission
    plot_transmission               = args.plot_transmission
    single_scattering_transmission  = args.single_scattering_transmission
    scattered_fields                = args.scattered_fields
    transmission_radius             = args.transmission_radius
    compute_DOS                     = args.compute_DOS
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
    
    if method == "hmatrices":
        # XXX Not sure if these do anything, will need to test it properly
        os.environ["JULIA_NUM_THREADS"] = str(n_cpus)
        os.environ["PYTHON_JULIACALL_THREADS"] = str(n_cpus) # https://docs.juliahub.com/PythonCall/WdXsa/0.9.7/juliacall/
        os.environ["OMP_NUM_THREADS"] = str(n_cpus)
    
    main(ndim,
        refractive_n = refractive_n,  phi=phi, regularize=regularize, N_raw=N, source = source, beam_waist=beam_waist, L=boxsize, size_subsample=size_subsample,
        k0range_args = k0range_args, thetarange_args=thetarange_args, input_files_args = input_files_args,
        cold_atoms=cold_atoms, kresonant_ = kresonant_, lattice=lattice, annulus = annulus, composite = composite, kick = kick, method = method,
        compute_transmission = compute_transmission, plot_transmission=plot_transmission, single_scattering_transmission=single_scattering_transmission, scattered_fields=scattered_fields, transmission_radius=transmission_radius,
        compute_DOS=compute_DOS, compute_interDOS=compute_interDOS, compute_SDOS=compute_SDOS, compute_LDOS=compute_LDOS, dos_sizes_args= dos_sizes_args, 
        compute_eigenmodes = compute_eigenmodes, number_eigenmodes = number_eigenmodes, plot_eigenmodes = plot_eigenmodes, sorting_type = sorting_type,
        intensity_fields = intensity_fields, amplitude_fields=amplitude_fields, phase_fields=phase_fields, just_compute_averages=just_compute_averages,
        dospoints=dospoints, spacing_factor=spacing_factor, idos_radius=idos_radius, write_eigenvalues=write_eigenvalues, write_ldos=write_ldos, gridsize=gridsize, window_width=window_width, batch_size = batch_size, angular_width=angular_width, plot_theta_index=plot_theta_index, adapt_scale = adapt_scale,
        output_directory=output_directory
        )
    sys.exit()


