import numpy as onp
import torch as np
import scipy as sp
import os
import sys
import hickle as hkl

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.colors as clr

import cmasher as cmr # https://github.com/1313e/CMasher

from Transmission2D import self_interaction_integral_scalar as self_interaction_integral_TM
from Transmission2D import self_interaction_integral_vector as self_interaction_integral_TE
from Transmission3D import self_interaction_integral_scalar, self_interaction_integral_vector

c = 3e8   #speed of light in vacuum, m/s

def alpha_cold_atoms_2d(k0range, omega0 = 3e15, Gamma = 5e16, Lfactor = 1e-4):
    """
    Typical polarizability of cold atoms, using an elastically bound electron model for the dielectric constant, in 2d space.
    Arguments:
    k0range: array of k values, in rad/m
    omega0: bare resonance pulsation, in rad/s
    Gamma: bare linewidth, in rad/s
    Lfactor: conversion factor for lengths, the values above being given for L = 100 µm
    """
    
    return (-2*Gamma/(omega0*(k0range*k0range-omega0*omega0/(c*c)+0.5j*Gamma*k0range*k0range/omega0)))/Lfactor**2

def alpha_cold_atoms_3d(k0range, omega0 = 3e15, Gamma = 5e16, Lfactor = 1e-4):
    """
    Typical polarizability of cold atoms, using an elastically bound electron model for the dielectric constant, in 3d space.
    Arguments:
    k0range: array of k values, in rad/m
    omega0: bare resonance pulsation, in rad/s
    Gamma: bare linewidth, in rad/s
    Lfactor: conversion factor for lengths, the values above being given for L = 100 µm
    """

    omegarange = k0range * c
    omega0sq = omega0*omega0
    return (-4*onp.pi*(c**3)*Gamma/(omega0sq*(omegarange*omegarange-omega0sq+1j*Gamma*omegarange*omegarange*omegarange/omega0sq)))/Lfactor**3


def alpha_Lorentz(k0range, volume, kresonant, kplasma, damping):
    """
    Bare Lorentz polarizability
    Arguments:
    k0range: array of k values
    volume: volume of scatterers
    kresonant: resonance value for k
    kplasma: plasma frequency converted to k-vector
    damping: non-radiative losses in k units
    """
    
    return volume * kplasma**2 / (kresonant**2 - k0range**2 - 1j * damping * k0range)

def alpha_small_dielectric_object(refractive_n, volume):
    """
    Bare static polarizability of a small dielectric object
    refractive_n: refractive index of the rods, can be complex
    volume: volume of the ball, in m^d
    """
    
    # Define the dielectric constant from the refractive index
    epsilon = refractive_n**2
    delta_epsilon = epsilon - 1 

    if onp.real(refractive_n) < 1.0:
        contrast = refractive_n - 1
        medium_n = onp.sqrt(1 - delta_epsilon)
        print("Real part of provided refractive_n is smaller than 1.0. Assuming dielectric contrast delta_epsilon = n_provided**2 - 1 ="+str(delta_epsilon)+" between medium and scatterers. We will assume n = 1.0 in scatterers and a medium with n_medium = sqrt(1 - delta_epsilon) = "+str(medium_n))

    return volume*delta_epsilon

def uniform_unit_disk_picking(n_points):
    """
    Generates an (N,2) tensor of n_points random points with a flat distribution inside the unit disk
    """
    
    U1 = onp.random.uniform(size = n_points)
    U2 = onp.random.uniform(size = n_points)
    X = onp.sqrt(U2) * onp.cos(2 * onp.pi * U1)
    Y = onp.sqrt(U2) * onp.sin(2 * onp.pi * U1)

    points = np.tensor(onp.vstack((X,Y)))

    return points.t()


def uniform_unit_ball_picking(n_points, dim):
    """
    Generates an (N,dim) tensor of n_points random points with a flat distribution inside the unit dim-ball
    https://mathworld.wolfram.com/BallPointPicking.html
    """

    normals = np.normal(0,1, size = (n_points, dim))
    exps = np.empty((n_points, 1))
    exps.exponential_()
    exps = np.sqrt(exps)
    proxies =  np.cat((normals, exps),1)
    
    points = normals/(np.linalg.norm(proxies, axis=-1)).reshape(n_points,1)


    return np.tensor(points, dtype = np.float64)

def fibonacci_sphere(samples=1000): 
    '''
    Returns a Fibonacci series sampling of the unit sphere with a set number of points.
    See https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    '''
    
    points = []
    golden = onp.pi * (onp.sqrt(5.) - 1.)  # golden angle in radians
    i = np.arange(samples)
    z = 1 - (i / (samples - 1)) * 2  # z goes from 1 to -1
    radii = onp.sqrt(1 - z**2)
    thetas = i * golden
    x = onp.cos(thetas) * radii
    y = onp.sin(thetas) * radii

    points = onp.stack([x,y,z]).transpose()
        
    # plot_3d_points(np.array(points), 'testfibo')

    return points.astype(onp.float64)

def plot_transmission_angularbeam(k0range, L, thetas, intensity, file_name_root,  n_thetas_trans = 0.0, adapt_scale = False, normalization = onp.array([]), appended_string=''):
    """
    Plots a radial version of the frequency-angle transmission plot given 
    k0range: list of wave vector moduli, in rad/m
    L: system sidelength, in m
    thetas: list of angles used for the orientation of the laser, in radians
    intensity: the relevant field intensity (dimensions: ks, detection angles, beam angles)
    file_name_root: prepended to the name of the file
    appended_string: possible postfix for the name of the file, e.g. "TM" or "TE"
    """

    freqs = onp.real(k0range*L/(2*onp.pi))

    # Define a matrix that encodes the width of the detector as a number of 1s every line around the central angle
    n_angles = intensity.shape[1]
    anglewidth_matrix = onp.diag(onp.ones(n_angles))
    if n_thetas_trans > 0:
        half_width = onp.int(onp.floor(n_thetas_trans/2))
        anglewidth_matrix = onp.fromfunction(lambda i, j: onp.abs( (i - j + n_angles/2)%n_angles - n_angles/2) <= half_width, (n_angles, n_angles))
    total_ = onp.sum(intensity*anglewidth_matrix,axis=1)

    #Normalize the field differently if needed
    if normalization.shape[0] != 0:
        total_norm = onp.sum(normalization,axis=1)
        total_ /= total_norm
    else:
        total_ /= n_thetas_trans + 1
    #     total_ /= onp.max(total_)
    
    if adapt_scale:
        vmin = None
        vmax = None
    else: 
        vmin = 1e-3
        vmax = 1e0
    
    fig, ax = plt.subplots(subplot_kw={'projection':'polar'})
    pc = ax.pcolormesh(thetas,freqs,total_,norm=clr.LogNorm(vmin=vmin,vmax=vmax), cmap=cmr.ember)#cmap=cmr.torch) #cmap='inferno')
    #ax.set_rmin(10.0)
    #ax.set_rticks([20,40])
    ax.set_axis_off()
    cbar = fig.colorbar(pc, location='left')
    cbar.ax.tick_params(labelsize=24)
    plt.savefig(file_name_root+'_transmission_angularbeam_'+appended_string+'.png', bbox_inches = 'tight',dpi=100, pad_inches = 0.1)
    plt.close()

    fig = plt.figure()
    ax = fig.gca()
    pc = ax.imshow(total_[:,:int(total_.shape[1]/2)], norm=clr.LogNorm(vmin=vmin,vmax=vmax), cmap=cmr.ember, extent =[0,180,freqs[0],freqs[-1]], origin='lower')
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'k_0L/2\pi')
    ax.set_aspect(180/(freqs[-1] - freqs[0]))
    fig.colorbar(pc)
    plt.savefig(file_name_root+'_transmission_beam_'+appended_string+'.png', bbox_inches = 'tight',dpi=100, pad_inches = 0.1)
    plt.close()
    
    avg_intensity = onp.mean(total_, axis=1)
    fig = plt.figure()
    ax = fig.gca()
    freqs = onp.real(k0range*L/(2*onp.pi))
    ax.plot(freqs, avg_intensity)
    ax.set_xlabel(r'$k_0L/2\pi$')
    ax.set_ylabel('Intensity')
    ax.legend()
    ax.set_yscale('log')
    plt.savefig(file_name_root+'_transmission_beam_avg'+appended_string+'.png', bbox_inches = 'tight',dpi=100, pad_inches = 0)
    plt.close()
    
    onp.savetxt(file_name_root+'_transmission_beam_avg'+appended_string+'.csv',onp.stack([freqs,avg_intensity]).T)

def plot_transmission_angularbeam_3d(k0range, L, thetas, intensity, measurement_points, file_name_root, angular_width = 1.0, adapt_scale = False, normalization = onp.array([]), appended_string=''):
    """
    Plots a radial version of the frequency-angle transmission plot given 
    k0range: list of wave vector moduli, in rad/m
    L: system sidelength, in m
    thetas: list of angles used for the orientation of the laser, in radians
    intensity: the relevant field intensity (dimensions: ks, detection angles, beam angles)
    file_name_root: prepended to the name of the file
    appended_string: possible postfix for the name of the file, e.g. "TM" or "TE"
    """

    freqs = onp.real(k0range*L/(2*onp.pi))
    cos_max_angle = onp.cos(angular_width * (onp.pi/2))
    u = onp.stack([onp.cos(thetas),onp.sin(thetas),onp.zeros(len(thetas))]).T
    u_out = measurement_points/onp.linalg.norm(measurement_points,axis=-1)[:,onp.newaxis]
    dotprod = onp.sum(u[:,onp.newaxis] * u_out, axis = -1)
    dotprod = dotprod.transpose()
    forward = dotprod >= cos_max_angle
    


    total_ = onp.sum(intensity*forward[onp.newaxis,:],axis=1)

    #Normalize the field differently if needed
    if normalization.shape[0] != 0:
        total_norm = onp.sum(normalization,axis=1)
        total_ /= total_norm
    else:
        total_ /= onp.sum(forward, axis=0)[onp.newaxis,:] + 1
    #     total_ /= onp.max(total_)
    
    if adapt_scale:
        vmin = None
        vmax = None
    else: 
        vmin = 1e-3
        vmax = 1e0
    
    fig, ax = plt.subplots(subplot_kw={'projection':'polar'})
    pc = ax.pcolormesh(thetas,freqs,total_,norm=clr.LogNorm(vmin=vmin,vmax=vmax), cmap=cmr.ember)
    #ax.set_rmin(10.0)
    #ax.set_rticks([20,40])
    ax.set_axis_off()
    cbar = fig.colorbar(pc)
    cbar.ax.tick_params(labelsize=24)
    plt.savefig(file_name_root+'_transmission_angularbeam_'+appended_string+'.png', bbox_inches = 'tight',dpi=100, pad_inches = 0.1)
    plt.close()

    fig = plt.figure()
    ax = fig.gca()
    pc = ax.imshow(total_[:,:int(total_.shape[1]/2)], norm=clr.LogNorm(vmin=vmin,vmax=vmax), cmap=cmr.ember, extent =[0,180,freqs[0],freqs[-1]], origin='lower')
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'k_0L/2\pi')
    ax.set_aspect(180/(freqs[-1] - freqs[0]))
    fig.colorbar(pc)
    plt.savefig(file_name_root+'_transmission_beam_'+appended_string+'.png', bbox_inches = 'tight',dpi=100, pad_inches = 0.1)
    plt.close()
    
    avg_intensity = onp.mean(total_, axis=1)
    fig = plt.figure()
    ax = fig.gca()
    freqs = onp.real(k0range*L/(2*onp.pi))
    ax.plot(freqs, avg_intensity)
    ax.set_xlabel(r'$k_0L/2\pi$')
    ax.set_ylabel('Intensity')
    ax.legend()
    ax.set_yscale('log')
    plt.savefig(file_name_root+'_transmission_beam_avg'+appended_string+'.png', bbox_inches = 'tight',dpi=100, pad_inches = 0)
    plt.close()
    
    onp.savetxt(file_name_root+'_transmission_beam_avg'+appended_string+'.csv',onp.stack([freqs,avg_intensity]).T)

def plot_transmission_flat(k0range, L, thetas, intensity, file_name_root,  n_thetas_trans = 0.0, adapt_scale = False, normalization = onp.array([]), appended_string=''):
    """
    Plots one a flattened version of the frequency-angle transmission plot given 
    k0range: list of wave vector moduli, in rad/m
    L: system sidelength, in m
    thetas: list of angles used for the orientation of the laser, in radians
    intensity: the relevant field intensity (dimensions: ks, detection angles, beam angles)
    file_name_root: prepended to the name of the file
    appended_string: possible postfix for the name of the file, e.g. "TM" or "TE"
    """
    freqs = onp.real(k0range*L/(2*onp.pi))

    # Define a matrix that encodes the width of the detector as a number of 1s every line around the central angle
    n_angles = intensity.shape[1]
    anglewidth_matrix = onp.diag(onp.ones(n_angles))
    if n_thetas_trans > 0:
        half_width = onp.int(onp.floor(n_thetas_trans/2))
        anglewidth_matrix = onp.fromfunction(lambda i, j: onp.abs( (i - j + n_angles/2)%n_angles - n_angles/2) <= half_width, (n_angles, n_angles))
    total_ = onp.sum(intensity*anglewidth_matrix,axis=1)
    
    if normalization.shape[0] != 0:
        total_norm = onp.sum(normalization,axis=1)
        total_ /= total_norm
    else:
        total_ /= n_thetas_trans + 1
    #     total_ /= onp.max(total_)
        
    if adapt_scale:
        vmin = None
        vmax = None
    else: 
        vmin = 1e-2
        vmax = 1e0
        
    fig = plt.figure()
    ax = fig.gca()
    pc = ax.imshow(total_[:,:int(total_.shape[1]/2)], norm=clr.LogNorm(vmin=vmin,vmax=vmax), cmap=cmr.ember, extent =[0,180,freqs[0],freqs[-1]], origin='lower')
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'k_0L/2\pi')
    ax.set_aspect(180/(freqs[-1] - freqs[0]))
    fig.colorbar(pc)
    plt.savefig(file_name_root+'_transmission_beam_'+appended_string+'.png', bbox_inches = 'tight',dpi=100, pad_inches = 0.1)
    plt.close()
    
    avg_intensity = onp.mean(total_, axis=1)
    fig = plt.figure()
    ax = fig.gca()
    freqs = onp.real(k0range*L/(2*onp.pi))
    ax.plot(freqs, avg_intensity)
    ax.set_xlabel(r'$k_0L/2\pi$')
    ax.set_ylabel('Intensity')
    ax.legend()
    ax.set_yscale('log')
    plt.savefig(file_name_root+'_transmission_beam_avg'+appended_string+'.png', bbox_inches = 'tight',dpi=100, pad_inches = 0)
    plt.close()
    
    onp.savetxt(file_name_root+'_transmission_beam_avg'+appended_string+'.csv',onp.stack([freqs,avg_intensity]).T)

def plot_transmission_linear(k0range, L,x, intensity, file_name_root,cmap='viridis', appended_string=''):
    """
    Plots one a flattened version of the frequency-angle transmission plot given 
    k0range: list of wave vector moduli, in rad/m
    L: system sidelength, in m
    thetas: list of angles used for the orientation of the laser, in radians
    intensity: the relevant field intensity (dimensions: ks, detection angles, beam angles)
    file_name_root: prepended to the name of the file
    appended_string: possible postfix for the name of the file, e.g. "TM" or "TE"
    """
    freqs = onp.real(k0range*L/(2*onp.pi))
    fig = plt.figure()
    ax = fig.gca()
    colors = onp.linspace(0,1,len(k0range))
    cmap = plt.get_cmap(cmap)
    for k in range(len(k0range)):
        print(freqs[k])
        ax.plot(x,intensity[k,:,0],c=cmap(colors[k]))
    ax.set_xlabel(r'$x$')
    ax.set_ylabel('Intensity')
    ax.set_yscale('log')
    plt.savefig(file_name_root+'_transmission_linear_'+appended_string+'.png', bbox_inches = 'tight',dpi=100, pad_inches = 0.1)
    plt.close()

def plot_singlebeam_angular_frequency_plot(k0range, L, thetas, intensity, file_name_root, n_thetas_trans = 0, normalization = onp.array([]), plot_theta_index = 0, appended_string=''):
    """
    Plots specific intensity for a single beam, in a radial frequency-angle plot 
    k0range: list of wave vector moduli, in rad/m
    L: system sidelength, in m
    thetas: list of angles used for the orientation of the laser, in radians
    intensity: the relevant field intensity (dimensions: ks, detection angles, beam angles)
    file_name_root: prepended to the name of the file
    appended_string: possible postfix for the name of the file, e.g. "TM" or "TE"
    """
    fig, ax = plt.subplots(subplot_kw={'projection':'polar'})
    freqs = onp.real(k0range*L/(2*onp.pi))
    total_ = intensity[:,:,plot_theta_index]

    if normalization.shape[0] != 0:
        total_norm = onp.sum(normalization,axis=1)
        total_ /= total_norm

    # XXX Use n_thetas_trans here as well on second dim if needed. Useful?
    # pc = ax.pcolormesh(thetas,freqs,total_,norm=clr.LogNorm(vmin=total_.min(),vmax=total_.max()), cmap=cmr.ember)
    pc = ax.pcolormesh(thetas,freqs,total_,norm=clr.LogNorm(vmin=1e-2,vmax=1e0), cmap=cmr.ember)
    ax.set_axis_off()
    # fig.colorbar(pc)
    plt.savefig(file_name_root+'_transmission_angular'+appended_string+'.png', bbox_inches = 'tight',dpi=100, pad_inches = 0)
    plt.close()

def plot_full_fields(field, ngridx, ngridy, k0_, angle_, intensity_fields, amplitude_fields, phase_fields, file_name_root, appended_string='', my_dpi = 1):

    if intensity_fields:

        intensity = onp.absolute(field)**2
        intensity = np.where(intensity <= 1e-10, 1e-10, intensity)

        fig = plt.figure(figsize = (ngridx/my_dpi, ngridy/my_dpi), dpi = my_dpi)
        ax = plt.gca()
        pc = ax.imshow(intensity, cmap='magma' , norm=clr.LogNorm(vmin=1e-3,vmax=2e0))
        ax.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
        plt.savefig(file_name_root+'_log_capped_intensity_k0'+str(k0_)+'_angle_'+str(angle_)+appended_string+'.png', bbox_inches = 'tight', dpi=my_dpi, pad_inches = 0)
        plt.close()

        fig = plt.figure(figsize=(ngridx/my_dpi, ngridy/my_dpi), dpi = my_dpi)
        ax = plt.gca()
        pc = ax.imshow(intensity, cmap='magma', vmin=0, vmax=2e0)
        ax.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
        plt.savefig(file_name_root+'_linear_capped_intensity_k0'+str(k0_)+'_angle_'+str(angle_)+appended_string+'.png', bbox_inches = 'tight',dpi=my_dpi, pad_inches = 0)
        plt.close()

    if amplitude_fields:
        amplitude = onp.real(field)
        scale_max = onp.sqrt(2.0)

        fig = plt.figure(figsize=(ngridx/my_dpi,ngridy/my_dpi), dpi=my_dpi)
        ax = plt.gca()
        pc = ax.imshow(amplitude, cmap=cmr.redshift, vmin=-scale_max, vmax=scale_max)
        ax.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
        plt.savefig(file_name_root+'_linear_capped_amplitude_k0'+str(k0_)+'_angle_'+str(angle_)+appended_string+'.png', bbox_inches = 'tight',dpi=my_dpi, pad_inches = 0)
        plt.close()

    if phase_fields:
        pure_phase = onp.angle(field)

        fig = plt.figure(figsize=(ngridx/my_dpi,ngridy/my_dpi), dpi=my_dpi)
        ax = plt.gca()
        pc = ax.imshow(pure_phase,cmap=cmr.emergency_s, vmin=-onp.pi,vmax=onp.pi,)
        ax.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
        plt.savefig(file_name_root+'_phase_k0'+str(k0_)+'_angle_'+str(angle_)+appended_string+'.png', bbox_inches = 'tight',dpi=my_dpi, pad_inches = 0)
        plt.close()


def plot_2d_field(intensity, ngrid, file_name_root, cmap=cmr.ember,logscale = True, vmin=1e-3, vmax=1e0,appended_string=''):
    fig = plt.figure()
    ax = plt.gca()
    if logscale:
        pc = ax.imshow(intensity.reshape(ngrid,ngrid),cmap=cmap,norm=clr.LogNorm(vmin=vmin,vmax=vmax))
    else: 
        pc = ax.imshow(intensity.reshape(ngrid,ngrid),cmap=cmap,vmin=vmin,vmax=vmax)
    fig.colorbar(pc)
    plt.savefig(file_name_root+'_intensity'+appended_string+'.png', bbox_inches = 'tight',dpi=100, pad_inches = 0)
    plt.close()

def plot_3d_points(points, file_name):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(points[:,0], points[:,1], points[:,2])

    plt.savefig(file_name+'_3dplot.png', dpi = 300)

def plot_2d_points(points, file_name):
    fig = plt.figure(figsize=(10,10),dpi=300)
    ax = fig.gca()

    ax.scatter(points[:,0], points[:,1], s = 2)

    plt.savefig(file_name+'_2dplot.png', dpi = 300)
    
def plot_IPR_damping_values(lambdas, IPRs, file_name, appended_string = '', logscale = False):
    
    fig = plt.figure(figsize=(10,10),dpi=300)
    ax = fig.gca()
    scatterplot = ax.scatter(np.real(lambdas), np.imag(lambdas), c=IPRs, s = 100, edgecolors='none',  cmap=cmr.bubblegum, vmin = 0, vmax = 0.5)
    cbar = plt.colorbar(scatterplot)
    cbar.set_label('IPR', rotation=270)
    ax.set_xlabel(r'$Re \Delta_n$')
    ax.set_ylabel(r'$Im \Delta_n$')
    if logscale:
        ax.set_yscale('log')
    plt.savefig(file_name+'_deltas_IPRs'+appended_string+'.png', dpi = 300)
    plt.close()
    
    fig = plt.figure(figsize=(10,10),dpi=300)
    ax = fig.gca()
    scatterplot = ax.scatter(np.imag(lambdas), IPRs, c=IPRs, s = 100, edgecolors='none',  cmap=cmr.bubblegum, vmin = 0, vmax = 0.5)
    ax.set_xlabel(r'$Im \Delta_n$')
    ax.set_ylabel(r'$IPR$')
    if logscale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    plt.savefig(file_name+'_damping_IPRs'+appended_string+'.png', dpi = 300)
    plt.close()

def plot_LDOS_2D(ldos_change,k0_,ngridx,ngridy,file_name,my_dpi=1, appended_string=''):

    # Matplotlib deals with figure sizes in a completely idiotic way, workaround https://stackoverflow.com/questions/13714454/specifying-and-saving-a-figure-with-exact-size-in-pixels
    fig = plt.figure(figsize=(ngridx/my_dpi,ngridy/my_dpi), dpi=my_dpi)
    ax = plt.gca()
    pc=ax.imshow(ldos_change.numpy().reshape(ngridy,ngridx),cmap=cmr.iceburn, vmin=-1.0, vmax=1.0)
    ax.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
    plt.savefig(file_name+'_k0_'+str(k0_)+'_ldos_capped'+appended_string+'.png', bbox_inches='tight', pad_inches=0., dpi=my_dpi)
    plt.close()

def plot_angular_averaged_transmission(k0range, L, intensities, file_name, appended_string=''):
    # Angular-averaged transmission
    intensities_ = onp.sum(intensities*onp.diag(onp.ones(intensities.shape[-1])),axis=1)
    avg_intensity = onp.mean(intensities_, axis=1)
    fig = plt.figure()
    ax = fig.gca()
    freqs = onp.real(k0range*L/(2*onp.pi))
    ax.plot(freqs, avg_intensity)
    ax.set_xlabel(r'$k_0L/2\pi$')
    ax.set_ylabel('Intensity')
    ax.legend()
    ax.set_yscale('log')
    plt.savefig(file_name+'_transmission_avg'+appended_string+'.png', bbox_inches = 'tight',dpi=100, pad_inches = 0)
    plt.close()
    
def plot_optical_thickness(k0range, L, alpharange, ndim, phi, volume, file_name, appended_string=''):
    # Determine and plot optical thickness against k for the system
    if ndim == 2:
        scattering_cross_section = 0.25 * k0range**3 * onp.absolute(alpharange)**2
    elif ndim == 3:
        scattering_cross_section = (1.0 / (6.0 * onp.pi)) * k0range**4 * onp.absolute(alpharange)**2
    rho = phi / volume
    scattering_mean_free_path_IS = 1. / ( rho * scattering_cross_section ) 
    optical_thickness = L / scattering_mean_free_path_IS
    
    fig = plt.figure()
    ax = fig.gca()
    freqs = onp.real(k0range*L/(2*onp.pi))
    ax.plot(freqs, optical_thickness)
    ax.set_xlabel(r'$k_0L/2\pi$')
    ax.set_ylabel('Optical thickness')
    ax.legend()
    plt.savefig(file_name+'_opticalthickness'+appended_string+'.png', bbox_inches = 'tight',dpi=100, pad_inches = 0)
    plt.close()

def plot_dressed_polarizability(k0range, L, alpharange, ndim, radius, volume, self_interaction, file_name, appended_string = '', scalar = False, self_interaction_type = "Rayleigh"):
    # Plot dressed polarizability taking into account self_interaction to find resonances
    
    if ndim == 3:
        
        alpha_d = alpharange.copy()
        
        if self_interaction:
            
            if scalar:
                self_int = self_interaction_integral_scalar(k0range, radius, self_interaction_type)
            else:
                self_int = self_interaction_integral_vector(k0range, radius, self_interaction_type)
                
            alpha_d /= (1 - k0range**2 * alpharange * self_int / volume)
        
        if scalar:
            scattering_cross_section = (1.0 / (4.0 * onp.pi)) * k0range**4 * onp.absolute(alpha_d)**2
        else:
            scattering_cross_section = (1.0 / (6.0 * onp.pi)) * k0range**4 * onp.absolute(alpha_d)**2
        extinction_cross_section = k0range * onp.imag(alpha_d)
        
        alpha_d = onp.absolute(alpha_d)
        
        max_alpha = onp.argmax(alpha_d)
        k0_max = k0range[max_alpha] * L / (2 * onp.pi)
        if max_alpha != alpha_d.shape[0] - 1:
            print("Resonance in the explored interval, at k0 = "+ str(k0_max) +"!")
            
        fig = plt.figure()
        ax = fig.gca()
        freqs = onp.real(k0range*L/(2*onp.pi))
        ax.plot(freqs, alpha_d, c = 'r', label = 'dressed')
        if self_interaction:
            ax.plot(freqs, onp.absolute(alpharange[0])*onp.ones(alpharange.shape), c='k', ls = '--', label = 'bare')
            deltaeps = alpharange[0]*onp.ones(alpharange.shape) / volume
            clausius = 3 * volume * deltaeps / (deltaeps + 3)
            ax.plot(freqs, onp.absolute(clausius), c='k', ls =':', label = 'Clausius-Mossotti')
        ax.set_xlabel(r'$k_0L/2\pi$')
        ax.set_ylabel(r"|\alpha_d|")
        ax.legend()
        plt.savefig(file_name+'_alphad'+appended_string+'.png', bbox_inches = 'tight',dpi=100, pad_inches = 0)
        plt.close()
        
        fig = plt.figure()
        ax = fig.gca()
        freqs = onp.real(k0range*L/(2*onp.pi))
        ax.plot(freqs, scattering_cross_section, c='r', label = "Rayleigh scattering CS")
        ax.plot(freqs, extinction_cross_section, c='k', label = "Rayleigh extinction CS")
        ax.plot(freqs, extinction_cross_section - scattering_cross_section, c='b', label = "Rayleigh absorption CS")
        ax.set_xlabel(r'$k_0L/2\pi$')
        ax.set_ylabel('Cross-sections')
        ax.legend()
        plt.savefig(file_name+'_crosssections'+appended_string+'.png', bbox_inches = 'tight',dpi=100, pad_inches = 0)
        plt.close()

        
    elif ndim == 2:
        
        alpha_d_TE = alpharange.copy()
        alpha_d_TM = alpharange.copy()
        
        if self_interaction:
            self_int_TM = self_interaction_integral_TM(k0range, radius, self_interaction_type)
            self_int_TE = self_interaction_integral_TE(k0range, radius, self_interaction_type)
            
            alpha_d_TM /= (1 - k0range**2 * alpharange * self_int_TM / volume)
            alpha_d_TE /= (1 - k0range**2 * alpharange * self_int_TE / volume)
            
        
        scattering_cross_section_TE = (1.0 / 8.0) * k0range**3 * onp.absolute(alpha_d_TE)**2
        extinction_cross_section_TE = k0range * onp.imag(alpha_d_TE)
        
        scattering_cross_section_TM = (1.0 / 4.0) * k0range**3 * onp.absolute(alpha_d_TM)**2
        extinction_cross_section_TM = k0range * onp.imag(alpha_d_TM)
        
        alpha_d_TE = onp.absolute(alpha_d_TE)
        alpha_d_TM = onp.absolute(alpha_d_TM)
        
        if scalar:
            max_TM = onp.argmax(alpha_d_TM)
            k0_max_TM = k0range[max_TM] * L / (2 * onp.pi)
            if onp.argmax(alpha_d_TM) != alpha_d_TM.shape[0] - 1:
                print("Resonance in the explored interval, at k0 = "+ str(k0_max_TM) +"!")
                
            fig = plt.figure()
            ax = fig.gca()
            freqs = onp.real(k0range*L/(2*onp.pi))
            ax.plot(freqs, alpha_d_TM, c = 'r')
            ax.plot(freqs, onp.absolute(alpharange[0])*onp.ones(alpharange.shape), c='k', ls ='--', label = 'bare')
            ax.set_xlabel(r'$k_0L/2\pi$')
            ax.set_ylabel(r"|\alpha_d|")
            ax.legend()
            plt.savefig(file_name+'_alphad'+appended_string+'.png', bbox_inches = 'tight',dpi=100, pad_inches = 0)
            plt.close()
            
            fig = plt.figure()
            ax = fig.gca()
            freqs = onp.real(k0range*L/(2*onp.pi))
            ax.plot(freqs, scattering_cross_section_TM, c='r', label = "Rayleigh scattering CS")
            ax.plot(freqs, extinction_cross_section_TM, c='k', label = "Rayleigh extinction CS")
            ax.plot(freqs, extinction_cross_section_TM - scattering_cross_section_TM, c='b', label = "Rayleigh absorption CS")
            ax.set_xlabel(r'$k_0L/2\pi$')
            ax.set_ylabel('Cross-sections')
            ax.legend()
            plt.savefig(file_name+'_crosssections'+appended_string+'.png', bbox_inches = 'tight',dpi=100, pad_inches = 0)
            plt.close()
            
        else:
            max_TE = onp.argmax(alpha_d_TE)
            k0_max_TE = k0range[max_TE] * L / (2 * onp.pi)
            if max_TE != alpha_d_TE.shape[0] - 1:
                print("TE Resonance in the explored interval, at k0 = "+ str(k0_max_TE) +"!")
                
            max_TM = onp.argmax(alpha_d_TM)
            k0_max_TM = k0range[max_TM] * L / (2 * onp.pi)
            if onp.argmax(alpha_d_TM) != alpha_d_TM.shape[0] - 1:
                print("TM Resonance in the explored interval, at k0 = "+ str(k0_max_TM) +"!")
                
            fig = plt.figure()
            ax = fig.gca()
            freqs = onp.real(k0range*L/(2*onp.pi))
            ax.plot(freqs, alpha_d_TE, c = 'r', label = 'TE')
            ax.plot(freqs, alpha_d_TM, c = 'b', label = 'TM')
            if self_interaction:
                ax.plot(freqs, onp.absolute(alpharange[0])*onp.ones(alpharange.shape), c='k', ls ='--', label = 'bare')
                deltaeps = alpharange[0]*onp.ones(alpharange.shape) / volume
                clausius = 2 * volume * deltaeps / (deltaeps + 2)
                ax.plot(freqs, onp.absolute(clausius), c='k', ls =':', label = 'Clausius-Mossotti')
            ax.set_xlabel(r'$k_0L/2\pi$')
            ax.set_ylabel(r"|\alpha_d|")
            ax.legend()
            plt.savefig(file_name+'_alphad'+appended_string+'.png', bbox_inches = 'tight',dpi=100, pad_inches = 0)
            plt.close()
            
            fig = plt.figure()
            ax = fig.gca()
            freqs = onp.real(k0range*L/(2*onp.pi))
            ax.plot(freqs, scattering_cross_section_TE, c='r', label = "Rayleigh scattering CS")
            ax.plot(freqs, extinction_cross_section_TE, c='k', label = "Rayleigh extinction CS")
            ax.plot(freqs, extinction_cross_section_TE - scattering_cross_section_TE, c='b', label = "Rayleigh absorption CS")
            ax.set_xlabel(r'$k_0L/2\pi$')
            ax.set_ylabel('Cross-sections')
            ax.legend()
            plt.savefig(file_name+'_crosssections_TE'+appended_string+'.png', bbox_inches = 'tight',dpi=100, pad_inches = 0)
            plt.close()
            
            fig = plt.figure()
            ax = fig.gca()
            freqs = onp.real(k0range*L/(2*onp.pi))
            ax.plot(freqs, scattering_cross_section_TM, c='r', label = "Rayleigh scattering CS")
            ax.plot(freqs, extinction_cross_section_TM, c='k', label = "Rayleigh extinction CS")
            ax.plot(freqs, extinction_cross_section_TM - scattering_cross_section_TM, c='b', label = "Rayleigh absorption CS")
            ax.set_xlabel(r'$k_0L/2\pi$')
            ax.set_ylabel('Cross-sections')
            ax.legend()
            plt.savefig(file_name+'_crosssections_TM'+appended_string+'.png', bbox_inches = 'tight',dpi=100, pad_inches = 0)
            plt.close()
        
    else:
        print("ndim not implemented!")
        sys.exit()

def plot_k_times_radius(k0range, radius, L, file_name, appended_string=''):
    # Plot the value of k times a to check whether hypotheses are still consistent
    ka = k0range * radius / (2 * onp.pi)
    
    fig = plt.figure()
    ax = fig.gca()
    freqs = onp.real(k0range*L/(2*onp.pi))
    ax.plot(freqs, ka)
    ax.set_xlabel(r'$k_0L/2\pi$')
    ax.set_ylabel(r'$k_0 a/2\pi$')
    ax.legend()
    plt.savefig(file_name+'_k_times_radius'+appended_string+'.png', bbox_inches = 'tight',dpi=100, pad_inches = 0)
    plt.close()

def plot_averaged_DOS(k0range, L, DOS, file_name, DOS_type, appended_string='', debug=False):
    # Averaged LDOS plot
    if debug:
        # XXX DEBUG: threshold values to -1
        DOS = onp.array(DOS)
        DOS = onp.where(DOS < -1, -1, DOS)
    fig = plt.figure()
    ax = fig.gca()
    freqs = onp.real(k0range*L/(2*onp.pi))
    ax.plot(freqs, DOS)
    ax.set_xlabel(r'$k_0L/2\pi$')
    ax.set_ylabel(r'$\delta\varrho$')
    ax.legend()
    plt.savefig(file_name+'_'+DOS_type+'_avg'+appended_string+'.png', bbox_inches = 'tight',dpi=100, pad_inches = 0)
    plt.close()

def loadpoints(file_path, ndim):
    
    if '.hkl' in file_path:
        points = hkl.load(file_path)[:,0:ndim]
    elif '.txt' in file_path:
        
        with open(file_path, 'r') as file:
            first_line = file.readline()
        # Determine the delimiter based on the first line
        if ',' in first_line:
            delimiter = ','
        elif ' ' in first_line:
            delimiter = ' '
        else:
            raise NotImplementedError("Delimiter not identified")
        
        points = onp.loadtxt(file_path, delimiter=delimiter)[:,0:ndim]
    else:
        print("Wrong file format")
        sys.exit()
        
    return points

def trymakedir(path):
    """this function deals with common race conditions"""
    while True:
        if not os.path.exists(path):
            try:
                os.makedirs(path)
                break
            except OSError as e:
                if e.errno != 17:
                    raise
                # time.sleep might help here
                pass
        else:
            break
