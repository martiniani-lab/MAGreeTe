import numpy as onp
import torch as np
import scipy as sp
import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.colors as clr

import cmasher as cmr # https://github.com/1313e/CMasher

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


    return points


def plot_transmission_angularbeam(k0range, L, thetas, intensity, file_name_root, appended_string=''):
    """
    Plots one a radial version of the frequency-angle transmission plot given 
    k0range: list of wave vector moduli, in rad/m
    L: system sidelength, in m
    thetas: list of angles used for the orientation of the laser, in radians
    intensity: the relevant field intensity
    file_name_root: prepended to the name of the file
    appended_string: possible postfix for the name of the file, e.g. "TM" or "TE"
    """
    freqs = onp.real(k0range*L/(2*onp.pi))
    total_ = onp.sum(intensity*onp.diag(onp.ones(intensity.shape[-1])),axis=1)
    fig, ax = plt.subplots(subplot_kw={'projection':'polar'})
    pc = ax.pcolormesh(thetas,freqs,total_,norm=clr.LogNorm(vmin=1e-2,vmax=1e0), cmap=cmr.ember)#cmap=cmr.torch) #cmap='inferno')
    #ax.set_rmin(10.0)
    #ax.set_rticks([20,40])
    ax.set_axis_off()
    cbar = fig.colorbar(pc)
    cbar.ax.tick_params(labelsize=24)
    plt.savefig(file_name_root+'_transmission_angularbeam_'+appended_string+'.png', bbox_inches = 'tight',dpi=100, pad_inches = 0.1)
    plt.close()

def plot_transmission_flat(k0range, L, thetas, intensity, file_name_root, appended_string=''):
    """
    Plots one a flattened version of the frequency-angle transmission plot given 
    k0range: list of wave vector moduli, in rad/m
    L: system sidelength, in m
    thetas: list of angles used for the orientation of the laser, in radians
    intensity: the relevant field intensity
    file_name_root: prepended to the name of the file
    appended_string: possible postfix for the name of the file, e.g. "TM" or "TE"
    """
    freqs = onp.real(k0range*L/(2*onp.pi))
    total_ = onp.sum(intensity*onp.diag(onp.ones(intensity.shape[-1])),axis=1)
    fig = plt.figure()
    ax = fig.gca()
    pc = ax.imshow(total_[:,:int(total_.shape[1]/2)], norm=clr.LogNorm(vmin=1e-2,vmax=1e0), cmap=cmr.ember, extent =[0,180,freqs[0],freqs[-1]], origin='lower')
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel('Frequency')
    ax.set_aspect(180/(freqs[-1] - freqs[0]))
    fig.colorbar(pc)
    plt.savefig(file_name_root+'_transmission_beam_'+appended_string+'.png', bbox_inches = 'tight',dpi=100, pad_inches = 0.1)
    plt.close()

def plot_transmission_linear(k0range, L,x, intensity, file_name_root,cmap='viridis', appended_string=''):
    """
    Plots one a flattened version of the frequency-angle transmission plot given 
    k0range: list of wave vector moduli, in rad/m
    L: system sidelength, in m
    thetas: list of angles used for the orientation of the laser, in radians
    intensity: the relevant field intensity
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

def plot_singlebeam_angular_frequency_plot(k0range, L, thetas, intensity, file_name_root, appended_string=''):
    """
    Plots specific intensity for a single beam, in a radial frequency-angle plot 
    k0range: list of wave vector moduli, in rad/m
    L: system sidelength, in m
    thetas: list of angles used for the orientation of the laser, in radians
    intensity: the relevant field intensity
    file_name_root: prepended to the name of the file
    appended_string: possible postfix for the name of the file, e.g. "TM" or "TE"
    """
    fig, ax = plt.subplots(subplot_kw={'projection':'polar'})
    freqs = onp.real(k0range*L/(2*onp.pi))
    total_ = intensity[:,:,0]
    pc = ax.pcolormesh(thetas,freqs,total_,norm=clr.LogNorm(vmin=total_.min(),vmax=total_.max()), cmap=cmr.ember)
    fig.colorbar(pc)
    plt.savefig(file_name_root+'_transmission_angular'+appended_string+'.png', bbox_inches = 'tight',dpi=100, pad_inches = 0)
    plt.close()

def plot_full_fields(field, ngridx, ngridy, k0_, angle_, intensity_fields, amplitude_fields, phase_fields, file_name_root, appended_string='', my_dpi = 1):

    if intensity_fields:

        intensity = onp.absolute(field)**2

        fig = plt.figure(figsize=(ngridx/my_dpi,ngridy/my_dpi), dpi=my_dpi)
        ax = plt.gca()
        pc = ax.imshow(intensity,cmap='magma' ,norm=clr.LogNorm(vmin=1e-3,vmax=1e0))
        ax.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
        plt.savefig(file_name_root+'_log_capped_intensity_k0'+str(k0_)+'_angle_'+str(angle_)+appended_string+'.png', bbox_inches = 'tight',dpi=my_dpi, pad_inches = 0)
        plt.close()

        fig = plt.figure(figsize=(ngridx/my_dpi,ngridy/my_dpi), dpi=my_dpi)
        ax = plt.gca()
        pc = ax.imshow(intensity,cmap='magma', vmin=1e-3,vmax=1e0)
        ax.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
        plt.savefig(file_name_root+'_linear_capped_intensity_k0'+str(k0_)+'_angle_'+str(angle_)+appended_string+'.png', bbox_inches = 'tight',dpi=my_dpi, pad_inches = 0)
        plt.close()

    if amplitude_fields:
        amplitude = onp.real(field)

        fig = plt.figure(figsize=(ngridx/my_dpi,ngridy/my_dpi), dpi=my_dpi)
        ax = plt.gca()
        pc = ax.imshow(amplitude,cmap=cmr.redshift, vmin=-1e0,vmax=1e0)
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

    plt.savefig(file_name+'_3dplot.png')

def plot_2d_points(points, file_name):
    fig = plt.figure(figsize=(10,10),dpi=300)
    ax = fig.gca()

    ax.scatter(points[:,0], points[:,1], s = 2)

    plt.savefig(file_name+'_2dplot.png')

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

def plot_averaged_DOS(k0range, L, DOS, file_name, DOS_type, appended_string=''):
    # Angular-averaged transmission
    fig = plt.figure()
    ax = fig.gca()
    freqs = onp.real(k0range*L/(2*onp.pi))
    ax.plot(freqs, DOS)
    ax.set_xlabel(r'$k_0L/2\pi$')
    ax.set_ylabel(r'$\delta\varrho$')
    ax.legend()
    plt.savefig(file_name+'_'+DOS_type+'_avg'+appended_string+'.png', bbox_inches = 'tight',dpi=100, pad_inches = 0)
    plt.close()


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