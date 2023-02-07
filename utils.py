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
    '''
    Typical polarizability of cold atoms, using an elastically bound electron model for the dielectric constant, in 2d space.
    Arguments:
    k0range: array of k values, in rad/m
    omega0: bare resonance pulsation, in rad/s
    Gamma: bare linewidth, in rad/s
    Lfactor: conversion factor for lengths, the values above being given for L = 100 µm
    '''

    return (-2*Gamma/(omega0*(k0range*k0range-omega0*omega0/(c*c)+0.5j*Gamma*k0range*k0range/omega0)))/Lfactor**2

def alpha_cold_atoms_3d(k0range, omega0 = 3e15, Gamma = 5e16, Lfactor = 1e-4):
    '''
    Typical polarizability of cold atoms, using an elastically bound electron model for the dielectric constant, in 3d space.
    Arguments:
    k0range: array of k values, in rad/m
    omega0: bare resonance pulsation, in rad/s
    Gamma: bare linewidth, in rad/s
    Lfactor: conversion factor for lengths, the values above being given for L = 100 µm
    '''

    omegarange = k0range * c
    omega0sq = omega0*omega0
    return (-4*onp.pi*(c**3)*Gamma/(omega0sq*(omegarange*omegarange-omega0sq+1j*Gamma*omegarange*omegarange*omegarange/omega0sq)))/Lfactor**3

def alpha_small_dielectric_object(refractive_n, volume):
    '''
    Bare static polarizability of a small dielectric object
    refractive_n: refractive index of the rods, can be complex
    volume: volume of the ball, in m^d
    '''
    
    # Define the dielectric constant from the refractive index
    epsilon = refractive_n**2
    delta_epsilon = epsilon - 1 

    return volume*delta_epsilon

def uniform_unit_disk_picking(n_points):
    '''
    Generates an (N,2) tensor of n_points random points with a flat distribution inside the unit disk
    '''
    
    U1 = onp.random.uniform(size = n_points)
    U2 = onp.random.uniform(size = n_points)
    X = onp.sqrt(U2) * onp.cos(2 * onp.pi * U1)
    Y = onp.sqrt(U2) * onp.sin(2 * onp.pi * U1)

    points = np.tensor(onp.vstack((X,Y)))

    return points.t()


def uniform_unit_ball_picking(n_points, dim):
    '''
    Generates an (N,dim) tensor of n_points random points with a flat distribution inside the unit dim-ball
    https://mathworld.wolfram.com/BallPointPicking.html
    '''

    normals = np.normal(0,1, size=(n_points, dim))
    exps = np.empty((n_points, 1))
    exps.exponential_()
    exps = np.sqrt(exps)
    proxies =  np.cat((normals, exps),1)
    
    points = normals/(np.linalg.norm(proxies, axis=-1)).reshape(-1,1,1)

    return points


def plot_transmission_angularbeam(k0range, L, thetas, intensity, file_name_root, appended_string=''):
    '''
    Plots one a radial version of the frequency-angle transmission plot given 
    k0range: list of wave vector moduli, in rad/m
    L: system sidelength, in m
    thetas: list of angles used for the orientation of the laser, in radians
    intensity: the relevant field intensity
    file_name_root: prepended to the name of the file
    appended_string: possible postfix for the name of the file, e.g. "TM" or "TE"
    '''
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
    '''
    Plots one a flattened version of the frequency-angle transmission plot given 
    k0range: list of wave vector moduli, in rad/m
    L: system sidelength, in m
    thetas: list of angles used for the orientation of the laser, in radians
    intensity: the relevant field intensity
    file_name_root: prepended to the name of the file
    appended_string: possible postfix for the name of the file, e.g. "TM" or "TE"
    '''
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

def plot_singlebeam_angular_frequency_plot(k0range, L, thetas, intensity, file_name_root, appended_string=''):
    '''
    Plots specific intensity for a single beam, in a radial frequency-angle plot 
    k0range: list of wave vector moduli, in rad/m
    L: system sidelength, in m
    thetas: list of angles used for the orientation of the laser, in radians
    intensity: the relevant field intensity
    file_name_root: prepended to the name of the file
    appended_string: possible postfix for the name of the file, e.g. "TM" or "TE"
    '''
    fig, ax = plt.subplots(subplot_kw={'projection':'polar'})
    freqs = onp.real(k0range*L/(2*onp.pi))
    total_ = intensity[:,:,0]
    pc = ax.pcolormesh(thetas,freqs,total_,norm=clr.LogNorm(vmin=total_.min(),vmax=total_.max()), cmap=cmr.ember)
    fig.colorbar(pc)
    plt.savefig(file_name_root+'_transmission_angular'+appended_string+'.png', bbox_inches = 'tight',dpi=100, pad_inches = 0)
    plt.close()

def plot_3d_points(points, file_name):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(points[:,0], points[:,1], points[:,2])

    plt.savefig(file_name+'_3dplot.png')

def plot_2d_points(points, file_name):
    fig = plt.figure()
    ax = fig.gca()

    ax.scatter(points[:,0], points[:,1])

    plt.savefig(file_name+'_2dplot.png')

def plot_LDOS_2D(ldos_change,k0_,ngridx,ngridy,file_name,my_dpi=1, appended_string=''):

    fig = plt.figure(figsize=(ngridx/my_dpi,ngridy/my_dpi), dpi=my_dpi)
    ax = plt.gca()
    pc=ax.imshow(ldos_change.numpy().reshape(ngridy,ngridx),cmap=cmr.iceburn, vmin=-1.0, vmax=1.0)
    ax.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
    plt.savefig(file_name+'_k0_'+str(k0_)+'_ldos_capped'+appended_string+'.png', bbox_inches='tight', pad_inches=0., dpi=my_dpi)
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