import numpy as onp
import torch as np
import scipy as sp

import matplotlib.pyplot as plt
import matplotlib.colors as clr

import cmasher as cmr # https://github.com/1313e/CMasher

c = 3e8   #speed of light in vacuum, m/s

def alpha_cold_atoms_2d(k0range, omega0 = 3e15, Gamma = 5e16):
    '''
    Typical polarizability of cold atoms, using an elastically bound electron model for the dielectric constant, in 2d space.
    Arguments:
    k0range: array of k values, in rad/m
    omega0: bare resonance pulsation, in rad/s
    Gamma: bare linewidth, in rad/s
    '''

    return -2*Gamma/(omega0*(k0range*k0range-omega0*omega0/(c*c)+0.5j*Gamma*k0range*k0range/omega0))

def alpha_cold_atoms_3d(k0range, omega0 = 3e15, Gamma = 5e16):
    '''
    Typical polarizability of cold atoms, using an elastically bound electron model for the dielectric constant, in 3d space.
    Arguments:
    k0range: array of k values, in rad/m
    omega0: bare resonance pulsation, in rad/s
    Gamma: bare linewidth, in rad/s
    '''

    omegarange = k0range * c
    omega0sq = omega0*omega0
    return -4*onp.pi*(c**3)*Gamma/(omega0sq*(omegarange*omegarange-omega0sq+1j*Gamma*omegarange*omegarange*omegarange/omega0sq))

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

def plot_transmission_angularbeam(k0range, L, thetas, intensity, file_name_root, appended_string=None):
    '''
    Plots one of the transmission_angularbeam plots given 
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
    ax.set_rmin(10.0)
    ax.set_rticks([20,40])
    ax.set_axis_off()
    cbar = fig.colorbar(pc)
    cbar.ax.tick_params(labelsize=24)
    plt.savefig(file_name_root+'_transmission_angularbeam_'+appended_string+'.png', bbox_inches = 'tight',dpi=100, pad_inches = 0.1)
    plt.close()
