import numpy as onp
import torch as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import colorsys
import hickle as hkl
import sys
import os
from utils import alpha_cold_atoms_2d
from Transmission2D import Transmission2D

w = 2.1e-5 #beam waist m
L = 100e-6 #box side length m

def main():
    phi = 0.6
    N = 4096
    size_ratio = 1.0
    a = -1.0
    k = 80
    ndim = 2

    if ndim == 2:
        volume = L*L*phi/N
        radius = onp.sqrt(volume/onp.pi )
    else:
        volume = L*L*L*phi/N
        radius = onp.cbrt(volume * 3.0 / (4.0 * onp.pi))

    k0range = onp.arange(40,81)*64/128*2*onp.pi/L
    alpharange = alpha_cold_atoms_2d(k0range)

    dname = '../HPY'+str(ndim)+'D/phi'+str(phi)+'/a'+str(a)+'/'
    file_name = 'HPY'+str(ndim)+'D_phi'+str(phi)+'_a'+str(a)+'_N'+str(N)+'_K'+str(k)
    #dname = '../hyperalg/sandbox/init_HSL/'
    #file_name = 'init_HSL2D_phi'+str(phi)+'_N'+str(N)
    file_name += '_points'
    #file_name += '_dual'
    #file_name = 'square'

    Ntheta = 360
    thetas = onp.arange(Ntheta)/Ntheta*2*np.pi
    meas_points = 2*L*onp.vstack([onp.cos(thetas),onp.sin(thetas)]).T

    i=0
    points = hkl.load(dname+file_name+'_'+str(i)+'.hkl')
    points = np.tensor(points[:,0:ndim]-0.5,dtype=np.double)
    idx = np.nonzero(np.linalg.norm(points,axis=-1)<=0.5)
    points = np.squeeze(points[idx])
    points *= L
    
    solver = Transmission2D(points)
    ETEall = []
    ETMall = []
    for k0, alpha in zip(k0range,alpharange):
        EjTE, EjTM = solver.run_EM(k0, alpha, thetas, radius)
        k0_ = onp.round(onp.real(k0*L/(2*onp.pi)),1)
        params = [alpha, k0]
        hkl.dump([onp.array(EjTE), onp.array(EjTM), onp.array(params),onp.array(points), onp.array(thetas)],file_name+'_Ek_k0_'+str(k0_)+'_'+str(i)+'.hkl')

        EkTE, EkTM = solver.calc_EM(meas_points, EjTE, EjTM, k0, alpha, thetas)
        EkTE = np.linalg.norm(EkTE,axis=1)
        ETEall.append(EkTE.numpy())
        ETMall.append(EkTM.numpy())
    ETEall = onp.array(ETEall)
    ETMall = onp.array(ETMall)

    TEtotal = onp.absolute(ETEall)**2
    TMtotal = onp.absolute(ETMall)**2

    freqs = onp.real(k0range*L/(2*onp.pi))
    total_ = onp.sum(TMtotal*onp.diag(onp.ones(TMtotal.shape[-1])),axis=1)
    fig, ax = plt.subplots(subplot_kw={'projection':'polar'})
    pc = ax.pcolormesh(thetas,freqs,total_,norm=clr.LogNorm(vmin=1e-2,vmax=1e0), cmap='inferno')
    ax.set_rmin(10.0)
    ax.set_rticks([20,40])
    ax.set_axis_off()
    cbar = fig.colorbar(pc)
    cbar.ax.tick_params(labelsize=24)
    plt.savefig(file_name+'_transmission_angularbeam_TM.png', bbox_inches = 'tight',dpi=100, pad_inches = 0.1)
    plt.close()

    total_ = onp.sum(TEtotal*onp.diag(onp.ones(TEtotal.shape[-1])),axis=1)
    fig, ax = plt.subplots(subplot_kw={'projection':'polar'})
    pc = ax.pcolormesh(thetas,freqs,total_, norm=clr.LogNorm(vmin=1e-2,vmax=1e0),cmap='inferno')
    ax.set_rmin(10.0)
    ax.set_rticks([20,40])
    ax.set_axis_off()
    cbar = fig.colorbar(pc)
    cbar.ax.tick_params(labelsize=24)
    plt.savefig(file_name+'_transmission_angularbeam_TE.png', bbox_inches = 'tight',dpi=100, pad_inches = 0.1)
    plt.close()       

if __name__ == '__main__':
    np.set_num_threads(32)
    np.device("cpu")
    main()
    sys.exit()


