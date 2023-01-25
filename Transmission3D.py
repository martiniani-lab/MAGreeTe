import sys
import numpy as onp
import torch as np
import scipy as sp
from scipy.special import hankel1
import hickle as hkl
from joblib import Parallel, delayed


c = 3e8   #speed of light in vacuum, m/s
omega0 = 3e15 #resonance frequency 1/s
Gamma = 5e16 #linewidth 1/s
w = 2.1e-5 #beam waist m
L = 100e-6 #box side length mi
I = np.tensor(onp.identity(3)).reshape(1,3,3) #identity matrix
#N = 1000 #number of scatterers



class Transmission2D:
    

    def __init__(self, points, source='beam'):
        self.r = points.reshape(-1,3)
        self.N = self.r.shape[0]
        self.source = source

    def greens(self,r,k0):
        R = np.linalg.norm(r,axis=-1)
        RxR = r.reshape(-1,1,3)*r.reshape(-1,3,1)
        return (I-RxR-(I-3*RxR)*(1/(1j*k0*R)+(k0*R)**-2))*np.exp(1j*k0*R)/(4*np.pi*R)

    def generate_source(self, points, k0, u, p):
        '''
        Generates the EM field of a source at a set of points

        points - (M,3)      coordinates of points
        k0     - (1)        frequency of source beam
        u      - (Ndirs, 3) propagation directions for the source
        p      - (Ndirs, 3) polarization directions for the source
        '''
        if self.source == 'beam':
            print('Calculating Beam Source')
            rpara = np.matmul(points,u.T)
            rperp = np.linalg.norm(points.reshape(-1,3,1) - rpara.reshape(points.shape[0],1,u.shape[0])*u.T.reshape(1,3,-1),axis=1)
            a = 2*rperp/(w*w*k0)
            E0j = np.exp(1j*rpara*k0-(rperp**2/(w*w*(1+1j*a))))/np.sqrt(1+1j*a)
            phi = np.arctan2(p[:,1], p[:,0]) #arctan(y/x)
            theta = np.arccos(p[:,2]) #arccos(z/r), r=1 for unit vector
            pvec = np.stack([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), p[:,2]])
        return E0j.reshape(points.shape[0],1,-1)*pvec.reshape(1,3,1)
 
    def calc(self, points, Ek, k0, alpha, u, p):
        '''
        Calculates the EM field at a set of measurement points

        points - (M,3)      coordinates of all measurement points
        Ek     - (N*3)      electromagnetic field at each scatterer
        k0     - (1)        frequency being measured
        alpha  - (1)        bare static polarizability at given k0
        u      - (Ndirs, 3) propagation directions for the source
        p      - (Ndirs, 3) polarization directions for the source
        '''
        points = np.tensor(points)
        
        # ensure u and p are all unit vectors
        u = np.tensor(u)/np.linalg.norm(u,axis=-1)
        p = np.tensor(p)/np.linalg.norm(p,axis=-1)
        
        # check polarization is orthogonal to propagation
        assert np.sum(np.absolute(np.sum(u*p,axis=-1))) == 0

        # generate source field for measurement points
        E0j = self.generate_source(points, k0, u, p) #(M,3,Ndirs)
        
        # calculate Ek field at all measurement points
        Ek = np.matmul(self.G0(points, k0, alpha), Ek).reshape(points.shape[0],3,-1) + E0j 
        return Ek
   
    def run(self, k0, alpha, u, p):
        '''
        Solves the EM field at each scatterer

        k0     - (1)        frequency being measured
        alpha  - (1)        bare static polarizability at given k0
        u      - (Ndirs, 3) propagation directions for the source
        p      - (Ndirs, 3) polarization directions for the source
        '''

        # generate source field for scatterer positions
        E0j = self.generate_source(self.r, k0, u, p) #(N,3,Ndirs)
        
        # calculate Ek field at each scatterer position
        G0 = self.G0(None, k0, alpha)
        G0.fill_diagonal_(-1)
        Ek = np.linalg.solve(G0, -E0j.reshape(3*self.N,-1)) 
        return Ek

    def G0(self, points, k0, alpha):
        '''
        Generate the Green's tensor for a set of positions

        points - (N,3)      set of point positions, None indicates the saved point pattern
        k0     - (1)        frequency being measured
        alpha  - (1)        bare static polarizability at given k0
        '''

        # check if None
        if points == None:
            points_ = self.r
        else:
            points_ = points
        print('Calculating greens tensor')
        # populate Green's tensor
        G0 = Parallel(n_jobs=32, require='sharedmem')(delayed(self.greensTE)(self.r.reshape(-1,3)-rr,k0) for rr in points_)
        G0 = np.stack(G0) #shape is (N,N,3,3)

        # replace NaN entries resulting from divergence (r-r'=0)
        if points == None:
            for idx in range(self.N):
                G0[idx,idx,:,:] = 0

        # shape into (N*3,N*3)
        G0 = np.transpose(G0,1,2).reshape(3*G0.shape[0],3*G0.shape[1]).to(np.complex128)
        G0 *= alpha*k0*k0
        return G0
