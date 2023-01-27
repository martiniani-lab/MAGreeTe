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
I = onp.identity(2).reshape(1,2,2) #identity matrix
#N = 1000 #number of scatterers



class Transmission2D:
    

    def __init__(self, points, source='beam'):
        self.r = points.reshape(-1,2)
        self.N = self.r.shape[0]
        self.source = source

    def greensTE(self,r,k0):
        R = onp.linalg.norm(r,axis=-1)
        RxR = r.numpy()[:,0:2].reshape(-1,1,2)*r.numpy()[:,0:2].reshape(-1,2,1)
        RxR /= (R*R).reshape(-1,1,1)
        R *= k0
        return np.tensor(0.25j*((I-RxR)*hankel1(0,R).reshape(-1,1,1)-(I-2*RxR)*(hankel1(1,R)/R).reshape(-1,1,1)),dtype=np.complex128)

    def greensTM(self,r,k0):
        R = onp.linalg.norm(r,axis=-1)
        return np.tensor(0.25j*hankel1(0,R*k0),dtype=np.complex128)

    def generate_source(self, points, k0, thetas):
        if self.source == 'beam':
            print('Calculating Beam Source')
            E0j = np.zeros((points.shape[0],len(thetas)),dtype=np.complex128)
            u = np.zeros((2,len(thetas)))
            for idx in range(len(thetas)):
                theta = thetas[idx]
                cost, sint = onp.cos(-theta),onp.sin(-theta)
                u[:,idx] = np.tensor([sint, cost])
                rot = np.tensor([[cost,-sint],[sint,cost]])
                rrot = np.matmul(rot,np.tensor(points.T)).T #(rparallel, rperp)
                a = 2*rrot[:,1]/(w*w*k0)
                E0j[:,idx] = np.exp(1j*rrot[:,0]*k0-(rrot[:,1]**2/(w*w*(1+1j*a))))/np.sqrt(1+1j*a)
        return E0j, u
 
    def calc_EM(self,points, EkTE, EkTM, k0, alpha, thetas, n_cpus=1):
        points = np.tensor(points)
        E0j, u = self.generate_source(points, k0, thetas)
        EkTM_ = np.matmul(self.G0_TM(points, k0, alpha, n_cpus=n_cpus), EkTM) + E0j
        E0j = E0j.reshape(points.shape[0],1,len(thetas))*u
        EkTE_ = np.matmul(self.G0_TE(points, k0, alpha, n_cpus=n_cpus), EkTE).reshape(points.shape[0],2,-1) + E0j 
        return EkTE_, EkTM_
   
    def run_EM(self, k0, alpha, thetas, radius, n_cpus=1, self_interaction=True):

        ### TM calculation
        E0j, u = self.generate_source(self.r, k0, thetas)
        G0 = self.G0_TM(self.r, k0, alpha, n_cpus=n_cpus)
        G0.fill_diagonal_(-1)
        if self_interaction:
            # Add self-interaction
            volume = onp.pi*radius*radius
            dims = G0.shape[0]
            self_int_TM = alpha*k0*k0*np.eye(dims) * (-1/(k0*k0*volume) + 0.5j*sp.special.hankel1(1,k0*radius)/(k0*radius))
            G0 += self_int_TM
        # Solve
        EkTM = np.linalg.solve(G0,-E0j)
        
        ### TE calculation
        E0j = E0j.reshape(self.N,1,len(thetas))*u
        G0 = self.G0_TE(None, k0, alpha, n_cpus=n_cpus)
        G0.fill_diagonal_(-1)
        if self_interaction:
            # Add self-interaction
            dims = G0.shape[0]
            self_int_TE = alpha*k0*k0*np.eye(dims) * (-1/(k0*k0*volume) + 0.25j*sp.special.hankel1(1,k0*radius)/(k0*radius))
            G0 += self_int_TE
        # Solve
        EkTE = np.linalg.solve(G0, -E0j.reshape(2*self.N,-1)) 
        return EkTE, EkTM

    def G0_TM(self, points, k0, alpha, n_cpus=1):
        #Green's function
        print('Calculating TM greens function')
        G0 = Parallel(n_jobs=n_cpus, require='sharedmem')(delayed(self.greensTM)(self.r.reshape(-1,2)-rr,k0) for rr in points)
        G0 = np.vstack(G0) #shape is (N,N)
        #Construct matrix form
        G0 *= alpha*k0*k0
        return G0

    def G0_TE(self, points, k0, alpha, n_cpus=1):
        #Green's function
        if points == None:
            points_ = self.r
        else:
            points_ = points
        print('Calculating TE greens function')
        G0 = Parallel(n_jobs=n_cpus, require='sharedmem')(delayed(self.greensTE)(self.r.reshape(-1,2)-rr,k0) for rr in points_)
        G0 = np.stack(G0) #shape is (N,N,2,2)
        #Construct matrix form
        if points == None:
            for idx in range(self.N):
                G0[idx,idx,:,:] = 0
        G0 = np.transpose(G0,1,2).reshape(2*G0.shape[0],2*G0.shape[1]).to(np.complex128)
        G0 *= alpha*k0*k0
        return G0
