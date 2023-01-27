import sys
import numpy as onp
import torch as np
import scipy as sp
from scipy.special import hankel1
import hickle as hkl
from joblib import Parallel, delayed


c = 3e8   #speed of light in vacuum, m/s
I = onp.identity(2).reshape(1,2,2) #identity matrix
#N = 1000 #number of scatterers
L = 100e-6 #box side length m
w = L/5


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

    def mean_DOS_measurements(self, measure_points, k0, alpha, radius, n_cpus=1, self_interaction= True):
        '''
        Computes the LDOS averaged at a list of measurement points, for TM and TE.
        This computation is a bit less expensive than the actual LDOS one,
        due to invariance of the trace under permutation and the use of Hadamard products
        measure_points      - (M,3)  coordinates of points where the LDOS is evaluated
        k0                  - (1)    frequency of source beam
        alpha               - (1)    bare static polarizability at given k0
        radius              - (1)    radius of the scatterers
        n_cpus              - (1)    number of cpus to multithread the generation of G0 over, defaults to 1
        self_interaction    - (bool) include or not self-interactions, defaults to True 
        '''

        Npoints = measure_points.shape[0]

        ### TM Calculation
        G0 = self.G0_TM(self.r, k0, alpha, n_cpus=n_cpus)
        G0.fill_diagonal_(-1)
        if self_interaction:
            # Add self-interaction
            volume = onp.pi*radius*radius
            dims = G0.shape[0]
            self_int_TM = alpha*k0*k0*np.eye(dims) * (-1/(k0*k0*volume) + 0.5j*sp.special.hankel1(1,k0*radius)/(k0*radius))
            G0 += self_int_TM
        # Invert the matrix 1 - k^2 alpha Green
        G0 *= -1
        Ainv = np.linalg.solve(G0, np.eye(len(G0), dtype=np.complex128))

        # Define the propagators from scatterers to measurement points
        G0_measure = self.G0_TM(measure_points, k0, alpha, n_cpus=n_cpus)
        #  Use cyclic invariance of the trace: tr(G A G^T) = tr (G^T G A)
        # symm_mat = onp.matmul(onp.transpose(G0_measure), G0_measure)
        #  Use that trace(A.B^T) = AxB with . = matrix product and x = Hadamard product, and that G^T G is symmetric,
        dos_factor_TM = ( np.matmul(G0_measure.t(), G0_measure) * Ainv ).sum()/Npoints
        dos_factor_TM *= 4.0 * k0*k0*alpha / onp.pi # For prefactor in systems invariant along z, see https://www.sciencedirect.com/science/article/pii/S1569441007000387
        dos_factor_TM = np.imag(dos_factor_TM)

        ### TE calculation
        G0 = self.G0_TE(None, k0, alpha, n_cpus=n_cpus)
        G0.fill_diagonal_(-1)
        if self_interaction:
            # Add self-interaction
            dims = G0.shape[0]
            self_int_TE = alpha*k0*k0*np.eye(dims) * (-1/(k0*k0*volume) + 0.25j*sp.special.hankel1(1,k0*radius)/(k0*radius))
            G0 += self_int_TE
        # Invert the matrix 1 - k^2 alpha Green
        G0 *= -1
        Ainv = np.linalg.solve(G0, np.eye(len(G0), dtype=np.complex128))

        # Define the propagators from scatterers to measurement points
        G0_measure = self.G0_TE(measure_points, k0, alpha, n_cpus=n_cpus)
        #  Use cyclic invariance of the trace: tr(G A G^T) = tr (G^T G A)
        # symm_mat = onp.matmul(onp.transpose(G0_measure), G0_measure)
        #  Use that trace(A.B^T) = AxB with . = matrix product and x = Hadamard product, and that G^T G is symmetric,
        dos_factor_TE = ( np.matmul(G0_measure.t(), G0_measure) * Ainv ).sum()/Npoints
        dos_factor_TE *= 4.0 * k0*k0* alpha / onp.pi
        dos_factor_TE = np.imag(dos_factor_TE)

        return dos_factor_TE, dos_factor_TM

    def LDOS_measurements(self, measure_points, k0, alpha, radius, n_cpus=1, self_interaction= True):
        '''
        Computes the LDOS at a list of measurement points, for TM and TE.
        This computation is fairly expensive, the number of measurement points should be 
        measure_points      - (M,3)  coordinates of points where the LDOS is evaluated
        k0                  - (1)    frequency of source beam
        alpha               - (1)    bare static polarizability at given k0
        radius              - (1)    radius of the scatterers
        n_cpus              - (1)    number of cpus to multithread the generation of G0 over, defaults to 1
        self_interaction    - (bool) include or not self-interactions, defaults to True 
        '''

        ### TM Calculation
        G0 = self.G0_TM(self.r, k0, alpha, n_cpus=n_cpus)
        G0.fill_diagonal_(-1)
        if self_interaction:
            # Add self-interaction
            volume = onp.pi*radius*radius
            dims = G0.shape[0]
            self_int_TM = alpha*k0*k0*np.eye(dims) * (-1/(k0*k0*volume) + 0.5j*sp.special.hankel1(1,k0*radius)/(k0*radius))
            G0 += self_int_TM
        # Invert the matrix 1 - k^2 alpha Green
        G0 *= -1
        Ainv = np.linalg.solve(G0, np.eye(len(G0), dtype=np.complex128))

        # Define the propagators from scatterers to measurement points
        G0_measure = self.G0_TM(measure_points, k0, alpha, n_cpus=n_cpus)
        # ldos_factor = onp.diagonal(onp.matmul(onp.matmul(G0_measure, Ainv),onp.transpose(G0_measure)))
        # Can be made better considering it's a diagonal https://stackoverflow.com/questions/17437817/python-how-to-get-diagonalab-without-having-to-perform-ab
        ldos_factor_TM = np.einsum('ij, ji->i',np.matmul(G0_measure, np.tensor(Ainv)), (G0_measure).t() )
        ldos_factor_TM *= 4.0 * k0*k0*alpha / onp.pi
        ldos_factor_TM = np.imag(ldos_factor_TM)

        ### TE calculation
        G0 = self.G0_TE(None, k0, alpha, n_cpus=n_cpus)
        G0.fill_diagonal_(-1)
        if self_interaction:
            # Add self-interaction
            dims = G0.shape[0]
            self_int_TE = alpha*k0*k0*np.eye(dims) * (-1/(k0*k0*volume) + 0.25j*sp.special.hankel1(1,k0*radius)/(k0*radius))
            G0 += self_int_TE
        # Invert the matrix 1 - k^2 alpha Green
        G0 *= -1
        Ainv = np.linalg.solve(G0, np.eye(len(G0), dtype=np.complex128))

        # Define the propagators from scatterers to measurement points
        G0_measure = self.G0_TE(measure_points, k0, alpha, n_cpus=n_cpus)
        # ldos_factor = onp.diagonal(onp.matmul(onp.matmul(G0_measure, Ainv),onp.transpose(G0_measure)))
        # Can be made better considering it's a diagonal https://stackoverflow.com/questions/17437817/python-how-to-get-diagonalab-without-having-to-perform-ab
        ldos_factor_TE = np.einsum('ij, ji->i',np.matmul(G0_measure, np.tensor(Ainv)), (G0_measure).t() )
        ldos_factor_TE *= 4.0 * k0*k0*alpha / onp.pi
        ldos_factor_TE = np.imag(ldos_factor_TE)

        return ldos_factor_TE, ldos_factor_TM