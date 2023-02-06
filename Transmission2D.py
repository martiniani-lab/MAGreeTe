import sys
import numpy as onp
import torch as np
import scipy as sp
from scipy.special import hankel1
import hickle as hkl


I = np.tensor(onp.identity(2)).reshape(1,1,2,2) #identity matrix


class Transmission2D:
    

    def __init__(self, points, source='beam'):
        self.r = points.reshape(-1,2)
        self.N = self.r.shape[0]
        self.source = source

    def torch_hankel1(self,nu, x):
        '''
        Torch implementation of hankel1(nu,x), for nu = 0 or 1.
        Uses the fact that Hankel(nu,z) = J(nu,z) + i Y(nu,z).
        Note: Y0 and Y1 do not appear in the official PyTorch documentation https://pytorch.org/docs/stable/special.html
        However, they are actually implemented, https://pytorch.org/cppdocs/api/file_torch_csrc_api_include_torch_special.h.html
        '''

        if nu == 0:
            # H0(z) = J0(z) + i Y0(z)
            return np.special.bessel_j0(x) +1j*np.special.bessel_y0(x)
        elif nu == 1:
            # H1(z) = J1(z) + i Y1(z)
            return np.special.bessel_j1(x) +1j*np.special.bessel_y1(x)
        else:
            exit("torch Hankel function only implemented for orders 0 and 1!")

    def torch_greensTE(self, r, k0):
        '''
        Torch implementation of the TE Green's function, taking tensors as entries
        '''
        N = r.shape[0]
        M = r.shape[1]
        R = np.linalg.norm(r,axis=-1)
        RxR = r[:,:,0:2].reshape(N,M,1,2)*r[:,:,0:2].reshape(N,M,2,1)
        RxR /= (R*R).reshape(N,M,1,1)
        R *= k0
        return 0.25j*((I-RxR)*self.torch_hankel1(0,R).reshape(N,M,1,1)-(I-2*RxR)*(self.torch_hankel1(1,R)/R).reshape(N,M,1,1))

    def torch_greensTM(self, r, k0):
        '''
        Torch implementation of the TM Green's function, taking tensors as entries
        '''
        R = np.linalg.norm(r, axis = -1)
        return 0.25j*self.torch_hankel1(0,R*k0)

    def generate_source(self, points, k0, thetas, w, print_statement=''):
        if self.source == 'beam':
            k0_ = onp.round(k0/(2.0*onp.pi),1)
            print('Calculating Beam Source at k0L/2pi = '+str(k0_)+' ('+print_statement+')')
            E0j = np.zeros((points.shape[0],len(thetas)),dtype=np.complex128)
            u = np.zeros((2,len(thetas)))
            for idx in range(len(thetas)):
                theta = thetas[idx]
                cost, sint = onp.cos(-theta),onp.sin(-theta)
                u[:,idx] = np.tensor([sint, cost])
                rot = np.tensor([[cost,-sint],[sint,cost]])
                rrot = np.matmul(rot,points.T).T #(rparallel, rperp)
                a = 2*rrot[:,1]/(w*w*k0)
                E0j[:,idx] = np.exp(1j*rrot[:,0]*k0-(rrot[:,1]**2/(w*w*(1+1j*a))))/np.sqrt(1+1j*a)
        return E0j, u
 
    def calc_EM(self,points, EkTE, EkTM, k0, alpha, thetas, beam_waist):
        points = np.tensor(points)
        E0j, u = self.generate_source(points, k0, thetas, beam_waist, print_statement='calc')
        EkTM_ = np.matmul(self.G0_TM(points, k0, alpha,print_statement='calc'), EkTM) + E0j
        E0j = E0j.reshape(points.shape[0],1,len(thetas))*u
        EkTE_ = np.matmul(self.G0_TE(points, k0, alpha, print_statement='calc'), EkTE).reshape(points.shape[0],2,-1) + E0j 
        # Take care of cases in which measurement points are exactly scatterer positions
        for j in np.argwhere(np.isnan(EkTM_[:,0])):
            EkTM_[j] = EkTM[np.nonzero(np.prod(self.r-points[j]==0,axis=-1))]
            EkTE_[j] = EkTE[np.nonzero(np.prod(self.r-points[j]==0,axis=-1))]
        return EkTE_, EkTM_
   
    def run_EM(self, k0, alpha, thetas, radius, beam_waist, self_interaction=True):

        ### TM calculation
        E0j, u = self.generate_source(self.r, k0, thetas, beam_waist, print_statement='run')
        G0 = self.G0_TM(self.r, k0, alpha, print_statement='run')
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
        G0 = self.G0_TE(None, k0, alpha, print_statement='run')
        G0.fill_diagonal_(-1)
        if self_interaction:
            # Add self-interaction
            dims = G0.shape[0]
            self_int_TE = alpha*k0*k0*np.eye(dims) * (-1/(k0*k0*volume) + 0.25j*sp.special.hankel1(1,k0*radius)/(k0*radius))
            G0 += self_int_TE
        # Solve
        EkTE = np.linalg.solve(G0, -E0j.reshape(2*self.N,-1)) 
        return EkTE, EkTM

    def G0_TM(self, points, k0, alpha, print_statement=''):
        #Green's function
        k0_ = onp.round(k0/(2.0*onp.pi),1)
        print("Calculating TM Green's function at k0L/2pi = "+str(k0_)+' ('+print_statement+')')
        G0 = self.torch_greensTM(points.reshape(-1,1,2) - self.r.reshape(1,-1,2), k0)
        #Construct matrix form
        G0 *= alpha*k0*k0
        return G0

    def G0_TE(self, points, k0, alpha, print_statement=''):
        #Green's function
        if points == None:
            points_ = self.r
        else:
            points_ = points
        k0_ = onp.round(k0/(2.0*onp.pi),1)
        print("Calculating TE Green's function at k0L/2pi = "+str(k0_)+' ('+print_statement+')')
        G0 = self.torch_greensTE(points_.reshape(-1,1,2) - self.r.reshape(1,-1,2), k0)
        #Construct matrix form
        if points == None:
            for idx in range(self.N):
                G0[idx,idx,:,:] = 0
        G0 = np.transpose(G0,1,2).reshape(2*G0.shape[0],2*G0.shape[1]).to(np.complex128)
        G0 *= alpha*k0*k0
        return G0

    def mean_DOS_measurements(self, measure_points, k0, alpha, radius, self_interaction= True):
        '''
        Computes the LDOS averaged at a list of measurement points, for TM and TE.
        This computation is a bit less expensive than the actual LDOS one,
        due to invariance of the trace under permutation and the use of Hadamard products.
        NB: This form of the calculation is only valid in the lossless case, alpha real.
        Imaginary parts of alpha lead to diverging parts of the DOS close to scatterers, and will be discarded.
        measure_points      - (M,3)  coordinates of points where the LDOS is evaluated
        k0                  - (1)    frequency of source beam
        alpha               - (1)    bare static polarizability at given k0
        radius              - (1)    radius of the scatterers
        self_interaction    - (bool) include or not self-interactions, defaults to True 
        '''

        Npoints = measure_points.shape[0]
        k0_ = onp.round(k0/(2.0*onp.pi),1)
        print("Computing mean DOS using "+str(Npoints)+" points at k0L/2pi = "+str(k0_))

        ### TM Calculation
        G0 = self.G0_TM(self.r, k0, alpha, print_statement='DOS inverse')
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
        G0_measure = self.G0_TM(measure_points, k0, alpha, print_statement='DOS measure')
        #  Use cyclic invariance of the trace: tr(G A G^T) = tr (G^T G A)
        # symm_mat = onp.matmul(onp.transpose(G0_measure), G0_measure)
        #  Use that trace(A.B^T) = AxB with . = matrix product and x = Hadamard product, and that G^T G is symmetric,
        dos_factor_TM = ( np.matmul(G0_measure.t(), G0_measure) * Ainv ).sum()/Npoints

        # Discard the imaginary part of alpha, only for the last part of the calculation https://www.jpier.org/pier/pier.php?paper=19111801
        alpha_ = onp.real(alpha)
        dos_factor_TM *= 4.0 * k0*k0*alpha_ / onp.pi # For prefactor in systems invariant along z, see https://www.sciencedirect.com/science/article/pii/S1569441007000387
        dos_factor_TM = np.imag(dos_factor_TM)

        ### TE calculation
        G0 = self.G0_TE(None, k0, alpha, print_statement='DOS inverse')
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
        G0_measure = self.G0_TE(measure_points, k0, alpha, print_statement='DOS measure')
        #  Use cyclic invariance of the trace: tr(G A G^T) = tr (G^T G A)
        # symm_mat = onp.matmul(onp.transpose(G0_measure), G0_measure)
        #  Use that trace(A.B^T) = AxB with . = matrix product and x = Hadamard product, and that G^T G is symmetric,
        dos_factor_TE = ( np.matmul(G0_measure.t(), G0_measure) * Ainv ).sum()/Npoints
        # Discard the imaginary part of alpha, only for the last part of the calculation https://www.jpier.org/pier/pier.php?paper=19111801
        alpha_ = onp.real(alpha)
        dos_factor_TE *= 4.0 * k0*k0* alpha_ / onp.pi
        dos_factor_TE = np.imag(dos_factor_TE)

        return dos_factor_TE, dos_factor_TM

    def LDOS_measurements(self, measure_points, k0, alpha, radius, self_interaction= True):
        '''
        Computes the LDOS at a list of measurement points, for TM and TE.
        This computation is fairly expensive, the number of measurement points should be small to avoid saturating resources.
        NB: This form of the calculation is only valid in the lossless case, alpha real.
        Imaginary parts of alpha lead to diverging parts of the DOS close to scatterers, and will be discarded.
        measure_points      - (M,3)  coordinates of points where the LDOS is evaluated
        k0                  - (1)    frequency of source beam
        alpha               - (1)    bare static polarizability at given k0
        radius              - (1)    radius of the scatterers
        self_interaction    - (bool) include or not self-interactions, defaults to True 
        '''

        M = measure_points.shape[0]

        ### TM Calculation
        G0 = self.G0_TM(self.r, k0, alpha, print_statement='LDOS inverse')
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
        G0_measure = self.G0_TM(measure_points, k0, alpha, print_statement='LDOS measure')
        # ldos_factor = onp.diagonal(onp.matmul(onp.matmul(G0_measure, Ainv),onp.transpose(G0_measure)))
        # Can be made better considering it's a diagonal https://stackoverflow.com/questions/17437817/python-how-to-get-diagonalab-without-having-to-perform-ab
        ldos_factor_TM = np.einsum('ij, ji->i',np.matmul(G0_measure, Ainv), (G0_measure).t() )
        # Discard the imaginary part of alpha, only for the last part of the calculation https://www.jpier.org/pier/pier.php?paper=19111801
        alpha_ = onp.real(alpha)
        ldos_factor_TM *= 4.0 * k0*k0*alpha_ / onp.pi
        ldos_factor_TM = np.imag(ldos_factor_TM)

        ### TE calculation
        G0 = self.G0_TE(None, k0, alpha, print_statement='LDOS inverse')
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
        G0_measure = self.G0_TE(measure_points, k0, alpha, print_statement='LDOS measure')
        # ldos_factor = onp.diagonal(onp.matmul(onp.matmul(G0_measure, Ainv),onp.transpose(G0_measure)))
        # Can be made better considering it's a diagonal https://stackoverflow.com/questions/17437817/python-how-to-get-diagonalab-without-having-to-perform-ab
        ldos_factor_TE = np.einsum('ij, ji->i',np.matmul(G0_measure, Ainv), (G0_measure).t() )
        # Discard the imaginary part of alpha, only for the last part of the calculation https://www.jpier.org/pier/pier.php?paper=19111801
        alpha_ = onp.real(alpha)
        ldos_factor_TE *= 4.0 * k0*k0*alpha_ / onp.pi
        ldos_factor_TE = np.imag(ldos_factor_TE)
        ldos_factor_TE = ldos_factor_TE.reshape(M,2,-1)
        ldos_factor_TE = np.sum(ldos_factor_TE, 1)

        return ldos_factor_TE, ldos_factor_TM
