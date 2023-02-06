import sys
import numpy as onp
import torch as np
import scipy as sp
from scipy.special import hankel1
import hickle as hkl


I = np.tensor(onp.identity(3)).reshape(1,3,3) #identity matrix
#N = 1000 #number of scatterers



class Transmission3D:
    

    def __init__(self, points, source='beam'):
        self.r = points.reshape(-1,3)
        self.N = self.r.shape[0]
        self.source = source
    
    def greens(self,r,k0):
        N = r.shape[0]
        M = r.shape[1]
        R = np.linalg.norm(r,axis=-1).reshape(N,M,1,1)
        RxR = r.reshape(N,M,1,3)*r.reshape(N,M,3,1)
        RxR /= R*R
        return (I-RxR-(I-3*RxR)*(1/(1j*k0*R)+(k0*R)**-2))*np.exp(1j*k0*R)/(4*onp.pi*R)

    def generate_source(self, points, k0, u, p, w, print_statement = ''):
        '''
        Generates the EM field of a source at a set of points

        points - (M,3)      coordinates of points
        k0     - (1)        frequency of source beam
        u      - (Ndirs, 3) propagation directions for the source
        p      - (Ndirs, 3) polarization directions for the source
        '''
        if self.source == 'beam':
            k0_ = onp.round(k0/(2.0*onp.pi),1)
            print('Calculating Beam Source at k0L/2pi = '+str(k0_)+' ('+print_statement+')')
            rpara = np.matmul(points,u.T)
            rperp = np.linalg.norm(points.reshape(-1,3,1) - rpara.reshape(points.shape[0],1,u.shape[0])*u.T.reshape(1,3,-1),axis=1)
            a = 2*rperp/(w*w*k0)
            E0j = np.exp(1j*rpara*k0-(rperp**2/(w*w*(1+1j*a))))/np.sqrt(1+1j*a)
            phi = np.arctan2(p[:,1], p[:,0]) #arctan(y/x)
            theta = np.arccos(p[:,2]) #arccos(z/r), r=1 for unit vector
            pvec = np.stack([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), p[:,2]])
        return E0j.reshape(points.shape[0],1,-1)*pvec.reshape(1,3,-1)
 
    def calc(self, points, Ek, k0, alpha, u, p, beam_waist):
        '''
        Calculates the EM field at a set of measurement points

        points - (M,3)      coordinates of all measurement points
        Ek     - (N*3)      electromagnetic field at each scatterer
        k0     - (1)        frequency being measured
        alpha  - (1)        bare static polarizability at given k0
        u      - (Ndirs, 3) propagation directions for the source
        p      - (Ndirs, 3) polarization directions for the source
        beam_waist - (1)    beam waist
        '''
        points = np.tensor(points)
        
        # ensure u and p are all unit vectors
        u /= np.linalg.norm(u,axis=-1).reshape(-1,1)
        p /= np.linalg.norm(p,axis=-1).reshape(-1,1)
        
        # check polarization is orthogonal to propagation
        assert np.sum(np.absolute(np.sum(u*p,axis=-1))) == 0

        # generate source field for measurement points
        E0j = self.generate_source(points, k0, u, p, beam_waist, print_statement='calc') #(M,3,Ndirs)
        
        # calculate Ek field at all measurement points
        Ek_ = np.matmul(self.G0(points, k0, alpha, print_statement='calc'), Ek).reshape(points.shape[0],3,-1) + E0j 

        # Take care of cases in which measurement points are exactly scatterer positions
        for j in np.argwhere(np.isnan(Ek_[:,0])):
            Ek_[j] = Ek[np.nonzero(np.prod(self.r-points[j]==0,axis=-1))]
        return Ek_
   
    def run(self, k0, alpha, u, p, radius, beam_waist, self_interaction=True):
        '''
        Solves the EM field at each scatterer

        k0                  - (1)           frequency being measured
        alpha               - (1)           bare static polarizability at given k0
        u                   - (Ndirs, 3)    propagation directions for the source
        p                   - (Ndirs, 3)    polarization directions for the source
        self_interaction    - (bool)        include or not self-interactions, defaults to True 
        '''

        # generate source field for scatterer positions
        E0j = self.generate_source(self.r, k0, u, p, beam_waist, print_statement='run') #(N,3,Ndirs)
        
        # calculate Ek field at each scatterer position
        G0 = self.G0(None, k0, alpha, print_statement='run')
        G0.fill_diagonal_(-1)
        if self_interaction:
            # Add self-interaction
            dims = G0.shape[0]
            volume = 4*onp.pi*(radius**3)/3
            self_int = (alpha/volume) * np.eye(dims)*((2.0/3.0)*onp.exp(1j*k0*radius)*(1- 1j*k0*radius) - 1.0) 
            G0 += self_int
        Ek = np.linalg.solve(G0, -E0j.reshape(3*self.N,-1)) 
        return Ek

    def G0(self, points, k0, alpha, print_statement=''):
        '''
        Generate the Green's tensor for a set of positions

        points          - (N,3)      set of point positions, None indicates the saved point pattern
        k0              - (1)        frequency being measured
        alpha           - (1)        bare static polarizability at given k0
        print_statement - str        disambiguating string used when printing (default = empty)
        '''

        # check if None
        if points == None:
            points_ = self.r
        else:
            points_ = points
        k0_ = onp.round(k0/(2.0*onp.pi),1)
        print("Calculating Green's function at k0L/2pi = "+str(k0_)+' ('+print_statement+')')
        # populate Green's tensor
        G0 = self.greens(points_.reshape(-1,1,3)-self.r.reshape(1,-1,3),k0) #shape is (M,N,3,3)

        # replace NaN entries resulting from divergence (r-r'=0)
        if points == None:
            for idx in range(self.N):
                G0[idx,idx,:,:] = 0

        # shape into (N*3,N*3)
        G0 = np.transpose(G0,1,2).reshape(3*G0.shape[0],3*G0.shape[1]).to(np.complex128)
        G0 *= alpha*k0*k0
        return G0

    def mean_DOS_measurements(self, measure_points, k0, alpha, radius, self_interaction= True):
        '''
        Computes the LDOS averaged at a list of measurement points.
        This computation is a bit less expensive than the actual LDOS one,
        due to invariance of the trace under permutation and the use of Hadamard products
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


        G0 = self.G0(None, k0, alpha, print_statement='DOS inverse')
        G0.fill_diagonal_(-1)
        if self_interaction:
            # Add self-interaction
            dims = G0.shape[0]
            volume = 4*onp.pi*(radius**3)/3
            self_int = (alpha/volume) * np.eye(dims)*((2.0/3.0)*onp.exp(1j*k0*radius)*(1- 1j*k0*radius) - 1.0) 
            G0 += self_int
        # Invert the matrix 1 - k^2 alpha Green
        G0 *= -1
        Ainv = np.linalg.solve(G0, np.eye(len(G0), dtype=np.complex128))

        # Define the propagators from scatterers to measurement points
        G0_measure = self.G0(measure_points, k0, alpha, print_statement='DOS measure')
        #  Use cyclic invariance of the trace: tr(G A G^T) = tr (G^T G A)
        # symm_mat = onp.matmul(onp.transpose(G0_measure), G0_measure)
        #  Use that trace(A.B^T) = AxB with . = matrix product and x = Hadamard product, and that G^T G is symmetric,
        dos_factor = ( np.matmul(G0_measure.t(), G0_measure) * Ainv ).sum()/Npoints
        # Discard the imaginary part of alpha, only for the last part of the calculation https://www.jpier.org/pier/pier.php?paper=19111801
        alpha_ = onp.real(alpha)
        dos_factor *= 2.0*k0*alpha_
        dos_factor = np.imag(dos_factor)

        return dos_factor

    def LDOS_measurements(self, measure_points, k0, alpha, radius, self_interaction= True):
        '''
        Computes the LDOS at a list of measurement points
        This computation is fairly expensive, the number of measurement points should be small to avoid saturating resources
        NB: This form of the calculation is only valid in the lossless case, alpha real.
        Imaginary parts of alpha lead to diverging parts of the DOS close to scatterers, and will be discarded.
        measure_points      - (M,3)  coordinates of points where the LDOS is evaluated
        k0                  - (1)    frequency of source beam
        alpha               - (1)    bare static polarizability at given k0
        radius              - (1)    radius of the scatterers
        self_interaction    - (bool) include or not self-interactions, defaults to True 
        '''

        M = measure_points.shape[0]

        G0 = self.G0(None, k0, alpha, print_statement='LDOS inverse')
        G0.fill_diagonal_(-1)
        if self_interaction:
            # Add self-interaction
            dims = G0.shape[0]
            volume = 4*onp.pi*(radius**3)/3
            self_int = (alpha/volume) * np.eye(dims)*((2.0/3.0)*onp.exp(1j*k0*radius)*(1- 1j*k0*radius) - 1.0) 
            G0 += self_int
        # Invert the matrix 1 - k^2 alpha Green
        G0 *= -1
        Ainv = np.linalg.solve(G0, np.eye(len(G0), dtype=np.complex128))

        # Define the propagators from scatterers to measurement points
        G0_measure = self.G0(measure_points, k0, alpha, print_statement='LDOS measure')
        # ldos_factor = onp.diagonal(onp.matmul(onp.matmul(G0_measure, Ainv),onp.transpose(G0_measure)))
        # Can be made better considering it's a diagonal https://stackoverflow.com/questions/17437817/python-how-to-get-diagonalab-without-having-to-perform-ab
        ldos_factor = np.einsum('ij, ji->i',np.matmul(G0_measure, Ainv), (G0_measure).t() )
        # Discard the imaginary part of alpha, only for the last part of the calculation https://www.jpier.org/pier/pier.php?paper=19111801
        alpha_ = onp.real(alpha)
        ldos_factor *= 2.0*k0*alpha_
        ldos_factor = np.imag(ldos_factor)
        ldos_factor = ldos_factor.reshape(M,3,-1)
        ldos_factor = np.sum(ldos_factor, 1)

        return ldos_factor
