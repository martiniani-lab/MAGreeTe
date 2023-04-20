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

    def torch_greensTE(self, r, k0,periodic = '', regularize = False, radius = 0.0):
        '''
        Torch implementation of the TE Green's function, taking tensors as entries
        r          - (M,2)      distances to propagate over
        k0         - (1)        wave-vector of source beam in vacuum
        periodic   - str        change boundary conditions: '' = free, ('x', 'y', 'xy') = choices of possible periodic directions
        regularize - bool       bring everything below a scatterer radius to the center value, to be consistent with approximations and avoid divergences
        '''

        N = r.shape[0]
        M = r.shape[1]
        if periodic == 'y':
            r[:,:,1] += 0.5
            r[:,:,1] %= 1
            r[:,:,1] -= 0.5
        elif periodic == 'x':
            r[:,:,0] += 0.5
            r[:,:,0] %= 1
            r[:,:,0] -= 0.5
            
        R = np.linalg.norm(r,axis=-1)
        RxR = r[:,:,0:2].reshape(N,M,1,2)*r[:,:,0:2].reshape(N,M,2,1)
        RxR /= (R*R).reshape(N,M,1,1)
        R *= k0

        if regularize:
            R = np.where(R < radius, 0.0, R)

        return 0.25j*((I-RxR)*self.torch_hankel1(0,R).reshape(N,M,1,1)-(I-2*RxR)*(self.torch_hankel1(1,R)/R).reshape(N,M,1,1))

    def torch_greensTM(self, r, k0, periodic='', regularize = False, radius = 0.0):
        '''
        Torch implementation of the TM Green's function, taking tensors as entries
        r          - (M,2)      distances to propagate over
        k0         - (1)        wave-vector of source beam in vacuum
        periodic   - str        change boundary conditions: '' = free, ('x', 'y', 'xy') = choices of possible periodic directions
        regularize - bool       bring everything below a scatterer radius to the center value, to be consistent with approximations and avoid divergences
        radius     - (1)        considered scatterer radius, only used for regularization
        '''

        if periodic == 'y':
            r[:,:,1] += 0.5
            r[:,:,1] %= 1
            r[:,:,1] -= 0.5
        elif periodic == 'x':
            r[:,:,0] += 0.5
            r[:,:,0] %= 1
            r[:,:,0] -= 0.5
        R = np.linalg.norm(r, axis = -1)

        if regularize:
            R = np.where(R < radius, 0.0, R)

        return 0.25j*self.torch_hankel1(0,R*k0)

    def generate_source(self, points, k0, thetas, w, print_statement=''):
        '''
        Generates the EM field of a source at a set of points

        points      - (M,2)      coordinates of points
        k0          - (1)        frequency of source beam
        thetas      - (Ndirs)    propagation directions for the source
        w           - (1)        beam waist for beam sources
        '''
        if self.source == 'beam':
            k0_ = onp.round(k0/(2.0*onp.pi),1)
            print('Calculating Beam Source at k0L/2pi = '+str(k0_)+' ('+print_statement+')')
            E0j = np.zeros((points.shape[0],len(thetas)),dtype=np.complex128)
            u = np.zeros((2,len(thetas)))
            for idx, theta in enumerate(thetas):
                cost, sint = onp.cos(-theta),onp.sin(-theta)
                u[:,idx] = np.tensor([sint, cost])
                rot = np.tensor([[cost,-sint],[sint,cost]])
                rrot = np.matmul(rot,points.T).T #(rparallel, rperp)
                a = 2*rrot[:,1]/(w*w*k0)
                E0j[:,idx] = np.exp(1j*rrot[:,0]*k0-(rrot[:,1]**2/(w*w*(1+1j*a))))/np.sqrt(1+1j*a)
        elif self.source == 'plane':
            k0_ = onp.round(k0/(2.0*onp.pi),1)
            print('Calculating Plane Source at k0L/2pi = '+str(k0_)+' ('+print_statement+')')
            E0j = np.zeros((points.shape[0],len(thetas)),dtype=np.complex128)
            u = np.zeros((2,len(thetas)))
            for idx in range(len(thetas)):
                theta = thetas[idx]
                cost, sint = onp.cos(-theta),onp.sin(-theta)
                u[:,idx] = np.tensor([sint, cost])
                rot = np.tensor([[cost,-sint],[sint,cost]])
                rrot = np.matmul(rot,points.T).T #(rparallel, rperp)
                E0j[:,idx] = np.exp(1j*rrot[:,0]*k0)
        return E0j, u
 
    def calc_EM(self,points, EkTE, EkTM, k0, alpha, thetas, beam_waist, regularize = False, radius = 0.0):
        '''
        Calculates the EM field at a set of measurement points

        points     - (M,2)      coordinates of all measurement points
        EkTE       - (N*2)      TE polarization component of the electromagnetic field at each scatterer
        EkTM       - (N)        TM polarization component of the electromagnetic field at each scatterer
        k0         - (1)        frequency being measured
        alpha      - (1)        bare static polarizability at given k0
        thetas     - (Ndirs)    propagation directions for the source
        beam_waist - (1)        beam waist
        regularize - bool       bring everything below a scatterer radius to the center value, to be consistent with approximations and avoid divergences
        radius     - (1)        considered scatterer radius, only used for regularization 
        '''

        points = np.tensor(points)
        E0j, u = self.generate_source(points, k0, thetas, beam_waist, print_statement='calc')

        EkTM_ = np.matmul(alpha*k0*k0* self.G0_TM(points, k0, print_statement='calc', regularize=regularize, radius=radius), EkTM) + E0j
        E0j = E0j.reshape(points.shape[0],1,len(thetas))*u
        EkTE_ = np.matmul(alpha*k0*k0* self.G0_TE(points, k0, print_statement='calc', regularize=regularize, radius=radius), EkTE).reshape(points.shape[0],2,-1) + E0j 

        # Take care of cases in which measurement points are exactly scatterer positions
        for j in np.argwhere(np.isnan(EkTM_[:,0])):
            if regularize:
                # If overlap, will just return the closest one
                possible_idx = np.nonzero(np.linalg.norm(self.r-points[j], axis = -1) <= radius)
                if possible_idx.shape[0] > 1:
                    idx = np.argmin(np.linalg.norm(self.r-points[j], axis = -1))
                else:
                    idx = possible_idx
                EkTM_[j] = EkTM[idx]
                EkTE_[j] = EkTE[idx]
            else:
                EkTM_[j] = EkTM[np.nonzero(np.prod(self.r-points[j]==0,axis=-1))]
                EkTE_[j] = EkTE[np.nonzero(np.prod(self.r-points[j]==0,axis=-1))]
        return EkTE_, EkTM_
   
    def run_EM(self, k0, alpha, thetas, radius, beam_waist, self_interaction=True):
        '''
        Solves the EM field at each scatterer

        k0                  - (1)           frequency being measured
        alpha               - (1)           bare static polarizability at given k0
        thetas              - (Ndirs)       propagation directions for the source
        radius              - (1)           radius of scatterers, used in self-interaction
        beam_waist          - (1)           beam waist of Gaussian beam source
        self_interaction    - (bool)        include or not self-interactions, defaults to True 
        '''

        ### TM calculation
        # Define the matrix M_tensor = I_tensor - k^2 alpha Green_tensor
        E0j, u = self.generate_source(self.r, k0, thetas, beam_waist, print_statement='run')
        M_tensor = -alpha*k0*k0* self.G0_TM(self.r, k0, print_statement='run')
        M_tensor.fill_diagonal_(1)
        if self_interaction:
            # Add self-interaction, (M_tensor)_ii = 1 - k^2 alpha self_int
            volume = onp.pi*radius*radius
            dims = M_tensor.shape[0]
            self_int_TM = np.eye(dims) * (-1/(k0*k0*volume) + 0.5j*sp.special.hankel1(1,k0*radius)/(k0*radius))
            M_tensor -= alpha*k0*k0*self_int_TM
        # Solve M_tensor.Ek = E0j
        EkTM = np.linalg.solve(M_tensor,E0j)
        
        ### TE calculation
        # Define the matrix M_tensor = I_tensor - k^2 alpha Green_tensor
        E0j = E0j.reshape(self.N,1,len(thetas))*u
        M_tensor = -alpha*k0*k0* self.G0_TE(None, k0, print_statement='run')
        M_tensor.fill_diagonal_(1)
        if self_interaction:
            # Add self-interaction
            dims = M_tensor.shape[0]
            self_int_TE = np.eye(dims) * (-1/(k0*k0*volume) + 0.25j*sp.special.hankel1(1,k0*radius)/(k0*radius))
            M_tensor -= alpha*k0*k0*self_int_TE
        # Solve M_tensor.Ek = E0j
        EkTE = np.linalg.solve(M_tensor, E0j.reshape(2*self.N,-1)) 
        return EkTE, EkTM
    
    def calc_EM_ss(self, points, k0, alpha, thetas, beam_waist, regularize = False, radius = 0.0):
        '''
        Calculates the EM field at a set of measurement points, using a single-scattering approximation

        points     - (M,2)      coordinates of all measurement points
        k0         - (1)        frequency being measured
        alpha      - (1)        bare static polarizability at given k0
        thetas     - (Ndirs)    propagation directions for the source
        beam_waist - (1)        beam waist
        regularize - bool       bring everything below a scatterer radius to the center value, to be consistent with approximations and avoid divergences
        radius     - (1)        considered scatterer radius, only used for regularization 
        '''

        points = np.tensor(points)
        E0_meas, u_meas = self.generate_source(points, k0, thetas, beam_waist, print_statement='calc_ss')
        E0_scat, u_scat = self.generate_source(self.r, k0, thetas, beam_waist, print_statement='calc_ss')
        EkTM_ = np.matmul(alpha*k0*k0* self.G0_TM(points, k0, print_statement='calc_ss', regularize=regularize, radius=radius), E0_scat) + E0_meas
        
        E0_meas = E0_meas.reshape(points.shape[0],1,len(thetas))*u_meas
        E0_scat = E0_scat.reshape(self.r.shape[0],1,len(thetas))*u_scat
        E0_scat = E0_scat.reshape(2*self.r.shape[0],-1)        
        EkTE_ = np.matmul(alpha*k0*k0* self.G0_TE(points, k0, print_statement='calc_ss', regularize=regularize, radius=radius), E0_scat).reshape(points.shape[0],2,-1) + E0_meas
        
        # Take care of cases in which measurement points are exactly scatterer positions
        for j in np.argwhere(np.isnan(EkTM_[:,0])):
            if regularize:
                # If overlap, will just return the closest one
                possible_idx = np.nonzero(np.linalg.norm(self.r-points[j], axis = -1) <= radius)
                if possible_idx.shape[0] > 1:
                    idx = np.argmin(np.linalg.norm(self.r-points[j], axis = -1))
                else:
                    idx = possible_idx
                EkTM_[j] = E0_meas[idx]
                EkTE_[j] = E0_meas[idx]
            else:
                EkTM_[j] = E0_meas[np.nonzero(np.prod(self.r-points[j]==0,axis=-1))]
                EkTE_[j] = E0_meas[np.nonzero(np.prod(self.r-points[j]==0,axis=-1))]
        return EkTE_, EkTM_

    def G0_TM(self, points, k0, print_statement='', regularize = False, radius=0.0):
        '''
        Returns a Green's tensor linking all points to all scatterers for the TM polarization
        '''
        #Green's function
        k0_ = onp.round(k0/(2.0*onp.pi),1)
        print("Calculating TM Green's function at k0L/2pi = "+str(k0_)+' ('+print_statement+')')
        G0 = self.torch_greensTM(points.reshape(-1,1,2) - self.r.reshape(1,-1,2), k0, regularize=regularize, radius=radius)
        return G0

    def G0_TE(self, points, k0, print_statement='', regularize = False, radius = 0.0):
        '''
        Returns a Green's tensor linking all points to all scatterers for the TE polarization
        '''
        #Green's function
        if points == None:
            points_ = self.r
        else:
            points_ = points
        k0_ = onp.round(k0/(2.0*onp.pi),1)
        print("Calculating TE Green's function at k0L/2pi = "+str(k0_)+' ('+print_statement+')')
        G0 = self.torch_greensTE(points_.reshape(-1,1,2) - self.r.reshape(1,-1,2), k0, regularize=regularize, radius=radius)
        #Construct matrix form
        if points == None:
            for idx in range(self.N):
                G0[idx,idx,:,:] = 0
        G0 = np.transpose(G0,1,2).reshape(2*G0.shape[0],2*G0.shape[1]).to(np.complex128)
        return G0

    def mean_DOS_measurements(self, measure_points, k0, alpha, radius, file_name, self_interaction= True, regularize = False, discard_absorption = False):
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
        # Define the matrix M_tensor = I_tensor - k^2 alpha Green_tensor
        M_tensor = -alpha*k0*k0* self.G0_TM(self.r, k0, print_statement='DOS inverse')
        M_tensor.fill_diagonal_(1)
        if self_interaction:
            # Add self-interaction, (M_tensor)_ii = 1 - k^2 alpha self_int
            volume = onp.pi*radius*radius
            dims = M_tensor.shape[0]
            self_int_TM = np.eye(dims) * (-1/(k0*k0*volume) + 0.5j*sp.special.hankel1(1,k0*radius)/(k0*radius))
            M_tensor -= alpha*k0*k0*self_int_TM
        # Compute W_tensor = inverse(M_tensor)
        W_tensor = np.linalg.solve(M_tensor, np.eye(len(M_tensor), dtype=np.complex128))

        # Define the propagators from scatterers to measurement points
        G0_measure = self.G0_TM(measure_points, k0, print_statement='DOS measure', regularize=regularize, radius=radius)
        # Check for measurement points falling exactly on scatterers
        for j in np.argwhere(np.isnan(G0_measure)):
            point_idx = j[0]
            scatter_idx = j[1]
            # At scatterers, replace G0(r_i, r_i) by self-interaction
            G0_measure[point_idx][scatter_idx] = 0
            if self_interaction:
                volume = onp.pi*radius*radius
                self_int_TM = (-1/(k0*k0*volume) + 0.5j*sp.special.hankel1(1,k0*radius)/(k0*radius))
                G0_measure[point_idx][scatter_idx] += self_int_TM
        #  Use cyclic invariance of the trace: tr(G A G^T) = tr (G^T G A)
        # symm_mat = onp.matmul(onp.transpose(G0_measure), G0_measure)
        #  Use that trace(A.B^T) = AxB with . = matrix product and x = Hadamard product, and that G^T G is symmetric,
        dos_factor_TM = ( np.matmul(G0_measure.t(), G0_measure) * W_tensor ).sum()/Npoints

        if discard_absorption:
            # Discard the imaginary part of alpha, only for the last part of the calculation https://www.jpier.org/pier/pier.php?paper=19111801
            alpha_ = onp.real(alpha)
        else:
            alpha_ = alpha
        dos_factor_TM *= 4.0 * k0*k0*alpha_ # For prefactor in systems invariant along z, see https://www.sciencedirect.com/science/article/pii/S1569441007000387
        dos_factor_TM = np.imag(dos_factor_TM)

        ### TE calculation
        # Define the matrix M_tensor = I_tensor - k^2 alpha Green_tensor
        M_tensor = -alpha*k0*k0*self.G0_TE(None, k0, print_statement='DOS inverse')
        M_tensor.fill_diagonal_(1)
        if self_interaction:
            # Add self-interaction, (M_tensor)_ii = 1 - k^2 alpha self_int
            dims = M_tensor.shape[0]
            self_int_TE = np.eye(dims) * (-1/(k0*k0*volume) + 0.25j*sp.special.hankel1(1,k0*radius)/(k0*radius))
            M_tensor -= alpha*k0*k0*self_int_TE
        # Compute W_tensor = inverse(M_tensor)
        W_tensor = np.linalg.solve(M_tensor, np.eye(len(M_tensor), dtype=np.complex128))

        # Define the propagators from scatterers to measurement points
        G0_measure = self.G0_TE(measure_points, k0, print_statement='DOS measure', regularize=regularize, radius=radius)
        # Check for measurement points falling exactly on scatterers
        for j in np.argwhere(np.isnan(G0_measure)):
            point_idx = j[0]
            scatter_idx = j[1]
            # At scatterers, replace G0(r_i, r_i) by self-interaction
            G0_measure[point_idx][scatter_idx] = 0
            if self_interaction:
                volume = onp.pi*radius*radius
                self_int_TE = (-1/(k0*k0*volume) + 0.25j*sp.special.hankel1(1,k0*radius)/(k0*radius))
                G0_measure[point_idx][scatter_idx] += self_int_TE
        #  Use cyclic invariance of the trace: tr(G A G^T) = tr (G^T G A)
        # symm_mat = onp.matmul(onp.transpose(G0_measure), G0_measure)
        #  Use that trace(A.B^T) = AxB with . = matrix product and x = Hadamard product, and that G^T G is symmetric,
        dos_factor_TE = ( np.matmul(G0_measure.t(), G0_measure) * W_tensor ).sum()/Npoints
        if discard_absorption:
            # Discard the imaginary part of alpha, only for the last part of the calculation https://www.jpier.org/pier/pier.php?paper=19111801
            alpha_ = onp.real(alpha)
        else:
            alpha_ = alpha
        dos_factor_TE *= 4.0 * k0*k0* alpha_
        dos_factor_TE = np.imag(dos_factor_TE)

        return dos_factor_TE, dos_factor_TM

    def LDOS_measurements(self, measure_points, k0, alpha, radius, self_interaction= True, regularize = False, discard_absorption = False):
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
        # Define the matrix M_tensor = I_tensor - k^2 alpha Green_tensor
        M_tensor = -alpha*k0*k0* self.G0_TM(self.r, k0, print_statement='LDOS inverse')
        M_tensor.fill_diagonal_(1)
        if self_interaction:
            # Add self-interaction, (M_tensor)_ii = 1 - k^2 alpha self_int
            volume = onp.pi*radius*radius
            dims = M_tensor.shape[0]
            self_int_TM = np.eye(dims) * (-1/(k0*k0*volume) + 0.5j*sp.special.hankel1(1,k0*radius)/(k0*radius))
            M_tensor -= alpha*k0*k0*self_int_TM
        # Compute W_tensor = inverse(M_tensor)
        W_tensor = np.linalg.solve(M_tensor, np.eye(len(M_tensor), dtype=np.complex128))

        # Define the propagators from scatterers to measurement points
        G0_measure = self.G0_TM(measure_points, k0, print_statement='LDOS measure', regularize=regularize, radius=radius)
        # Check for measurement points falling exactly on scatterers
        for j in np.argwhere(np.isnan(G0_measure)):
            point_idx = j[0]
            scatter_idx = j[1]
            # At scatterers, replace G0(r_i, r_i) by self-interaction
            G0_measure[point_idx][scatter_idx] = 0
            if self_interaction:
                volume = onp.pi*radius*radius
                self_int_TM = (-1/(k0*k0*volume) + 0.5j*sp.special.hankel1(1,k0*radius)/(k0*radius))
                G0_measure[point_idx][scatter_idx] += self_int_TM
        # ldos_factor = onp.diagonal(onp.matmul(onp.matmul(G0_measure, W_tensor),onp.transpose(G0_measure)))
        # Can be made better considering it's a diagonal https://stackoverflow.com/questions/17437817/python-how-to-get-diagonalab-without-having-to-perform-ab
        ldos_factor_TM = np.einsum('ij, ji->i',np.matmul(G0_measure, W_tensor), (G0_measure).t() )
        if discard_absorption:
            # Discard the imaginary part of alpha, only for the last part of the calculation https://www.jpier.org/pier/pier.php?paper=19111801
            alpha_ = onp.real(alpha)
        else:
            alpha_ = alpha
        ldos_factor_TM *= 4.0 * k0*k0*alpha_
        ldos_factor_TM = np.imag(ldos_factor_TM)

        ### TE calculation
        # Define the matrix M_tensor = I_tensor - k^2 alpha Green_tensor
        M_tensor = -alpha*k0*k0* self.G0_TE(None, k0, print_statement='LDOS inverse')
        M_tensor.fill_diagonal_(1)
        if self_interaction:
            # Add self-interaction, (M_tensor)_ii = 1 - k^2 alpha self_int
            dims = M_tensor.shape[0]
            self_int_TE = np.eye(dims) * (-1/(k0*k0*volume) + 0.25j*sp.special.hankel1(1,k0*radius)/(k0*radius))
            M_tensor -= alpha*k0*k0*self_int_TE
        # Compute W_tensor = inverse(M_tensor)
        W_tensor = np.linalg.solve(M_tensor, np.eye(len(M_tensor), dtype=np.complex128))

        # Define the propagators from scatterers to measurement points
        G0_measure = self.G0_TE(measure_points, k0, print_statement='LDOS measure', regularize=regularize, radius=radius)
        # Check for measurement points falling exactly on scatterers
        for j in np.argwhere(np.isnan(G0_measure)):
            point_idx = j[0]
            scatter_idx = j[1]
            # At scatterers, replace G0(r_i, r_i) by self-interaction
            G0_measure[point_idx][scatter_idx] = 0
            if self_interaction:
                volume = onp.pi*radius*radius
                self_int_TE = (-1/(k0*k0*volume) + 0.25j*sp.special.hankel1(1,k0*radius)/(k0*radius))
                G0_measure[point_idx][scatter_idx] += self_int_TE
        # ldos_factor = onp.diagonal(onp.matmul(onp.matmul(G0_measure, Ainv),onp.transpose(G0_measure)))
        # Can be made better considering it's a diagonal https://stackoverflow.com/questions/17437817/python-how-to-get-diagonalab-without-having-to-perform-ab
        ldos_factor_TE = np.einsum('ij, ji->i',np.matmul(G0_measure, W_tensor), (G0_measure).t() )
        if discard_absorption:
            # Discard the imaginary part of alpha, only for the last part of the calculation https://www.jpier.org/pier/pier.php?paper=19111801
            alpha_ = onp.real(alpha)
        else:
            alpha_ = alpha
        ldos_factor_TE *= 4.0 * k0*k0*alpha_
        ldos_factor_TE = np.imag(ldos_factor_TE)
        ldos_factor_TE = ldos_factor_TE.reshape(M,2,-1)
        ldos_factor_TE = np.sum(ldos_factor_TE, 1)

        return ldos_factor_TE, ldos_factor_TM

    def compute_eigenvalues_and_scatterer_LDOS(self, k0, alpha, radius, file_name, self_interaction= True, write_eigenvalues=True):
        '''
        Computes the eigenvalues of the Green's matrix, and the corresponding LDOS at scatterers, for TM and TE.
        This computation is way less expensive than the other LDOS, due to simple dependence on the eigenvalues
        measure_points      - (M,3)  coordinates of points where the LDOS is evaluated
        k0                  - (1)    frequency of source beam
        alpha               - (1)    bare static polarizability at given k0
        radius              - (1)    radius of the scatterers
        self_interaction    - (bool) include or not self-interactions, defaults to True 
        '''

        Npoints = self.r.shape[0]
        print(self.r.shape)
        k0_ = onp.round(k0/(2.0*onp.pi),1)
        print("Computing spectrum and scatterer LDOS using "+str(Npoints)+" points at k0L/2pi = "+str(k0_))

        ### TM Calculation
        # Define the matrix M_tensor = I_tensor - k^2 alpha Green_tensor
        M_tensor = -alpha*k0*k0*self.G0_TM(self.r, k0, print_statement='DOS eigvals')
        M_tensor.fill_diagonal_(1)
        if self_interaction:
            # Add self-interaction, (M_tensor)_ii = 1 - k^2 alpha self_int
            volume = onp.pi*radius*radius
            dims = M_tensor.shape[0]
            self_int_TM = np.eye(dims) * (-1/(k0*k0*volume) + 0.5j*sp.special.hankel1(1,k0*radius)/(k0*radius))
            M_tensor -= alpha*k0*k0*self_int_TM
        # Compute the spectrum of the M_tensor
        lambdas = np.linalg.eigvals(M_tensor)

        if write_eigenvalues:
            onp.savetxt(file_name+'_lambdas_'+str(k0_)+'_TM.csv', onp.stack([np.real(lambdas).numpy(), np.imag(lambdas).numpy()]).T)

        # Compute the trace part here
        dos_factor_TM = ((1 - lambdas)**2 / lambdas).sum()/Npoints
        dos_factor_TM *= 4.0 / ( k0**2 * alpha) # For prefactor in systems invariant along z, see https://www.sciencedirect.com/science/article/pii/S1569441007000387
        dos_factor_TM = np.imag(dos_factor_TM)

        ### TE calculation
        # Define the matrix M_tensor = I_tensor - k^2 alpha Green_tensor
        M_tensor = -alpha*k0*k0*self.G0_TE(None, k0, print_statement='DOS eigvals')
        M_tensor.fill_diagonal_(1)
        if self_interaction:
            # Add self-interaction, (M_tensor)_ii = 1 - k^2 alpha self_int
            dims = M_tensor.shape[0]
            self_int_TE = np.eye(dims) * (-1/(k0*k0*volume) + 0.25j*sp.special.hankel1(1,k0*radius)/(k0*radius))
            M_tensor -= alpha*k0*k0*self_int_TE
        # Compute the spectrum of the M_tensor
        lambdas = np.linalg.eigvals(M_tensor)

        if write_eigenvalues:
            onp.savetxt(file_name+'_lambdas_'+str(k0_)+'_TE.csv', onp.stack([np.real(lambdas).numpy(), np.imag(lambdas).numpy()]).T)

        # Compute the trace part here
        dos_factor_TE = ((1 - lambdas)**2 / lambdas).sum()/Npoints
        dos_factor_TE *= 4.0 / ( k0**2 * alpha) # For prefactor in systems invariant along z, see https://www.sciencedirect.com/science/article/pii/S1569441007000387
        dos_factor_TE = np.imag(dos_factor_TE)

        return dos_factor_TE, dos_factor_TM