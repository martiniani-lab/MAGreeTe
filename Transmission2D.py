import sys
import numpy as onp
import torch as np
import scipy as sp
from scipy.special import hankel1
import hickle as hkl

import utils

I = np.tensor(onp.identity(2)).reshape(1,1,2,2) #identity matrix

def self_interaction_integral_scalar(k0, radius, self_interaction_type = "Rayleigh"):
    
    volume = onp.pi * radius**2
    
    if self_interaction_type == "full":
        self_int_scalar = (-1/(k0*k0) + 0.5j * volume * sp.special.hankel1(1,k0*radius)/(k0*radius))
    elif self_interaction_type == "Rayleigh":
        self_int_scalar = (-1.0 * radius**2 / 4.0) * (2.0 * onp.euler_gamma - 1.0 + 2.0 * onp.log(k0 * radius / 2.0) - 1j * onp.pi)
    else:
        raise NotImplementedError
    
    return self_int_scalar

def self_interaction_integral_vector(k0, radius, self_interaction_type = "Rayleigh"):
    
    volume = onp.pi * radius**2
    
    if self_interaction_type == "full":
        self_int_vector = (-1/(k0*k0) + 0.25j* volume * sp.special.hankel1(1,k0*radius)/(k0*radius))
    elif self_interaction_type == "Rayleigh":
        self_int_vector = (-1/(2.0 *k0*k0) - (radius**2 / 8.0) * (2.0 * onp.euler_gamma - 1.0 + 2.0 * onp.log(k0 * radius / 2.0) - 1j * onp.pi))
    else:
        raise NotImplementedError
    
    return self_int_vector

class Transmission2D_vector:
    

    def __init__(self, points, source = "beam"):
        self.r = points.reshape(-1,2)
        self.N = self.r.shape[0]
        self.source = source
        
    def torch_hankel1(self, nu, x):
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

    def torch_greens(self, r, k0, periodic = '', regularize = False, radius = 0.0):
        '''
        Torch implementation of the TE Green's function, taking tensors as entries
        r          - (N,M,2)      distances to propagate over
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
    
    def generate_source(self, points, k0, thetas, w, print_statement=''):
        '''
        Generates the EM field of a source at a set of points

        points      - (M,2)      coordinates of points
        k0          - (1)        frequency of source beam
        thetas      - (Ndirs)    propagation directions for the source
        w           - (1)        beam waist for beam sources
        '''
        
        if self.source == "beam":
            # Collimated beam with zero curvature,
            # from solution of the the paraxial approximation of Maxwell-Helmholtz,
            # see https://en.wikipedia.org/wiki/Gaussian_beam
            
            k0_ = onp.round(k0/(2.0*onp.pi),1)
            print('Calculating Beam Source at k0L/2pi = '+str(k0_)+' ('+print_statement+')')
            
            E0 = np.zeros((points.shape[0],len(thetas)),dtype=np.complex128)
            u = np.zeros((2,len(thetas)))
            for idx, theta in enumerate(thetas):
                # Rotate the system by - theta rather than the expression of the source
                cost, sint = onp.cos(-theta),onp.sin(-theta)
                u[:,idx] = np.tensor([sint, cost])
                rot = np.tensor([[cost,-sint],[sint,cost]])
                rrot = np.matmul(rot,points.T).T #(rparallel, rperp)
                a = 2*rrot[:,1]/(w*w*k0)
                E0[:,idx] = np.exp(1j*rrot[:,0]*k0-(rrot[:,1]**2/(w*w*(1+1j*a))))/np.sqrt(1+1j*a)
                
            # TE is just TM but with in-plane polarization perpendicular to prop
            E0 = E0.reshape(points.shape[0],1,len(thetas))*u
                
        elif self.source == 'plane':
            # Infinitely extended Plane wave
            
            k0_ = onp.round(k0/(2.0*onp.pi),1)
            print('Calculating Plane Source at k0L/2pi = '+str(k0_)+' ('+print_statement+')')
            
            E0 = np.zeros((points.shape[0],len(thetas)),dtype=np.complex128)
            u = np.zeros((2,len(thetas)))
            for idx in range(len(thetas)):
                # Rotate the system by - theta rather than the expression of the source
                theta = thetas[idx]
                cost, sint = onp.cos(-theta),onp.sin(-theta)
                u[:,idx] = np.tensor([sint, cost])
                rot = np.tensor([[cost,-sint],[sint,cost]])
                rrot = np.matmul(rot,points.T).T #(rparallel, rperp)
                E0[:,idx] = np.exp(1j*rrot[:,0]*k0)
                
            # TE is just TM but with in-plane polarization perpendicular to prop
            E0 = E0.reshape(points.shape[0],1,len(thetas))*u
                
        elif self.source == 'point':
            # One electric point dipole emitting light at source_distance * L away
            source_distance = 2.0
            source_intensity = 1.0 * k0 * source_distance * 2.0 * onp.pi
            
            k0_ = onp.round(k0/(2.0*onp.pi),1)
            print('Calculating Point Source at k0L/2pi = '+str(k0_)+' ('+print_statement+')')
            
            E0 = np.zeros((points.shape[0],2,len(thetas)),dtype=np.complex128)
            u = np.zeros((2,len(thetas)))
            for idx in range(len(thetas)):
                theta = thetas[idx]
                cost, sint = onp.cos(-theta),onp.sin(-theta)
                u[:,idx] = np.tensor([sint, cost])
                rot = np.tensor([[cost,-sint],[sint,cost]])
                rrot = np.matmul(rot,points.T).T #(rparallel, rperp)
                source_location = source_distance * np.tensor([-1.0, 0.0])
                dipole_moment = onp.sqrt(source_intensity) * np.tensor([0.0,1.0])
                # Compute TE field in modified coordinates
                E0[:,:,idx] = np.matmul(self.torch_greens(rrot.reshape(-1,1,2) - source_location.reshape(1,-1,2), k0), dipole_moment.type(np.complex128)).squeeze()
                # Rotate the TE polarization at the end to match actual coordinates
                E0[:,:,idx] = np.matmul(rot.type(np.complex128).T, E0[:,:,idx].T).T

        elif self.source is None:
            E0 = np.zeros((points.shape[0],2,len(thetas)),dtype=np.complex128)

        else:
            raise NotImplementedError
                
        return E0
 
    def propagate(self, points, Ek, k0, alpha, E0, regularize = False, radius = 0.0):
        '''
        Calculates the EM field at a set of measurement points

        points           - (M,2)         coordinates of all measurement points
        Ek             - (N*2)           TE polarization component of the electromagnetic field at each scatterer
        k0               - (1)           frequency being measured
        alpha            - (1)           bare static polarizability at given k0
        thetas           - (Ndirs)       propagation directions for the source
        beam_waist       - (1)           beam waist
        regularize       - bool          bring everything below a scatterer radius to the center value, to be consistent with approximations and avoid divergences
        radius           - (1)           considered scatterer radius, only used for regularization
        '''

        points = np.tensor(points)
        Ek_ = np.matmul(alpha*k0*k0* self.G0_vector(points, k0, print_statement='propagate', regularize=regularize, radius=radius), Ek).reshape(points.shape[0],2,-1) + E0

        # Take care of cases in which measurement points are exactly scatterer positions
        for j in np.argwhere(np.isnan(Ek_[:,0,0])):
            if regularize:
                # If overlap, will just return the closest one
                possible_idx = np.nonzero(np.linalg.norm(self.r-points[j], axis = -1) <= radius)
                if possible_idx.shape[0] > 1:
                    idx = np.argmin(np.linalg.norm(self.r-points[j], axis = -1))
                else:
                    idx = possible_idx
                Ek_[j] = Ek[idx]
            else:
                Ek_[j] = Ek[np.nonzero(np.prod(self.r-points[j]==0,axis=-1))]
                
        return Ek_
   
    def solve(self, k0, alpha, radius, E0, self_interaction = True, self_interaction_type = "Rayleigh"):
        '''
        Solves the EM field at each scatterer

        k0                    - (1)           frequency being measured
        alpha                 - (1)           bare static polarizability at given k0
        thetas                - (Ndirs)       propagation directions for the source
        radius                - (1)           radius of scatterers, used in self-interaction
        beam_waist            - (1)           beam waist of Gaussian beam source
        self_interaction      - (bool)        include or not self-interactions, defaults to True 
        self_interaction_type - (string)      what order of approximation of S to use, "Rayleigh" or "full"
        '''
        
        ### TE calculation
        # Define the matrix M_tensor = I_tensor - k^2 alpha Green_tensor
        M_tensor = -alpha*k0*k0* self.G0_vector(None, k0, print_statement='solve')
        M_tensor.fill_diagonal_(1)
        if self_interaction:
            # Add self-interaction
            volume = onp.pi*radius*radius
            dims = M_tensor.shape[0]
            M_tensor -= alpha*k0*k0*self_interaction_integral_vector(k0, radius, self_interaction_type) /volume * np.eye(dims)
        # Solve M_tensor.Ek = E0j
        Ek = np.linalg.solve(M_tensor, E0.reshape(2*self.N,-1))
        return Ek
    
    def propagate_ss(self, points, k0, alpha, E0_meas, E0_scat, regularize = False, radius = 0.0):
        '''
        Calculates the EM field at a set of measurement points, using a single-scattering approximation

        points           - (M,2)      coordinates of all measurement points
        k0               - (1)        frequency being measured
        alpha            - (1)        bare static polarizability at given k0
        thetas           - (Ndirs)    propagation directions for the source
        beam_waist       - (1)        beam waist
        regularize       - bool       bring everything below a scatterer radius to the center value, to be consistent with approximations and avoid divergences
        radius           - (1)        considered scatterer radius, only used for regularization
        '''

        points = np.tensor(points)
        Ek_ = np.matmul(alpha*k0*k0* self.G0_vector(points, k0, print_statement='propagate_ss', regularize=regularize, radius=radius), E0_scat).reshape(points.shape[0],2,-1) + E0_meas
        
        # Take care of cases in which measurement points are exactly scatterer positions
        for j in np.argwhere(np.isnan(Ek_[:,0,0])):
            if regularize:
                # If overlap, will just return the closest one
                possible_idx = np.nonzero(np.linalg.norm(self.r-points[j], axis = -1) <= radius)
                if possible_idx.shape[0] > 1:
                    idx = np.argmin(np.linalg.norm(self.r-points[j], axis = -1))
                else:
                    idx = possible_idx
                Ek_[j] = E0_meas[idx]
            else:
                Ek_[j] = E0_meas[np.nonzero(np.prod(self.r-points[j]==0,axis=-1))]
                
        return Ek_

    def G0_vector(self, points, k0, print_statement = '', regularize = False, radius = 0.0):
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
        G0 = self.torch_greens(points_.reshape(-1,1,2) - self.r.reshape(1,-1,2), k0, regularize=regularize, radius=radius)
        #Construct matrix form
        if points == None:
            for idx in range(self.N):
                G0[idx,idx,:,:] = 0
        G0 = np.transpose(G0,1,2).reshape(2*G0.shape[0],2*G0.shape[1]).to(np.complex128)
        return G0

    def mean_DOS_measurements(self, measure_points, k0, alpha, radius, self_interaction = True, self_interaction_type = "Rayleigh", regularize = False, discard_absorption = False):
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
        self_interaction_type - (string)      what order of approximation of S to use, "Rayleigh" or "full"
        '''

        Npoints = measure_points.shape[0]
        k0_ = onp.round(k0/(2.0*onp.pi),1)
        print("Computing mean DOS using "+str(Npoints)+" points at k0L/2pi = "+str(k0_))

        ### TE calculation
        # Define the matrix M_tensor = I_tensor - k^2 alpha Green_tensor
        M_tensor = -alpha*k0*k0*self.G0_vector(None, k0, print_statement='DOS inverse')
        M_tensor.fill_diagonal_(1)
        if self_interaction:
            # Add self-interaction, (M_tensor)_ii = 1 - k^2 alpha self_int
            volume = onp.pi*radius*radius
            dims = M_tensor.shape[0]
            M_tensor -= alpha*k0*k0*self_interaction_integral_vector(k0, radius, self_interaction_type) /volume * np.eye(dims)
        # Compute W_tensor = inverse(M_tensor)
        W_tensor = np.linalg.solve(M_tensor, np.eye(len(M_tensor), dtype=np.complex128))

        # Define the propagators from scatterers to measurement points
        G0_measure = self.G0_vector(measure_points, k0, print_statement='DOS measure', regularize=regularize, radius=radius)
        # Check for measurement points falling exactly on scatterers
        for j in np.argwhere(np.isnan(G0_measure)):
            point_idx = j[0]
            scatter_idx = j[1]
            # At scatterers, replace G0(r_i, r_i) by self-interaction
            G0_measure[point_idx][scatter_idx] = 0
            if self_interaction:
                volume = onp.pi*radius*radius
                G0_measure[point_idx][scatter_idx] += self_interaction_integral_vector(k0, radius, self_interaction_type) / volume
        #  Use cyclic invariance of the trace: tr(G A G^T) = tr (G^T G A)
        # symm_mat = onp.matmul(onp.transpose(G0_measure), G0_measure)
        #  Use that trace(A.B^T) = AxB with . = matrix product and x = Hadamard product, and that G^T G is symmetric,
        dos_factor_vector = ( np.matmul(G0_measure.t(), G0_measure) * W_tensor ).sum()/Npoints
        if discard_absorption:
            # Discard the imaginary part of alpha, only for the last part of the calculation https://www.jpier.org/pier/pier.php?paper=19111801
            alpha_ = onp.real(alpha)
        else:
            alpha_ = alpha
        dos_factor_vector *= 4.0 * k0*k0* alpha_
        dos_factor_vector = np.imag(dos_factor_vector)

        return dos_factor_vector

    def LDOS_measurements(self, measure_points, k0, alpha, radius, self_interaction = True, self_interaction_type = "Rayleigh", regularize = False, discard_absorption = False):
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
        self_interaction_type - (string)      what order of approximation of S to use, "Rayleigh" or "full"
        '''

        M = measure_points.shape[0]

        ### TE calculation
        # Define the matrix M_tensor = I_tensor - k^2 alpha Green_tensor
        M_tensor = -alpha*k0*k0* self.G0_vector(None, k0, print_statement='LDOS inverse')
        M_tensor.fill_diagonal_(1)
        if self_interaction:
            # Add self-interaction, (M_tensor)_ii = 1 - k^2 alpha self_int
            volume = onp.pi*radius*radius
            dims = M_tensor.shape[0]
            M_tensor -= alpha*k0*k0*self_interaction_integral_vector(k0, radius, self_interaction_type)/volume * np.eye(dims)
        # Compute W_tensor = inverse(M_tensor)
        W_tensor = np.linalg.solve(M_tensor, np.eye(len(M_tensor), dtype=np.complex128))

        # Define the propagators from scatterers to measurement points
        G0_measure = self.G0_vector(measure_points, k0, print_statement='LDOS measure', regularize=regularize, radius=radius)
        # Check for measurement points falling exactly on scatterers
        for j in np.argwhere(np.isnan(G0_measure)):
            point_idx = j[0]
            scatter_idx = j[1]
            # At scatterers, replace G0(r_i, r_i) by self-interaction
            G0_measure[point_idx][scatter_idx] = 0
            if self_interaction:
                volume = onp.pi*radius*radius
                G0_measure[point_idx][scatter_idx] += self_interaction_integral_vector(k0, radius, self_interaction_type)/volume
        # ldos_factor = onp.diagonal(onp.matmul(onp.matmul(G0_measure, Ainv),onp.transpose(G0_measure)))
        # Can be made better considering it's a diagonal https://stackoverflow.com/questions/17437817/python-how-to-get-diagonalab-without-having-to-perform-ab
        ldos_factor_vector = np.einsum('ij, ji->i',np.matmul(G0_measure, W_tensor), (G0_measure).t() )
        if discard_absorption:
            # Discard the imaginary part of alpha, only for the last part of the calculation https://www.jpier.org/pier/pier.php?paper=19111801
            alpha_ = onp.real(alpha)
        else:
            alpha_ = alpha
        ldos_factor_vector *= 4.0 * k0*k0*alpha_
        ldos_factor_vector = np.imag(ldos_factor_vector)
        ldos_factor_vector = ldos_factor_vector.reshape(M,2,-1)
        ldos_factor_vector = np.sum(ldos_factor_vector, 1)

        return ldos_factor_vector

    def compute_eigenvalues_and_scatterer_LDOS(self, k0, alpha, radius, file_name, self_interaction = True, self_interaction_type = "Rayleigh", write_eigenvalues = True):
        '''
        Computes the eigenvalues of the Green's matrix, and the corresponding LDOS at scatterers, for TM and TE.
        This computation is way less expensive than the other LDOS, due to simple dependence on the eigenvalues
        measure_points      - (M,3)  coordinates of points where the LDOS is evaluated
        k0                  - (1)    frequency of source beam
        alpha               - (1)    bare static polarizability at given k0
        radius              - (1)    radius of the scatterers
        self_interaction    - (bool) include or not self-interactions, defaults to True 
        self_interaction_type - (string)      what order of approximation of S to use, "Rayleigh" or "full"
        '''

        Npoints = self.r.shape[0]
        print(self.r.shape)
        k0_ = onp.round(k0/(2.0*onp.pi),1)
        print("Computing spectrum and scatterer LDOS using "+str(Npoints)+" points at k0L/2pi = "+str(k0_))

        ### TE calculation
        # Define the matrix M_tensor = I_tensor - k^2 alpha Green_tensor
        M_tensor = -alpha*k0*k0*self.G0_vector(None, k0, print_statement='DOS eigvals')
        M_tensor.fill_diagonal_(1)
        if self_interaction:
            # Add self-interaction, (M_tensor)_ii = 1 - k^2 alpha self_int
            volume = onp.pi*radius*radius
            dims = M_tensor.shape[0]
            M_tensor -= alpha*k0*k0*self_interaction_integral_vector(k0, radius, self_interaction_type)/volume * np.eye(dims)
        # Compute the spectrum of the M_tensor
        deltas = np.linalg.eigvals(M_tensor)

        if write_eigenvalues:
            onp.savetxt(file_name+'_deltas_'+str(k0_)+'_vector.csv', onp.stack([np.real(deltas).numpy(), np.imag(deltas).numpy()]).T)

        # Compute the trace part here
        dos_factor_vector = ((1 - deltas)**2 / deltas).sum()/Npoints
        dos_factor_vector *= 4.0 / ( k0**2 * alpha) # For prefactor in systems invariant along z, see https://www.sciencedirect.com/science/article/pii/S1569441007000387
        dos_factor_vector = np.imag(dos_factor_vector)

        return dos_factor_vector
    
    def compute_eigenmodes_IPR(self, k0, alpha, radius, file_name, self_interaction = True, self_interaction_type = "Rayleigh", number_eigenmodes = 1, write_eigenvalues = True, sorting_type = 'IPR'):
    
        Npoints = self.r.shape[0]
        k0_ = onp.round(k0/(2.0*onp.pi),1)
        print("Computing spectrum and scatterer LDOS using "+str(Npoints)+" points at k0L/2pi = "+str(k0_))


        ### TE Calculation
        # Define the matrix M_tensor = I_tensor - k^2 alpha Green_tensor
        M_tensor = -alpha*k0*k0*self.G0_vector(None, k0, print_statement='DOS eigvals')
        M_tensor.fill_diagonal_(1)
        if self_interaction:
            # Add self-interaction, (M_tensor)_ii = 1 - k^2 alpha self_int
            volume = onp.pi*radius*radius
            dims = M_tensor.shape[0]
            M_tensor -= alpha*k0*k0*self_interaction_integral_vector(k0, radius, self_interaction_type)/volume * np.eye(dims)
        # Compute the spectrum of the M_tensor
    
        # Works, maybe consider scipy.schur instead, and output IPRs + one / some eigenvector(s) for plotting purposes
        deltas, eigenvectors = np.linalg.eig(M_tensor)
        IPRs = np.sum(np.abs(eigenvectors**4), axis = 0) / (np.sum(np.abs(eigenvectors**2), axis = 0))**2
        
        
        deltas = 1.0 - k0**2 * alpha * deltas
        utils.plot_IPR_damping_values(deltas, IPRs, file_name+'_deltas', logscale=True, appended_string=str(k0_))
        # utils.plot_IPR_damping_values(1-deltas, IPRs, file_name+'_test'+extra_string, logscale=True, appended_string=str(k0_))
        
        if write_eigenvalues:
            onp.savetxt(file_name+'_deltas_'+str(k0_)+'.csv', onp.stack([np.real(deltas).numpy(), np.imag(deltas).numpy(), IPRs]).T)
            
            
        if sorting_type == 'IPR':
            IPRs, indices = np.sort(IPRs, descending=True)
            deltas = deltas[indices]
            eigenvectors = eigenvectors[:,indices]
        elif sorting_type == 'damping':
            indices = np.argsort(np.imag(deltas), descending= False) # Want SMALL dampings first
            deltas = deltas[indices]
            IPRs = IPRs[indices]
            eigenvectors = eigenvectors[:,indices]
        else:
            raise NotImplementedError
            
        
        returned_eigenvectors = eigenvectors[:, 0:number_eigenmodes]

        # Debug plots
        # returned_eigenvalues = deltas[0:number_eigenmodes]
        # print(returned_eigenvalues)
        # print(IPRs.amax())
        # print(IPRs[0])
        
        return deltas, returned_eigenvectors, IPRs
class Transmission2D_scalar:
    

    def __init__(self, points, source = "beam"):
        self.r = points.reshape(-1,2)
        self.N = self.r.shape[0]
        self.source = source

    def torch_hankel1(self, nu, x):
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
            
    def torch_greens(self, r, k0, periodic = '', regularize = False, radius = 0.0):
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

    def generate_source(self, points, k0, thetas, w, print_statement = ''):
        '''
        Generates the EM field of a source at a set of points

        points      - (M,2)      coordinates of points
        k0          - (1)        frequency of source beam
        thetas      - (Ndirs)    propagation directions for the source
        w           - (1)        beam waist for beam sources
        '''
        
        if self.source == "beam":
            # Collimated beam with zero curvature,
            # from solution of the the paraxial approximation of Maxwell-Helmholtz,
            # see https://en.wikipedia.org/wiki/Gaussian_beam
            
            k0_ = onp.round(k0/(2.0*onp.pi),1)
            print('Calculating Beam Source at k0L/2pi = '+str(k0_)+' ('+print_statement+')')
            
            E0j = np.zeros((points.shape[0],len(thetas)),dtype=np.complex128)
            for idx, theta in enumerate(thetas):
                # Rotate the system by - theta rather than the expression of the source
                cost, sint = onp.cos(-theta),onp.sin(-theta)
                rot = np.tensor([[cost,-sint],[sint,cost]])
                rrot = np.matmul(rot,points.T).T #(rparallel, rperp)
                a = 2*rrot[:,1]/(w*w*k0)
                E0j[:,idx] = np.exp(1j*rrot[:,0]*k0-(rrot[:,1]**2/(w*w*(1+1j*a))))/np.sqrt(1+1j*a)
                
        elif self.source == 'plane':
            # Infinitely extended Plane wave
            
            k0_ = onp.round(k0/(2.0*onp.pi),1)
            print('Calculating Plane Source at k0L/2pi = '+str(k0_)+' ('+print_statement+')')
            
            E0j = np.zeros((points.shape[0],len(thetas)),dtype=np.complex128)
            for idx in range(len(thetas)):
                # Rotate the system by - theta rather than the expression of the source
                theta = thetas[idx]
                cost, sint = onp.cos(-theta),onp.sin(-theta)
                rot = np.tensor([[cost,-sint],[sint,cost]])
                rrot = np.matmul(rot,points.T).T #(rparallel, rperp)
                E0j[:,idx] = np.exp(1j*rrot[:,0]*k0)
                
        elif self.source == 'point':
            # One electric point dipole emitting light at source_distance * L away
            source_distance = 2.0
            source_intensity = 1.0 * k0 * source_distance * 2.0 * onp.pi
            
            k0_ = onp.round(k0/(2.0*onp.pi),1)
            print('Calculating Point Source at k0L/2pi = '+str(k0_)+' ('+print_statement+')')
            
            E0j = np.zeros((points.shape[0],len(thetas)),dtype=np.complex128)
            for idx in range(len(thetas)):
                theta = thetas[idx]
                cost, sint = onp.cos(-theta),onp.sin(-theta)
                rot = np.tensor([[cost,-sint],[sint,cost]])
                rrot = np.matmul(rot,points.T).T #(rparallel, rperp)
                source_location = source_distance * np.tensor([-1.0, 0.0])
                E0j[:,idx] = onp.sqrt(source_intensity) * self.torch_greens(rrot.reshape(-1,1,2) - source_location.reshape(1,-1,2), k0).reshape(points.shape[0])
                
        elif self.source is None:
            E0j = np.zeros((points.shape[0],len(thetas)),dtype=np.complex128)

        else:
            raise NotImplementedError
                
        return E0j
 
    def propagate(self, points, Ek, k0, alpha, E0j, regularize = False, radius = 0.0):
        '''
        Calculates the EM field at a set of measurement points

        points           - (M,2)         coordinates of all measurement points
        Ek               - (N)           field at each scatterer
        k0               - (1)           frequency being measured
        alpha            - (1)           bare static polarizability at given k0
        thetas           - (Ndirs)       propagation directions for the source
        beam_waist       - (1)           beam waist
        regularize       - bool          bring everything below a scatterer radius to the center value, to be consistent with approximations and avoid divergences
        radius           - (1)           considered scatterer radius, only used for regularization
        '''

        points = np.tensor(points)
        Ek_ = np.matmul(alpha*k0*k0* self.G0(points, k0, print_statement='propagate', regularize=regularize, radius=radius), Ek) + E0j
        
        # Take care of cases in which measurement points are exactly scatterer positions
        for j in np.argwhere(np.isnan(Ek_[:,0])):
            if regularize:
                # If overlap, will just return the closest one
                possible_idx = np.nonzero(np.linalg.norm(self.r-points[j], axis = -1) <= radius)
                if possible_idx.shape[0] > 1:
                    idx = np.argmin(np.linalg.norm(self.r-points[j], axis = -1))
                else:
                    idx = possible_idx
                Ek_[j] = Ek[idx]
            else:
                Ek_[j] = Ek[np.nonzero(np.prod(self.r-points[j]==0,axis=-1))]
                
        return Ek_
   
    def solve(self, k0, alpha, radius, E0j, self_interaction = True, self_interaction_type = "Rayleigh"):
        '''
        Solves for the field at each scatterer

        k0                  - (1)           frequency being measured
        alpha               - (1)           bare static polarizability at given k0
        thetas              - (Ndirs)       propagation directions for the source
        radius              - (1)           radius of scatterers, used in self-interaction
        beam_waist          - (1)           beam waist of Gaussian beam source
        self_interaction    - (bool)        include or not self-interactions, defaults to True 
        self_interaction_type - (string)      what order of approximation of S to use, "Rayleigh" or "full"
        '''

        ### TM calculation
        # Define the matrix M_tensor = I_tensor - k^2 alpha Green_tensor
        M_tensor = -alpha*k0*k0* self.G0(self.r, k0, print_statement='solve')
        M_tensor.fill_diagonal_(1)
        if self_interaction:
            # Add self-interaction, (M_tensor)_ii = 1 - k^2 alpha self_int
            volume = onp.pi*radius*radius
            dims = M_tensor.shape[0]
            M_tensor -= alpha*k0*k0*self_interaction_integral_scalar(k0, radius, self_interaction_type)/volume * np.eye(dims)
        # Solve M_tensor.Ek = E0j
        # NB: this uses an LU decomposition according to torch https://pytorch.org/docs/stable/generated/torch.linalg.lu.html
        Ek = np.linalg.solve(M_tensor,E0j)
        
        return Ek
    
    def propagate_ss(self, points, k0, alpha, E0_meas, E0_scat, regularize = False, radius = 0.0):
        '''
        Calculates the field at a set of measurement points, using a single-scattering approximation

        points           - (M,2)      coordinates of all measurement points
        k0               - (1)        frequency being measured
        alpha            - (1)        bare static polarizability at given k0
        thetas           - (Ndirs)    propagation directions for the source
        beam_waist       - (1)        beam waist
        regularize       - bool       bring everything below a scatterer radius to the center value, to be consistent with approximations and avoid divergences
        radius           - (1)        considered scatterer radius, only used for regularization
        '''

        points = np.tensor(points)
        Ek_ = np.matmul(alpha*k0*k0* self.G0(points, k0, print_statement='propagate_ss', regularize=regularize, radius=radius), E0_scat) + E0_meas
        
        # Take care of cases in which measurement points are exactly scatterer positions
        for j in np.argwhere(np.isnan(Ek_[:,0])):
            if regularize:
                # If overlap, will just return the closest one
                possible_idx = np.nonzero(np.linalg.norm(self.r-points[j], axis = -1) <= radius)
                if possible_idx.shape[0] > 1:
                    idx = np.argmin(np.linalg.norm(self.r-points[j], axis = -1))
                else:
                    idx = possible_idx
                Ek_[j] = E0_meas[idx]
            else:
                Ek_[j] = E0_meas[np.nonzero(np.prod(self.r-points[j]==0,axis=-1))]
                
        return Ek_

    def G0(self, points, k0, print_statement = '', regularize = False, radius=0.0):
        '''
        Returns a Green's tensor linking all points to all scatterers for the TM polarization
        '''
        #Green's function
        k0_ = onp.round(k0/(2.0*onp.pi),1)
        print("Calculating TM Green's function at k0L/2pi = "+str(k0_)+' ('+print_statement+')')
        G0_scalar = self.torch_greens(points.reshape(-1,1,2) - self.r.reshape(1,-1,2), k0, regularize=regularize, radius=radius)
        return G0_scalar

    def mean_DOS_measurements(self, measure_points, k0, alpha, radius, self_interaction= True, self_interaction_type = "Rayleigh", regularize = False, discard_absorption = False):
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
        self_interaction_type - (string)      what order of approximation of S to use, "Rayleigh" or "full"
        '''

        Npoints = measure_points.shape[0]
        k0_ = onp.round(k0/(2.0*onp.pi),1)
        print("Computing mean DOS using "+str(Npoints)+" points at k0L/2pi = "+str(k0_))

        # Define the matrix M_tensor = I_tensor - k^2 alpha Green_tensor
        M_tensor = -alpha*k0*k0* self.G0(self.r, k0, print_statement='DOS inverse')
        M_tensor.fill_diagonal_(1)
        if self_interaction:
            # Add self-interaction, (M_tensor)_ii = 1 - k^2 alpha self_int
            volume = onp.pi*radius*radius
            dims = M_tensor.shape[0]
            M_tensor -= alpha*k0*k0*self_interaction_integral_scalar(k0, radius, self_interaction_type)/volume * np.eye(dims)
        # Compute W_tensor = inverse(M_tensor)
        W_tensor = np.linalg.solve(M_tensor, np.eye(len(M_tensor), dtype=np.complex128))

        # Define the propagators from scatterers to measurement points
        G0_measure = self.G0(measure_points, k0, print_statement='DOS measure', regularize=regularize, radius=radius)
        # Check for measurement points falling exactly on scatterers
        for j in np.argwhere(np.isnan(G0_measure)):
            point_idx = j[0]
            scatter_idx = j[1]
            # At scatterers, replace G0(r_i, r_i) by self-interaction
            G0_measure[point_idx][scatter_idx] = 0
            if self_interaction:
                volume = onp.pi*radius*radius
                G0_measure[point_idx][scatter_idx] += self_interaction_integral_scalar(k0, radius, self_interaction_type)/volume
        #  Use cyclic invariance of the trace: tr(G A G^T) = tr (G^T G A)
        # symm_mat = onp.matmul(onp.transpose(G0_measure), G0_measure)
        #  Use that trace(A.B^T) = AxB with . = matrix product and x = Hadamard product, and that G^T G is symmetric,
        dos_factor_scalar = ( np.matmul(G0_measure.t(), G0_measure) * W_tensor ).sum()/Npoints

        if discard_absorption:
            # Discard the imaginary part of alpha, only for the last part of the calculation https://www.jpier.org/pier/pier.php?paper=19111801
            alpha_ = onp.real(alpha)
        else:
            alpha_ = alpha
        dos_factor_scalar *= 4.0 * k0*k0*alpha_ # For prefactor in systems invariant along z, see https://www.sciencedirect.com/science/article/pii/S1569441007000387
        dos_factor_scalar = np.imag(dos_factor_scalar)

        return dos_factor_scalar

    def LDOS_measurements(self, measure_points, k0, alpha, radius, self_interaction= True, self_interaction_type = "Rayleigh", regularize = False, discard_absorption = False):
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
        self_interaction_type - (string)      what order of approximation of S to use, "Rayleigh" or "full"
        '''

        M = measure_points.shape[0]

        # Define the matrix M_tensor = I_tensor - k^2 alpha Green_tensor
        M_tensor = -alpha*k0*k0* self.G0(self.r, k0, print_statement='LDOS inverse')
        M_tensor.fill_diagonal_(1)
        if self_interaction:
            # Add self-interaction, (M_tensor)_ii = 1 - k^2 alpha self_int
            volume = onp.pi*radius*radius
            dims = M_tensor.shape[0]
            M_tensor -= alpha*k0*k0*self_interaction_integral_scalar(k0, radius, self_interaction_type)/volume * np.eye(dims)
        # Compute W_tensor = inverse(M_tensor)
        W_tensor = np.linalg.solve(M_tensor, np.eye(len(M_tensor), dtype=np.complex128))

        # Define the propagators from scatterers to measurement points
        G0_measure = self.G0(measure_points, k0, print_statement='LDOS measure', regularize=regularize, radius=radius)
        # Check for measurement points falling exactly on scatterers
        for j in np.argwhere(np.isnan(G0_measure)):
            point_idx = j[0]
            scatter_idx = j[1]
            # At scatterers, replace G0(r_i, r_i) by self-interaction
            G0_measure[point_idx][scatter_idx] = 0
            if self_interaction:
                volume = onp.pi*radius*radius
                G0_measure[point_idx][scatter_idx] += self_interaction_integral_scalar(k0, radius, self_interaction_type)/volume
        # ldos_factor = onp.diagonal(onp.matmul(onp.matmul(G0_measure, W_tensor),onp.transpose(G0_measure)))
        # Can be made better considering it's a diagonal https://stackoverflow.com/questions/17437817/python-how-to-get-diagonalab-without-having-to-perform-ab
        ldos_factor_scalar = np.einsum('ij, ji->i',np.matmul(G0_measure, W_tensor), (G0_measure).t() )
        if discard_absorption:
            # Discard the imaginary part of alpha, only for the last part of the calculation https://www.jpier.org/pier/pier.php?paper=19111801
            alpha_ = onp.real(alpha)
        else:
            alpha_ = alpha
        ldos_factor_scalar *= 4.0 * k0*k0*alpha_
        ldos_factor_scalar = np.imag(ldos_factor_scalar)

        return ldos_factor_scalar

    def compute_eigenvalues_and_scatterer_LDOS(self, k0, alpha, radius, file_name, self_interaction = True, self_interaction_type = "Rayleigh", write_eigenvalues = True):
        '''
        Computes the eigenvalues of the Green's matrix, and the corresponding LDOS at scatterers, for TM and TE.
        This computation is way less expensive than the other LDOS, due to simple dependence on the eigenvalues
        measure_points      - (M,3)  coordinates of points where the LDOS is evaluated
        k0                  - (1)    frequency of source beam
        alpha               - (1)    bare static polarizability at given k0
        radius              - (1)    radius of the scatterers
        self_interaction    - (bool) include or not self-interactions, defaults to True
        self_interaction_type - (string)      what order of approximation of S to use, "Rayleigh" or "full"
        '''

        Npoints = self.r.shape[0]
        print(self.r.shape)
        k0_ = onp.round(k0/(2.0*onp.pi),1)
        print("Computing spectrum and scatterer LDOS using "+str(Npoints)+" points at k0L/2pi = "+str(k0_))

        ### TM Calculation
        # Define the matrix M_tensor = I_tensor - k^2 alpha Green_tensor
        M_tensor = -alpha*k0*k0*self.G0(self.r, k0, print_statement='DOS eigvals')
        M_tensor.fill_diagonal_(1)
        if self_interaction:
            # Add self-interaction, (M_tensor)_ii = 1 - k^2 alpha self_int
            volume = onp.pi*radius*radius
            dims = M_tensor.shape[0]
            M_tensor -= alpha*k0*k0*self_interaction_integral_scalar(k0, radius, self_interaction_type)/volume * np.eye(dims)
        # Compute the spectrum of the M_tensor
        deltas = np.linalg.eigvals(M_tensor)

        if write_eigenvalues:
            onp.savetxt(file_name+'_deltas_'+str(k0_)+'_scalar.csv', onp.stack([np.real(deltas).numpy(), np.imag(deltas).numpy()]).T)

        # Compute the trace part here
        dos_factor_scalar = ((1 - deltas)**2 / deltas).sum()/Npoints
        dos_factor_scalar *= 4.0 / ( k0**2 * alpha) # For prefactor in systems invariant along z, see https://www.sciencedirect.com/science/article/pii/S1569441007000387
        dos_factor_scalar = np.imag(dos_factor_scalar)

        return dos_factor_scalar
    
    def compute_eigenmodes_IPR(self, k0, alpha, radius, file_name, self_interaction = True, self_interaction_type = "Rayleigh", number_eigenmodes = 1, write_eigenvalues = True, sorting_type = 'IPR'):
    
        Npoints = self.r.shape[0]
        k0_ = onp.round(k0/(2.0*onp.pi),1)
        print("Computing spectrum and scatterer LDOS using "+str(Npoints)+" points at k0L/2pi = "+str(k0_))

        ### TM Calculation
        # Define the matrix M_tensor = I_tensor - k^2 alpha Green_tensor
        M_tensor = -alpha*k0*k0*self.G0(self.r, k0, print_statement='DOS eigvals')
        M_tensor.fill_diagonal_(1)
        if self_interaction:
            # Add self-interaction, (M_tensor)_ii = 1 - k^2 alpha self_int
            volume = onp.pi*radius*radius
            dims = M_tensor.shape[0]
            M_tensor -= alpha*k0*k0*self_interaction_integral_scalar(k0, radius, self_interaction_type)/volume * np.eye(dims)
    
        # Works, maybe consider scipy.schur instead, and output IPRs + one / some eigenvector(s) for plotting purposes
        deltas, eigenvectors = np.linalg.eig(M_tensor)
        IPRs = np.sum(np.abs(eigenvectors**4), axis = 0) / (np.sum(np.abs(eigenvectors**2), axis = 0))**2
        
        deltas = 1.0 - k0**2 * alpha * deltas
        utils.plot_IPR_damping_values(deltas, IPRs, file_name+'_deltas', logscale=True, appended_string=str(k0_))
        # utils.plot_IPR_damping_values(1-deltas, IPRs, file_name+'_test'+extra_string, logscale=True, appended_string=str(k0_))
        
        if write_eigenvalues:
            onp.savetxt(file_name+'_deltas_'+str(k0_)+'.csv', onp.stack([np.real(deltas).numpy(), np.imag(deltas).numpy(), IPRs]).T)
            
            
        if sorting_type is 'IPR':
            IPRs, indices = np.sort(IPRs, descending=True)
            deltas = deltas[indices]
            eigenvectors = eigenvectors[:,indices]
        elif sorting_type is 'damping':
            indices = np.argsort(np.imag(deltas), descending= False) # Want SMALL dampings first
            deltas = deltas[indices]
            IPRs = IPRs[indices]
            eigenvectors = eigenvectors[:,indices]
        else:
            raise NotImplementedError
            
        
        returned_eigenvectors = eigenvectors[:, 0:number_eigenmodes]

        # Debug plots
        # returned_eigenvalues = deltas[0:number_eigenmodes]
        # print(returned_eigenvalues)
        # print(IPRs.amax())
        # print(IPRs[0])
        
        return deltas, returned_eigenvectors, IPRs