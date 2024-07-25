import sys
import numpy as onp
import torch as np
import scipy as sp
from scipy.special import hankel1
import hickle as hkl
from juliacall import Main as jl
from juliacall import Pkg as jlPkg

import utils


I = np.tensor(onp.identity(3)).reshape(1,3,3) #identity matrix

def self_interaction_integral_vector(k0, radius, self_interaction_type = "Rayleigh"):
    
    volume = 4.0 * onp.pi * radius**3 / 3.0
    
    if self_interaction_type == "full":
        self_int = (1.0/k0**2) * ((2.0 / 3.0) * onp.exp(1j*k0*radius)*(1- 1j*k0*radius) - 1.0)
    elif self_interaction_type == "Rayleigh":
        self_int = -1.0/(3.0 * k0**2) + radius**2 / 3.0 + 1j * k0 * volume / (6.0 * onp.pi)
    else:
        raise NotImplementedError
    
    return self_int

def self_interaction_integral_scalar(k0, radius, self_interaction_type = "Rayleigh"):
    
    volume = 4.0 * onp.pi * radius**3 / 3.0
    
    if self_interaction_type == "full":
        self_int = (1.0/k0**2) * (onp.exp(1j*k0*radius)*(1- 1j*k0*radius) - 1.0)
    elif self_interaction_type == "Rayleigh":
        self_int = radius**2 / 2.0 + 1j * k0 * volume / (4.0 * onp.pi)
    else:
        raise NotImplementedError
    
    return self_int

class Transmission3D:
    

    def __init__(self, points, source='beam'):
        self.r = points.reshape(-1,3)
        self.N = self.r.shape[0]
        self.source = source
    
    def greens(self,r,k0,periodic = '', regularize = False, radius=0.0):
        '''
        Torch implementation of the 3d Green's function, taking tensors as entries
        r          - (M,2)      distances to propagate over
        k0         - (1)        wave-vector of source beam in vacuum
        periodic   - str        change boundary conditions: '' = free, ('x', 'y', 'xy') = choices of possible periodic directions
        regularize - bool       bring everything below a scatterer radius to the center value, to be consistent with approximations and avoid divergences
        radius     - (1)        considered scatterer radius, only used for regularization
        '''
        N = r.shape[0]
        M = r.shape[1]
        if 'x' in periodic:
            r[:,:,0] += 0.5
            r[:,:,0] %= 1
            r[:,:,0] -= 0.5
        if 'y' in periodic:
            r[:,:,1] += 0.5
            r[:,:,1] %= 1
            r[:,:,1] -= 0.5
        if 'z' in periodic:
            r[:,:,2] += 0.5
            r[:,:,2] %= 1
            r[:,:,2] -= 0.5
        R = np.linalg.norm(r,axis=-1).reshape(N,M,1,1)
        RxR = r.reshape(N,M,1,3)*r.reshape(N,M,3,1)
        RxR /= R*R

        if regularize:
            R = np.where(R < radius, 0.0, R)

        return (I-RxR-(I-3*RxR)*(1/(1j*k0*R)+(k0*R)**-2))*np.exp(1j*k0*R)/(4*onp.pi*R)

    def generate_source(self, points, k0, u, p, w, print_statement = ''):
        '''
        Generates the EM field of a source at a set of points

        points - (M,3)      coordinates of points
        k0     - (1)        frequency of source beam
        u      - (Ndirs, 3) propagation directions for the source
        p      - (Ndirs, 3) polarization directions for the source
        w      - (1)        beam waist for beam sources
        '''
        
        if self.source == 'beam':
            # Collimated beam with zero curvature,
            # from solution of the the paraxial approximation of Maxwell-Helmholtz,
            # see https://en.wikipedia.org/wiki/Gaussian_beam
            
            k0_ = onp.round(k0/(2.0*onp.pi),1)
            print('Calculating Beam Source at k0L/2pi = '+str(k0_)+' ('+print_statement+')')
            
            rpara = np.matmul(points,u.T)
            rperp = np.linalg.norm(points.reshape(-1,3,1) - rpara.reshape(points.shape[0],1,u.shape[0])*u.T.reshape(1,3,-1),axis=1)
            a = 2*rperp/(w*w*k0)
            E0j = np.exp(1j*rpara*k0-(rperp**2/(w*w*(1+1j*a))))/np.sqrt(1+1j*a)
            phi = np.arctan2(p[:,1], p[:,0]) #arctan(y/x)
            theta = np.arccos(p[:,2]) #arccos(z/r), r=1 for unit vector
            pvec = np.stack([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), p[:,2]])
            
            E0j = E0j.reshape(points.shape[0],1,-1)*pvec.reshape(1,3,-1)
            
        elif self.source == 'plane':
            # Infinitely extended Plane wave
            
            k0_ = onp.round(k0/(2.0*onp.pi),1)
            print('Calculating Plane Source at k0L/2pi = '+str(k0_)+' ('+print_statement+')')
            
            rpara = np.matmul(points,u.T)
            E0j = np.exp(1j*rpara*k0)
            phi = np.arctan2(p[:,1], p[:,0]) #arctan(y/x)
            theta = np.arccos(p[:,2]) #arccos(z/r), r=1 for unit vector
            pvec = np.stack([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), p[:,2]])
            
            E0j = E0j.reshape(points.shape[0],1,-1)*pvec.reshape(1,3,-1)
            
        elif self.source == 'point':
            # One electric point dipole emitting light at source_distance * L away
            source_distance = 2.0
            source_intensity = 1.0 * (k0 * source_distance)**2 * 4.0 * onp.pi
            
            k0_ = onp.round(k0/(2.0*onp.pi),1)
            print('Calculating Point Source at k0L/2pi = '+str(k0_)+' ('+print_statement+')')
            
            rpara = np.matmul(points,u.T)
            rperp = np.linalg.norm(points.reshape(-1,3,1) - rpara.reshape(points.shape[0],1,u.shape[0])*u.T.reshape(1,3,-1),axis=1)
            
            source_location = source_distance * (-u)
            dipole_moment = onp.sqrt(source_intensity) * p
            
            E0j = np.matmul(self.torch_greensTE(points.reshape(-1,1,3) - source_location.reshape(1,-1,3), k0), dipole_moment.type(np.complex128)).squeeze()

        elif self.source is None:
            
            E0j = np.zeros((points.shape[0],u.shape[1],u.shape[0]),dtype=np.complex128)
            
        else:
            raise NotImplementedError
        
        return E0j
 
    def propagate(self, points, Ek, k0, alpha, u, p, beam_waist, regularize = False, radius=0.0):
        '''
        Calculates the EM field at a set of measurement points

        points           - (M,3)         coordinates of all measurement points
        Ek               - (N*3)         electromagnetic field at each scatterer
        k0               - (1)           frequency being measured
        alpha            - (1)           bare static polarizability at given k0
        u                - (Ndirs, 3)    propagation directions for the source
        p                - (Ndirs, 3)    polarization directions for the source
        beam_waist       - (1)           beam waist
        regularize       - bool          bring everything below a scatterer radius to the center value, to be consistent with approximations and avoid divergences
        radius           - (1)           considered scatterer radius, only used for regularization 
        '''
        points = np.tensor(points)
        
        # ensure u and p are all unit vectors
        u /= np.linalg.norm(u,axis=-1).reshape(-1,1)
        p /= np.linalg.norm(p,axis=-1).reshape(-1,1)
        
        # check polarization is orthogonal to propagation
        assert np.sum(np.absolute(np.sum(u*p,axis=-1))) == 0

        # generate source field for measurement points
        E0j = self.generate_source(points, k0, u, p, beam_waist, print_statement='propagate') #(M,3,Ndirs)
        
        # calculate Ek field at all measurement points
        Ek_ = np.matmul(alpha*k0*k0*self.G0(points, k0, print_statement='propagate', regularize=regularize, radius=radius), Ek).reshape(points.shape[0],3,-1) + E0j 

        # Take care of cases in which measurement points are exactly scatterer positions
        for j in np.argwhere(np.isnan(Ek_[:,0,0])):
            if regularize:
                # If overlap, will just return the closest one
                possible_idx = np.nonzero(np.linalg.norm(self.r-points[j],axis=-1) <= radius)
                if possible_idx.shape[0] > 1:
                    idx = np.argmin(np.linalg.norm(self.r-points[j], axis = -1))
                else:
                    idx = possible_idx
                Ek_[j] = Ek.reshape(points.shape[0],3,-1)[idx]
            else:
                idx = np.nonzero(np.prod(self.r-points[j]==0,axis=-1))
                Ek_[j] = Ek.reshape(points.shape[0],3,-1)[idx]
                
        return Ek_
   
    def solve(self, k0, alpha, u, p, radius, beam_waist, self_interaction = True, self_interaction_type = "Rayleigh"):
        '''
        Solves the EM field at each scatterer

        k0                  - (1)           frequency being measured
        alpha               - (1)           bare static polarizability at given k0
        u                   - (Ndirs, 3)    propagation directions for the source
        p                   - (Ndirs, 3)    polarization directions for the source
        radius              - (1)           radius of scatterers, used in self-interaction
        beam_waist          - (1)           beam waist of Gaussian beam source
        self_interaction    - (bool)        include or not self-interactions, defaults to True 
        self_interaction_type - (string)      what order of approximation of S to use, "Rayleigh" or "full"
        '''

        # Generate source field for scatterer positions
        E0j = self.generate_source(self.r, k0, u, p, beam_waist, print_statement='solve') #(N,3,Ndirs)
        
        ### Calculate Ek field at each scatterer position
        # Define the matrix M_tensor = I_tensor - k^2 alpha Green_tensor
        M_tensor = -alpha*k0*k0*self.G0(None, k0, print_statement='solve')
        M_tensor.fill_diagonal_(1)
        if self_interaction:
            # Add self-interaction, (M_tensor)_ii = 1 - k^2 alpha self_interaction / volume
            dims = M_tensor.shape[0]
            volume = 4*onp.pi*(radius**3)/3
            M_tensor -= k0**2 * alpha * self_interaction_integral_vector(k0, radius, self_interaction_type) / volume * np.eye(dims)
        # Solve M_tensor.Ek = E0j
        Ek = np.linalg.solve(M_tensor, E0j.reshape(3*self.N,-1)) 
        return Ek
    
    def propagate_ss(self, points, k0, alpha, u, p, beam_waist, regularize = False, radius = 0.0):
        '''
        Calculates the EM field at a set of measurement points, using a single-scattering approximation

        points           - (M,3)         coordinates of all measurement points
        k0               - (1)           frequency being measured
        alpha            - (1)           bare static polarizability at given k0
        u                - (Ndirs, 3)    propagation directions for the source
        p                - (Ndirs, 3)    polarization directions for the source
        beam_waist       - (1)           beam waist
        regularize       - bool          bring everything below a scatterer radius to the center value, to be consistent with approximations and avoid divergences
        radius           - (1)           considered scatterer radius, only used for regularization 
        '''

        points = np.tensor(points)
        E0_meas = self.generate_source(points, k0, u, p, beam_waist, print_statement='propagate_ss')
        E0_scat = self.generate_source(self.r, k0, u, p, beam_waist, print_statement='propagate_ss')
        E0_scat = E0_scat.reshape(3*self.r.shape[0],-1)        
        Ek_ = np.matmul(alpha*k0*k0* self.G0(points, k0, print_statement='propagate_ss', regularize=regularize, radius=radius), E0_scat).reshape(points.shape[0],3,-1) + E0_meas
        
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

    def G0(self, points, k0, print_statement='', regularize = False, radius = 0.0):
        '''
        Generate the Green's tensor for a set of positions

        points          - (N,3)      set of point positions, None indicates the saved point pattern
        k0              - (1)        frequency being measured
        print_statement - str        disambiguating string used when printing (default = empty)
        regularize - bool       bring everything below a scatterer radius to the center value, to be consistent with approximations and avoid divergences
        radius     - (1)        considered scatterer radius, only used for regularization
        '''

        # check if None
        if points == None:
            points_ = self.r
        else:
            points_ = points
        k0_ = onp.round(k0/(2.0*onp.pi),1)
        print("Calculating Green's function at k0L/2pi = "+str(k0_)+' ('+print_statement+')')
        # populate Green's tensor
        G0 = self.greens(points_.reshape(-1,1,3)-self.r.reshape(1,-1,3), k0, regularize=regularize, radius=radius) #shape is (M,N,3,3)

        # replace NaN entries resulting from divergence (r-r'=0)
        if points == None:
            for idx in range(self.N):
                G0[idx,idx,:,:] = 0

        # shape into (N*3,N*3)
        G0 = np.transpose(G0,1,2).reshape(3*G0.shape[0],3*G0.shape[1]).to(np.complex128)
        return G0

    def mean_DOS_measurements(self, measure_points, k0, alpha, radius, self_interaction= True, self_interaction_type = "Rayleigh", regularize = False, discard_absorption = False):
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
        regularize - bool       bring everything below a scatterer radius to the center value, to be consistent with approximations and avoid divergences
        '''

        Npoints = measure_points.shape[0]
        k0_ = onp.round(k0/(2.0*onp.pi),1)
        print("Computing mean DOS using "+str(Npoints)+" points at k0L/2pi = "+str(k0_))

        # Define the matrix M_tensor = I_tensor - k^2 alpha Green_tensor
        M_tensor = -alpha*k0*k0*self.G0(None, k0, print_statement='DOS inverse')
        M_tensor.fill_diagonal_(1)
        if self_interaction:
            # Add self-interaction, (M_tensor)_ii = 1 - k^2 alpha self_interaction / volume
            dims = M_tensor.shape[0]
            volume = 4*onp.pi*(radius**3)/3
            M_tensor -= k0**2 * alpha * self_interaction_integral_vector(k0, radius, self_interaction_type) / volume * np.eye(dims)
        # Compute W_tensor = inverse(M_tensor)
        W_tensor = np.linalg.solve(M_tensor, np.eye(len(M_tensor), dtype=np.complex128))

        # Define the propagators from scatterers to measurement points
        G0_measure = self.G0(measure_points, k0, print_statement='DOS measure', regularize=regularize, radius = radius)
        # Check for measurement points falling exactly on scatterers
        for j in np.argwhere(np.isnan(G0_measure)):
            point_idx = j[0]
            scatter_idx = j[1]
            # At scatterers, replace G0(r_i, r_i) by self-interaction
            G0_measure[point_idx][scatter_idx] = 0
            if self_interaction:
                volume = 4*onp.pi*(radius**3)/3
                G0_measure[point_idx][scatter_idx] += self_interaction_integral_vector(k0, radius, self_interaction_type) / volume
        #  Use cyclic invariance of the trace: tr(G A G^T) = tr (G^T G A)
        # symm_mat = onp.matmul(onp.transpose(G0_measure), G0_measure)
        #  Use that trace(A.B^T) = AxB with . = matrix product and x = Hadamard product, and that G^T G is symmetric,
        dos_factor = ( np.matmul(G0_measure.t(), G0_measure) * W_tensor ).sum()/Npoints
        if discard_absorption:
            # Discard the imaginary part of alpha, only for the last part of the calculation https://www.jpier.org/pier/pier.php?paper=19111801
            alpha_ = onp.real(alpha)
        else:
            alpha_ = alpha
        dos_factor *= 2.0*onp.pi*k0*alpha_
        dos_factor = np.imag(dos_factor)

        return dos_factor

    def LDOS_measurements(self, measure_points, k0, alpha, radius, self_interaction= True, self_interaction_type = "Rayleigh", regularize = False, discard_absorption = False):
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
        regularize          - bool   bring everything below a scatterer radius to the center value, to be consistent with approximations and avoid divergences
        '''

        M = measure_points.shape[0]

        # Define the matrix M_tensor = I_tensor - k^2 alpha Green_tensor
        M_tensor = -alpha*k0*k0*self.G0(None, k0, print_statement='DOS inverse')
        M_tensor.fill_diagonal_(1)
        if self_interaction:
            # Add self-interaction, (M_tensor)_ii = 1 - k^2 alpha self_interaction / volume
            dims = M_tensor.shape[0]
            volume = 4*onp.pi*(radius**3)/3
            M_tensor -= k0**2 * alpha * self_interaction_integral_vector(k0, radius, self_interaction_type) / volume * np.eye(dims)
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
                volume = 4*onp.pi*(radius**3)/3
                G0_measure[point_idx][scatter_idx] += self_interaction_integral_vector(k0, radius, self_interaction_type) / volume
        # ldos_factor = onp.diagonal(onp.matmul(onp.matmul(G0_measure, W_tensor),onp.transpose(G0_measure)))
        # Can be made better considering it's a diagonal https://stackoverflow.com/questions/17437817/python-how-to-get-diagonalab-without-having-to-perform-ab
        ldos_factor = np.einsum('ij, ji->i',np.matmul(G0_measure, W_tensor), (G0_measure).t() )
        if discard_absorption:
            # Discard the imaginary part of alpha, only for the last part of the calculation https://www.jpier.org/pier/pier.php?paper=19111801
            alpha_ = onp.real(alpha)
        else:
            alpha_ = alpha
        ldos_factor *= 2.0*onp.pi*k0*alpha_
        ldos_factor = np.imag(ldos_factor)
        ldos_factor = ldos_factor.reshape(M,3,-1)
        ldos_factor = np.sum(ldos_factor, 1)

        return ldos_factor

    def compute_eigenvalues_and_scatterer_LDOS(self, k0, alpha, radius, file_name, self_interaction= True, self_interaction_type = "Rayleigh", write_eigenvalues=True):
        '''
        Computes the eigenvalues of the Green's matrix, and the corresponding LDOS at scatterers.
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

        # Define the matrix M_tensor = I_tensor - k^2 alpha Green_tensor
        M_tensor = -alpha*k0*k0*self.G0(None, k0, print_statement='DOS inverse')
        M_tensor.fill_diagonal_(1)
        if self_interaction:
            # Add self-interaction, (M_tensor)_ii = 1 - k^2 alpha self_interaction / volume
            dims = M_tensor.shape[0]
            volume = 4*onp.pi*(radius**3)/3
            M_tensor -= k0**2 * alpha * self_interaction_integral_vector(k0, radius, self_interaction_type) / volume * np.eye(dims)
        # Compute the spectrum of the M_tensor
        deltas = np.linalg.eigvals(M_tensor)

        if write_eigenvalues:
            onp.savetxt(file_name+'_deltas_'+str(k0_)+'.csv', onp.stack([np.real(deltas).numpy(), np.imag(deltas).numpy()]).T)

        # Compute the trace part here
        dos_factor = ((1 - deltas)**2 / deltas).sum()/Npoints
        dos_factor *= 2.0 * onp.pi / (k0**3 * alpha)
        dos_factor = np.imag(dos_factor)

        return dos_factor
    
    def compute_eigenmodes_IPR(self, k0, alpha, radius, file_name, self_interaction = True, self_interaction_type = "Rayleigh", number_eigenmodes = 1, write_eigenvalues = True, sorting_type = 'IPR'):
    
        Npoints = self.r.shape[0]
        k0_ = onp.round(k0/(2.0*onp.pi),1)
        print("Computing spectrum and scatterer LDOS using "+str(Npoints)+" points at k0L/2pi = "+str(k0_))

        # Define the matrix M_tensor = I_tensor - k^2 alpha Green_tensor
        M_tensor = -alpha*k0*k0*self.G0(None, k0, print_statement='DOS inverse')
        M_tensor.fill_diagonal_(1)
        if self_interaction:
            # Add self-interaction, (M_tensor)_ii = 1 - k^2 alpha self_interaction / volume
            dims = M_tensor.shape[0]
            volume = 4*onp.pi*(radius**3)/3
            M_tensor -= k0**2 * alpha * self_interaction_integral_vector(k0, radius, self_interaction_type) / volume * np.eye(dims)
        # Compute the spectrum of the M_tensor
    
        # Works, maybe consider scipy.schur instead, and output IPRs + one / some eigenvector(s) for plotting purposes
        deltas, eigenvectors = np.linalg.eig(M_tensor)
        IPRs = np.sum(np.abs(eigenvectors**4), axis = 0) / (np.sum(np.abs(eigenvectors**2), axis = 0))**2
        
        lambdas = (1.0 - deltas) / (k0**2 * alpha)
        
        kares = onp.sqrt(1.0 + 3.0/onp.real(alpha/volume))
        gamma0 = (2.0 / 9.0) * (1 + 3.0 / onp.real(alpha/volume))
        
        # Dedimensionalize like in Monsarrat
        lambdas = 6.0 * volume * k0**2 * lambdas / (kares * gamma0)
        
        # utils.plot_IPR_damping_values(deltas, IPRs, file_name+'_deltas', logscale=True, appended_string=str(k0_))
        utils.plot_IPR_damping_values(lambdas, IPRs, file_name+'_lambdas', logscale=True, appended_string=str(k0_))
        # utils.plot_IPR_damping_values(1-deltas, IPRs, file_name+'_test'+extra_string, logscale=True, appended_string=str(k0_))
        
        if write_eigenvalues:
            onp.savetxt(file_name+'_lambdas_'+str(k0_)+'.csv', onp.stack([np.real(lambdas).numpy(), np.imag(lambdas).numpy(), IPRs]).T)
            
            
        if sorting_type == 'IPR':
            IPRs, indices = np.sort(IPRs, descending=True)
            lambdas = lambdas[indices]
            eigenvectors = eigenvectors[:,indices]
        elif sorting_type == 'damping':
            indices = np.argsort(np.imag(lambdas), descending= False) # Want SMALL dampings first
            lambdas = lambdas[indices]
            IPRs = IPRs[indices]
            eigenvectors = eigenvectors[:,indices]
        else:
            raise NotImplementedError
            
        
        returned_eigenvectors = eigenvectors[:, 0:number_eigenmodes]

        gamman = gamma0 * onp.imag(lambdas) / 2
        omegan = kares - gamma0 * onp.real(lambdas) / 2
        ratio = (gamman / 2) / ( (k0 * radius - omegan)**2 + (gamman / 2)**2 )
        print(np.mean(ratio)/onp.pi)
        # Debug plots
        # returned_eigenvalues = deltas[0:number_eigenmodes]
        # print(returned_eigenvalues)
        # print(IPRs.amax())
        # print(IPRs[0])
        
        return lambdas, returned_eigenvectors, IPRs
    
class Transmission3D_hmatrices:
    

    def __init__(self, points, source='beam'):
        self.r = points.reshape(-1,3)
        self.N = self.r.shape[0]
        self.source = source
        jlPkg.activate("Transmission3D")
        jlPkg.instantiate()
        jl.seval("using Transmission3D")
    
    
    def generate_source(self, points, k0, u, p, w, print_statement = ''):
        '''
        Generates the EM field of a source at a set of points

        points - (M,3)      coordinates of points
        k0     - (1)        frequency of source beam
        u      - (Ndirs, 3) propagation directions for the source
        p      - (Ndirs, 3) polarization directions for the source
        w      - (1)        beam waist for beam sources
        '''
        
        if self.source == 'beam':
            # Collimated beam with zero curvature,
            # from solution of the the paraxial approximation of Maxwell-Helmholtz,
            # see https://en.wikipedia.org/wiki/Gaussian_beam
            
            k0_ = onp.round(k0/(2.0*onp.pi),1)
            print('Calculating Beam Source at k0L/2pi = '+str(k0_)+' ('+print_statement+')')
            
            rpara = np.matmul(points,u.T)
            rperp = np.linalg.norm(points.reshape(-1,3,1) - rpara.reshape(points.shape[0],1,u.shape[0])*u.T.reshape(1,3,-1),axis=1)
            a = 2*rperp/(w*w*k0)
            E0j = np.exp(1j*rpara*k0-(rperp**2/(w*w*(1+1j*a))))/np.sqrt(1+1j*a)
            phi = np.arctan2(p[:,1], p[:,0]) #arctan(y/x)
            theta = np.arccos(p[:,2]) #arccos(z/r), r=1 for unit vector
            pvec = np.stack([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), p[:,2]])
            
            E0j = E0j.reshape(points.shape[0],1,-1)*pvec.reshape(1,3,-1)
            
        elif self.source == 'plane':
            # Infinitely extended Plane wave
            
            k0_ = onp.round(k0/(2.0*onp.pi),1)
            print('Calculating Plane Source at k0L/2pi = '+str(k0_)+' ('+print_statement+')')
            
            rpara = np.matmul(points,u.T)
            E0j = np.exp(1j*rpara*k0)
            phi = np.arctan2(p[:,1], p[:,0]) #arctan(y/x)
            theta = np.arccos(p[:,2]) #arccos(z/r), r=1 for unit vector
            pvec = np.stack([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), p[:,2]])
            
            E0j = E0j.reshape(points.shape[0],1,-1)*pvec.reshape(1,3,-1)
            
        elif self.source == 'point':
            # One electric point dipole emitting light at source_distance * L away
            source_distance = 2.0
            source_intensity = 1.0 * (k0 * source_distance)**2 * 4.0 * onp.pi
            
            k0_ = onp.round(k0/(2.0*onp.pi),1)
            print('Calculating Point Source at k0L/2pi = '+str(k0_)+' ('+print_statement+')')
            
            rpara = np.matmul(points,u.T)
            rperp = np.linalg.norm(points.reshape(-1,3,1) - rpara.reshape(points.shape[0],1,u.shape[0])*u.T.reshape(1,3,-1),axis=1)
            
            source_location = source_distance * (-u)
            dipole_moment = onp.sqrt(source_intensity) * p
            
            E0j = np.matmul(self.torch_greensTE(points.reshape(-1,1,3) - source_location.reshape(1,-1,3), k0), dipole_moment.type(np.complex128)).squeeze()

        elif self.source is None:
            
            E0j = np.zeros((points.shape[0],u.shape[1],u.shape[0]),dtype=np.complex128)

        else:
            raise NotImplementedError
        
        return E0j

    def solve(self, k0, alpha, u, p, radius, beam_waist, self_interaction = True, self_interaction_type = "Rayleigh"):
        '''
        Solves the EM field at each scatterer

        k0                  - (1)           frequency being measured
        alpha               - (1)           bare static polarizability at given k0
        u                   - (Ndirs, 3)    propagation directions for the source
        p                   - (Ndirs, 3)    polarization directions for the source
        radius              - (1)           radius of scatterers, used in self-interaction
        beam_waist          - (1)           beam waist of Gaussian beam source
        self_interaction    - (bool)        include or not self-interactions, defaults to True
        self_interaction_type - (string)      what order of approximation of S to use, "Rayleigh" or "full"
        '''

        # Generate source field for scatterer positions
        E0j = self.generate_source(self.r, k0, u, p, beam_waist, print_statement='solve') #(N,3,Ndirs)
        
        # Julia-side solver with Abstract Hierarchical Matrices
        regularize = False # Not needed for solve part, writing it as a variable to make it clear what it is
        use_lu = True # Whether to use an LU decomposition then solve from it, or to solve anew at every angle
        atol = 0 # Absolute tolerance used in HMatrices
        rtol = 1e-3 # Relative tolerance
        debug = False
        Ek = jl.Transmission3D.solve(self.r.numpy(), E0j.reshape(3*self.N,-1).numpy(), k0, alpha, radius, self_interaction, regularize = regularize, self_interaction_type = self_interaction_type, use_lu = use_lu, atol = atol, rtol = rtol, debug=debug)
        
        return Ek
    
    def propagate(self, points, Ek, k0, alpha, u, p, beam_waist, regularize = False, radius=0.0):
        '''
        Calculates the EM field at a set of measurement points

        points           - (M,3)         coordinates of all measurement points
        Ek               - (N*3)         electromagnetic field at each scatterer
        k0               - (1)           frequency being measured
        alpha            - (1)           bare static polarizability at given k0
        u                - (Ndirs, 3)    propagation directions for the source
        p                - (Ndirs, 3)    polarization directions for the source
        beam_waist       - (1)           beam waist
        regularize       - bool          bring everything below a scatterer radius to the center value, to be consistent with approximations and avoid divergences
        radius           - (1)           considered scatterer radius, only used for regularization 
        '''
        points = np.tensor(points)
        
        # ensure u and p are all unit vectors
        u /= np.linalg.norm(u,axis=-1).reshape(-1,1)
        p /= np.linalg.norm(p,axis=-1).reshape(-1,1)
        
        # check polarization is orthogonal to propagation
        assert np.sum(np.absolute(np.sum(u*p,axis=-1))) == 0

        # generate source field for measurement points
        E0j = self.generate_source(points, k0, u, p, beam_waist, print_statement='propagate') #(M,3,Ndirs)
        
        # Compute full field
        Ek_ = np.tensor(jl.Transmission3D.propagate(self.r.numpy(), points.numpy(), Ek, k0, alpha, radius, regularize).to_numpy()).reshape(E0j.shape) + E0j
        
        # Take care of cases in which measurement points are exactly scatterer positions
        for j in np.argwhere(np.isnan(Ek_[:,0,0])):
            if regularize:
                # If overlap, will just return the closest one
                possible_idx = np.nonzero(np.linalg.norm(self.r-points[j],axis=-1) <= radius)
                if possible_idx.shape[0] > 1:
                    idx = np.argmin(np.linalg.norm(self.r-points[j], axis = -1))
                else:
                    idx = possible_idx
                Ek_[j] = Ek.reshape(points.shape[0],3,-1)[idx]
            else:
                idx = np.nonzero(np.prod(self.r-points[j]==0,axis=-1))
                Ek_[j] = Ek.reshape(points.shape[0],3,-1)[idx]
                
        return Ek_
    
    def propagate_ss(self, points, k0, alpha, u, p, beam_waist, regularize = False, radius = 0.0):
        '''
        Calculates the EM field at a set of measurement points, using a single-scattering approximation

        points           - (M,3)         coordinates of all measurement points
        k0               - (1)           frequency being measured
        alpha            - (1)           bare static polarizability at given k0
        u                - (Ndirs, 3)    propagation directions for the source
        p                - (Ndirs, 3)    polarization directions for the source
        beam_waist       - (1)           beam waist
        regularize       - bool          bring everything below a scatterer radius to the center value, to be consistent with approximations and avoid divergences
        radius           - (1)           considered scatterer radius, only used for regularization 
        '''

        points = np.tensor(points)
        E0_meas = self.generate_source(points, k0, u, p, beam_waist, print_statement='propagate_ss')
        E0_scat = self.generate_source(self.r, k0, u, p, beam_waist, print_statement='propagate_ss')
        E0_scat = E0_scat.reshape(3*self.r.shape[0],-1)        
        Ek_ = np.tensor(jl.Transmission3D.propagate(self.r.numpy(), points.numpy(), E0_scat.numpy(), k0, alpha, radius, regularize).to_numpy().reshape(E0_meas.shape)) + E0_meas
        
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
    
    def mean_DOS_measurements(self, measure_points, k0, alpha, radius, self_interaction = True, self_interaction_type = "Rayleigh", regularize = False, discard_absorption = False):
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
        regularize - bool       bring everything below a scatterer radius to the center value, to be consistent with approximations and avoid divergences
        '''

        Npoints = measure_points.shape[0]
        k0_ = onp.round(k0/(2.0*onp.pi),1)
        print("Computing mean DOS using "+str(Npoints)+" points at k0L/2pi = "+str(k0_))

        ### Calculation
        dos_factor = jl.Transmission3D.mean_dos(self.r.numpy(), measure_points.numpy(), k0, alpha, radius, self_interaction, self_interaction_type = self_interaction_type, regularize = regularize, discard_absorption = discard_absorption)

        return dos_factor

    def LDOS_measurements(self, measure_points, k0, alpha, radius, self_interaction= True, self_interaction_type = "Rayleigh", regularize = False, discard_absorption = False):
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
        regularize          - bool   bring everything below a scatterer radius to the center value, to be consistent with approximations and avoid divergences
        '''
        
        ### Calculation
        ldos_factor = jl.Transmission3D.ldos(self.r.numpy(), measure_points.numpy(), k0, alpha, radius, self_interaction, self_interaction_type = self_interaction_type, regularize = regularize, discard_absorption = discard_absorption)
        ldos_factor = np.tensor(ldos_factor).unsqueeze(1)

        return ldos_factor

    def compute_eigenvalues_and_scatterer_LDOS(self, k0, alpha, radius, file_name, self_interaction = True, self_interaction_type = "Rayleigh", write_eigenvalues = True):
        '''
        Computes the eigenvalues of the Green's matrix, and the corresponding LDOS at scatterers, for TM and TE.
        This computation is way less expensive than the other LDOS, due to simple dependence on the eigenvalues
        measure_points      - (M,3)  coordinates of points where the LDOS is evaluated
        k0                  - (1)    frequency of source beam
        alpha               - (1)    bare static polarizability at given k0
        radius              - (1)    radius of the scatterers
        self_interaction    - (bool) include or not self-interactions, defaults to True 
        '''

        print("eigvals not implemented in HMatrices")
        sys.exit()

        k0_ = onp.round(k0/(2.0*onp.pi),1)
        Npoints = self.r.shape[0]
        print("Computing spectrum and scatterer LDOS using "+str(Npoints)+" points at k0L/2pi = "+str(k0_))

        deltas = jl.Transmission3D.spectrum(self.r.numpy(), k0, alpha, radius, self_interaction, self_interaction_type = self_interaction_type)
        
        if write_eigenvalues:
            onp.savetxt(file_name+'_deltas_'+str(k0_)+'.csv', onp.stack([np.real(deltas).numpy(), np.imag(deltas).numpy()]).T)

        # Compute the trace part here
        dos_factor = ((1 - deltas)**2 / deltas).sum()/Npoints
        dos_factor *= 2.0 * onp.pi / (k0**3 * alpha)
        dos_factor = np.imag(dos_factor)

        return dos_factor
    
    
class Transmission3D_scalar:
    

    def __init__(self, points, source='beam'):
        self.r = points.reshape(-1,3)
        self.N = self.r.shape[0]
        self.source = source
    
    def greens(self, r, k0, periodic = '', regularize = False, radius = 0.0):
        '''
        Torch implementation of the 3d Green's function for scalar waves, taking tensors as entries
        r          - (M,2)      distances to propagate over
        k0         - (1)        wave-vector of source beam in vacuum
        periodic   - str        change boundary conditions: '' = free, ('x', 'y', 'xy') = choices of possible periodic directions
        regularize - bool       bring everything below a scatterer radius to the center value, to be consistent with approximations and avoid divergences
        radius     - (1)        considered scatterer radius, only used for regularization
        '''
        N = r.shape[0]
        M = r.shape[1]
        if 'x' in periodic:
            r[:,:,0] += 0.5
            r[:,:,0] %= 1
            r[:,:,0] -= 0.5
        if 'y' in periodic:
            r[:,:,1] += 0.5
            r[:,:,1] %= 1
            r[:,:,1] -= 0.5
        if 'z' in periodic:
            r[:,:,2] += 0.5
            r[:,:,2] %= 1
            r[:,:,2] -= 0.5
        R = np.linalg.norm(r,axis=-1).reshape(N,M)

        if regularize:
            R = np.where(R < radius, 0.0, R)

        return np.exp(1j*k0*R)/(4*onp.pi*R)

    def generate_source(self, points, k0, u, w, print_statement = ''):
        '''
        Generates the EM field of a source at a set of points

        points - (M,3)      coordinates of points
        k0     - (1)        frequency of source beam
        u      - (Ndirs, 3) propagation directions for the source
        w      - (1)        beam waist for beam sources
        '''
        
        if self.source == 'beam':
            # Collimated beam with zero curvature,
            # from solution of the the paraxial approximation of Maxwell-Helmholtz,
            # see https://en.wikipedia.org/wiki/Gaussian_beam
            
            k0_ = onp.round(k0/(2.0*onp.pi),1)
            print('Calculating Beam Source at k0L/2pi = '+str(k0_)+' ('+print_statement+')')
            
            rpara = np.matmul(points,u.T)
            rperp = np.linalg.norm(points.reshape(-1,3,1) - rpara.reshape(points.shape[0],1,u.shape[0])*u.T.reshape(1,3,-1),axis=1)
            a = 2*rperp/(w*w*k0)
            E0j = np.exp(1j*rpara*k0-(rperp**2/(w*w*(1+1j*a))))/np.sqrt(1+1j*a)
            
        elif self.source == 'plane':
            # Infinitely extended Plane wave
            
            k0_ = onp.round(k0/(2.0*onp.pi),1)
            print('Calculating Plane Source at k0L/2pi = '+str(k0_)+' ('+print_statement+')')
            
            rpara = np.matmul(points,u.T)
            E0j = np.exp(1j*rpara*k0)
            
        elif self.source == 'point':
            # One electric point dipole emitting light at source_distance * L away
            source_distance = 2.0
            source_intensity = 1.0 * (k0 * source_distance)**2 * 4.0 * onp.pi
            
            k0_ = onp.round(k0/(2.0*onp.pi),1)
            print('Calculating Point Source at k0L/2pi = '+str(k0_)+' ('+print_statement+')')
            
            rpara = np.matmul(points,u.T)
            rperp = np.linalg.norm(points.reshape(-1,3,1) - rpara.reshape(points.shape[0],1,u.shape[0])*u.T.reshape(1,3,-1),axis=1)
            
            source_location = source_distance * (-u)
            dipole_moment = onp.sqrt(source_intensity)
            
            E0j = self.greens(points.reshape(-1,1,3) - source_location.reshape(1,-1,3), k0) * dipole_moment.type(np.complex128)
        
        elif self.source is None:
            
            E0j = np.zeros((points.shape[0],u.shape[1]),dtype=np.complex128)
        
        else:
            raise NotImplementedError
        
        return E0j
 
    def propagate(self, points, Ek, k0, alpha, u, beam_waist, regularize = False, radius = 0.0):
        '''
        Calculates the EM field at a set of measurement points

        points           - (M,3)         coordinates of all measurement points
        Ek               - (N*3)         electromagnetic field at each scatterer
        k0               - (1)           frequency being measured
        alpha            - (1)           bare static polarizability at given k0
        u                - (Ndirs, 3)    propagation directions for the source
        beam_waist       - (1)           beam waist
        regularize       - bool          bring everything below a scatterer radius to the center value, to be consistent with approximations and avoid divergences
        radius           - (1)           considered scatterer radius, only used for regularization 
        '''
        points = np.tensor(points)
        
        # ensure u are all unit vectors
        u /= np.linalg.norm(u,axis=-1).reshape(-1,1)

        # generate source field for measurement points
        E0j = self.generate_source(points, k0, u, beam_waist, print_statement='propagate') #(M,Ndirs)
        
        # calculate Ek field at all measurement points
        Ek_ = np.matmul(alpha*k0*k0*self.G0(points, k0, print_statement='propagate', regularize=regularize, radius=radius), Ek).reshape(points.shape[0],u.shape[0]) + E0j 

        # Take care of cases in which measurement points are exactly scatterer positions
        for j in np.argwhere(np.isnan(Ek_[:,0])):
            if regularize:
                # If overlap, will just return the closest one
                possible_idx = np.nonzero(np.linalg.norm(self.r-points[j],axis=-1) <= radius)
                if possible_idx.shape[0] > 1:
                    idx = np.argmin(np.linalg.norm(self.r-points[j], axis = -1))
                else:
                    idx = possible_idx
                Ek_[j] = Ek.reshape(points.shape[0],-1)[idx]
            else:
                idx = np.nonzero(np.prod(self.r-points[j]==0,axis=-1))
                Ek_[j] = Ek.reshape(points.shape[0],-1)[idx]
                
        return Ek_
   
    def solve(self, k0, alpha, u, radius, beam_waist, self_interaction = True, self_interaction_type = "Rayleigh"):
        '''
        Solves the EM field at each scatterer

        k0                  - (1)           frequency being measured
        alpha               - (1)           bare static polarizability at given k0
        u                   - (Ndirs, 3)    propagation directions for the source
        radius              - (1)           radius of scatterers, used in self-interaction
        beam_waist          - (1)           beam waist of Gaussian beam source
        self_interaction    - (bool)        include or not self-interactions, defaults to True 
        self_interaction_type - (string)      what order of approximation of S to use, "Rayleigh" or "full"

        '''

        # Generate source field for scatterer positions
        E0j = self.generate_source(self.r, k0, u, beam_waist, print_statement='solve') #(N,3,Ndirs)
        
        ### Calculate Ek field at each scatterer position
        # Define the matrix M_tensor = I_tensor - k^2 alpha Green_tensor
        M_tensor = -alpha*k0*k0*self.G0(None, k0, print_statement='solve')
        M_tensor.fill_diagonal_(1)
        if self_interaction:
            # Add self-interaction, (M_tensor)_ii = 1 - k^2 alpha self_interaction / volume
            dims = M_tensor.shape[0]
            volume = 4*onp.pi*(radius**3)/3
            M_tensor -= k0**2 * alpha * self_interaction_integral_scalar(k0, radius, self_interaction_type) / volume * np.eye(dims)
        # Solve M_tensor.Ek = E0j
        Ek = np.linalg.solve(M_tensor, E0j.reshape(self.N,-1)) 
        return Ek
    
    def propagate_ss(self, points, k0, alpha, u, beam_waist, regularize = False, radius = 0.0):
        '''
        Calculates the EM field at a set of measurement points, using a single-scattering approximation

        points           - (M,3)         coordinates of all measurement points
        k0               - (1)           frequency being measured
        alpha            - (1)           bare static polarizability at given k0
        u                - (Ndirs, 3)    propagation directions for the source
        beam_waist       - (1)           beam waist
        regularize       - bool          bring everything below a scatterer radius to the center value, to be consistent with approximations and avoid divergences
        radius           - (1)           considered scatterer radius, only used for regularization 
        '''

        points = np.tensor(points)
        E0_meas = self.generate_source(points, k0, u, beam_waist, print_statement='propagate_ss')
        E0_scat = self.generate_source(self.r, k0, u, beam_waist, print_statement='propagate_ss')
        E0_scat = E0_scat.reshape(self.r.shape[0],-1)        
        Ek_ = np.matmul(alpha*k0*k0* self.G0(points, k0, print_statement='propagate_ss', regularize=regularize, radius=radius), E0_scat).reshape(points.shape[0],-1) + E0_meas
        
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

    def G0(self, points, k0, print_statement='', regularize = False, radius = 0.0):
        '''
        Generate the Green's tensor for a set of positions

        points          - (N,3)      set of point positions, None indicates the saved point pattern
        k0              - (1)        frequency being measured
        print_statement - str        disambiguating string used when printing (default = empty)
        regularize - bool       bring everything below a scatterer radius to the center value, to be consistent with approximations and avoid divergences
        radius     - (1)        considered scatterer radius, only used for regularization
        '''

        # check if None
        if points == None:
            points_ = self.r
        else:
            points_ = points
        k0_ = onp.round(k0/(2.0*onp.pi),1)
        print("Calculating Green's function at k0L/2pi = "+str(k0_)+' ('+print_statement+')')
        # populate Green's tensor
        G0 = self.greens(points_.reshape(-1,1,3)-self.r.reshape(1,-1,3), k0, regularize=regularize, radius=radius) #shape is (M,N)

        # replace NaN entries resulting from divergence (r-r'=0)
        if points == None:
            for idx in range(self.N):
                G0[idx,idx] = 0

        return G0

    def mean_DOS_measurements(self, measure_points, k0, alpha, radius, self_interaction = True, self_interaction_type = "Rayleigh", regularize = False, discard_absorption = False):
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
        regularize - bool       bring everything below a scatterer radius to the center value, to be consistent with approximations and avoid divergences
        '''

        Npoints = measure_points.shape[0]
        k0_ = onp.round(k0/(2.0*onp.pi),1)
        print("Computing mean DOS using "+str(Npoints)+" points at k0L/2pi = "+str(k0_))

        # Define the matrix M_tensor = I_tensor - k^2 alpha Green_tensor
        M_tensor = -alpha*k0*k0*self.G0(None, k0, print_statement='DOS inverse')
        M_tensor.fill_diagonal_(1)
        if self_interaction:
            # Add self-interaction, (M_tensor)_ii = 1 - k^2 alpha self_interaction / volume
            dims = M_tensor.shape[0]
            volume = 4*onp.pi*(radius**3)/3
            M_tensor -= k0**2 * alpha * self_interaction_integral_scalar(k0, radius, self_interaction_type) / volume * np.eye(dims)
        # Compute W_tensor = inverse(M_tensor)
        W_tensor = np.linalg.solve(M_tensor, np.eye(len(M_tensor), dtype=np.complex128))

        # Define the propagators from scatterers to measurement points
        G0_measure = self.G0(measure_points, k0, print_statement='DOS measure', regularize=regularize, radius = radius)
        # Check for measurement points falling exactly on scatterers
        for j in np.argwhere(np.isnan(G0_measure)):
            point_idx = j[0]
            scatter_idx = j[1]
            # At scatterers, replace G0(r_i, r_i) by self-interaction
            G0_measure[point_idx][scatter_idx] = 0
            if self_interaction:
                volume = 4*onp.pi*(radius**3)/3
                G0_measure[point_idx][scatter_idx] += self_interaction_integral_scalar(k0, radius, self_interaction_type) / volume
        #  Use cyclic invariance of the trace: tr(G A G^T) = tr (G^T G A)
        # symm_mat = onp.matmul(onp.transpose(G0_measure), G0_measure)
        #  Use that trace(A.B^T) = AxB with . = matrix product and x = Hadamard product, and that G^T G is symmetric,
        dos_factor = ( np.matmul(G0_measure.t(), G0_measure) * W_tensor ).sum()/Npoints
        if discard_absorption:
            # Discard the imaginary part of alpha, only for the last part of the calculation https://www.jpier.org/pier/pier.php?paper=19111801
            alpha_ = onp.real(alpha)
        else:
            alpha_ = alpha
        dos_factor *= 4.0*onp.pi*k0*alpha_
        dos_factor = np.imag(dos_factor)

        return dos_factor

    def LDOS_measurements(self, measure_points, k0, alpha, radius, self_interaction = True, self_interaction_type = "Rayleigh", regularize = False, discard_absorption = False):
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
        regularize          - bool   bring everything below a scatterer radius to the center value, to be consistent with approximations and avoid divergences
        '''

        M = measure_points.shape[0]

        # Define the matrix M_tensor = I_tensor - k^2 alpha Green_tensor
        M_tensor = -alpha*k0*k0*self.G0(None, k0, print_statement='DOS inverse')
        M_tensor.fill_diagonal_(1)
        if self_interaction:
            # Add self-interaction, (M_tensor)_ii = 1 - k^2 alpha self_interaction / volume
            dims = M_tensor.shape[0]
            volume = 4*onp.pi*(radius**3)/3
            M_tensor -= k0**2 * alpha * self_interaction_integral_scalar(k0, radius, self_interaction_type) / volume * np.eye(dims)
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
                volume = 4*onp.pi*(radius**3)/3
                G0_measure[point_idx][scatter_idx] += self_interaction_integral_scalar(k0, radius, self_interaction_type) / volume
        # ldos_factor = onp.diagonal(onp.matmul(onp.matmul(G0_measure, W_tensor),onp.transpose(G0_measure)))
        # Can be made better considering it's a diagonal https://stackoverflow.com/questions/17437817/python-how-to-get-diagonalab-without-having-to-perform-ab
        ldos_factor = np.einsum('ij, ji->i',np.matmul(G0_measure, W_tensor), (G0_measure).t() )
        if discard_absorption:
            # Discard the imaginary part of alpha, only for the last part of the calculation https://www.jpier.org/pier/pier.php?paper=19111801
            alpha_ = onp.real(alpha)
        else:
            alpha_ = alpha
        ldos_factor *= 4.0*onp.pi*k0*alpha_
        ldos_factor = np.imag(ldos_factor)
        ldos_factor = ldos_factor.reshape(M,-1)
        ldos_factor = np.sum(ldos_factor, 1)

        return ldos_factor

    def compute_eigenvalues_and_scatterer_LDOS(self, k0, alpha, radius, file_name, self_interaction = True, self_interaction_type = "Rayleigh", write_eigenvalues = True):
        '''
        Computes the eigenvalues of the Green's matrix, and the corresponding LDOS at scatterers.
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

        # Define the matrix M_tensor = I_tensor - k^2 alpha Green_tensor
        M_tensor = -alpha*k0*k0*self.G0(None, k0, print_statement='DOS inverse')
        M_tensor.fill_diagonal_(1)
        if self_interaction:
            # Add self-interaction, (M_tensor)_ii = 1 - k^2 alpha self_interaction / volume
            dims = M_tensor.shape[0]
            volume = 4*onp.pi*(radius**3)/3
            M_tensor -= k0**2 * alpha * self_interaction_integral_scalar(k0, radius, self_interaction_type) / volume * np.eye(dims)
        # Compute the spectrum of the M_tensor
        deltas = np.linalg.eigvals(M_tensor)

        if write_eigenvalues:
            onp.savetxt(file_name+'_deltas_'+str(k0_)+'.csv', onp.stack([np.real(deltas).numpy(), np.imag(deltas).numpy()]).T)

        # Compute the trace part here
        dos_factor = ((1 - deltas)**2 / deltas).sum()/Npoints
        dos_factor *= 4.0 * onp.pi / (k0**3 * alpha)
        dos_factor = np.imag(dos_factor)

        return dos_factor
    
    def compute_eigenmodes_IPR(self, k0, alpha, radius, file_name, self_interaction = True, self_interaction_type = "Rayleigh", number_eigenmodes = 1, write_eigenvalues = True, sorting_type = 'IPR'):
    
        Npoints = self.r.shape[0]
        k0_ = onp.round(k0/(2.0*onp.pi),1)
        print("Computing spectrum and scatterer LDOS using "+str(Npoints)+" points at k0L/2pi = "+str(k0_))

        # Define the matrix M_tensor = I_tensor - k^2 alpha Green_tensor
        M_tensor = -alpha*k0*k0*self.G0(None, k0, print_statement='DOS inverse')
        M_tensor.fill_diagonal_(1)
        if self_interaction:
            # Add self-interaction, (M_tensor)_ii = 1 - k^2 alpha self_interaction / volume
            dims = M_tensor.shape[0]
            volume = 4*onp.pi*(radius**3)/3
            M_tensor -= k0**2 * alpha * self_interaction_integral_scalar(k0, radius, self_interaction_type) / volume * np.eye(dims)
        # Compute the spectrum of the M_tensor
    
        # Works, maybe consider scipy.schur instead, and output IPRs + one / some eigenvector(s) for plotting purposes
        deltas, eigenvectors = np.linalg.eig(M_tensor)
        IPRs = np.sum(np.abs(eigenvectors**4), axis = 0) / (np.sum(np.abs(eigenvectors**2), axis = 0))**2
        
        lambdas = (1.0 - deltas) / (k0**2 * alpha)
        
        kares = onp.sqrt(2.0/onp.real(alpha/volume))
        gamma0 = 4.0 / (3.0 * onp.real(alpha/volume))
        
        # Dedimensionalize like in Monsarrat
        lambdas = 4.0 * volume * k0**2 * lambdas / (kares * gamma0)
        
        # utils.plot_IPR_damping_values(deltas, IPRs, file_name+'_deltas', logscale=True, appended_string=str(k0_))
        utils.plot_IPR_damping_values(lambdas, IPRs, file_name+'_lambdas', logscale=True, appended_string=str(k0_))
        # utils.plot_IPR_damping_values(1-deltas, IPRs, file_name+'_test'+extra_string, logscale=True, appended_string=str(k0_))
        
        if write_eigenvalues:
            onp.savetxt(file_name+'_lambdas_'+str(k0_)+'.csv', onp.stack([np.real(lambdas).numpy(), np.imag(lambdas).numpy(), IPRs]).T)
            
            
        if sorting_type == 'IPR':
            IPRs, indices = np.sort(IPRs, descending=True)
            lambdas = lambdas[indices]
            eigenvectors = eigenvectors[:,indices]
        elif sorting_type == 'damping':
            indices = np.argsort(np.imag(lambdas), descending= False) # Want SMALL dampings first
            lambdas = lambdas[indices]
            IPRs = IPRs[indices]
            eigenvectors = eigenvectors[:,indices]
        else:
            raise NotImplementedError
            
        
        returned_eigenvectors = eigenvectors[:, 0:number_eigenmodes]


        gamman = gamma0 * onp.imag(lambdas) / 2
        omegan = kares - gamma0 * onp.real(lambdas) / 2
        ratio = (gamman / 2) / ( (k0 * radius - omegan)**2 + (gamman / 2)**2 )
        print(np.mean(ratio)/onp.pi)
        
        gn_order = np.argsort(np.real(lambdas), descending=False)
        gn_order = np.where(gn_order == lambdas.shape[0]-1, gn_order - 1, gn_order)
        gn = np.imag(lambdas[gn_order]) / (np.real(lambdas[gn_order+1] - np.real(lambdas[gn_order])))
        utils.plot_IPR_damping_values(lambdas, gn, file_name+'_lambdas_thouless', logscale=True, appended_string=str(k0_))

        # Debug plots
        # returned_eigenvalues = deltas[0:number_eigenmodes]
        # print(returned_eigenvalues)
        # print(IPRs.amax())
        # print(IPRs[0])
        
        return lambdas, returned_eigenvectors, IPRs
    
class Transmission3D_scalar_hmatrices:
    

    def __init__(self, points, source='beam'):
        self.r = points.reshape(-1,3)
        self.N = self.r.shape[0]
        self.source = source
        jlPkg.activate("Transmission3D")
        jl.seval("using Transmission3D")
        
    
    
    def generate_source(self, points, k0, u, w, print_statement = ''):
        '''
        Generates the EM field of a source at a set of points

        points - (M,3)      coordinates of points
        k0     - (1)        frequency of source beam
        u      - (Ndirs, 3) propagation directions for the source
        w      - (1)        beam waist for beam sources
        '''
        
        if self.source == 'beam':
            # Collimated beam with zero curvature,
            # from solution of the the paraxial approximation of Maxwell-Helmholtz,
            # see https://en.wikipedia.org/wiki/Gaussian_beam
            
            k0_ = onp.round(k0/(2.0*onp.pi),1)
            print('Calculating Beam Source at k0L/2pi = '+str(k0_)+' ('+print_statement+')')
            
            rpara = np.matmul(points,u.T)
            rperp = np.linalg.norm(points.reshape(-1,3,1) - rpara.reshape(points.shape[0],1,u.shape[0])*u.T.reshape(1,3,-1),axis=1)
            a = 2*rperp/(w*w*k0)
            E0j = np.exp(1j*rpara*k0-(rperp**2/(w*w*(1+1j*a))))/np.sqrt(1+1j*a)
            
        elif self.source == 'plane':
            # Infinitely extended Plane wave
            
            k0_ = onp.round(k0/(2.0*onp.pi),1)
            print('Calculating Plane Source at k0L/2pi = '+str(k0_)+' ('+print_statement+')')
            
            rpara = np.matmul(points,u.T)
            E0j = np.exp(1j*rpara*k0)
            
        elif self.source == 'point':
            # One electric point dipole emitting light at source_distance * L away
            source_distance = 2.0
            source_intensity = 1.0 * (k0 * source_distance)**2 * 4.0 * onp.pi
            
            k0_ = onp.round(k0/(2.0*onp.pi),1)
            print('Calculating Point Source at k0L/2pi = '+str(k0_)+' ('+print_statement+')')
            
            rpara = np.matmul(points,u.T)
            rperp = np.linalg.norm(points.reshape(-1,3,1) - rpara.reshape(points.shape[0],1,u.shape[0])*u.T.reshape(1,3,-1),axis=1)
            
            source_location = source_distance * (-u)
            dipole_moment = onp.sqrt(source_intensity)
            
            E0j = self.greens(points.reshape(-1,1,3) - source_location.reshape(1,-1,3), k0) * dipole_moment.type(np.complex128)
        
        elif self.source is None:
            
            E0j = np.zeros((points.shape[0],u.shape[1],u.shape[0]),dtype=np.complex128)
        
        else:
            raise NotImplementedError
        
        return E0j.reshape(points.shape[0], u.shape[0])

    def solve(self, k0, alpha, u, radius, beam_waist, self_interaction = True, self_interaction_type = "Rayleigh"):
        '''
        Solves the EM field at each scatterer

        k0                  - (1)           frequency being measured
        alpha               - (1)           bare static polarizability at given k0
        u                   - (Ndirs, 3)    propagation directions for the source
        radius              - (1)           radius of scatterers, used in self-interaction
        beam_waist          - (1)           beam waist of Gaussian beam source
        self_interaction    - (bool)        include or not self-interactions, defaults to True 
        self_interaction_type - (string)      what order of approximation of S to use, "Rayleigh" or "full"
        
        '''

        # Generate source field for scatterer positions
        E0j = self.generate_source(self.r, k0, u, beam_waist, print_statement='solve') #(N,3,Ndirs)
        
        # Julia-side solver with Abstract Hierarchical Matrices
        regularize = False # Not needed for solve part, writing it as a variable to make it clear what it is
        use_lu = True # Whether to use an LU decomposition then solve from it, or to solve anew at every angle
        atol = 0 # Absolute tolerance used in HMatrices
        rtol = 1e-3 # Relative tolerance
        debug = False
        Ek = jl.Transmission3D.solve_scalar(self.r.numpy(), E0j.reshape(self.N,-1).numpy(), k0, alpha, radius, self_interaction, self_interaction_type = self_interaction_type, regularize = regularize, use_lu = use_lu, atol = atol, rtol = rtol, debug=debug)
        
        return Ek
    
    def propagate(self, points, Ek, k0, alpha, u, beam_waist, regularize = False, radius = 0.0):
        '''
        Calculates the EM field at a set of measurement points

        points           - (M,3)         coordinates of all measurement points
        Ek               - (N*3)         electromagnetic field at each scatterer
        k0               - (1)           frequency being measured
        alpha            - (1)           bare static polarizability at given k0
        u                - (Ndirs, 3)    propagation directions for the source
        beam_waist       - (1)           beam waist
        regularize       - bool          bring everything below a scatterer radius to the center value, to be consistent with approximations and avoid divergences
        radius           - (1)           considered scatterer radius, only used for regularization 
        '''
        points = np.tensor(points)
        
        # ensure u and p are all unit vectors
        u /= np.linalg.norm(u,axis=-1).reshape(-1,1)

        # generate source field for measurement points
        E0j = self.generate_source(points, k0, u, beam_waist, print_statement='propagate') #(M,3,Ndirs)
        
        # Compute full field
        Ek_ = np.tensor(jl.Transmission3D.propagate_scalar(self.r.numpy(), points.numpy(), Ek, k0, alpha, radius, regularize).to_numpy()).reshape(E0j.shape) + E0j
        
        # Take care of cases in which measurement points are exactly scatterer positions
        for j in np.argwhere(np.isnan(Ek_[:,0])):
            if regularize:
                # If overlap, will just return the closest one
                possible_idx = np.nonzero(np.linalg.norm(self.r-points[j],axis=-1) <= radius)
                if possible_idx.shape[0] > 1:
                    idx = np.argmin(np.linalg.norm(self.r-points[j], axis = -1))
                else:
                    idx = possible_idx
                Ek_[j] = Ek.reshape(points.shape[0],-1)[idx]
            else:
                idx = np.nonzero(np.prod(self.r-points[j]==0,axis=-1))
                Ek_[j] = Ek.reshape(points.shape[0],-1)[idx]
                
        return Ek_
    
    def propagate_ss(self, points, k0, alpha, u, beam_waist, regularize = False, radius = 0.0):
        '''
        Calculates the EM field at a set of measurement points, using a single-scattering approximation

        points           - (M,3)         coordinates of all measurement points
        k0               - (1)           frequency being measured
        alpha            - (1)           bare static polarizability at given k0
        u                - (Ndirs, 3)    propagation directions for the source
        beam_waist       - (1)           beam waist
        regularize       - bool          bring everything below a scatterer radius to the center value, to be consistent with approximations and avoid divergences
        radius           - (1)           considered scatterer radius, only used for regularization 
        '''

        points = np.tensor(points)
        E0_meas = self.generate_source(points, k0, u, beam_waist, print_statement='propagate_ss')
        E0_scat = self.generate_source(self.r, k0, u, beam_waist, print_statement='propagate_ss')
        E0_scat = E0_scat.reshape(self.r.shape[0],-1)        
        Ek_ = np.tensor(jl.Transmission3D.propagate_scalar(self.r.numpy(), points.numpy(), E0_scat.numpy(), k0, alpha, radius, regularize).to_numpy().reshape(E0_meas.shape)) + E0_meas
        
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
    
    def mean_DOS_measurements(self, measure_points, k0, alpha, radius, self_interaction = True, self_interaction_type = "Rayleigh", regularize = False, discard_absorption = False):
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
        regularize - bool       bring everything below a scatterer radius to the center value, to be consistent with approximations and avoid divergences
        '''

        Npoints = measure_points.shape[0]
        k0_ = onp.round(k0/(2.0*onp.pi),1)
        print("Computing mean DOS using "+str(Npoints)+" points at k0L/2pi = "+str(k0_))

        ### Calculation
        dos_factor = jl.Transmission3D.mean_dos_scalar(self.r.numpy(), measure_points.numpy(), k0, alpha, radius, self_interaction, self_interaction_type = self_interaction_type, regularize = regularize, discard_absorption=discard_absorption)

        return dos_factor

    def LDOS_measurements(self, measure_points, k0, alpha, radius, self_interaction = True, self_interaction_type = "Rayleigh", regularize = False, discard_absorption = False):
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
        regularize          - bool   bring everything below a scatterer radius to the center value, to be consistent with approximations and avoid divergences
        '''
        
        ### Calculation
        ldos_factor = jl.Transmission3D.ldos_scalar(self.r.numpy(), measure_points.numpy(), k0, alpha, radius, self_interaction, self_interaction_type = self_interaction_type, regularize = regularize, discard_absorption = discard_absorption)
        ldos_factor = np.tensor(ldos_factor).unsqueeze(1)

        return ldos_factor

    def compute_eigenvalues_and_scatterer_LDOS(self, k0, alpha, radius, file_name, self_interaction = True, write_eigenvalues = True):
        '''
        Computes the eigenvalues of the Green's matrix, and the corresponding LDOS at scatterers, for TM and TE.
        This computation is way less expensive than the other LDOS, due to simple dependence on the eigenvalues
        measure_points      - (M,3)  coordinates of points where the LDOS is evaluated
        k0                  - (1)    frequency of source beam
        alpha               - (1)    bare static polarizability at given k0
        radius              - (1)    radius of the scatterers
        self_interaction    - (bool) include or not self-interactions, defaults to True 
        '''

        print("eigvals not implemented in HMatrices")
        sys.exit()

        k0_ = onp.round(k0/(2.0*onp.pi),1)
        Npoints = self.r.shape[0]
        print("Computing spectrum and scatterer LDOS using "+str(Npoints)+" points at k0L/2pi = "+str(k0_))

        deltas = jl.Transmission3D.spectrum(self.r.numpy(), k0, alpha, radius, self_interaction, self_interaction_type = self_interaction_type)
        
        if write_eigenvalues:
            onp.savetxt(file_name+'_deltas_'+str(k0_)+'.csv', onp.stack([np.real(deltas).numpy(), np.imag(deltas).numpy()]).T)

        # Compute the trace part here
        dos_factor = ((1 - deltas)**2 / deltas).sum()/Npoints
        dos_factor *= 2.0 * onp.pi / (k0**3 * alpha)
        dos_factor = np.imag(dos_factor)

        return dos_factor
