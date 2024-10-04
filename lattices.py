import numpy as onp
import torch as np
import sys
from scipy.spatial import Delaunay, Voronoi
from utils import uniform_unit_disk_picking, uniform_unit_ball_picking

def cut_circle(r, rad=0.5):
    idx = np.nonzero(np.linalg.norm(r,axis=-1)<=rad)
    r = np.squeeze(r[idx])
    return r

def exclude_circle(r, rad=0.25):
    idx = np.nonzero(np.linalg.norm(r,axis=-1)>rad)
    r = np.squeeze(r[idx])
    return r

def add_displacement(r, dr=1e-6):
    if r.shape[1]==2:
        disp = dr * uniform_unit_disk_picking(r.shape[0])
    else:
        disp = dr * uniform_unit_ball_picking(r.shape[0], 3)
    return r + disp

def poisson(N, ndim):
    r = np.rand(N, ndim, dtype=np.double) - 0.5
    return r

def square(Nside=65,centered=True,disp=0):
    N = Nside*Nside
    x = np.arange(Nside)
    grid = np.zeros((Nside,Nside,2),dtype=np.double)
    grid[:,:,0] = x.reshape(-1,1)
    grid[:,:,1] = x.reshape(1,-1)
    r = grid.reshape(-1,2)
    if centered:
        r /= Nside-1
    else:
        r /= Nside
    r -= 0.5
    if disp != 0:
        r = add_displacement(r,dr=disp)
    return r

def triangular(Nx=71,Ny=41,disp=0):
    N = Nx*Ny
    x = np.arange(-Nx,Nx+1,dtype=np.double)*onp.sqrt(3)/2
    y = np.arange(-Ny,Ny+1,dtype=np.double)
    grid = np.zeros((x.shape[0],y.shape[0],2))
    grid[:,:,0] = x.reshape(-1,1)
    grid[:,:,1] = y.reshape(1,-1)
    grid[::2,:,1] += 0.5
    r = grid.reshape(-1,2)
    #r += onp.random.random(2)
    r -= np.mean(r)
    r /= np.max(r[:,0])
    if disp != 0:
        r = add_displacement(r,dr=disp)
    r = r.to(np.double)
    return r

def honeycomb(Nx=71,Ny=41,disp=0):
    x = np.arange(Nx)*onp.sqrt(3)/2
    y = np.arange(Ny)*1.5
    grid = np.zeros((x.shape[0],y.shape[0],2),dtype=np.double)
    grid[:,:,0] = x.reshape(-1,1)
    grid[:,:,1] = y.reshape(1,-1)
    grid[::2,::2,1] += 0.5
    grid[1::2,1::2,1] += 0.5
    r = grid.reshape(-1,2)
    #r += onp.random.random(2)
    r /= Ny*1.5
    r -= 0.5
    if disp != 0:
        r = add_displacement(r,dr=disp)
    return r

def vogel_spiral(N = 1000):

    golden = (1 + 5 ** 0.5) / 2
    psi = 2.0 * np.pi / (golden**2)

    n = np.arange(N, dtype=np.double)

    x = np.cos(n * psi) * np.sqrt(n)
    y = np.sin(n * psi) * np.sqrt(n)

    r = np.vstack([x,y]).T
    r /= r.amax()
    r /= 2.0
    
    return r

def icosahedral_quasicrystal(nspan = 4, N=4096, offset = None, disp=0):
    '''
    Generate an icosahedral quasicrystal in 3d from a projection of a 6d lattice
    Adapted from https://github.com/joshcol9232/tiling/tree/1825644190ff08786c4ada890b088b533244c4b6
    '''
    
    if offset == None:
        offset = np.zeros(6) # Thin rhomboids at center?
    
    # From: https://physics.princeton.edu//~steinh/QuasiPartII.pdf
    sqrt5 = onp.sqrt(5)
    icos = [
        onp.array([(2.0 / sqrt5) * onp.cos(2 * onp.pi * n / 5),
                  (2.0 / sqrt5) * onp.sin(2 * onp.pi * n / 5),
                  1.0 / sqrt5])
        for n in range(5)
    ]
    icos.append(onp.array([0.0, 0.0, 1.0]))
    
    ico_basis = Basis(onp.array(icos), offset)
    
    # Run the algorithm. k_ranges sets the number of construction planes used in the method.
    # The function outputs a list of Cell objects.
    cells = dualgrid_method(ico_basis, nspan)
    
    r = []
    unique_indices = []  # Edges will be when distance between indices is 1
    for c in cells:
        for arr_index, i in enumerate(c.indices):
            i = list(i)
            if i not in unique_indices:
                unique_indices.append(i)
                r.append(c.verts[arr_index])
    r = onp.vstack(r)
    r = onp.unique(r, axis = 0)
    
    r /= r.max()
    rabs = onp.absolute(r)
    points = r[onp.nonzero((rabs[:,0]<=0.5)*(rabs[:,1]<=0.5))]+0.5
    r = np.from_numpy(r)

    # Get smaller numbers
    # XXX could make the construction above a bit more flexible, but need some logic in the nspan
    N_measured = points.shape[0]
    ratio = onp.cbrt(N/N_measured)
    if ratio > 1:
        print("Can't currently make a quasicrystal that big!")
        # sys.exit()
        print("Capping to "+str(N_measured)+" particles")
    else:
        #Multiply the surface area by ratio, or the max radius by sqrt(ratio)
        r = cut_circle(r, rad = ratio * 0.5)
        r *= 0.5/ratio
    if disp != 0:
        r = add_displacement(r,dr=disp)
        
    r /= r.max()
    
    return r

def quasicrystal(N = 4096, nspan=46, ndirs=5, mode="",disp=0, offset = None):
    # http://www.gregegan.net/APPLETS/12/deBruijnNotes.html
    if ndirs < 4:
        print("A quasicrystal needs at least 4-fold symmetry!")
        sys.exit()
    if mode != "":
        nspan=33
        
    if offset == None:
        offset = np.zeros(ndirs) # Thin rhombi at center
        # offset = np.ones(ndirs)/ndirs # Anti-Penrose? http://www.jcrystal.com/steffenweber/ftgallery/ftgallery.html column2
        # offset = 1/2 - np.ones(ndirs)/(2*ndirs) # Penrose column1
    
    intersectrange = onp.arange(-nspan+1, nspan)
    sizes = (ndirs, intersectrange.shape[0], ndirs, intersectrange.shape[0])

    x = onp.fromfunction(lambda k,j,s,q: (bjk_factor(j - nspan+1,k,offset,ndirs)*onp.sin(s*onp.pi/ndirs) - bjk_factor(q - nspan+1, s, offset, ndirs) * onp.sin(k*onp.pi/ndirs) ) / onp.sin( (k - s) * onp.pi / ndirs ), sizes, dtype = int  )
    y = onp.fromfunction(lambda k,j,s,q: (-bjk_factor(j - nspan+1,k,offset,ndirs)*onp.cos(s*onp.pi/ndirs) + bjk_factor(q - nspan+1, s, offset, ndirs) * onp.cos(k*onp.pi/ndirs) ) / onp.sin( (k - s) * onp.pi / ndirs ), sizes, dtype = int  )
    
    r = onp.vstack([x.ravel(),y.ravel()]).T
    r = r[onp.isfinite(x.ravel())*onp.isfinite(y.ravel())]
    r = onp.unique(r,axis=0)
    
    print("Raw quasicrystal contains "+str(r.shape[0])+" points")
    
    if mode == 'quasidual':
        centroids = []
        tri = Delaunay(r)
        for t in tri.simplices:
            p = onp.mean(r[t],axis=0)
            centroids.append(p)
        r = onp.asarray(centroids)
    elif mode == 'quasivoro':
        voro = Voronoi(r)
        r = onp.asarray(voro.vertices)
    elif mode == 'deBruijndual':
        centroids = []
        tri = Delaunay(r)
        for t in tri.simplices:
            p = onp.mean(r[t],axis=0)
            centroids.append(p)
        centroids = onp.asarray(centroids)
        r = project_dual_luftalla(centroids, ndirs, offset)
        
    r /= r.max()
    rabs = onp.absolute(r)
    points = r[onp.nonzero((rabs[:,0]<=0.5)*(rabs[:,1]<=0.5))]+0.5
    r = np.tensor(r)

    # Get smaller numbers
    # XXX could make the construction above a bit more flexible, but need some logic in the nspan
    N_measured = points.shape[0]
    ratio = onp.sqrt(N/N_measured)
    if ratio > 1:
        print("Can't currently make a quasicrystal that big!")
        # sys.exit()
        print("Capping to "+str(N_measured)+" particles")
    else:
        #Multiply the surface area by ratio, or the max radius by sqrt(ratio)
        r = cut_circle(r, rad = ratio * 0.5)
        r *= 0.5/ratio
    if disp != 0:
        r = add_displacement(r,dr=disp)
        
    r /= r.max()
    return r

def bjk_factor(j, k, offset, ndirs):

    b = onp.power(-1,k) * (offset.numpy()[k] - j - 1/2) * onp.sqrt(ndirs/2)
    
    return b

def project_dual_luftalla(points, ndirs, offset, angle_offset = 0):
    # Implements dual relation from https://drops.dagstuhl.de/opus/volltexte/2021/14018/pdf/OASIcs-AUTOMATA-2021-9.pdf
    # The dual is written for shifts of 1 between lines, while the Egan construction above uses sqrt(ndirs/2)
    scaling_factor = onp.sqrt(ndirs/2) 
    
    if ndirs % 2 == 1:
        angles = onp.arange(ndirs)*2.0*onp.pi/ndirs + angle_offset
    else:
        print("\n\nEven values can have issues for some offsets! 0 is the safest.\n\n")
        angles = onp.arange(ndirs)*1.0*onp.pi/ndirs + angle_offset
        offset *= onp.power(-1, range(ndirs))
        
    term1 = onp.outer(points[:,0], onp.cos(angles)) + onp.outer(points[:,1], onp.sin(angles))
    term1 /= scaling_factor
    term1 += offset.numpy()-0.5
    term1 = onp.ceil(term1)
    x = term1 * onp.cos(angles)
    y = term1 * onp.sin(angles)
    x = onp.sum(x, axis = 1)
    y = onp.sum(y, axis = 1)
    
    r = onp.vstack([x,y]).T
    r = r[onp.isfinite(x.ravel())*onp.isfinite(y.ravel())]
    r = onp.unique(r,axis=0)
    r /= scaling_factor # Not sure about that scaling though, but it doesn't matter for this generation code
    
    return(r)

def cubic(Nside=17,centered=True,disp=0,normalize=True):
    x = np.arange(Nside)
    grid = np.zeros((Nside,Nside,Nside,3),dtype=np.double)
    grid[:,:,:,0] = x.reshape(-1,1,1)
    grid[:,:,:,1] = x.reshape(1,-1,1)
    grid[:,:,:,2] = x.reshape(1,1,-1)
    r = grid.reshape(-1,3)
    if normalize:
        if centered:
            r /= Nside-1
        else:
            r /= Nside
        r -= 0.5
    if disp != 0:
        r = add_displacement(r,dr=disp)
    return r

def bcc(Nside=17,disp=0,normalize=True):
    r = cubic(Nside, normalize=False)
    r = np.cat((r,r+np.tensor([0.5,0.5,0.5]).reshape(1,3)),0)
    if normalize:
        r /= Nside-1
        r -= 0.5
    if disp != 0:
        r = add_displacement(r,dr=disp)
    return r

def fcc(Nside=17,disp=0,normalize=True):
    r = cubic(Nside,normalize=False)
    r = np.cat((r,r+np.tensor([0.5,0.5,0]).reshape(1,3),r+np.tensor([0.5,0,0.5]).reshape(1,3),r+np.tensor([0,0.5,0.5]).reshape(1,3)),0)
    if normalize:
        r /= Nside-1
        r -= 0.5
    if disp != 0:
        r = add_displacement(r,dr=disp)
    return r

def diamond(Nside=17,disp=0, normalize=True):
    r = fcc(Nside,normalize=False)
    r = np.cat((r,r+np.tensor([0.25,0.25,0.25]).reshape(1,3)),0)
    if normalize:
        r /= Nside-1
        r -= 0.5
    if disp != 0:
        r = add_displacement(r,dr=disp)
    return r

def simple_hexagonal(Nx=17,Ny=15, Nz=15, disp=0):
    N = Nx*Ny
    x = np.arange(-Nx,Nx+1,dtype=np.double)*onp.sqrt(3)/2
    y = np.arange(-Ny,Ny+1,dtype=np.double)
    z = np.arange(-Nz,Nz+1,dtype=np.double)
    grid = np.zeros((x.shape[0],y.shape[0],z.shape[0],3))
    grid[:,:,:,0] = x.reshape(-1,1,1)
    grid[:,:,:,1] = y.reshape(1,-1,1)
    grid[:,:,:,2] = y.reshape(1,1,-1)
    grid[::2,:,:,1] += 0.5
    r = grid.reshape(-1,3)
    #r += onp.random.random(2)
    r -= np.mean(r)
    r /= np.max(r[:,0])
    if disp != 0:
        r = add_displacement(r,dr=disp)
    r = r.to(np.double)
    return r


# XXX The following needs rewriting
# For now, this is a minimal working unit from https://github.com/joshcol9232/tiling/tree/1825644190ff08786c4ada890b088b533244c4b6

import itertools
from multiprocessing import Pool
from functools import partial
class Cell:
    """
    Class to hold a set of four vertices, along with additional information
    """
    def __init__(self, vertices, indices, intersection):
        """
        verts: Corner vertices of the real tile/cell.
        indices: The "grid space" indices of each vertex.
        
        """
        self.verts = vertices
        self.indices = indices
        self.intersection = intersection # The intersection which caused this cell's existance. Used for plotting

    def __repr__(self):
        return "Cell(%s)" % (self.indices[0])

    def __eq__(self, other):
        return self.indices == other.indices

    def is_in_filter(self, *args, **kwargs):
        """
        Utility function for checking whever the rhombohedron is in rendering distance
        `fast` just checks the first vertex and exits, otherwise if any of the vertices are inside the filter
        then the whole cell is inside filter
        """
        def run_filter(filter, filter_centre, filter_args=[], filter_indices=False, fast=False, invert_filter=False):
            if fast:
                if filter_indices:
                    return filter(self.indices[0], onp.zeros_like(self.indices), *filter_args)
                else:
                    return filter(self.verts[0], filter_centre, *filter_args)
            else:
                if filter_indices:
                    zero_centre = onp.zeros_like(self.indices)
                    for i in self.indices:
                        if filter(i, zero_centre, *filter_args):
                            return True
                    return False
                else:
                    for v in self.verts:
                        if filter(v, filter_centre, *filter_args):
                            return True
                    return False
        
        result = run_filter(*args, **kwargs)
        if kwargs["invert_filter"]:
            return not result
        else:
            return result
    
class Basis:
    """
    Utility class for defining a set of basis vectors. Has conversion functions between different spaces.
    """
    def __init__(self, vecs, offsets):
        self.vecs = vecs
        self.dimensions = len(self.vecs[0])
        self.offsets = offsets

    def realspace(self, indices):
        """
        Gives position of given indices in real space.
        """
        out = onp.zeros(self.dimensions, dtype=float)
        for j, e in enumerate(self.vecs):
            out += e * indices[j]

        return out

    def gridspace(self, r):
        """
        Returns where a "real" point lies in grid space.
        """
        out = onp.zeros(len(self.vecs), dtype=int)

        for j, e in enumerate(self.vecs):
            out[j] = int(np.ceil( onp.dot( r, self.vecs[j] ) - self.offsets[j] ))

        return out

    def get_possible_cells(self, decimals):
        """
        Function that finds all possible cell shapes in the final mesh.
        Number of decimal places required for finite hash keys (floats are hard to == )
        Returns a dictionary of volume : [all possible combinations of basis vector to get that volume]
        """
        shapes = {}  # volume : set indices

        for inds in itertools.combinations(range(len(self.vecs)), self.dimensions):
            vol = abs(onp.linalg.det(onp.matrix([self.vecs[j] for j in inds]))) # Determinant ~ volume

            if vol != 0:
                vol = onp.around(vol, decimals=decimals)
                if vol not in shapes.keys():
                    shapes[vol] = [inds]
                else:
                    shapes[vol].append(inds)

        return shapes
    
def dualgrid_method(basis, k_range, shape_accuracy=4, single_threaded=False):
    """
    de Bruijn dual grid method.
    Generates and returns cells from basis given in the range given.
    Shape accuracy is the number of decimal places used to classify cell shapes
    Returns: cells, possible cell shapes
    """
    # Get each set of parallel planes
    construction_sets = construction_sets_from_basis(basis)

    # `j_combos` corresponds to a list of construction sets to compare. For a 2D basis of length 3, it would be:
    #   (0, 1), (0, 2), (1, 2)
    # Which covers all possible combinations.
    j_combos = itertools.combinations(range(len(construction_sets)), basis.dimensions)

    # Find intersections between each of the plane sets, and retrive cells
    cells = []
    if single_threaded:
        for js in j_combos:
            cells.append(_get_cells_from_construction_sets(construction_sets, k_range, basis, shape_accuracy, js))
    else:
        # Use a `Pool` to distribute work between CPU cores.
        p = Pool()
        work_func = partial(_get_cells_from_construction_sets, construction_sets, k_range, basis, shape_accuracy)
        cells = p.map(work_func, j_combos)
        p.close()

    # Cells is a list of lists -> flatten to a flat 1D list
    return [cell_list for worker_result in cells for cell_list in worker_result]


def _get_cells_from_construction_sets(construction_sets, k_range, basis, shape_accuracy, js):
    """
    Retrieves all intersections between the first construction set in the index list, and the rest.
    """
    intersections, k_combos = construction_sets[js[0]].get_intersections_with(k_range, [construction_sets[j] for j in js[1:]])

    cells = []
    for i, intersection in enumerate(intersections):
        # Calculate neighbours for this intersection
        indices_set = _get_neighbours(intersection, js, k_combos[i], basis)
        vertices_set = []

        for indices in indices_set:
            vertex = basis.realspace(indices)
            vertices_set.append(vertex)

        vertices_set = onp.array(vertices_set)
        c = Cell(vertices_set, indices_set, intersection)
        cells.append(c)

    return cells


def _get_neighbours(intersection, js, ks, basis):
    """
    For a given intersection, this function returns the grid-space indices of the spaces surrounding the intersection.
    A "grid-space index" is an N dimensional vector of integer values where N is the number of basis vectors. Each element
    corresponds to an integer multiple of a basis vector, which gives the final location of the tile vertex.

    There will always be a set number of neighbours depending on the number of dimensions. For 2D this is 4 (to form a tile),
    for 3D this is 8 (to form a cube), etc...
    """
    # Each possible neighbour of intersection. See eq. 4.5 in de Bruijn paper
    # For example:
    # [0, 0], [0, 1], [1, 0], [1, 1] for 2D
    directions = onp.array(list(itertools.product(*[[0, 1] for _i in range(basis.dimensions)])))

    indices = basis.gridspace(intersection)

    # Load known indices into indices array
    for index, j in enumerate(js):
        indices[j] = ks[index]

    # Copy the intersection indices. This is then incremented for the remaining indices depending on what neighbour it is.
    neighbours = [ onp.array([ v for v in indices ]) for _i in range(len(directions)) ]

    # Quick note: Kronecker delta function -> (i == j) = False (0) or True (1) in python. Multiplication of bool is allowed
    # Also from de Bruijn paper 1.
    deltas = [onp.array([(j == js[i]) * 1 for j in range(len(basis.vecs))]) for i in range(basis.dimensions)]

    # Apply equation 4.5 in de Bruijn's paper 1, expanded for any basis len and extra third dimension
    for i, e in enumerate(directions): # e Corresponds to epsilon in paper
        neighbours[i] += onp.dot(e, deltas)

    return neighbours

def construction_sets_from_basis(basis):
    return [ ConstructionSet(e, basis.offsets[i]) for (i, e) in enumerate(basis.vecs) ]


class ConstructionSet:
    """
    A class to represent a set of parallel lines / planes / n dimensional parallel structure.
    It implements a single method to return all intersections with another ConstructionSet.
    """
    def __init__(self, normal, offset):
        """
        normal: Normal vector to this construction set.
        offset: Offset of these lines from the origin.
        """
        self.normal = normal
        self.offset = offset

    def get_intersections_with(self, k_range, others):
        """
        Calculates all intersections between this set of lines/planes and another.
        """
        dimensions = len(self.normal)
        # Pack Cartesian coefficients into matrix.
        # E.g ax + by + cz = d.     a, b, c for each
        coef_matrix = onp.array([self.normal, *[ o.normal for o in others ]])

        # Check for singular matrix
        if onp.linalg.det(coef_matrix) == 0:
            print("WARNING: Unit vectors form singular matrices.")
            return [], []

        # get inverse of coefficient matrix
        coef_inv = onp.linalg.inv(coef_matrix)

        k_combos = _get_k_combos(k_range, dimensions)

        # last part (d) of Cartiesian form.
        # Pack offsets into N dimensional vector, then + [integers] to get specific planes within set
        base_offsets = onp.array([self.offset, *[ o.offset for o in others ]])

        ds = k_combos + base_offsets # remaining part of cartesian form (d)
        intersections = onp.asarray( (coef_inv * onp.asmatrix(ds).T).T )

        return intersections, k_combos

def _get_k_combos(k_range, dimensions):
    """ 
    Returns all possible comparison between two sets of lines for dimension number "dimensions" and max k_range (index range)
    E.g for 2D with a k range of 1 this is: 

      k_combos = [[-1 -1] [-1  0] [-1  1] [ 0 -1] [ 0  0] [ 0  1] [ 1 -1] [ 1  0] [ 1  1]]
     
    Then, when comparing two 2D construction sets, this compares line (-1) of set 1, with line (-1) of set 2, etc...
    """
    return onp.array(list(itertools.product(*[ [k for k in range(1-k_range, k_range)] for _d in range(dimensions) ])))