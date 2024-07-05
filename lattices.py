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
