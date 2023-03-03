import numpy as onp
import torch as np
from scipy.spatial import Delaunay, Voronoi

def cut_circle(r, rad=0.5):
    idx = np.nonzero(np.linalg.norm(r,axis=-1)<=rad)
    r = np.squeeze(r[idx])
    return r

def add_displacement(r, dr=1e-6):
    disp = onp.random.random(r.shape[0])*2*onp.pi
    if r.shape[1]==2:
        disp = onp.vstack([onp.cos(disp),onp.sin(disp)]).T*dr
    else:
        disp = (onp.random.random(r.shape)-0.5)*dr
    return r + np.tensor(disp)

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


def quasicrystal(nspan=46, ndirs=5, mode=None,disp=0):
    N = 4096
    if mode != None:
        nspan=33
    dirs = onp.arange(ndirs).reshape(-1,1)
    angles = dirs*2*onp.pi/ndirs
    vx = onp.cos(angles)
    vy = onp.sin(angles)
    mm = vy/vx
    span = onp.arange(nspan).reshape(1,-1)
    # 0.1427, 0.1913
    y1 = vy*dirs*0.2 - vx*(span-(nspan/2))
    x1 = vx*dirs*0.2 + vy*(span-(nspan/2))
    b = y1 - mm*x1
    #[5,61]
    x0 = (b[:-1,:].reshape(ndirs-1,-1,1,1) - b[1:,:].reshape(1,1,ndirs-1,-1))/(mm[1:].reshape(1,1,ndirs-1,1)-mm[:-1].reshape(ndirs-1,1,1,1))
    #[4,61,4,61]
    y0 = mm[:-1].reshape(ndirs-1,1,1,1)*x0 + b[:-1,:].reshape(ndirs-1,-1,1,1)
    index = onp.trunc(vy.reshape(1,1,1,1,ndirs)*onp.expand_dims(x0,axis=-1)- vx.reshape(1,1,1,1,ndirs)*(onp.expand_dims(y0,axis=-1)-b[:,0].reshape(1,1,1,1,ndirs)))
    #[4,61,4,61,5]
    points = []
    for idx in range(4):
        index[idx,:,:,:,idx] = span.reshape(-1,1,1)-1
        index[:,:,idx,:,idx+1] = span.reshape(1,1,-1)-1
    points.append(onp.trunc(index))
    for idx in range(4):
        index[idx,:,:,:,idx] = span.reshape(-1,1,1)
        index[:,:,idx,:,idx+1] = span.reshape(1,1,-1)-1
    points.append(onp.trunc(index))
    for idx in range(4):
        index[idx,:,:,:,idx] = span.reshape(-1,1,1)-1
        index[:,:,idx,:,idx+1] = span.reshape(1,1,-1)
    points.append(onp.trunc(index))
    for idx in range(4):
        index[idx,:,:,:,idx] = span.reshape(-1,1,1)
        index[:,:,idx,:,idx+1] = span.reshape(1,1,-1)
    points.append(onp.trunc(index))
    index = onp.array(points).reshape(-1,ndirs)
    index[index<0] = np.nan
    index[index>nspan-3] = np.nan
    index = onp.unique(index,axis=0)
    x = onp.dot(vx.ravel(),index.T)
    y = onp.dot(vy.ravel(),index.T)
    #y += 0.5*vy[:-1].reshape(-1,1,1,1) + 0.5*vy[1:].reshape(1,1,-1,1)
    r = onp.vstack([x.ravel(),y.ravel()]).T
    r = r[onp.isfinite(x.ravel())*onp.isfinite(y.ravel())]
    r = onp.unique(r,axis=0)
    r /= r.max()
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
    rabs = onp.absolute(r)
    points = r[onp.nonzero((rabs[:,0]<=0.5)*(rabs[:,1]<=0.5))]+0.5
    r = np.tensor(r)
    if disp != 0:
        r = add_displacement(r,dr=disp)
    return r

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
