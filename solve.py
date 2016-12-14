from __future__ import division
import numpy as np
import ode as ode
import time
import numexpr as ne
import scipy
import scipy.special
import bottleneck as bn
import files

def vnorbit(r):
    r1 = r[None,:]
    r2 = r[:,None]
    R = ne.evaluate("r1-r2")
    ss = bn.ss(R,axis=2)
    d = ne.evaluate("sqrt(ss+0.01)**3")[:,:,None]
    F = ne.evaluate("R/d")
    sF = bn.nansum(F,axis=1)
    return sF

def gen_blackhole(N,c,v):
    r0 = np.zeros([N,3])
    v0 = np.zeros([N,3])
    r0 += c
    v0 += v 
    return r0,v0

def gen_galaxy(N,omega,vrange,c,v,vconst,rot=False):
    bh = 70 
    d0 = np.random.normal(0,omega,N)
    th0 = np.random.rand(N)*np.pi*2
    r0 = np.empty([N,3])
    r0[:,0] = d0*np.cos(th0)
    r0[:,1] = d0*np.sin(th0)
    r0[:,2] = np.random.normal(0,omega/10,N)
    r0n = r0/np.linalg.norm(r0,axis=1)[:,None]
    r0nm = np.linalg.norm(r0n[:,:-1],axis=1)**2
    r0n2 = np.concatenate((r0n[:,:-1],np.array([r0nm/r0n[:,2]]).T),axis=1)
    r0n2[:,1] *= -1
    r0n2 = r0n2/np.linalg.norm(r0n2,axis=1)[:,None]
    v0 = np.cross(r0n,r0n2)*np.sign(r0n[:,2])[:,None]
    myr = np.linalg.norm(r0,axis=1)
    base = np.sqrt(myr)
    xpowx = 0.2*(60*myr)**(1/(15*myr))
    bhc = base
    bhc[myr>=0.25] = 0 
    v0 *= (vconst*scipy.special.erf(np.linalg.norm(r0,axis=1)/(np.sqrt(2)*omega))+bh*bhc)[:,None]
    v0[:,:-1] += (np.random.rand(N,2)*vrange*2-vrange)*vconst*0.5*scipy.special.erf(np.linalg.norm(r0,axis=1)/(np.sqrt(2)*omega))[:,None]
    r0 += c
    v0 += v
    if rot:  
        sq = 1/np.sqrt(2)
        rotm = np.array([[1,0,0],[0,sq,-sq],[0,sq,sq]])
        for i in range(len(r0)):
            r0[i]=np.dot(rotm,r0[i])
            v0[i]=np.dot(rotm,v0[i])
    return r0,v0

if __name__ == "__main__":
    N = 1000
    tf = 1.5 
    nstep = 20000

    omega = 0.15
    vrange = 1

    orbitv = 18
    orbitvf = 1
    omega2=1e-5
    r0,v0 = gen_galaxy(N,omega,vrange,[-3,0,0],[0.5,0,0],32,rot=True)
    r1,v1 = gen_blackhole(N//100,[-3,0,0],[0.5,0,0])

    r0 = np.append(r0,r1,axis=0)
    v0 = np.append(v0,v1,axis=0)
    r1,v1 = gen_galaxy(N,omega,vrange,[3,0,0],[-0.5,0,0],32)
    r0 = np.append(r0,r1,axis=0)
    v0 = np.append(v0,v1,axis=0)
    r1,v1 = gen_blackhole(N//100,[3,0,0],[-0.5,0,0])
    r0 = np.append(r0,r1,axis=0)
    v0 = np.append(v0,v1,axis=0)


    tt,rr = ode.symplectic(vnorbit,r0,v0,0,tf,nstep)

    save = np.vstack([tt,rr])
    files.hdf5Save(save,"final.hdf5")
