from __future__ import division
import numpy as np
import leastsquares as ls
import numexpr as ne
import numba
import time
from itertools import izip, count

def rk4(f,r0,a,b,N):
    tpoints,h = np.linspace(a,b,num=N+1,retstep=True)
    r = np.asarray(r0,float)
    rt = np.empty(tpoints.shape,np.ndarray) 
    for i in np.arange(tpoints.size):
        t = tpoints[i]
        rt[i] = r
        k1 = h*f(r,t)
        k2 = h*f(r+0.5*k1,t+0.5*h)
        k3 = h*f(r+0.5*k2,t+0.5*h)
        k4 = h*f(r+k3,t+h)
        r = r + (k1+2*k2+2*k3+k4)/6
        print i
    return tpoints,np.transpose(np.vstack(rt))

def ark4(f,r0,a,b,d):
    # d = d/(b-a)
    N = 10 
    h = (b-a)/N
    t = a 
    r = np.asarray(r0,float)
    rt = [r]
    tt = []
    def do_rk4(mr,mt,mh):
        k1 = mh*f(mr,mt)
        k2 = mh*f(mr+0.5*k1,mt+0.5*mh)
        k3 = mh*f(mr+0.5*k2,mt+0.5*mh)
        k4 = mh*f(mr+k3,mt+mh)
        return mr + (k1+2*k2+2*k3+k4)/6
    while t<b:
        rt = np.append(rt,[r],axis=0)
        tt = np.append(tt,t)
        rho = 0
        while rho<1:
            h0 = h
            r11 = do_rk4(r,t,h)
            r1 = do_rk4(r11,t+h,h)
            r2 = do_rk4(r,t,2*h)
            ersq = np.square(np.abs(r1-r2))
            total = np.sqrt(ersq[::4]+ersq[1::4])
            #total[total>30]=0
            euclid = np.max(total)/30
            #euclid = np.sqrt(np.sum(ersq))
            if euclid==0:
                h = 4*h
                h0 = h
                r11 = do_rk4(r,t,h)
                r1 = do_rk4(r11,t+h,h)
                while t+h0==t:
                    h = 4*h
                    h0 = h
                    r11 = do_rk4(r,t,h)
                    r1 = do_rk4(r11,t+h,h)
                break
            rho = d*h/euclid
            h = h*rho**0.25
        rt = np.append(rt,[r11],axis=0)
        tt = np.append(tt,t+h0)
        r = r1
        t = t+2*h0
        print t/b
    if tt[-1]<b:
        rt = np.append(rt,[r],axis=0)
        tt = np.append(tt,t)
    rt = np.delete(rt,0,axis=0)
    return tt,np.transpose(rt)

def rk2(f,r0,a,b,N):
    tpoints,h = np.linspace(a,b,num=N+1,retstep=True)
    r = np.asarray(r0,float)
    rt = np.empty(tpoints.shape,np.ndarray) 
    for i in np.arange(tpoints.size):
        t = tpoints[i]
        rt[i] = r
        k1 = h*f(r,t)
        k2 = h*f(r+0.5*k1,t+0.5*h)
        r = r + k2
    return tpoints,np.transpose(np.vstack(rt))

def euler(f,r0,a,b,N):
    tpoints,h = np.linspace(a,b,num=N+1,retstep=True)
    r = np.asarray(r0,float)
    rt = np.empty(tpoints.shape,np.ndarray) 
    for i in np.arange(tpoints.size):
        t = tpoints[i]
        rt[i] = r
        r = r + h*f(r,t) 
    return tpoints,np.transpose(np.vstack(rt))

#@numba.vectorize([numba.float64(numba.float64,numba.float64)])
def symplectic(f,r0,v0,a,b,N):

    d13 = 1/(2-2**(1/3))
    c14 = d13/2
    c23 = (1-2**(1/3))*c14
    d2 = -d13*2**(1/3)
    
    c2 = c23
    c3 = c23
    d1 = d13
    d3 = d13
    c1 = c14
    c4 = c14
    d4 = 0
    
    #d1 = 0.9196615230173999
    #d2 = 0.25/d1-d1/2
    #d3 = 1-d1-d2
    #c1 = d3
    #c2 = d2
    #c3 = d1
    
    #c1 = 0.0617588581356263250
    #c2 = 0.3389780265536433551
    #c3 = 0.6147913071755775662
    #c4 = -0.1405480146593733802
    #c5 = 0.1250198227945261338
    #d1 = 0.2051776615422863869
    #d2 = 0.4030212816042145870
    #d3 = -0.1209208763389140082
    #d4 = 0.5127219331924130343

    tpoints,h = np.linspace(a,b,num=N+1,retstep=True)
    r = np.array(r0,dtype=np.float64)
    v = np.array(v0,dtype=np.float64)
    #r = np.array(r0,dtype=np.float32)
    #v = np.array(v0,dtype=np.float32)
    dim = len(r[0])
    npoints = len(tpoints)
    rt = np.empty([npoints,len(r),dim],dtype=np.float64)
    vt = np.empty([npoints,len(r),dim],dtype=np.float64)
    #rt = np.empty([npoints,len(r),dim],dtype=np.float32)
    #vt = np.empty([npoints,len(r),dim],dtype=np.float32)
    for i in xrange(npoints):
        rt[i] = r
        vt[i] = v
        r += c1*h*v
        v += d1*h*f(r)
        r += c2*h*v
        v += d2*h*f(r)
        r += c3*h*v
        v += d3*h*f(r)
        r += c4*h*v
        #v += d4*h*f(r)
        #r += c5*h*v
        #v += d5*h*f(r)
        print i
        
    rt = rt.reshape((npoints,dim*len(r)))
    vt = vt.reshape((npoints,dim*len(r)))

    rp = np.transpose(np.vstack(rt))
    vp = np.transpose(np.vstack(vt))
    out = np.empty(rp.shape)
    out = np.append(out,out,axis=0)
    for i in xrange(len(rp)//dim):
        for j in xrange(dim):
            out[2*dim*i+j] = rp[dim*i+j]
            out[2*dim*i+j+dim] = vp[dim*i+j]
    return tpoints,out

def verlet(f,r0,v0,a,b,N):
    tpoints,h = np.linspace(a,b,num=N+1,retstep=True)
    r = np.array(r0,dtype=np.float_)
    v = np.array(v0,dtype=np.float_)
    vh = v+0.5*h*f(r)
    dim = len(r[0])
    npoints = len(tpoints)
    rt = np.empty([npoints,len(r),dim])
    vt = np.empty([npoints,len(r),dim])

    for i in xrange(npoints):
        rt[i] = r
        vt[i] = v
        r = r+h*vh
        k = h*f(r)
        v = vh+0.5*k
        vh = vh+k
        print i
    rt = rt.reshape((npoints,dim*len(r)))
    vt = vt.reshape((npoints,dim*len(r)))

    rp = np.transpose(np.vstack(rt))
    vp = np.transpose(np.vstack(vt))
    out = np.empty(rp.shape)
    out = np.append(out,out,axis=0)
    for i in range(len(rp)//dim):
        for j in range(dim):
            out[2*dim*i+j] = rp[dim*i+j]
            out[2*dim*i+j+dim] = vp[dim*i+j]
    return tpoints,out

def at_t(r,t,t0):
    if t0 in t:
        return r.T[np.where(t==t0)].flatten()
    i = 0
    while t[i]<t0:
        i += 1
    i -= 1
    x = np.array([t[i],t[i+1]])
    rt0 = []
    for s in r:
        m,c = ls.leastsquares(x,np.array([s[i],s[i+1]]))
        rt0.append(m*t0+c)
    return np.asarray(rt0)

if __name__ == "__main__":

    def f(r,t):
        x = r[0]
        y = r[1]
        fx = x*y*np.sin(x)-x-x*x
        fy = y-x*y*y*x*np.sin(x)*np.cos(y)
        return np.array([fx,fy],float)

    def f2(r,t):
        x = r[0]
        y = r[1]
        vx = r[2]
        vy = r[3]
        fx = vx
        fy = vy
        fvx = x/np.sqrt(x*x+y*y)**3+vx
        fvy = y/np.sqrt(x*x+y*y)**3
        return np.array([fx,fy,fvx,fvy],float)

    def f22(r,v,t):
        x = r[0]
        y = r[1]
        vx = v[0]
        vy = v[1]
        fx = x/np.sqrt(x*x+y*y)**3+vx
        fy = y/np.sqrt(x*x+y*y)**3
        return np.array([fx,fy],float)

    def f3(r,t):
        x = r[0]
        vx = r[1]
        fx = vx
        fvx = x+vx
        return np.array([fx,fvx],float)
    # x''=x+x'
    # x(0)=0, x'(0)=1
    # x(5)=1458.8986

    t,r = rk4(f,[1,1],0,10,385)
    print t.size
    print at_t(r,t,10)
    t,r = ark4(f,[1,1],0,10,1e-2)
    print at_t(r,t,10)
    print t.size
    t,r = rk4(f,[1,1],0,10,10000)
    print at_t(r,t,10)
    t,r = rk2(f,[1,1],0,10,1000)
    print r
    t,r = euler(f,[1,1],0,10,1000)
    print "ohno",r
    t,r = rk4(f3,[0,1],0,5,1000)
    print r
    t,r = rk4(f2,[1,1,-1,1],0,5,10000)
    print r
    t,r = verlet(f22,[1,1],[-1,1],0,5,10000)
    print r
