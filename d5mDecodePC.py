import pickle
import numpy as _N
import matplotlib.pyplot as _plt
import kde as _kde
from rtree import index

#  Decode unsorted spike train

#  In our case with no history dependence
#  p(X_t | dN_t, m_t) 

def reshapedInd(inds, dims, Nd):
    ind1D = 0
    for dm in xrange(Nd):
        ind1D += inds[dm]*_N.product(dims[dm+1:])
    return ind1D

class simDecode():
    pX_Nm = None
    kde   = None
    lmd   = None
    def init(self, usekde=False):
        oo = self
        with open("marks.dump", "rb") as f:
            lm = pickle.load(f)
        
        if usekde:
            with open("kde.dump", "rb") as f:
                kde = pickle.load(f)

            oo.Nx = kde.Nx;        oo.Nm = kde.Nm
            oo.xA = kde.xA;        oo.mA = kde.mA
            oo.Ntr= kde.Ntr
            oo.lmd= kde.kde
        else:
            oo.Nx = lm.Nx;         oo.Nm = lm.Nm
            oo.xA = lm.xA;         oo.mA = lm.mA
            oo.lmd=lm.lmd
        oo.mdim  = lm.mdim
        oo.lmdRS = oo.lmd.reshape(oo.Nx, oo.Nm**oo.mdim)
        oo.N  = lm.N
        oo.dt = lm.dt
        oo.sd = lm.sd
        oo.x  = lm.x

        oo.dat = _N.empty(oo.N, dtype=list)
        for dat in lm.marks:
            print oo.x[dat[0]]
            if oo.dat[dat[0]] is None:
                oo.dat[dat[0]] = [dat[1]]
            else:
                oo.dat[dat[0]] = [dat[2]]
        oo.dx= (2*oo.xA)/float(oo.Nx)
        oo.dm= (oo.mA)/float(oo.Nm)
        oo.marks = lm.marks

    def decode(self):
        oo = self
        ##  Lmd_xk
        ##  each 

        oo.pX_Nm = _N.zeros((oo.N, oo.Nx))
        #oo.pX_Nm[0] = 1./(2*oo.xA)
        oo.pX_Nm[0, 0:10] = 151/5.
        A = _N.trapz(oo.pX_Nm[0], dx=oo.dx)
        oo.pX_Nm[0] /= A


        oo.intgrd= _N.empty(oo.Nx)
        oo.intgrl = _N.empty(oo.Nx)
        oo.xTrs  = _N.empty((oo.Nx, oo.Nx))      #  Gaussian

        x  = _N.linspace(-oo.xA, oo.xA, oo.Nx)
        ##  (xk - a xk1)   


        i = 0
        F = 0.98
        for x1 in x:
            j = 0
            for x0 in x:
                oo.xTrs[i, j]  = _N.exp(-((x1-F*x0)**2)/(2*oo.sd*oo.sd))
                j += 1
            i += 1


        oo.Lam_xk = oo.lmd
        for d in xrange(oo.mdim-1, -1, -1):
            oo.Lam_xk = _N.sum(oo.Lam_xk, axis=(d+1))*(float(oo.mA)/oo.Nm)  #  marginalize over mark space

        #  keep in mind that k_{k-1} is not treated as a value with a correct answer.
        #  integrate over all possible values of x_{k-1}

        #  Need value of integrand for all x_{k-1}
        #  I will perform integral L times for each time step
        #  multiply integral with p(\Delta N_k, m_k | x_k)


        pNkmk0   = _N.exp(-oo.dt * oo.Lam_xk)
        pNkmk    = _N.ones(oo.Nx)

        oo.dcd       = _N.empty(oo.N)
        dims     = _N.ones(oo.mdim, dtype=_N.int)*oo.Nm
        for t in xrange(1, 2500):
            pNkmk[:] = pNkmk0

            if oo.dat[t] is None:
                pNkmk[:] = pNkmk0
            else:
                print "t %(t)d     %(x).3f" % {"t" : t, "x" : oo.x[t]}
                pNkmk.fill(1)
                nSpks = len(oo.dat[t])
                print "nSpks %d" % nSpks
                
                inds = _N.array(oo.dat[t][0]/oo.dm, dtype=_N.int)
                pNkmk = pNkmk0 * oo.lmdRS[:, reshapedInd(inds, dims, oo.mdim)]
                for ns in xrange(1, nSpks):
                    print "> 1 spk at %d" % t
                    inds = _N.array(oo.dat[t][ns]/oo.dm, dtype=_N.int)
                    pNkmk[:] *= oo.lmdRS[:, reshapedInd(inds, dims, oo.mdim)]

            for ixk in xrange(oo.Nx):
                _N.multiply(oo.xTrs[ixk], oo.pX_Nm[t-1], out=oo.intgrd)
                oo.intgrl[ixk] = _N.trapz(oo.intgrd)

            oo.pX_Nm[t] = oo.intgrl * pNkmk
            A = _N.trapz(oo.pX_Nm[t], dx=oo.dx)
            oo.pX_Nm[t] /= A

