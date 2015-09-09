import pickle
import numpy as _N
import matplotlib.pyplot as _plt
import kde as _kde
import scipy.stats as _ss
#import fitMvNormE as fMN   
import fitMvNorm as fMN   

#  Decode unsorted spike train

#  In our case with no history dependence
#  p(X_t | dN_t, m_t) 

def reshapedInd(inds, dims, Nd):
    ind1D = 0
    for dm in xrange(Nd):
        ind1D += inds[dm]*_N.product(dims[dm+1:])
    return ind1D

def estimate_posFstd(x):   #  AR coefficients for position data
    N    = len(x)
    F0AA = _N.dot(x[0:-1], x[0:-1])
    F0BB = _N.dot(x[0:-1], x[1:])

    q2 = 0.1

    a_q2         = 1e-1;          B_q2         = 1e-4

    ITER = 150
    Fs   = _N.empty(ITER)
    F0   = 0.999
    q2s   = _N.empty(ITER)
    for it in xrange(ITER):
        F0std= _N.sqrt(q2/F0AA)
        F0a, F0b  = (-1 - F0BB/F0AA) / F0std, (1 - F0BB/F0AA) / F0std
        F0=F0BB/F0AA+F0std*_ss.truncnorm.rvs(F0a, F0b)
    
        #   sample q2
        a = a_q2 + 0.5*N  #  N + 1 - 1
        rsd_stp = x[1:] - F0*x[0:-1]
        BB = B_q2 + 0.5 * _N.dot(rsd_stp, rsd_stp)
        q2 = _ss.invgamma.rvs(a, scale=BB)
        Fs[it] = F0
        q2s[it] = q2

    return _N.mean(Fs[ITER/2:]), _N.mean(q2s[ITER/2:])

class simDecode():
    bWmaze= False
    pX_Nm = None
    kde   = None
    lmd   = None
    lmd0  = None

    #  xp   position grid.  need this in decode
    xp    = None
    dxp   = None

    #  current posterior model parameters
    po_us    = None
    po_covs  = None
    po_ws    = None

    #  initting fitMvNorm
    sepHash  = False
    pctH     = None
    MS       = None

    def init(self, usekde=False):
        oo = self
        with open("marks.dump", "rb") as f:
            lm = pickle.load(f)
        oo.marks = lm.marks

        #with open("kde.dump", "rb") as f:
        #    kde = pickle.load(f)

        oo.Nx = lm.Nx;        oo.Nm = lm.Nm
        oo.xA = lm.xA;        oo.mA = lm.mA
        #oo.Ntr= kde.Ntr
        oo.mdim  = lm.k#kde.mdim

        ####  spatial grid for evaluating firing rates
        oo.xp   = _N.linspace(-oo.xA, oo.xA, oo.Nx)  #  space points
        #  bin space for occupation histogram.  same # intvs as space points
        oo.dxp   = oo.xp[1] - oo.xp[0]
        oo.xb    = _N.empty(oo.Nx+1)
        oo.xb[0:oo.Nx] = oo.xp - 0.5*oo.dxp
        oo.xb[oo.Nx] = oo.xp[-1]+ 0.5*oo.dxp
        ####

        shp  = [oo.Nx]    #  shape of kde
        shp.extend([oo.Nm]*oo.mdim)
        #oo.lmd= kde.kde.reshape(shp)

        #oo.lmdFLaT = oo.lmd.reshape(oo.Nx, oo.Nm**oo.mdim)
        oo.dt = lm.dt

        oo.pos  = lm.pos
        oo.mvpos  = lm.mvpos
        #oo.Fp, oo.q2p = 0.995, 0.001#estimate_posFstd(oo.pos)
        oo.Fp, oo.q2p = 1, 0.005 #estimate_posFstd(oo.pos)

        oo.pX_Nm = _N.zeros((oo.pos.shape[0], oo.Nx))
        oo.mvNrm   = fMN.fitMvNorm(oo.ITERS, oo.M, oo.mdim + 1)
        oo.mvNrm.ITERS = oo.ITERS

    def encode(self, t0, t1, initPriors=False):
        oo = self
        oo.N = t1-t0

        dat = _N.empty(oo.N, dtype=list)
        stpos  = []   #  pos  @ time of spikes
        marks  = []   #  mark @ time of spikes

        nspks  = 0
        for n in xrange(t0, t1):         # oo.marks   "list of arrays"
            if oo.marks[n] is not None:  # [arr(k-dim mark1), arr(k-dim mark2)]
                themarks = oo.marks[n]
                for l in xrange(len(themarks)):
                    stpos.append(oo.pos[n])
                    marks.append(themarks[l])
                    nspks += 1

        print "nspikes %(s)d  from %(t0)d   %(t1)d" % {"t0" : t0, "t1" : t1, "s" : nspks}

        if initPriors:
            oo.mvNrm.init0(oo.M, oo.mdim+1, stpos, marks, 0, nspks, sepHash=oo.sepHash, pctH=oo.pctH, MS=oo.MS)
        oo.mvNrm.fit(oo.mvNrm.M, oo.mdim+1, stpos, marks, 0, nspks)
        oo.mvNrm.set_priors_and_initial_values()
        oo.setLmd0(t0, t1, nspks)

    def decode(self, t0, t1):
        oo = self
        ##  each 


        oo.pX_Nm[t0] = 1. / oo.Nx
        #oo.pX_Nm[t0, 0:10] = 151/5.
        A = _N.trapz(oo.pX_Nm[t0], dx=oo.dxp)
        oo.pX_Nm[t0] /= A

        oo.intgrd= _N.empty(oo.Nx)
        oo.intgrl = _N.empty(oo.Nx)
        oo.xTrs  = _N.empty((oo.Nx, oo.Nx))      #  Gaussian

        x  = _N.linspace(-oo.xA, oo.xA, oo.Nx)
        ##  (xk - a xk1)

        i = 0

        if not oo.bWmaze:
            for x1 in x:
                j = 0
                for x0 in x:  #  j is current position
                    oo.xTrs[i, j]  = _N.exp(-((x1-oo.Fp*x0)**2)/(2*oo.q2p)) 
                    j += 1
                i += 1

        else:
            #  second index of xTrs is old position, first index is "to"
            for j in xrange(oo.Nx):
                x0  = x[j]
                for i in xrange(oo.Nx):
                    x1  = x[i]
                    oo.xTrs[i, j]  = _N.exp(-((x1-x0)**2)/(2*oo.q2p))   # 

                if x0 > 0:  #  right turns
                    for i in xrange(oo.Nx):
                        x1  = x[i]
                        if x1 > 0:   #  x1 = 1, x0=5  (x1-(5-6))**2 
                            oo.xTrs[i, j]  += _N.exp(-(x1-(x0-oo.xA))**2/(2*oo.q2p))   # 
                        elif x1 < 0:#  x1 = -1, x0=5  (x1-(6-5))**2
                            oo.xTrs[i, j]  += _N.exp(-(x1-(oo.xA-x0))**2/(2*oo.q2p))   # 

                    if (x0 > 5):  #  turning around like x0 = -5.3  --> 0.7
                        x0r = x0-oo.xA
                        ib4M1  = _N.where((x[0:-1] < x0r) & ( x[1:] > x0r))[0]
                        
                        for i in xrange(0, ib4M1[0]+1):
                            x1  = x[i]
                            oo.xTrs[i, j]  += _N.exp(-(x1-x0r)**2/(2*oo.q2p)) 


                elif x0 < 0:  #  left turns
                    for i in xrange(oo.Nx):
                        x1  = x[i]
                        if x1 > 0:   #  x1 = 1, x0=5  (x1-(5-6))**2 
                            oo.xTrs[i, j]  += _N.exp(-(x1-(-oo.xA-x0))**2/(2*oo.q2p))   # 
                        elif x1 < 0:#  x1 = -1, x0=5  (x1-(6-5))**2
                            oo.xTrs[i, j]  += _N.exp(-(x1-(x0-(-oo.xA)))**2/(2*oo.q2p))   # 

                    if (x0 < -5):  #  turning around like x0 = -5.3  --> 0.7
                        x0r = oo.xA + x0
                        ib4P1  = _N.where((x[0:-1] < x0r) & ( x[1:] > x0r))[0]

                        for i in xrange(ib4P1[0]+1, oo.Nx):
                            x1  = x[i]
                            oo.xTrs[i, j]  += _N.exp(-(x1-x0r)**2/(2*oo.q2p)) 

                #oo.xTrs[:, j] += _N.mean(oo.xTrs[:, j])*0.01
                A = _N.trapz(oo.xTrs[:, j])*((2.*oo.xA)/oo.Nx)
                oo.xTrs[:, j] /= A



        #  keep in mind that k_{k-1} is not treated as a value with a correct answer.
        #  integrate over all possible values of x_{k-1}

        #  Need value of integrand for all x_{k-1}
        #  I will perform integral L times for each time step
        #  multiply integral with p(\Delta N_k, m_k | x_k)

        pNkmk0   = _N.exp(-oo.dt * oo.Lam_xk)
        pNkmk    = _N.ones(oo.Nx)

        dims     = _N.ones(oo.mdim, dtype=_N.int)*oo.Nm

        fxdMks = _N.empty((oo.Nx, oo.mdim+1))  #  for each pos, a fixed mark
        fxdMks[:, 0] = oo.xp

        for t in xrange(t0+1,t1): # start at 1 because initial condition
            pNkmk[:] = pNkmk0

            #  build likelihood
            if oo.marks[t] is None:
                pNkmk[:] = pNkmk0
            else:
                nSpks = len(oo.marks[t])

                for ns in xrange(nSpks):
                    fxdMks[:, 1:] = oo.marks[t][ns]
                    pNkmk[:] *= oo.mvNrm.evalAtFxdMks(fxdMks)*oo.lmd0 * oo.iocc

            #  transition convolved with previous posterior

            for ixk in xrange(oo.Nx):
                _N.multiply(oo.xTrs[ixk], oo.pX_Nm[t-1], out=oo.intgrd)
                oo.intgrl[ixk] = _N.trapz(oo.intgrd, dx=oo.dxp)

            oo.pX_Nm[t] = oo.intgrl * pNkmk
            A = _N.trapz(oo.pX_Nm[t], dx=oo.dxp)
            oo.pX_Nm[t] /= A
    
    def setLmd0(self, t0, t1, nspks):
        """
        Lmd0.  
        """
        oo = self
        #####  

        cmps   = _N.zeros((oo.mvNrm.M, oo.Nx))
        for m in xrange(oo.mvNrm.M):
            var = oo.mvNrm.covs[m, 0, 0]
            ivar = 1./var
            cmps[m] = (1/_N.sqrt(2*_N.pi*var)) * _N.exp(-0.5*ivar*(oo.xp - oo.mvNrm.us[m, 0])**2)

        y  = _N.sum(oo.mvNrm.ms*cmps, axis=0)
        occ, tmp = _N.histogram(oo.mvpos, bins=oo.xb)

        Tot_occ  = _N.sum(occ)

        oo.iocc = 1. / (occ + Tot_occ*0.01)
        MargLam = y * oo.iocc
        oo.lmd0    = (nspks / ((t1-t0)*0.001)) / _N.trapz(MargLam, dx=oo.dxp)
        oo.Lam_xk = oo.lmd0 * MargLam

    def getMarks(self, t0, t1):
        oo = self
        """
        """
        L = len(oo.marks)

        store = []
        for t in xrange(t0, t1):
            if oo.marks[t] is not None:
                for i in xrange(len(oo.marks[t])):
                    store.append(oo.marks[t][i])

        return _N.array(store)

    def lklhd(self, t0, t1):
        oo = self
        pNkmk   = _N.empty((t1-t0, oo.Nx))

        for t in xrange(t0, t1):
            pNkmk[t-t0]    = _N.exp(-oo.dt * oo.Lam_xk)

            fxdMks = _N.empty((oo.Nx, oo.mdim+1))  #  for each pos, a fixed mark
            fxdMks[:, 0] = oo.xp

            if oo.marks[t] is not None:
                fxdMks[:, 1:] = oo.marks[t][0]
                pNkmk[t-t0] *= oo.mvNrm.evalAtFxdMks(fxdMks)*oo.lmd0 * oo.iocc
        return pNkmk
