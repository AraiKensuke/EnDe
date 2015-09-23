import pickle
import numpy as _N
import matplotlib.pyplot as _plt
import kde as _kde
import scipy.stats as _ss
#import fitMvNormE as fMN   
import fitMvNorm as fMN   
import time as _tm
import kdeutil as _ku
import scipy.integrate as _si

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
    nTets   = 1
    bWmaze= False
    pX_Nm = None
    Lklhd = None
    kde   = None
    lmd   = None
    lmd0  = None

    #  xp   position grid.  need this in decode
    xp    = None
    xpr   = None   # reshaped xp
    dxp   = None

    #  current posterior model parameters
    po_us    = None
    po_covs  = None
    po_ws    = None

    #  initting fitMvNorm
    sepHash  = False
    sepHashMthd=0
    pctH     = None
    MS       = None
    kde      = False

    Bx       = None;     bx     = None;     Bm    = None

    def init(self, kde=False, bx=None, Bx=None, Bm=None):
        oo = self
        oo.kde = kde
        with open("marks.dump", "rb") as f:
            lm = pickle.load(f)
        oo.marks = lm.marks
        oo.nTets = oo.marks.shape[1]

        oo.Nx = lm.Nx;        oo.Nm = lm.Nm
        oo.xA = lm.xA;        oo.mA = lm.mA
        oo.mdim  = lm.k#kde.mdim
        oo.mvpos  = lm.mvpos
        try:
            if lm.mvpos_t is not None:
                oo.mvpos_t = lm.mvpos_t
        except AttributeError:
            oo.mvpos_t = None

        ####  spatial grid for evaluating firing rates
        oo.xp   = _N.linspace(-oo.xA, oo.xA, oo.Nx)  #  space points
        oo.xpr  = oo.xp.reshape((oo.Nx, 1))
        #  bin space for occupation histogram.  same # intvs as space points
        oo.dxp   = oo.xp[1] - oo.xp[0]
        oo.xb    = _N.empty(oo.Nx+1)
        oo.xb[0:oo.Nx] = oo.xp - 0.5*oo.dxp
        oo.xb[oo.Nx] = oo.xp[-1]+ 0.5*oo.dxp\
        ####

        shp  = [oo.Nx]    #  shape of kde
        shp.extend([oo.Nm]*oo.mdim)
        #oo.lmd= kde.kde.reshape(shp)

        #oo.lmdFLaT = oo.lmd.reshape(oo.Nx, oo.Nm**oo.mdim)
        oo.dt = lm.dt

        oo.pos  = lm.pos
        oo.Fp, oo.q2p = 1, 0.005

        oo.pX_Nm = _N.zeros((oo.pos.shape[0], oo.Nx))
        oo.Lklhd = _N.zeros((oo.nTets, oo.pos.shape[0], oo.Nx))

        oo.decmth = "kde"
        if not oo.kde:
            oo.decmth = "mxn"
            oo.mvNrm= []
            for nt in xrange(oo.nTets):
                oo.mvNrm.append(fMN.fitMvNorm(oo.ITERS, oo.M, oo.mdim + 1))
                oo.mvNrm[nt].ITERS = oo.ITERS

    def encode(self, t0, t1, initPriors=False, doTouchUp=False, MF=None):
        oo = self
        tt1 = _tm.time()
        oo.N = t1-t0


        if oo.mvpos_t is not None:   #  mvpos
            mt = _N.array(oo.mvpos_t)
            iis = _N.where((mt >= t0) & (mt <= t1))[0]
            oo.mvpos = _N.array(oo.pos[mt[iis]])

        dat = _N.empty(oo.N, dtype=list)
        stpos  = []   #  pos  @ time of spikes
        marks  = []   #  mark @ time of spikes
        nspks  = _N.zeros(oo.nTets, dtype=_N.int)
        for nt in xrange(oo.nTets):
            marks.append([])
            stpos.append([])
            nspks[nt]  = 0
            for n in xrange(t0, t1):         # oo.marks   "list of arrays"
                if oo.marks[n, nt] is not None:  # [arr(k-dim mark1), arr(k-dim mark2)]
                    themarks = oo.marks[n, nt]
                    for l in xrange(len(themarks)):
                        stpos[nt].append(oo.pos[n])
                        marks[nt].append(themarks[l])
                        nspks[nt] += 1

            print "nspikes tet %(tet)d %(s)d  from %(t0)d   %(t1)d" % {"t0" : t0, "t1" : t1, "s" : nspks[nt], "tet" : nt}

        oo.all_pos = oo.mvpos

        if oo.kde:
            oo.tr_pos   = []
            oo.tr_marks = []

            for nt in xrange(oo.nTets):
                oo.tr_pos.append(_N.array(stpos[nt]))
                oo.tr_marks.append(_N.array(marks[nt]))
        else:
            tt2 = _tm.time()
            if initPriors:
                for nt in xrange(oo.nTets):
                    oo.mvNrm[nt].init0(stpos[nt], marks[nt], 0, nspks[nt], sepHash=oo.sepHash, pctH=oo.pctH, MS=oo.MS, sepHashMthd=oo.sepHashMthd, doTouchUp=doTouchUp, MF=MF)
            tt3 = _tm.time()
            for nt in xrange(oo.nTets):
                oo.mvNrm[nt].fit(oo.mvNrm[nt].M, stpos[nt], marks[nt], 0, nspks[nt])
                oo.mvNrm[nt].set_priors_and_initial_values()
            tt4 = _tm.time()
            print (tt2-tt1)
            print (tt3-tt2)
            print (tt4-tt3)
        oo.setLmd0(t0, t1, nspks)

    def decode(self, t0, t1):
        oo = self
        ##  each 


        oo.pX_Nm[t0] = 1. / oo.Nx
        #oo.pX_Nm[t0, 0:10] = 151/5.
        A = _N.trapz(oo.pX_Nm[t0], dx=oo.dxp)
        oo.pX_Nm[t0] /= A

        oo.intgrd= _N.empty(oo.Nx)
        oo.intgrd2d= _N.empty((oo.Nx, oo.Nx))
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

        pNkmk0   = _N.exp(-oo.dt * oo.Lam_xk)  #  one for each tetrode
        pNkmk    = _N.ones(oo.Nx)

        dims     = _N.ones(oo.mdim, dtype=_N.int)*oo.Nm

        fxdMks = _N.empty((oo.Nx, oo.mdim+1))  #  for each pos, a fixed mark
        fxdMks[:, 0] = oo.xp

        pNkmk = _N.empty((oo.Nx, oo.nTets))


        tStart = _tm.time()
        for t in xrange(t0+1,t1): # start at 1 because initial condition
            #tt1 = _tm.time()
            for nt in xrange(oo.nTets):
                oo.Lklhd[nt, t] = pNkmk0[:, nt]

                #  build likelihood
                if oo.marks[t, nt] is not None:
                    nSpks = len(oo.marks[t, nt])

                    for ns in xrange(nSpks):
                        fxdMks[:, 1:] = oo.marks[t, nt][ns]
                        if oo.kde:
                            #(atMark, fld_x, tr_pos, tr_mks, all_pos, mdim, Bx, cBm, bx)
                            #_ku.kerFr(fxdMks[0, 1:], fxdMks[:, 0], oo.tr_pos, oo.tr_mks, oo.mvpos, oo.mdim, oo.Bx, oo.cBm, oo.bx)
                            oo.Lklhd[nt, t] *= _ku.kerFr(fxdMks[0, 1:], oo.xpr, oo.tr_pos[nt], oo.tr_marks[nt], oo.all_pos, oo.mdim, oo.Bx, oo.Bm, oo.bx)* oo.iocc
                        else:
                            oo.Lklhd[nt, t] *= oo.mvNrm[nt].evalAtFxdMks(fxdMks)*oo.lmd0[nt] * oo.iocc

            ttt1 =0
            ttt2 =0
            ttt3 =0

            #tt2 = _tm.time()

            #  transition convolved with previous posterior

            _N.multiply(oo.xTrs, oo.pX_Nm[t-1], out=oo.intgrd2d)   
            oo.intgrl = _N.trapz(oo.intgrd2d, dx=oo.dxp, axis=1)
            #for ixk in xrange(oo.Nx):   #  above trapz over 2D array
            #    oo.intgrl[ixk] = _N.trapz(oo.intgrd2d[ixk], dx=oo.dxp)

            #tt3 = _tm.time()
            oo.pX_Nm[t] = oo.intgrl * _N.product(oo.Lklhd[:, t], axis=0)
            A = _N.trapz(oo.pX_Nm[t], dx=oo.dxp)
            oo.pX_Nm[t] /= A
            #tt4 = _tm.time()
            #print "%(1).3e   %(2).3e   %(3).3e" % {"1" : (tt2-tt1), "2" : (tt3-tt2), "3" : (tt4-tt3)}
            #print "%(1).3e   %(2).3e" % {"1" : ttt1, "2" : ttt2}
        tEnd = _tm.time()
        print "decode   %(1).3e" % {"1" : (tEnd-tStart)}
    
    def setLmd0(self, t0, t1, nspks):
        """
        Lmd0.  
        """
        oo = self
        #####  
        oo.lmd0 = _N.empty(oo.nTets)
        oo.Lam_xk = _N.ones((oo.Nx, oo.nTets))
        
        if oo.kde:
            oo.marks
            ibx2 = 1./ (oo.bx*oo.bx)
            for nt in xrange(oo.nTets):
                oo.Lam_xk[:, nt] = _ku.Lambda(oo.xpr, oo.tr_pos[nt], oo.all_pos, oo.Bx, oo.bx)

            occ    = _N.sum(_N.exp(-0.5*ibx2*(oo.xpr - oo.all_pos)**2), axis=1)  #  this piece doesn't need to be evaluated for every new spike
            Tot_occ  = _N.sum(occ)
            oo.iocc = 1./(occ + Tot_occ*0.01)
        else:  #####  fit mix gaussian
            for nt in xrange(oo.nTets):
                cmps   = _N.zeros((oo.mvNrm[nt].M, oo.Nx))
                for m in xrange(oo.mvNrm[nt].M):
                    var = oo.mvNrm[nt].covs[m, 0, 0]
                    ivar = 1./var
                    cmps[m] = (1/_N.sqrt(2*_N.pi*var)) * _N.exp(-0.5*ivar*(oo.xp - oo.mvNrm[nt].us[m, 0])**2)

                y  = _N.sum(oo.mvNrm[nt].ms*cmps, axis=0)
                occ, tmp = _N.histogram(oo.mvpos, bins=oo.xb)

                Tot_occ  = _N.sum(occ)

            oo.iocc = 1. / (occ + Tot_occ*0.01)
            MargLam = y * oo.iocc
            oo.lmd0[nt]    = (nspks[nt] / ((t1-t0)*0.001)) / _N.trapz(MargLam, dx=oo.dxp)
            oo.Lam_xk[:, nt] = oo.lmd0[nt] * MargLam

    """
    def getMarks(self, t0, t1):
        oo = self
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
    """
