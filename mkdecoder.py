import numpy as _N
from EnDedirs import resFN, datFN
import kdeutil as _ku
import time as _tm

class mkdecoder:
    nTets   = 1
    pX_Nm = None   #  p(X | Nm)
    Lklhd = None
    kde   = None
    lmd   = None
    lmd0  = None
    mltMk = 1    #  multiply mark values to 

    marksObserved = None   #  observed in this encode epoch

    #  xp   position grid.  need this in decode
    xp    = None
    xpr   = None   # reshaped xp
    dxp   = None

    #  current posterior model parameters
    u_    = None
    covs_ = None
    f_    = None
    q2_   = None
    l0_   = None

    #  initting fitMvNorm
    kde      = False
    Bx       = None;     bx     = None;     Bm    = None

    tetfile  = "marks.pkl"
    usetets  = None
    utets_str= ""

    tt0      = None
    tt1      = None

    dbgMvt   = False
    spdMult   = 0.5

    Nx       = 50

    xLo      = 0
    xHi      = 3
    mLo      = -2
    mHi      = 8

    ##  X_   and _X
    def __init__(self, kde=False, bx=None, Bx=None, Bm=None, mkfns=None, encfns=None, K=None, nTets=None):
        """
        """
        oo = self
        oo.kde = kde

        oo.bx = bx;   oo.Bx = Bx;   oo.Bm = Bm
        oo.mkpos = []
        #  read mkfns
        for fn in mkfns:
            dat = _N.loadtxt(datFN("%s.dat" % fn))
            K   = dat.shape[1] - 2
            oo.mkpos.append(dat)

        oo.nTets = len(oo.mkpos)
        oo.mdim  = K
        oo.pos  = dat[:, 0]         #  length of 

        if not kde:
            oo.mdim  = K
            oo.nTets  = nTets
        
        ####  spatial grid for evaluating firing rates
        oo.xp   = _N.linspace(oo.xLo, oo.xHi, oo.Nx)  #  space points
        oo.xpr  = oo.xp.reshape((oo.Nx, 1))
        #  bin space for occupation histogram.  same # intvs as space points
        oo.dxp   = oo.xp[1] - oo.xp[0]
        oo.xb    = _N.empty(oo.Nx+1)
        oo.xb[0:oo.Nx] = oo.xp - 0.5*oo.dxp
        oo.xb[oo.Nx] = oo.xp[-1]+ 0.5*oo.dxp
        ####

        #oo.lmdFLaT = oo.lmd.reshape(oo.Nx, oo.Nm**oo.mdim)
        oo.dt = 0.001

        oo.pX_Nm = _N.zeros((oo.pos.shape[0], oo.Nx))
        oo.Lklhd = _N.zeros((oo.nTets, oo.pos.shape[0], oo.Nx))

        oo.decmth = "kde"
        #################################################################

        oo.intgrd= _N.empty(oo.Nx)
        oo.intgrd2d= _N.empty((oo.Nx, oo.Nx))
        oo.intgrl = _N.empty(oo.Nx)
        oo.xTrs  = _N.zeros((oo.Nx, oo.Nx))      #  Gaussian

        x  = _N.linspace(oo.xLo, oo.xHi, oo.Nx)
        ##  (xk - a xk1)

        i = 0

        grdsz = (float(oo.xHi-oo.xLo)/oo.Nx)
        spdGrdUnts = _N.diff(oo.pos) / grdsz  # unit speed ( per ms ) in grid units

        #  avg. time it takes to move 1 grid is 1 / _N.mean(_N.abs(spdGrdUnts))
        #  p(i+1, i) = 1/<avg spdGrdUnts>
        p1 = 2*_N.mean(_N.abs(spdGrdUnts))*oo.spdMult
        #  assume Nx is even
        #k2 = 0.02
        k2 = 0.1
        k3 = 0.1


        for i in xrange(0, oo.Nx):  #  indexing of xTrs  [to, from]
            oo.xTrs[i, i] = 1-p1   
            if i == 0:
                #oo.xTrs[0, oo.Nx-1] = p1*0.5
                oo.xTrs[0, oo.Nx-1] = p1*0.9  #  backwards
            if i >= 0:
                oo.xTrs[i-1, i] = p1*0.1
                oo.xTrs[i, i-1] = p1*0.9
            if i == oo.Nx-1:
                oo.xTrs[oo.Nx-1, 0] = p1*0.1

        """
        ##  W-maze
        for i in xrange(0, oo.Nx/2):
            oo.xTrs[i, i] = 1-p1
            if i > 0:
                oo.xTrs[i-1, i] = p1
            if i > 1:        ##  next nearest neighbor
                oo.xTrs[i-2, i] = p1*k2
                oo.xTrs[i+1, i] = p1*k2*k3
            elif i == 1:
                oo.xTrs[oo.Nx/2-1, 1] = p1*k2/2
                oo.xTrs[oo.Nx/2, 1]   = p1*k2/2
                oo.xTrs[i+1, i] = p1*k2*k3

        oo.xTrs[oo.Nx/2-1, 0] = p1/2
        oo.xTrs[oo.Nx/2, 0]   = p1/2
        for i in xrange(oo.Nx/2, oo.Nx):
            oo.xTrs[i, i] = 1-p1
            if i < oo.Nx - 1:
                oo.xTrs[i+1, i] = p1
            if i < oo.Nx - 2:
                oo.xTrs[i-1, i] = p1*k2*k3
                oo.xTrs[i+2, i] = p1*k2
            elif i == oo.Nx-2:
                oo.xTrs[i-1, i] = p1*k2*k3
                oo.xTrs[oo.Nx/2-1, oo.Nx-2] = p1*k2/2
                oo.xTrs[oo.Nx/2, oo.Nx-2]   = p1*k2/2
        oo.xTrs[oo.Nx/2-1, oo.Nx-1] = p1/2
        oo.xTrs[oo.Nx/2, oo.Nx-1]   = p1/2

            #oo.xTrs[:, j] += _N.mean(oo.xTrs[:, j])*0.01
        for i in xrange(oo.Nx):
            A = _N.trapz(oo.xTrs[:, i])*((oo.xHi-oo.xLo)/float(oo.Nx))
            oo.xTrs[:, i] /= A
        """

    def init_pX_Nm(self, t):
        oo = self
        oo.pX_Nm[t] = 1. / oo.Nx
        if oo.dbgMvt:
            oo.pX_Nm[t, 20:30] = 151/5.
        A = _N.trapz(oo.pX_Nm[t], dx=oo.dxp)
        oo.pX_Nm[t] /= A

    def decodeMoG(self, prms, uFE, t0, t1):
        """
        uFE    which epoch fit to use for encoding model
        prms posterior params
        use params to decode marks from t0 to t1
        """
        oo = self
        ##  each 

        for nt in xrange(oo.nTets):
            l0s   = prms[nt][0][uFE]
            us    = prms[nt][1][uFE]
            covs  = prms[nt][2][uFE]
            M     = covs.shape[0]

            iSgs  = _N.linalg.inv(covs)
            i2pidcovs = 1/_N.sqrt(2*_N.pi*_N.linalg.det(covs))
            i2pidcovsr= i2pidcovs.reshape((M, 1))

        oo.init_pX_Nm(t0)   #  flat pX_Nm  init cond at start of decode
        oo.setLmd0(0, t0, prms=prms, uFE=uFE)
        pNkmk0   = _N.exp(-oo.dt * oo.Lam_xk)  #  one for each tetrode

        fxdMks = _N.empty((oo.Nx, oo.mdim+1))  #  for each pos, a fixed mark
        fxdMks[:, 0] = oo.xp

        for t in xrange(t0+1,t1): # start at 1 because initial condition
            #tt1 = _tm.time()
            for nt in xrange(oo.nTets):
                oo.Lklhd[nt, t] = pNkmk0[:, nt]

                if (oo.mkpos[nt][t, 1] == 1):
                    fxdMks[:, 1:] = oo.mkpos[nt][t, 2:]
                    oo.Lklhd[nt, t] *= _ku.evalAtFxdMks_new(fxdMks, l0s, us, covs, iSgs, i2pidcovsr)*oo.lmd0[nt] * oo.dt

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
        #print "decode   %(1).3e" % {"1" : (tEnd-tStart)}

    def setLmd0(self, t0, t1, uFE=None, prms=None):
        """
        Lmd0.  
        """
        oo = self
        #####  
        oo.lmd0 = _N.empty(oo.nTets)
        oo.Lam_xk = _N.ones((oo.Nx, oo.nTets))

        if oo.kde:
            ibx2 = 1./ (oo.bx*oo.bx)        
            occ    = _N.sum(_N.exp(-0.5*ibx2*(oo.xpr - oo.pos[t0:t1])**2), axis=1)  #  this piece doesn't need to be evaluated for every new spike
            Tot_occ  = _N.sum(occ)
            oo.iocc = 1./occ

            for nt in xrange(oo.nTets):
                oo.Lam_xk[:, nt] = _ku.Lambda(oo.xpr, oo.tr_pos[nt], oo.pos, oo.Bx, oo.bx)
        else:  #####  fit mix gaussian
            for nt in xrange(oo.nTets):
                l0s   = prms[nt][0][uFE]
                us    = prms[nt][1][uFE]
                covs  = prms[nt][2][uFE]
                M     = covs.shape[0]

                cmps   = _N.zeros((M, oo.Nx))
                for m in xrange(M):
                    var = covs[m, 0, 0]
                    ivar = 1./var
                    cmps[m] = (1/_N.sqrt(2*_N.pi*var)) * _N.exp(-0.5*ivar*(oo.xp - us[m, 0])**2)
                print l0s*cmps
                oo.Lam_xk[:, nt]   = _N.sum(l0s*cmps, axis=0)

    def decodeKDE(self, t0, t1):
        oo = self

        oo.init_pX_Nm(t0)   #  flat pX_Nm  init cond at start of decode
        oo.setLmd0(0, t0)

        ##  each 

        #  k_{k-1} is not treated as a value with a correct answer.
        #  integrate over all possible values of x_{k-1}
        #  Need value of integrand for all x_{k-1}
        #  I will perform integral L times for each time step
        #  multiply integral with p(\Delta N_k, m_k | x_k)

        pNkmk0   = _N.exp(-oo.dt * oo.Lam_xk)  #  one for each tetrode
        pNkmk    = _N.ones(oo.Nx)

        fxdMks = _N.empty((oo.Nx, oo.mdim+1))  #  fixed mark for each field pos.
        fxdMks[:, 0] = oo.xp   

        pNkmk = _N.empty((oo.Nx, oo.nTets))

        tStart = _tm.time()

        if oo.kde:
            occ    = 1./oo.iocc
            iBx2   = 1. / (oo.Bx * oo.Bx)
            sptl   =  []
            for nt in xrange(oo.nTets):
                sptl.append(-0.5*iBx2*(oo.xpr - oo.tr_pos[nt])**2)  #  this piece doesn't need to be evalu
        for t in xrange(t0+1,t1): # start at 1 because initial condition
            #tt1 = _tm.time()
            for nt in xrange(oo.nTets):
                oo.Lklhd[nt, t] = pNkmk0[:, nt]

                if (oo.mkpos[nt][t, 1] == 1):
                    fxdMks[:, 1:] = oo.mkpos[nt][t, 2:]
                        #(atMark, fld_x, tr_pos, tr_mks, all_pos, mdim, Bx, cBm, bx)
                    oo.Lklhd[nt, t] *= _ku.kerFr(fxdMks[0, 1:], sptl[nt], oo.tr_marks[nt], oo.mdim, oo.Bx, oo.Bm, oo.bx)* oo.iocc*oo.dt

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

    def prepareDecKDE(self, t0, t1, telapse=0):
        #preparae decoding step for KDE
        oo = self

        oo.tr_pos   = []
        oo.tr_marks = []

        for nt in xrange(oo.nTets):
            sts = _N.where(oo.mkpos[nt][t0:t1, 1] == 1)[0] + t0

            oo.tr_pos.append(_N.array(oo.mkpos[nt][sts, 0]))
            oo.tr_marks.append(_N.array(oo.mkpos[nt][sts, 2:]))

    def spkts(self, nt, t0, t1):
        return _N.where(self.mkpos[nt][t0:t1, 1] == 1)[0] + t0
