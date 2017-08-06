import numpy as _N
from EnDedirs import resFN, datFN
import kdeutil as _ku
import time as _tm
import matplotlib.pyplot as _plt
import hc_bcast as _hb

mz_CRCL = 0
mz_W    = 1

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

    maze     = None

    dbgMvt   = False
    spdMult   = 0.5

    Nx       = 61

    xLo      = 0
    xHi      = 3
    mLo      = -2
    mHi      = 8

    sts_per_tet = None
    _sts_per_tet = None
    svMkIntnsty = None   #  save just the mark intensities

    ##  X_   and _X
    def __init__(self, kde=False, bx=None, Bx=None, Bm=None, mkfns=None, encfns=None, K=None, nTets=None, xLo=0, xHi=3, maze=mz_CRCL, spdMult=0.1, ignorespks=False):
        """
        """
        oo = self
        oo.maze = maze
        oo.kde = kde

        oo.spdMult = spdMult
        oo.ignorespks = ignorespks
        oo.bx = bx;   oo.Bx = Bx;   oo.Bm = Bm
        oo.mkpos = []
        #  read mkfns
        _sts   = []#  a mark on one of the several tetrodes
        oo._sts_per_tet = []
        for fn in mkfns:   #  for each tetrode filename

            _dat = _N.loadtxt(datFN("%s.dat" % fn))
            if K is None:
                K   = _dat.shape[1] - 2
                dat = _dat
            else:
                dat = _dat[:, 0:2+K]
            oo.mkpos.append(dat)
            spkts = _N.where(dat[:, 1] == 1)[0]
            oo._sts_per_tet.append(spkts)
            _sts.extend(spkts)
        oo.sts = _N.unique(_sts)

        oo.nTets = len(oo.mkpos)
        oo.mdim  = K
        oo.pos  = dat[:, 0]         #  length of 

        if not kde:
            oo.mdim  = K
            oo.nTets  = nTets
        
        oo.xLo = xLo;     oo.xHi = xHi
        
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
        oo.intnstyAtMrk= None    # instead of saving everything

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
        p1 = 1.5*_N.mean(_N.abs(spdGrdUnts))*oo.spdMult
        #  assume Nx is even
        #k2 = 0.02
        k2 = 0.1
        k3 = 0.1

        if maze == mz_CRCL:
            ##  circular maze
            for i in xrange(0, oo.Nx):  #  indexing of xTrs  [to, from]
                oo.xTrs[i, i] = 1-p1   
                if i == 0:
                    #oo.xTrs[0, oo.Nx-1] = p1*0.5
                    oo.xTrs[0, oo.Nx-1] = p1*0.8  #  backwards
                if i >= 0:
                    oo.xTrs[i-1, i] = p1*0.2
                    oo.xTrs[i, i-1] = p1*0.8
                if i == oo.Nx-1:
                    oo.xTrs[oo.Nx-1, 0] = p1*0.2
        elif maze == mz_W:
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
        print "epoch used for encoding: %d" % uFE
        oo = self
        ##  each 

        oo.svMkIntnsty = []

        for nt in xrange(oo.nTets):
            l0s   = prms[nt][0][uFE]
            us    = prms[nt][1][uFE]
            covs  = prms[nt][2][uFE]
            M     = covs.shape[0]

            iSgs  = _N.linalg.inv(covs)
            i2pidcovs = (1/_N.sqrt(2*_N.pi))**(oo.mdim+1)*(1./_N.sqrt(_N.linalg.det(covs)))
            i2pidcovsr= i2pidcovs.reshape((M, 1))
            oo.svMkIntnsty.append([])

        oo.init_pX_Nm(t0)   #  flat pX_Nm  init cond at start of decode
        tt1 = _tm.time()
        oo.LmdMargOvrMrks(0, t0, prms=prms, uFE=uFE)
        tt2 = _tm.time()
        print "tt2-tt1  %.3f" % (tt2-tt1)
        pNkmk0   = _N.exp(-oo.dt * oo.Lam_MoMks)  #  one for each tetrode

        fxdMks = _N.empty((oo.Nx, oo.mdim+1))  #  for each pos, a fixed mark
        #fxdMks = _N.empty(oo.mdim)  #  for each pos, a fixed mark
        fxdMks[:, 0] = oo.xp

        dens = []

        ones = _N.ones(oo.Nx)

        #tspk = 0
        for t in xrange(t0+1,t1): # start at 1 because initial condition
            #print "t %d" % t

            for nt in xrange(oo.nTets):
                oo.Lklhd[nt, t] = pNkmk0[:, nt]

                if (oo.mkpos[nt][t, 1] == 1):
                    fxdMks[:, 1:] = oo.mkpos[nt][t, 2:]
                    if not oo.ignorespks:
                        #mkint = _ku.evalAtFxdMks_new(fxdMks, l0s, us, covs, iSgs, i2pidcovsr)*oo.dt
                        # print fxdMks.shape
                        # print l0s.shape
                        # print us.shape
                        # print iSgs.shape
                        # print i2pidcovs.shape
                        #tt1 = _tm.time()
                        l0sr = _N.array(l0s[:, 0])
                        #mkint2 = _hb.evalAtFxdMks_new(fxdMks, l0sr, us, iSgs, i2pidcovs, M, oo.Nx, oo.mdim + 1)*oo.dt
                        mkint = _hb.evalAtFxdMks_new(fxdMks, l0sr, us, iSgs, i2pidcovs, M, oo.Nx, oo.mdim + 1)*oo.dt
                        #print mkint1
                        #print mkint2
                        #tt2 = _tm.time()
                        oo.Lklhd[nt, t] *= mkint
                        oo.svMkIntnsty[nt].append(mkint)
                        #tspk += tt2-tt1


            ttt1 =0
            ttt2 =0
            ttt3 =0

            #tt2 = _tm.time()

            #  transition convolved with previous posterior

            ####  INSTEAD OF THIS   #  
            if oo.maze == mz_W:
                _N.multiply(oo.xTrs, oo.pX_Nm[t-1], out=oo.intgrd2d)   
                oo.intgrl = _N.trapz(oo.intgrd2d, dx=oo.dxp, axis=1)
            else:
                ####  DO THIS
                oo.intgrl = _N.dot(oo.xTrs, oo.pX_Nm[t-1])
                oo.intgrl /= _N.sum(oo.intgrl)
            #for ixk in xrange(oo.Nx):   #  above trapz over 2D array
            #    oo.intgrl[ixk] = _N.trapz(oo.intgrd2d[ixk], dx=oo.dxp)

            #tt3 = _tm.time()
            oo.pX_Nm[t] = oo.intgrl * _N.product(oo.Lklhd[:, t], axis=0)   #  product of all tetrodes
            A = _N.trapz(oo.pX_Nm[t], dx=oo.dxp)
            if A == 0:
                print "A is %(A).5f at time %(t)d" % {"A": A, "t" : t}
                fig = _plt.figure()

                #_plt.plot(_N.product(oo.Lklhd[:, t], axis=0))
                #print "t=%d" % t
                for tet in xrange(4):
                    #print oo.Lklhd[tet, t]
                    fig = _plt.figure()
                    _plt.plot(oo.Lklhd[tet, t])

            if _N.isnan(A):
                print "nan" 
                print mkint
            assert A > 0, "A   %(A).4f,  t is %(t)d" % {"A" : A, "t" : t}

            oo.pX_Nm[t] /= A
            #tt4 = _tm.time()
            #print "%(1).3e   %(2).3e   %(3).3e" % {"1" : (tt2-tt1), "2" : (tt3-tt2), "3" : (tt4-tt3)}
            #print "%(1).3e   %(2).3e" % {"1" : ttt1, "2" : ttt2}
        #print "tspk  is %.3f" % tspk
        tEnd = _tm.time()
        #print "decode   %(1).3e" % {"1" : (tEnd-tStart)}
        #_N.savetxt("densGT", _N.array(dens))

    def LmdMargOvrMrks(self, enc_t0, enc_t1, uFE=None, prms=None):
        """
        0:t0   used for encode
        Lmd0.  
        """
        oo = self
        #####  
        oo.lmd0 = _N.empty(oo.nTets)
        oo.Lam_MoMks = _N.ones((oo.Nx, oo.nTets))

        if oo.kde:  #  also calculate the occupation. Nothing to do with LMoMks
            ibx2 = 1./ (oo.bx*oo.bx)        
            occ    = _N.sum((1/_N.sqrt(2*_N.pi*oo.bx*oo.bx))*_N.exp(-0.5*ibx2*(oo.xpr - oo.pos[enc_t0:enc_t1])**2), axis=1)*oo.dxp  #  this piece doesn't need to be evaluated for every new spike
            #  _plt.hist(mkd.pos, bins=mkd.xp) == _plt.plot(mkd.xp, occ)  
            #  len(occ) = total time of observation in ms.
            oo.occ = occ  
            oo.iocc = 1./occ  

            ###  Lam_MoMks  is a function of space
            for nt in xrange(oo.nTets):
                oo.Lam_MoMks[:, nt] = _ku.Lambda(oo.xpr, oo.tr_pos[nt], oo.pos[enc_t0:enc_t1], oo.Bx, oo.bx, oo.dxp, occ)
        else:  #####  fit mix gaussian
            for nt in xrange(oo.nTets):
                l0s   = prms[nt][0][uFE]    #  M x 1
                us    = prms[nt][1][uFE]
                covs  = prms[nt][2][uFE]
                M     = covs.shape[0]

                cmps   = _N.zeros((M, oo.Nx))
                for m in xrange(M):
                    var = covs[m, 0, 0]
                    ivar = 1./var
                    cmps[m] = (1/_N.sqrt(2*_N.pi*var)) * _N.exp(-0.5*ivar*(oo.xp - us[m, 0])**2)
                oo.Lam_MoMks[:, nt]   = _N.sum(l0s*cmps, axis=0)


    def decodeKDE(self, t0, t1):
        """
        decode activity from [t0:t1]
        """
        oo = self

        oo.init_pX_Nm(t0)   #  flat pX_Nm  init cond at start of decode

        tt1 = _tm.time()
        oo.LmdMargOvrMrks(0, t0)
        tt2 = _tm.time()
        print "tt2-tt1  %.3f" % (tt2-tt1)



        ##  each 

        #  k_{k-1} is not treated as a value with a correct answer.
        #  integrate over all possible values of x_{k-1}
        #  Need value of integrand for all x_{k-1}
        #  I will perform integral L times for each time step
        #  multiply integral with p(\Delta N_k, m_k | x_k)

        pNkmk0   = _N.exp(-oo.dt * oo.Lam_MoMks)  #  one for each tetrode
        pNkmk    = _N.ones(oo.Nx)

        fxdMks = _N.empty((oo.Nx, oo.mdim+1))  #  fixed mark for each field pos.
        fxdMks[:, 0] = oo.xp   

        pNkmk = _N.empty((oo.Nx, oo.nTets))

        ibx2   = 1. / (oo.bx * oo.bx)
        sptl   =  []
        dens   = []

        oo.svMkIntnsty = []

        for nt in xrange(oo.nTets):
            lst = []

            sptl.append(-0.5*ibx2*(oo.xpr - oo.tr_pos[nt])**2)  #  this piece doesn't need to be evalu
            oo.svMkIntnsty.append([])

        #tspk = 0
        ###############################
        for t in xrange(t0+1,t1): # start at 1 because initial condition

            for nt in xrange(oo.nTets):
                oo.Lklhd[nt, t] = pNkmk0[:, nt]

                if (oo.mkpos[nt][t, 1] == 1):
                    fxdMks[:, 1:] = oo.mkpos[nt][t, 2:]
                        #(atMark, fld_x, tr_pos, tr_mks, all_pos, mdim, Bx, cBm, bx)
                    #tt1 = _tm.time()
                    mkint = _ku.kerFr(fxdMks[0, 1:], sptl[nt], oo.tr_marks[nt], oo.mdim, oo.Bx, oo.Bm, oo.bx, oo.dxp, oo.occ)
                    #tt2 = _tm.time()
                    #tspk += tt2-tt1

                    oo.svMkIntnsty[nt].append(mkint)
                    #dens.append(_ku.kerFr(fxdMks[0, 1:], sptl[nt], oo.tr_marks[nt], oo.mdim, oo.Bx, oo.Bm, oo.bx, oo.dxp, oo.occ))
                    if _N.sum(mkint) > 0:  #  if firing rate is 0, ignore this spike
                        oo.Lklhd[nt, t] *= mkint
                    else:
                        print "mark so far away that mk intensity is 0.  ignoring"

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
        #print "tspk  is %.3f" % tspk
        tEnd = _tm.time()

        #_N.savetxt("densKDE", _N.array(dens))

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


