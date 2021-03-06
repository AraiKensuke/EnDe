import cPickle as pickle
import numpy as _N
import matplotlib.pyplot as _plt
import kde as _kde
import scipy.stats as _ss
#import fitMvNormE as fMN   
import fitMvNormTet as fMN   
import time as _tm
import kdeutil as _ku
import scipy.integrate as _si

#  Decode unsorted spike train

#  In our case with no history dependence
#  p(X_t | dN_t, m_t) 

class simDecode():
    nTets   = 1
    bWmaze= False
    pX_Nm = None
    Lklhd = None
    kde   = None
    lmd   = None
    lmd0  = None
    mltMk = 1    #  multiply mark values to 
    AR    = 0.5
    AR0    = 1

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

    tetfile  = "marks.pkl"
    usetets  = None
    utets_str= ""

    tt0      = None
    tt1      = None

    dbgMvt   = False

    minds    = None   #  

    encN     = 0  #  how many time points used in encode

    snpsht_us  = []
    snpsht_covs= []
    snpsht_ms  = []
    snpsht_gz  = []

    def init(self, kde=False, bx=None, Bx=None, Bm=None):
        oo = self
        oo.kde = kde
        with open(oo.tetfile, "rb") as f:
            lm = pickle.load(f)
        f.close()

        if oo.usetets is None:
            oo.tetlist = lm.tetlist
            oo.marks = lm.marks
            oo.utets_str = "all"
        else:
            stetlist= set(oo.usetets)

            mis     = [i for i, item in enumerate(lm.tetlist) if item in stetlist]
            oo.marks   =  lm.marks[:, mis]

            print mis
            for l in xrange(len(mis)):
                sc = "" if (l == len(mis)-1) else ","
                oo.utets_str += "%(l)s%(c)s" % {"l" : lm.tetlist[mis[l]], "c" : sc}

        if oo.mltMk != 1:
            nons = _N.equal(oo.marks, None)
            for nt in xrange(oo.nTets):
                spks = _N.where(nons[:, nt] == False)[0]

                for l in xrange(len(spks)):
                    mklst = oo.marks[spks[l], nt][0]

                    for ll in xrange(len(mklst)):
                        mklst[ll] *= oo.mltMk
             
        oo.nTets = oo.marks.shape[1]

        oo.Nx = lm.Nx;        oo.Nm = lm.Nm
        oo.xA = lm.xA;        oo.mA = lm.mA
            
        oo.mdim  = lm.k#kde.mdim

        try:
            if lm.minds is not None:
                oo.minds = lm.minds
        except AttributeError:
            pass

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
        oo.Fp, oo.q2p = 1, 0.003

        oo.pX_Nm = _N.zeros((oo.pos.shape[0], oo.Nx))
        oo.Lklhd = _N.zeros((oo.nTets, oo.pos.shape[0], oo.Nx))

        oo.decmth = "kde"
        if not oo.kde:
            oo.decmth = "mxn"
            oo.mvNrm= []
            for nt in xrange(oo.nTets):
                oo.mvNrm.append(fMN.fitMvNorm(oo.ITERS, oo.M, oo.mdim + 1))
                oo.mvNrm[nt].ITERS = oo.ITERS
                oo.mvNrm[nt].AR = oo.AR
                oo.mvNrm[nt].AR0 = oo.AR0

    def encode(self, t0=None, t1=None, encIntvs=None, initPriors=False, doTouchUp=False, MF=None, kmeansinit=True, telapse=0):
        """
        eIntvs: array of times whose spikes we use to create encoding model
        """
        oo = self

        if encIntvs is None:
            encIntvs = _N.array([[t0, t1]])
            oo.N = t1-t0
        else:
            oo.N = encIntvs[-1, 1] - encIntvs[0, 0]
            
        oo.tt0 = encIntvs[0, 0]
        oo.tt1 = encIntvs[-1, 1]
        tt1 = _tm.time()

        oo.encN = _N.sum(encIntvs[:, 1] - encIntvs[:, 0])

        if initPriors:
            oo.all_pos = _N.empty(oo.encN)
            ii = 0
            for ein in xrange(encIntvs.shape[0]):
                oo.all_pos[ii:ii+(encIntvs[ein,1]-encIntvs[ein,0])] = oo.pos[encIntvs[ein,0]:encIntvs[ein,1]]
                ii += encIntvs[ein,1]-encIntvs[ein,0]
        else:     #  just make this longer
            tmp = _N.empty(oo.encN)
            ii = 0
            for ein in xrange(encIntvs.shape[0]):
                tmp[ii:ii+(encIntvs[ein,1]-encIntvs[ein,0])] = oo.pos[encIntvs[ein,0]:encIntvs[ein,1]]
                ii += encIntvs[ein,1]-encIntvs[ein,0]

            oo.all_pos = _N.array(oo.all_pos.tolist() + tmp.tolist())

        dat = _N.empty(oo.N, dtype=list)       # include times when no mvt.
        stpos  = []   #  pos  @ time of spikes
        marks  = []   #  mark @ time of spikes
        nspks  = _N.zeros(oo.nTets, dtype=_N.int)
        for nt in xrange(oo.nTets):
            marks.append([])
            stpos.append([])
            nspks[nt]  = 0
            for ein in xrange(encIntvs.shape[0]):
                t0, t1 = encIntvs[ein]

                for n in xrange(t0, t1):         # oo.marks   "list of arrays"
                    if oo.marks[n, nt] is not None:  # [arr(k-dim mark1), arr(k-dim mark2)]
                        themarks = oo.marks[n, nt]
                        for l in xrange(len(themarks)):
                            stpos[nt].append(oo.pos[n])
                            marks[nt].append(themarks[l])
                            nspks[nt] += 1

            print "nspikes tet %(tet)d %(s)d  from %(t0)d   %(t1)d" % {"t0" : t0, "t1" : t1, "s" : nspks[nt], "tet" : nt}

        oo.tr_pos   = []
        oo.tr_marks = []

        for nt in xrange(oo.nTets):
            oo.tr_pos.append(_N.array(stpos[nt]))
            oo.tr_marks.append(_N.array(marks[nt]))


        if not oo.kde:
            oo.snpsht_us.append([])
            oo.snpsht_covs.append([])
            oo.snpsht_ms.append([])
            oo.snpsht_gz.append([])

            tt2 = _tm.time()
            if initPriors:
                for nt in xrange(oo.nTets):
                    print "initPriors for tet %d" % nt
                    oo.mvNrm[nt].init0(stpos[nt], marks[nt], 0, nspks[nt], sepHash=oo.sepHash, pctH=oo.pctH, MS=oo.MS, sepHashMthd=oo.sepHashMthd, doTouchUp=doTouchUp, MF=MF, kmeansinit=kmeansinit)
                    for m in xrange(oo.mvNrm[nt].M):
                        oo.mvNrm[nt].us[m] = oo.mvNrm[nt].smu[0, m]
                        oo.mvNrm[nt].covs[m] = oo.mvNrm[nt].scov[0, m]
                        oo.mvNrm[nt].ms[m] = oo.mvNrm[nt].sm[0, m]

        oo.setLmd0(nspks)

    def decode(self, t0, t1):
        oo = self
        ##  each 

        oo.pX_Nm[t0] = 1. / oo.Nx
        if oo.dbgMvt:
            oo.pX_Nm[t0, 20:30] = 151/5.
        A = _N.trapz(oo.pX_Nm[t0], dx=oo.dxp)
        oo.pX_Nm[t0] /= A

        oo.intgrd= _N.empty(oo.Nx)
        oo.intgrd2d= _N.empty((oo.Nx, oo.Nx))
        oo.intgrl = _N.empty(oo.Nx)
        oo.xTrs  = _N.zeros((oo.Nx, oo.Nx))      #  Gaussian

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
            grdsz = (12./oo.Nx)

            spdGrdUnts = _N.diff(oo.pos) / grdsz  # unit speed ( per ms ) in grid units

            #  avg. time it takes to move 1 grid is 1 / _N.mean(_N.abs(spdGrdUnts))
            #  p(i+1, i) = 1/<avg spdGrdUnts>
            p1 = _N.mean(_N.abs(spdGrdUnts))*1
            #  assume Nx is even
            #k2 = 0.02
            k2 = 0.2
            k3 = 0.1
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
                A = _N.trapz(oo.xTrs[:, i])*((2.*oo.xA)/oo.Nx)
                oo.xTrs[:, i] /= A

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


        if oo.kde:
            occ    = 1./oo.iocc
            iBx2   = 1. / (oo.Bx * oo.Bx)
            sptl   =  []
            for nt in xrange(oo.nTets):
                sptl.append(-0.5*iBx2*(oo.xpr - oo.tr_pos[nt])**2)  #  this piece doesn't need to be evalu
        if not oo.kde:
            for nt in xrange(oo.nTets):
                oo.mvNrm[nt].iSgs  = _N.linalg.inv(oo.mvNrm[nt].covs)
                
                oo.mvNrm[nt].i2pidcovs = 1/_N.sqrt(2*_N.pi*_N.linalg.det(oo.mvNrm[nt].covs))
                oo.mvNrm[nt].i2pidcovsr= oo.mvNrm[nt].i2pidcovs.reshape((oo.mvNrm[nt].M, 1))
        for t in xrange(t0+1,t1): # start at 1 because initial condition
            #tt1 = _tm.time()
            for nt in xrange(oo.nTets):
                oo.Lklhd[nt, t] = pNkmk0[:, nt]

                #  build likelihood
                if (oo.marks[t, nt] is not None) and (not oo.dbgMvt):
                    nSpks = len(oo.marks[t, nt])

                    for ns in xrange(nSpks):
                        fxdMks[:, 1:] = oo.marks[t, nt][ns]
                        if oo.kde:
                            #(atMark, fld_x, tr_pos, tr_mks, all_pos, mdim, Bx, cBm, bx)
                            #_ku.kerFr(fxdMks[0, 1:], fxdMks[:, 0], oo.tr_pos, oo.tr_mks, oo.mvpos, oo.mdim, oo.Bx, oo.cBm, oo.bx)
                            oo.Lklhd[nt, t] *= _ku.kerFr(fxdMks[0, 1:], sptl[nt], oo.tr_marks[nt], oo.mdim, oo.Bx, oo.Bm, oo.bx)* oo.iocc*oo.dt
                        else:
                            oo.Lklhd[nt, t] *= oo.mvNrm[nt].evalAtFxdMks_new(fxdMks)*oo.lmd0[nt] * oo.iocc * oo.dt

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
    
    def setLmd0(self, nspks):
        """
        Lmd0.  
        """
        oo = self
        #####  
        if oo.dbgMvt:
            occ = _N.ones(oo.Nx) / oo.Nx
            oo.iocc = 1./occ
            oo.Lam_xk = _N.zeros((oo.Nx, oo.nTets))
            return 
        oo.lmd0 = _N.empty(oo.nTets)
        oo.Lam_xk = _N.ones((oo.Nx, oo.nTets))

        ibx2 = 1./ (oo.bx*oo.bx)        
        #occ    = _N.sum(_N.exp(-0.5*ibx2*(oo.xpr - oo.all_pos[t0:t1])**2), axis=1)  #  this piece doesn't need to be evaluated for every new spike
        occ    = _N.sum(_N.exp(-0.5*ibx2*(oo.xpr - oo.all_pos)**2), axis=1)  #  this piece doesn't need to be evaluated for every new spike
        Tot_occ  = _N.sum(occ)
        #oo.iocc = 1./(occ + Tot_occ*0.01)
        oo.iocc = 1./occ

        if oo.kde:
            for nt in xrange(oo.nTets):
                oo.Lam_xk[:, nt] = _ku.Lambda(oo.xpr, oo.tr_pos[nt], oo.all_pos, oo.Bx, oo.bx)
        else:  #####  fit mix gaussian
            for nt in xrange(oo.nTets):
                cmps   = _N.zeros((oo.mvNrm[nt].M, oo.Nx))
                for m in xrange(oo.mvNrm[nt].M):
                    var = oo.mvNrm[nt].covs[m, 0, 0]
                    print "m is %(m)d    var is %(var).3f" % {"m" : m, "var" : var}
                    ivar = 1./var
                    cmps[m] = (1/_N.sqrt(2*_N.pi*var)) * _N.exp(-0.5*ivar*(oo.xp - oo.mvNrm[nt].us[m, 0])**2)

                y  = _N.sum(oo.mvNrm[nt].ms*cmps, axis=0)
                MargLam = y * oo.iocc
                oo.lmd0[nt]    = (nspks[nt] / (oo.encN*0.001)) / _N.trapz(MargLam, dx=oo.dxp)
                oo.Lam_xk[:, nt] = oo.lmd0[nt] * MargLam

    def scoreDecode(self, dt0, dt1):
        oo     = self
        maxPos = _N.max(oo.pX_Nm[dt0:dt1], axis=1)

        inds   = _N.empty(dt1-dt0, dtype=_N.int)

        for t in xrange(dt0, dt1):
            inds[t-dt0] = _N.where(oo.pX_Nm[t] == maxPos[t-dt0])[0][0]

        diffPos = oo.xp[inds] - oo.pos[dt0:dt1]
        return _N.sum(_N.abs(diffPos)) / (dt1-dt0)

    def dump(self, dir):
        oo    = self
        pcklme = {}
        pcklme["posteriors"] = oo.pX_Nm
        pcklme["Lklhds"] = oo.Lklhd
        if not oo.kde:
            pcklme["snpsht_us"] = oo.snpsht_us
            pcklme["snpsht_covs"] = oo.snpsht_covs
            pcklme["snpsht_ms"] = oo.snpsht_ms

        dmp = open("%s/dec.dump" % dir, "wb")
        pickle.dump(pcklme, dmp, -1)
        dmp.close()

        # import pickle
        # with open("mARp.dump", "rb") as f:
        # lm = pickle.load(f)
