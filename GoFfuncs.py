import numpy as _N
from EnDedirs import resFN, datFN
import kdeutil as _ku
import time as _tm
import matplotlib.pyplot as _plt
import hc_bcast as _hb

mz_CRCL = 0
mz_W    = 1

class GoFfuncs:
    nTets   = 1
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
    chmins   = None

    sts_per_tet = None
    _sts_per_tet = None
    svMkIntnsty = None   #  save just the mark intensities

    ##  X_   and _X
    def __init__(self, Nx=61, kde=False, bx=None, Bx=None, Bm=None, mkfns=None, encfns=None, K=None, nTets=None, xLo=0, xHi=3, maze=mz_CRCL, spdMult=0.1, ignorespks=False, chmins=None, rotate=False):
        """
        """
        oo = self
        oo.Nx = Nx
        oo.maze = maze
        oo.kde = kde
        if chmins is not None:
            oo.chmins = chmins
        else:
            oo.chmins = _N.ones(K)*-10000

        oo.spdMult = spdMult
        oo.ignorespks = ignorespks
        oo.bx = bx;   oo.Bx = Bx;   oo.Bm = Bm
        oo.mkpos = []
        #  read mkfns
        _sts   = []#  a mark on one of the several tetrodes
        oo._sts_per_tet = []

        #  rotation about axis 1
        th1 = _N.pi/4
        rot1  = _N.array([[1, 0, 0,            0],
                          [0, 1, 0,            0],
                          [0, 0, _N.cos(th1),  _N.sin(th1)],
                          [0, 0, -_N.sin(th1), _N.cos(th1)]])

        #  roation about axis 4
        th4  = (54.738/180.)*_N.pi
        rot4  = _N.array([[1, 0, 0,            0],
                          [0, _N.cos(th4), _N.sin(th4), 0],
                          [0, -_N.sin(th4), _N.cos(th4), 0],
                          [0,            0,      0, 1]])


        th3   = (60.0/180.)*_N.pi
        rot3  = _N.array([[_N.cos(th3), _N.sin(th3), 0, 0],
                          [-_N.sin(th3), _N.cos(th3), 0, 0],
                          [0,            0,      1, 0],
                          [0,            0,      0, 1]]
        )

        for fn in mkfns:   #  for each tetrode filename
            _dat = _N.loadtxt(datFN("%s.dat" % fn))
            if K is None:
                K   = _dat.shape[1] - 2
                dat = _dat
            else:
                dat = _dat[:, 0:2+K]

            oo.mkpos.append(dat)
            spkts = _N.where(dat[:, 1] == 1)[0]

            if rotate:
                for t in spkts:
                    dat[t, 2:] = _N.dot(rot3, _N.dot(rot4, dat[t, 2:]))
                    
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
                l0s   = prms[nt][uFE][0]    #  M x 1
                us    = prms[nt][uFE][1]
                covs  = prms[nt][uFE][2]
                M     = covs.shape[0]

                cmps   = _N.zeros((M, oo.Nx))
                for m in xrange(M):
                    var = covs[m, 0, 0]
                    ivar = 1./var
                    cmps[m] = (1/_N.sqrt(2*_N.pi*var)) * _N.exp(-0.5*ivar*(oo.xp - us[m, 0])**2)
                oo.Lam_MoMks[:, nt]   = _N.sum(l0s*cmps, axis=0)

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


    #####################   GoF tools
    def mark_ranges(self, g_M):
        oo = self
        mk_ranges = _N.empty((oo.nTets, oo.mdim, g_M))

        for nt in xrange(oo.nTets):
            for im in xrange(oo.mdim):
                sts = _N.where(oo.mkpos[nt][:, 1] == 1)[0]

                mL  = _N.min(oo.mkpos[nt][sts, 2+im])
                mH  = _N.max(oo.mkpos[nt][sts, 2+im])
                A   = mH - mL
                mk_ranges[nt, im] = _N.linspace(mL - 0.03*A, mH + 0.03*A, g_M, endpoint=True)

        return mk_ranges

    def rescale_spikes(self, prms, uFE, t0, t1, kde=False):
        """
        uFE    which epoch fit to use for encoding model
        prms posterior params
        use params to decode marks from t0 to t1
        """
        print "epoch used for encoding: %d" % uFE
        oo = self
        ##  each 

        disc_pos = _N.array((oo.pos - oo.xLo) * (oo.Nx/(oo.xHi-oo.xLo)), dtype=_N.int)
        oo.svMkIntnsty = []
        l0s = []
        us  = []
        covs= []
        M   = []
        iSgs= []
        i2pidcovs  = []
        i2pidcovsr = []
        sptl= []

        if not kde:
            for nt in xrange(oo.nTets):
                l0s.append(prms[nt][uFE][0])
                us.append(prms[nt][uFE][1])
                covs.append(prms[nt][uFE][2])
                M.append(covs[nt].shape[0])

                print covs[nt]
                iSgs.append(_N.linalg.inv(covs[nt]))
                i2pidcovs.append((1/_N.sqrt(2*_N.pi))**(oo.mdim+1)*(1./_N.sqrt(_N.linalg.det(covs[nt]))))
                #i2pidcovsr.append(i2pidcovs.reshape((M, 1)))
                oo.svMkIntnsty.append([])
                l0sr = _N.array(l0s[nt][:, 0])
        else:
            ibx2   = 1. / (oo.bx * oo.bx)
            for nt in xrange(oo.nTets):
                sptl.append(-0.5*ibx2*(oo.xpr - oo.tr_pos[nt])**2)  #  this piece doesn't need to be evalu


        fxdMks = _N.empty((oo.Nx, oo.mdim+1))  #  for each pos, a fixed mark
        fxdMks[:, 0] = oo.xp

        nt    = 0
        rscld = []

        for t in xrange(t0+1, t1): # start at 1 because initial condition
            if (oo.mkpos[nt][t, 1] == 1):
                fxdMks[:, 1:] = oo.mkpos[nt][t, 2:]
                if kde:   #  kerFr returns lambda(m, x) for all x
                    mkint = _ku.kerFr(fxdMks[0, 1:], sptl[nt], oo.tr_marks[nt], oo.mdim, oo.Bx, oo.Bm, oo.bx, oo.dxp, oo.occ)
                else:     #  evalAtFxdMks returns lambda(m, x) for all x
                    mkint = _hb.evalAtFxdMks_new(fxdMks, l0sr, us[nt], iSgs[nt], i2pidcovs[nt], M[nt], oo.Nx, oo.mdim + 1)*oo.dt

                lst = [_N.sum(mkint[disc_pos[t0+1:t1]]), _N.sum(mkint[disc_pos[t0+1:t]])]
                lst.extend(oo.mkpos[nt][t, 2:].tolist())

                rscld.append(lst)

        return rscld

    def max_rescaled_T_at_mark(self, mrngs, g_M, prms, uFE, t0, t1, rad, smpld_marks=None, kde=False):
        """
        uFE    which epoch fit to use for encoding model
        prms posterior params
        use params to decode marks from t0 to t1
        """
        print "epoch used for encoding: %d" % uFE
        oo = self
        ##  each 

        disc_pos = _N.array((oo.pos - oo.xLo) * (oo.Nx/(oo.xHi-oo.xLo)), dtype=_N.int)
        oo.svMkIntnsty = []
        l0s = []
        us  = []
        covs= []
        fs  = []
        q2s= []

        M   = []
        iSgs= []
        i2pidcovs = []
        i2pidcovsr = []
        sptl= []

        if not kde:
            for nt in xrange(oo.nTets):
                l0s.append(prms[nt][uFE][0])
                us.append(prms[nt][uFE][1])
                covs.append(prms[nt][uFE][2])
                fs.append(prms[nt][uFE][3])
                iq2s.append(1./prms[nt][uFE][4])
                fs_us.append(prms[nt][uFE][5])
                q2s_covs.append(prms[nt][uFE][6])

                M.append(covs[nt].shape[0])

                iSgs.append(_N.linalg.inv(q2s_covs[nt]))
                i2pidcovs.append((1/_N.sqrt(2*_N.pi))**(oo.mdim+1)*(1./_N.sqrt(_N.linalg.det(q2s_covs[nt]))))
                #i2pidcovsr.append(i2pidcovs.reshape((M, 1)))
                oo.svMkIntnsty.append([])
            l0sr = _N.array(l0s[0][:, 0])  # for nt==0
        else:
            ibx2   = 1. / (oo.bx * oo.bx)
            for nt in xrange(oo.nTets):
                sptl.append(-0.5*ibx2*(oo.xpr - oo.tr_pos[nt])**2)  #  this piece doesn't need to be evalu

        nt    = 0

        if oo.mdim == 1:
            O = _N.zeros(g_M)   #  where lambda is near 0, so is O
            O01 = _N.zeros(g_M, dtype=_N.bool)   #  where lambda is near 0, so is O
        elif oo.mdim == 2:
            O = _N.zeros([g_M, g_M])
            O01 = _N.zeros([g_M, g_M], dtype=_N.bool)   #  where lambda is near 0, so is O
        elif oo.mdim == 4:
            O = _N.zeros([g_M, g_M, g_M, g_M])
            O01 = _N.zeros([g_M, g_M, g_M, g_M], dtype=_N.bool)   #  where lambda is near 0, so is O

        mk = _N.empty((oo.Nx, oo.mdim+1))
        mk[:, 0] = oo.xp

        disc_pos_t0t1 = _N.array(disc_pos[t0+1:t1])

        if not kde:
            usnt          = _N.array(us[nt])
            iSgsnt        = _N.array(iSgs[nt])
            i2pidcovsnt   = _N.array(i2pidcovs[nt])
            Msnt          = _N.array(M[nt])

        LLcrnr = mrngs[nt, :, 0]   # lower left hand corner

        dm = _N.diff(mrngs[0])[:, 0]
        
        if (smpld_marks is not None) and (oo.mdim > 1):
            tt0 = _tm.time()
            sN = smpld_marks.shape[0]
            for s in xrange(sN):
                print "s  %d" % s
                icnt = 0
                ttt0 = _tm.time()
                inds = _N.array((smpld_marks[s] - LLcrnr) / dm, dtype=_N.int)

                if oo.mdim == 2:
                    i0 = inds[0]
                    i1 = inds[1]

                    if O01[i0, i1] == False:
                        O01[i0, i1] = True
                        i0_l = i0 - rad if i0 >= rad else 0
                        i0_h = i0 + rad+1 if i0 + rad < g_M else g_M
                        i1_l = i1 - rad if i1 >= rad else 0
                        i1_h = i1 + rad+1 if i1 + rad < g_M else g_M

                        if not kde:
                            for ii0 in xrange(i0_l, i0_h):
                                for ii1 in xrange(i1_l, i1_h):
                                    if O[ii0, ii1] == 0:
                                        #mk[:, 1:] = _N.array([mrngs[nt, 0, ii0], mrngs[nt, 1, ii1]])
                                        p_mk[:, 1] = mrngs[nt, 0, ii0]
                                        p_mk[:, 2] = mrngs[nt, 1, ii1]

                                        mkint = _hb.evalAtFxdMks_new(mk, l0sr, usnt, fsnt, iSgsnt, i2pidcovsnt, Msnt, oo.Nx, oo.mdim + 1)*oo.dt
                                        O[ii0, ii1] = _N.sum(mkint[disc_pos_t0t1])
                        else:
                            for ii0 in xrange(i0_l, i0_h):
                                for ii1 in xrange(i1_l, i1_h):
                                    if O[ii0, ii1] == 0:
                                        mk[:, 1:] = _N.array([mrngs[nt, 0, ii0], mrngs[nt, 1, ii1]])
                                        mkint = _ku.kerFr(mk[0, 1:], sptl[nt], oo.tr_marks[nt], oo.mdim, oo.Bx, oo.Bm, oo.bx, oo.dxp, oo.occ)
                                        O[ii0, ii1] = _N.sum(mkint[disc_pos_t0t1])

                elif oo.mdim == 4:
                    i0 = inds[0]
                    i1 = inds[1]
                    i2 = inds[2]
                    i3 = inds[3]

                    if O01[i0, i1, i2, i3] == False:
                        O01[i0, i1, i2, i3] = True
                        i0_l = i0 - rad if i0 >= rad else 0
                        i0_h = i0 + rad+1 if i0 + rad < g_M else g_M
                        i1_l = i1 - rad if i1 >= rad else 0
                        i1_h = i1 + rad+1 if i1 + rad < g_M else g_M
                        i2_l = i2 - rad if i2 >= rad else 0
                        i2_h = i2 + rad+1 if i2 + rad < g_M else g_M
                        i3_l = i3 - rad if i3  >= rad else 0
                        i3_h = i3 + rad+1 if i3 + rad < g_M else g_M


                        if not kde:
                            for ii0 in xrange(i0_l, i0_h):
                                for ii1 in xrange(i1_l, i1_h):
                                    for ii2 in xrange(i2_l, i2_h):
                                        for ii3 in xrange(i3_l, i3_h):
                                            if O01[ii0, ii1, ii2, ii3] == False:
                                                O01[ii0, ii1, ii2, ii3] = True
                                                icnt += 1
                                                mk[:, 1:] = _N.array([mrngs[nt, 0, ii0], mrngs[nt, 1, ii1], mrngs[nt, 2, ii2], mrngs[nt, 3, ii3]])
                                                mkint = _hb.evalAtFxdMks_new(mk, l0sr, usnt, iSgsnt, i2pidcovsnt, Msnt, oo.Nx, oo.mdim + 1)*oo.dt
                                                O[ii0, ii1, ii2, ii3] = _N.sum(mkint[disc_pos_t0t1])
                        else:
                            for ii0 in xrange(i0_l, i0_h):
                                for ii1 in xrange(i1_l, i1_h):
                                    for ii2 in xrange(i2_l, i2_h):
                                        for ii3 in xrange(i3_l, i3_h):
                                            if O[ii0, ii1, ii2, ii3] == 0:
                                                mk[:, 1:] = _N.array([mrngs[nt, 0, ii0], mrngs[nt, 1, ii1], mrngs[nt, 2, ii2], mrngs[nt, 3, ii3]])
                                                mkint = _ku.kerFr(mk[0, 1:], sptl[nt], oo.tr_marks[nt], oo.mdim, oo.Bx, oo.Bm, oo.bx, oo.dxp, oo.occ)
                                                O[ii0, ii1, ii2, ii3] = _N.sum(mkint[disc_pos_t0t1])

                ttt1 = _tm.time()
                print "%(1).3f      %(icnt)d" % {"1" : (ttt1-ttt0), "icnt" : icnt}



            tt1 = _tm.time()
            print "done   %.4f" % (tt1-tt0)
        else:   #  not smpld_marks.  brute force, calculate over entire grid
            if oo.mdim == 1:
                if not kde:
                    for im1 in xrange(g_M):
                        tt0 = _tm.time()
                        mk[:, 1:] = _N.array([mrngs[nt, 0, im1]])

                        mkint = _hb.evalAtFxdMks_new(mk, l0sr, usnt, iSgsnt, i2pidcovsnt, Msnt, oo.Nx, oo.mdim + 1)*oo.dt
                        O[im1] = _N.sum(mkint[disc_pos_t0t1])
                else:
                    for im1 in xrange(g_M):
                        tt0 = _tm.time()
                        mk[:, 1:] = _N.array([mrngs[nt, 0, im1]])

                        mkint = _ku.kerFr(mk[0, 1:], sptl[nt], oo.tr_marks[nt], oo.mdim, oo.Bx, oo.Bm, oo.bx, oo.dxp, oo.occ)
                        O[im1] = _N.sum(mkint[disc_pos_t0t1])

            if oo.mdim == 2:
                if not kde:
                    for im1 in xrange(g_M):
                        print "%d" % im1
                        tt0 = _tm.time()
                        for im2 in xrange(g_M):
                            mk[:, 1:] = _N.array([mrngs[nt, 0, im1], mrngs[nt, 1, im2]])

                            mkint = _hb.evalAtFxdMks_new(mk, l0sr, usnt, iSgsnt, i2pidcovsnt, Msnt, oo.Nx, oo.mdim + 1)*oo.dt
                            O[im1, im2] = _N.sum(mkint[disc_pos_t0t1])
                        tt1 = _tm.time()
                        print (tt1-tt0)
                else:
                    for im1 in xrange(g_M):
                        print "%d" % im1
                        tt0 = _tm.time()
                        for im2 in xrange(g_M):
                            mk[:, 1:] = _N.array([mrngs[nt, 0, im1], mrngs[nt, 1, im2]])

                            mkint = _ku.kerFr(mk[0, 1:], sptl[nt], oo.tr_marks[nt], oo.mdim, oo.Bx, oo.Bm, oo.bx, oo.dxp, oo.occ)
                            O[im1, im2] = _N.sum(mkint[disc_pos_t0t1])
                        tt1 = _tm.time()
                        print (tt1-tt0)


            elif oo.mdim == 4:
                if not kde:
                    for im1 in xrange(g_M):
                        print "%d" % im1
                        tt0 = _tm.time()
                        for im2 in xrange(g_M):
                            for im3 in xrange(g_M):
                                for im4 in xrange(g_M):
                                    mk[:, 1:] = _N.array([mrngs[nt, 0, im1], mrngs[nt, 1, im2], mrngs[nt, 2, im3], mrngs[nt, 3, im4]])

                                    mkint = _ku.kerFr(mk[0, 1:], sptl[nt], oo.tr_marks[nt], oo.mdim, oo.Bx, oo.Bm, oo.bx, oo.dxp, oo.occ)
                                    O[im1, im2, im3, im4] = _N.sum(mkint[disc_pos_t0t1])
                        tt1 = _tm.time()
                        print (tt1-tt0)
                else:
                    for im1 in xrange(g_M):
                        print "%d" % im1
                        tt0 = _tm.time()
                        for im2 in xrange(g_M):
                            for im3 in xrange(g_M):
                                for im4 in xrange(g_M):
                                    mk[:, 1:] = _N.array([mrngs[nt, 0, im1], mrngs[nt, 1, im2], mrngs[nt, 2, im3], mrngs[nt, 3, im4]])

                                    mkint = _hb.evalAtFxdMks_new(mk, l0sr, usnt, iSgsnt, i2pidcovsnt, Msnt, oo.Nx, oo.mdim + 1)*oo.dt
                                    O[im1, im2, im3, im4] = _N.sum(mkint[disc_pos_t0t1])
                        tt1 = _tm.time()
                        print (tt1-tt0)

        return O

    def calc_volrat(self, O, g_Mf, g_Tf, fg_Mf, fg_Tf, m1, m2, t, dtf, O_z, vlr_z):
        #  changes in rescaled-time direction is abrupt, while over marks may not be so abrupt.  Cut box in mark direction in 4 

        #  assumption O[m1+1, m2+1] = O[m1, m2] + dO_m1
        dO_m1 = O[m1+1, m2] - O[m1, m2]
        dO_m2 = O[m1, m2+1] - O[m1, m2]

        #  make a finer grid for O_z
        for im1f in xrange(g_Mf):
            for im2f in xrange(g_Mf):
                O_z[im1f, im2f] = O[m1, m2] + (im1f/(fg_Mf-1))*dO_m1 + (im2f/(fg_Mf-1))*dO_m2
        #O_z[g_Mf-1, g_Mf-1] = O[m1+1, m2+1]

        for im1f in xrange(g_Mf-1):
            for im2f in xrange(g_Mf-1):
                for itf in xrange(g_Tf-1):
                    tL = t + itf * dtf
                    tH = t + (itf+1) * dtf 

                    d1h = tH - O_z[im1f, im2f] 
                    d2h = tH - O_z[im1f+1, im2f] 
                    d3h = tH - O_z[im1f, im2f+1] 
                    d4h = tH - O_z[im1f+1, im2f+1]
                    d1l = O_z[im1f, im2f] - tL
                    d2l = O_z[im1f+1, im2f] - tL
                    d3l = O_z[im1f, im2f+1] - tL
                    d4l = O_z[im1f+1, im2f+1] - tL

                    if (((d1h > 0) or (d2h > 0) or \
                         (d3h > 0) or (d4h > 0)) and \
                        ((d1l > 0) or (d2l > 0) or \
                         (d3l > 0) or (d4l > 0))):
                        #  a border
                        if d1h > 0:
                            r1h = 1 if (d1h > dtf) else d1h / dtf
                        else:
                            r1h = 0.01  #  don't set to 0
                        if d2h > 0:
                            r2h = 1 if (d2h > dtf) else d2h / dtf
                        else:
                            r2h = 0.01 #  don't set to 0
                        if d3h > 0:
                            r3h = 1 if (d3h > dtf) else d3h / dtf
                        else:
                            r3h = 0.01  #  don't set to 0
                        if d4h > 0:
                            r4h = 1 if (d4h > dtf) else d4h / dtf
                        else:
                            r4h = 0.01  #  don't set to 0


                        vlr_z[im1f, im2f, itf] = r1h*r2h*r3h*r4h
                    else:  #  not a border
                        if ((d1h < 0) and (d2h < 0) and \
                            (d3h < 0) and (d4h < 0)):
                            vlr_z[im1f, im2f, itf] = 1
                        else:
                            vlr_z[im1f, im2f, itf] = 0

                    # if (((O_z[im1f, im2f] < tH) or (O_z[im1f+1, im2f] < tH) or \
                    #     (O_z[im1f, im2f+1] < tH) or (O_z[im1f+1, im2f+1] < tH)) and \
                    #     ((O_z[im1f, im2f] > tL) or (O_z[im1f+1, im2f] > tL) or \
                    #      (O_z[im1f, im2f+1] > tL) or (O_z[im1f+1, im2f+1] > tL))):
                    #     #  a border
                    #     vlr_z[im1f, im2f, itf] = 
                    # else:  #  not a border
                    #     if ((O_z[im1f, im2f] > tH) and (O_z[im1f+1, im2f] > tH) and \
                    #         (O_z[im1f, im2f+1] > tH) and (O_z[im1f+1, im2f+1] > tH)):
                    #         vlr_z[im1f, im2f, itf] = 1
                    #     else:
                    #         vlr_z[im1f, im2f, itf] = 0

        return _N.mean(vlr_z)

                        
                    
        
        
        
