"""
V1.2   use adaptive range for integrating over f
variance   0.0001 
"""
import stats_util as s_u
import scipy.stats as _ss
import os
import time as _tm
from ig_prmLib import ig_prmsUV
import numpy as _N
import matplotlib.pyplot as _plt
from EnDedirs import resFN, datFN
import pickle
from fitutil import  emMKPOS_sep1A, sepHashEM, sepHash, colorclusters
from posteriorUtil import MAPvalues2
from filter import gauKer

import clrs 

class MarkAndRF:
    ky_p_l0 = 0;    ky_p_f  = 1;    ky_p_q2 = 2
    ky_h_l0_a = 0;  ky_h_l0_B=1;
    ky_h_f_u  = 2;  ky_h_f_q2=3;
    ky_h_q2_a = 4;  ky_h_q2_B=5;

    ky_p_u = 0;       ky_p_Sg = 1;
    ky_h_u_u = 0;     ky_h_u_Sg=1;
    ky_h_Sg_nu = 2;   ky_h_Sg_PSI=3;

    dt      = 0.001
    #  position dependent firing rate
    ######################################  PRIORS
    twpi = 2*_N.pi

    #  sizes of arrays
    Nupx = 300      #   # points to sample position with  (uniform lam(x)p(x))
    fss = 100       #  sampling at various values of f
    q2ss = 150      #  sampling at various values of q2

    intvs = None    #  
    dat   = None

    resetClus = True

    diffPerMin = 1.  #  diffusion per minute
    epochs   = None

    outdir   = None
    polyFit  = True
    xLo      = -6
    xHi      = 6
    
    def __init__(self, outdir, fn, intvfn, xLo=0, xHi=3):
        oo     = self
        ######################################  DATA input, define intervals
        # bFN = fn[0:-4]
        oo.outdir = outdir

        # if not os.access(bFN, os.F_OK):
        #     os.mkdir(bFN)

        oo.dat    = _N.loadtxt("%s.dat" % datFN(fn, create=False))
        #oo.datprms= _N.loadtxt("%s_prms.dat" % datFN(fn, create=False))

        intvs     = _N.loadtxt("%s.dat" % datFN(intvfn, create=False))

        oo.intvs  = _N.array(intvs*oo.dat.shape[0], dtype=_N.int)
        oo.epochs    = oo.intvs.shape[0] - 1
        
        NT     = oo.dat.shape[0]
        oo.xLo = xLo
        oo.xHi = xHi

    def gibbs(self, ITERS, K, ep1=0, ep2=None, savePosterior=True, gtdiffusion=False, Mdbg=None, doSepHash=True):
        """
        gtdiffusion:  use ground truth center of place field in calculating variance of center.  Meaning of diffPerMin different
        """
        print "gibbs"
        oo = self
        twpi     = 2*_N.pi
        pcklme   = {}

        gkMAP    = gauKer(2)

        ep2 = oo.epochs if (ep2 == None) else ep2
        oo.epochs = ep2-ep1


        ######################################  GRID for calculating
        ####  #  points in sum.  
        ####  #  points in uniform sampling of exp(x)p(x)   (non-spike interals)
        ####  #  points in sampling of f  for conditional posterior distribution
        ####  #  points in sampling of q2 for conditional posterior distribution
        ####  NSexp, Nupx, fss, q2ss

        #  numerical grid
        ux = _N.linspace(oo.xLo, oo.xHi, oo.Nupx, endpoint=False)   # uniform x position
        q2x    = _N.exp(_N.linspace(_N.log(0.0001), _N.log(100), oo.q2ss))  #  5 orders of
        d_q2x  = _N.diff(q2x)
        q2x_m1 = _N.array(q2x[0:-1])
        lq2x    = _N.log(q2x)
        iq2x    = 1./q2x
        q2xr     = q2x.reshape((oo.q2ss, 1))
        iq2xr     = 1./q2xr
        sqrt_2pi_q2x   = _N.sqrt(twpi*q2x)
        l_sqrt_2pi_q2x = _N.log(sqrt_2pi_q2x)

        freeClstr = None
        x      = oo.dat[:, 0]
        mks    = oo.dat[:, 2:]

        q2rate = oo.diffPerEpoch**2  #  unit of minutes  
        ######################################  PRECOMPUTED
        posbins  = _N.linspace(oo.xLo, oo.xHi, oo.Nupx+1)

        for epc in xrange(ep1, ep2):
            print "^^^^^^^^^^^^^^^^^^^^^^^^    epoch %d" % epc

            t0 = oo.intvs[epc]
            t1 = oo.intvs[epc+1]

            Asts    = _N.where(oo.dat[t0:t1, 1] == 1)[0]   #  based at 0
            Ants    = _N.where(oo.dat[t0:t1, 1] == 0)[0]

            n1   = len(Asts)
            n0   = 0

            if epc == ep1:   ###  initialize
                _x   = _N.empty((n1-n0, K+1))
                _x[:, 0]    = x[Asts+t0]
                _x[:, 1:]   = mks[Asts+t0]

                if not doSepHash:
                    unonhash = _N.arange(len(Asts))
                    hashsp   = _N.array([])
                    hashthresh = _N.min(_x[:, 1:], axis=0)
                else:
                    unonhash, hashsp, hashthresh = sepHash(_x, BINS=10, blksz=5, xlo=0, xhi=3)
                #  hashthresh is dim 2
                
                    # fig = _plt.figure(figsize=(5, 10))
                    # fig.add_subplot(3, 1, 1)
                    # _plt.scatter(_x[hashsp, 1], _x[hashsp, 2], color="red")
                    # _plt.scatter(_x[unonhash, 1], _x[unonhash, 2], color="black")
                    # fig.add_subplot(3, 1, 2)
                    # _plt.scatter(_x[hashsp, 0], _x[hashsp, 1], color="red")
                    # _plt.scatter(_x[unonhash, 0], _x[unonhash, 1], color="black")
                    # fig.add_subplot(3, 1, 3)
                    # _plt.scatter(_x[hashsp, 0], _x[hashsp, 2], color="red")
                    # _plt.scatter(_x[unonhash, 0], _x[unonhash, 2], color="black")

                if (len(unonhash) > 0) and (len(hashsp) > 0):
                    labS, labH, clstrs = emMKPOS_sep1A(_x[unonhash], _x[hashsp])
                elif len(unonhash) == 0:
                    labS, labH, clstrs = emMKPOS_sep1A(None, _x[hashsp], TR=5)
                else:
                    labS, labH, clstrs = emMKPOS_sep1A(_x[unonhash], None, TR=5)
                colorclusters(_x[hashsp], labH, clstrs[1])
                colorclusters(_x[unonhash], labS, clstrs[0])

                #fig = _plt.figure(figsize=(7, 10))
                #fig.add_subplot(2, 1, 1)

                flatlabels = _N.ones(n1-n0, dtype=_N.int)*-1
                cls = clrs.get_colors(clstrs[0] + clstrs[1])
                for i in xrange(clstrs[0]):
                    these = _N.where(labS == i)[0]
                    
                    flatlabels[unonhash[these]] = i
                    #_plt.scatter(_x[unonhash[these], 0], _x[unonhash[these], 1], color=cls[i])
                for i in xrange(clstrs[1]):
                    these = _N.where(labH == i)[0]

                    flatlabels[hashsp[these]] = i + clstrs[0]
                    #_plt.scatter(_x[hashsp[these], 0], _x[hashsp[these], 1], color=cls[i+clstrs[0]])

                MF     = clstrs[0] + clstrs[1]
                M = int(MF * 1.1) + 2   #  20% more clusters
                print "cluters:  %d" % M

                freeClstr = _N.empty(M, dtype=_N.bool)   #  Actual cluster
                freeClstr[:] = False



                #  PRIORS
                #  priors  prefixed w/ _
                #_f_u   = _N.zeros(M);    _f_q2  = _N.ones(M)*4 #  wide
                _f_u   = _N.zeros(M);    _f_q2  = _N.ones(M)*1 #  wide
                #  inverse gamma
                _q2_a  = _N.ones(M)*1e-4;    _q2_B  = _N.ones(M)*1e-3
                #_plt.plot(q2x, q2x**(-_q2_a-1)*_N.exp(-_q2_B / q2x))
                _l0_a = _N.ones(M)*1.1;     _l0_B  = _N.ones(M)*(1/30.)

                #  
                _u_u   = _N.zeros((M, K));  
                #_u_Sg = _N.tile(_N.identity(K), M).T.reshape((M, K, K))*9
                _u_Sg = _N.tile(_N.identity(K), M).T.reshape((M, K, K))*1
                _u_iSg = _N.linalg.inv(_u_Sg)
                _Sg_nu = _N.ones((M, 1));  
                _Sg_PSI = _N.tile(_N.identity(K), M).T.reshape((M, K, K))*0.1

                #####  MODES  - find from the sampling
                oo.sp_prmPstMd = _N.zeros((oo.epochs, 3*M))   # mode of params
                oo.sp_hypPstMd  = _N.zeros((oo.epochs, (2+2+2)*M)) # hyperparam
                oo.mk_prmPstMd = [_N.zeros((oo.epochs, M, K)),
                                  _N.zeros((oo.epochs, M, K, K))]
                                  # mode of params
                oo.mk_hypPstMd  = [_N.zeros((oo.epochs, M, K)),
                                   _N.zeros((oo.epochs, M, K, K)), # hyperparam
                                   _N.zeros((oo.epochs, M, 1)), # hyperparam
                                   _N.zeros((oo.epochs, M, K, K))]

                #  Gibbs sampling
                #  parameters l0, f, q2

                ######################################  GIBBS samples, need for MAP estimate
                smp_sp_prms = _N.zeros((3, ITERS, M))  
                smp_mk_prms = [_N.zeros((K, ITERS, M)), _N.zeros((K, K, ITERS, M))]
                smp_sp_hyps = _N.zeros((6, ITERS, M))  
                smp_mk_hyps = [_N.zeros((K, ITERS, M)), _N.zeros((K, K, ITERS, M)),
                               _N.zeros((1, ITERS, M)), _N.zeros((K, K, ITERS, M))]

                ######################################  INITIAL VALUE OF PARAMS
                l0       = _N.array([11.,]*M)
                q2       = _N.array([0.04]*M)
                #f        = _N.array([1.1]*M)
                f        = _N.empty(M)
                #f        = _N.array([0.6, 2.3])
                ######################################  MARK PARAMS
                u       = _N.zeros((M, K))   #  center
                # u       = _N.array([[5, 4.5],
                #                     [1.5, 2.5]])   #  center

                Sg      = _N.tile(_N.identity(K), M).T.reshape((M, K, K))*0.1

                ######  the hyperparameters for u need to be saved
                u_u_     = _N.zeros((M, K))
                u_Sg_    = _N.array(_u_Sg)
                Sg_nu_   = _N.zeros((M, 1))
                Sg_PSI_  = _N.array(_Sg_PSI)

                rat      = _N.zeros(M+1)
                pc       = _N.zeros(M)

                ##################
                lab      = _N.array(labS.tolist() + (labH + clstrs[0]).tolist())
                tmpx     = _N.empty((n1-n0, K+1))

                strt = 0
                if len(unonhash) > 0:
                    tmpx[0:len(unonhash)] = _x[unonhash]
                    strt = len(unonhash)
                if len(hashsp) > 0:
                    tmpx[strt:]  = _x[hashsp]

                for im in xrange(M):  #if lab < 0, these marks not used for init
                    if im < MF:
                        kinds = _N.where(flatlabels == im)[0]  #  inds
                        f[im]  = _N.mean(x[Asts[kinds]+t0], axis=0)
                        u[im]  = _N.mean(mks[Asts[kinds]+t0], axis=0)
                        q2[im] = 0.05
                        Sg[im] = _N.identity(K)*0.1
                        l0[im] = 10

                    else:
                        f[im]  = _N.random.rand()*3
                        u[im]  = _N.random.rand(K)
                        q2[im] = 0.05
                        Sg[im] = _N.identity(K)*0.1
                        l0[im] = 2

                oo.sp_prmPstMd[0, oo.ky_p_l0::3] = l0
                oo.sp_prmPstMd[0, oo.ky_p_f::3] = f
                oo.sp_prmPstMd[0, oo.ky_p_q2::3] = q2
                oo.mk_prmPstMd[oo.ky_p_u][0] = u
                oo.mk_prmPstMd[oo.ky_p_Sg][0] = Sg


            if gtdiffusion:
                #  tell me how much diffusion to expect per min.
                #  [(EXPECT) x (# of minutes between epochs)]**2 
                q2rate = (oo.dat[t1-1,2]-oo.dat[t0,2])**2*oo.diffPerMin

            NSexp   = t1-t0    #  length of position data  #  # of no spike positions to sum
            xt0t1 = _N.array(x[t0:t1])

            px, xbns = _N.histogram(xt0t1, bins=posbins, normed=True)

            nSpks    = len(Asts)
            gz   = _N.zeros((ITERS, nSpks, M), dtype=_N.bool)
            print "spikes %d" % nSpks

            dSilenceX = (NSexp/float(oo.Nupx))*(oo.xHi-oo.xLo)

            xAS  = x[Asts + t0]   #  position @ spikes
            mAS  = mks[Asts + t0]   #  position @ spikes
            xASr = xAS.reshape((1, nSpks))
            mASr = mAS.reshape((nSpks, 1, K))
            econt = _N.empty((M, nSpks))
            rat   = _N.zeros((M+1, nSpks))

            qdrMKS = _N.empty((M, nSpks))
            ################################  GIBBS ITERS ITERS ITERS

            #  linalgerror
            #_iSg_Mu = _N.einsum("mjk,mk->mj", _N.linalg.inv(_u_Sg), _u_u)

            clusSz = _N.zeros(M, dtype=_N.int)
            #print "---prior covariance"
            #print _u_Sg
            #print "---prior covariance"

            # print "q2_a, q2_B is" 
            # print _q2_a
            # print _q2_B

            for iter in xrange(ITERS):
                tt1 = _tm.time()
                if (iter % 100) == 0:    print "iter  %d" % iter

                ur         = u.reshape((1, M, K))
                fr         = f.reshape((M, 1))    # centers
                iq2        = 1./q2
                iSg        = _N.linalg.inv(Sg)
                iq2r       = iq2.reshape((M, 1))  
                try:
                    ##  warnings because l0 is 0
                    isN = _N.where(q2 <= 0)[0]
                    if len(isN) > 0:
                        q2[isN] = 0.3

                    is0 = _N.where(l0 <= 0)[0]
                    if len(is0) > 0:
                        l0[is0] = 0.001

                    pkFR       = _N.log(l0) - 0.5*_N.log(_N.sqrt(twpi*q2))
                except RuntimeWarning:
                    print "WARNING"
                    print l0
                    print q2

                mkNrms = _N.log(1/_N.sqrt(twpi*_N.linalg.det(Sg)))
                mkNrms = mkNrms.reshape((M, 1))

                rnds       = _N.random.rand(nSpks)

                pkFRr      = pkFR.reshape((M, 1))
                dmu        = (mASr - ur)

                _N.einsum("nmj,mjk,nmk->mn", dmu, iSg, dmu, out=qdrMKS)
                qdrSPC     = (fr - xASr)*(fr - xASr)*iq2r  #  M x nSpks


                ###  how far is closest cluster to each newly observed mark

                realCl = _N.where(freeClstr == False)[0]
                nNrstMKS_d = _N.sqrt(_N.min(qdrMKS[realCl], axis=0)/K)  #  dim len(sts)
                nNrstSPC_d = _N.sqrt(_N.min(qdrSPC[realCl], axis=0))


                #  mAS = mks[Asts+t0] 
                #  xAS = x[Asts + t0]   #  position @ spikes
                
                if (epc > 0) and (iter == 0):
                    # print "---------------qdrSPC"
                    # print qdrSPC
                    # print "---------------qdrSPC"
                    # fig = _plt.figure()
                    # fig.add_subplot(2, 1, 1)
                    # _plt.hist(nNrstMKS_d, bins=30)
                    # fig.add_subplot(2, 1, 2)
                    # _plt.hist(nNrstSPC_d, bins=30)

                    # fig = _plt.figure()
                    # fig.add_subplot(1, 1, 1)
                    # _plt.hist(x[Asts+t0], bins=30)

                    abvthrEachCh = mks[Asts+t0] > hashthresh
                    abvthrAtLeast1Ch = _N.sum(abvthrEachCh, axis=1) > 0
                    abvthrInds   = _N.where(abvthrAtLeast1Ch)[0]

                    print "MKS"
                    farMKS = _N.where((nNrstMKS_d > 1) & abvthrAtLeast1Ch)[0]
                    print "SPACE"
                    farSPC  = _N.where((nNrstSPC_d > 2))[0]
                    print "len(farSPC) is %d" % len(farSPC)
                    print "len(farMK) is %d" % len(farMKS)


                    iused = 0  #  use up to 3
                    bDone = False
                    fig = _plt.figure(figsize=(8, 5))
                    fig.add_subplot(2, 1, 1)
                    _plt.scatter(x[Asts + t0], mks[Asts+t0, 0], color="black", s=2)
                    bDoMKS = (len(farMKS) > 0)
                    bDoSPC = (len(farSPC) > 0)

                    for m in xrange(M):
                        #if freeClstr[m] and (not bDone) and bDoMKS:
                        if freeClstr[m]:
                            these     = (Asts+t0)[abvthrInds]
                            _f_u[m] = _N.mean(x[these], axis=0)
                            _u_u[m]   = _N.mean(mks[these], axis=0)
                            l0[m]     = _N.random.rand()*10
                            if (iused < 2) and bDoMKS:
                                these     = (Asts+t0)[farMKS]
                                f[m]      = _N.mean(x[these], axis=0)
                                u[m]      = _N.mean(mks[these], axis=0)
                            else:
                                f[m]      = _N.mean(x[these], axis=0)
                                u[m]      = _N.mean(mks[these], axis=0)

                                # print "BEG far markise"
                                # _plt.scatter(x[these], mks[these, 0], color="red", s=4)
                                # print "f[m]=%(1)s   u[m]=%(2)s" % {"1" : str(f[m]), "2" : str(u[m])}
                                freeClstr[m] = False
                                iused += 1
                        # else:
                        #     print "NOTTTTTTT far markise"
                        #     print "f[m]=%(1)s   u[m]=%(2)s" % {"1" : str(f[m]), "2" : str(u[m])}

                    bDone = False
                    # fig.add_subplot(2, 1, 2)
                    # _plt.scatter(x[Asts + t0], mks[Asts+t0, 0], color="black", s=3)
                    iused = 0
                    for m in xrange(M):
                        #if freeClstr[m] and (not bDone) and bDoSPC:
                        if freeClstr[m]:
                            these     = (Asts+t0)[abvthrInds]
                            _f_u[m] = _N.mean(x[these], axis=0)
                            _u_u[m]   = _N.mean(mks[these], axis=0)
                            l0[m]     = _N.random.rand()*10
                            if (iused < 2) and bDoSPC:
                                these     = (Asts+t0)[farSPC]
                                f[m]      = _N.mean(x[these], axis=0)
                                u[m]      = _N.mean(mks[these], axis=0)
                            else:
                                f[m]      = _N.mean(x[these], axis=0)
                                u[m]      = _N.mean(mks[these], axis=0)

                                # _plt.scatter(x[these], mks[these, 0], color="red", s=4)
                                # print "BEG far spatial"
                                # print "f[m]=%(1)s   u[m]=%(2)s" % {"1" : str(f[m]), "2" : str(u[m])}
                                freeClstr[m] = False
                                iused += 1

                        # else:
                        #     print "NOTTTTTTT far spatial"
                        #     print "f[m]=%(1)s   u[m]=%(2)s" % {"1" : str(f[m]), "2" : str(u[m])}
                        

                cont       = pkFRr + mkNrms - 0.5*(qdrSPC + qdrMKS)
                
                mcontr     = _N.max(cont, axis=0).reshape((1, nSpks))  
                cont       -= mcontr
                _N.exp(cont, out=econt)

                for m in xrange(M):
                    rat[m+1] = rat[m] + econt[m]

                rat /= rat[M]
                """
                # print f
                # print u
                # print q2
                # print Sg
                # print l0
                """

                # print rat

                M1 = rat[1:] >= rnds
                M2 = rat[0:-1] <= rnds

                gz[iter] = (M1&M2).T

                ###############  FOR EACH CLUSTER

                for m in xrange(M):
                    ttc1 = _tm.time()
                    iiq2 = 1./q2[m]
                    minds = _N.where(gz[iter, :, m] == 1)[0]
                    sts  = Asts[minds] + t0
                    nSpksM   = len(sts)
                    clusSz[m] = nSpksM

                    #  prior described by hyper-parameters.
                    #  prior described by function

                    #  likelihood

                    """
                    ############################################
                    """
                    mcs = _N.empty((M, K))   # cluster sample means

                    #u_Sg_[m] = _N.linalg.inv(_u_Sg[m] + nSpksM*iSg[m])

                    if nSpksM > 0:
                        #try:
                        u_Sg_[m] = _N.linalg.inv(_N.linalg.inv(_u_Sg[m]) + nSpksM*iSg[m])
                        # except _N.linalg.linalg.LinAlgError:
                        #     print m
                        #     print _u_Sg[m]
                        #     print iSg[m]
                        #     raise
                        clstx    = mks[sts]

                        mcs[m]       = _N.mean(clstx, axis=0)
                        u_u_[m] = _N.einsum("jk,k->j", u_Sg_[m], _N.dot(_N.linalg.inv(_u_Sg[m]), _u_u[m]) + nSpksM*_N.dot(iSg[m], mcs[m]))
                        u[m] = _N.random.multivariate_normal(u_u_[m], u_Sg_[m])

                        # hyp
                        ########  POSITION
                        ##  mean of posterior distribution of cluster means
                        #  sigma^2 and mu are the current Gibbs-sampled values

                        ##  mean of posterior distribution of cluster means

                    smp_mk_prms[oo.ky_p_u][:, iter, m] = u[m]
                    #print ".   %d" % nSpksM
                    smp_mk_hyps[oo.ky_h_u_u][:, iter, m] = u_u_[m]
                    smp_mk_hyps[oo.ky_h_u_Sg][:, :, iter, m] = u_Sg_[m]

                    #ttc1a = _tm.time()
                    # dot(MATRIX, vector)   
                    """
                    ############################################
                    """
                    ###############  CONDITIONAL f
                    q2pr = _f_q2[m] if (_f_q2[m] > q2rate) else q2rate
                    #ttc1b = _tm.time()
                    if nSpksM > 0:  #  spiking portion likelihood x prior
                        fs  = (1./nSpksM)*_N.sum(xt0t1[sts-t0])
                        fq2 = q2[m]/nSpksM
                        U   = (fs*q2pr + _f_u[m]*fq2) / (q2pr + fq2)
                        FQ2 = (q2pr*fq2) / (q2pr + fq2)
                    else:
                        U   = _f_u[m]
                        FQ2 = q2pr
                    #ttc1c = _tm.time()
                    FQ    = _N.sqrt(FQ2)
                    fx    = _N.linspace(U - FQ*60, U + FQ*60, oo.fss)
                    fxr     = fx.reshape((oo.fss, 1))
                    #ttc1d = _tm.time()
                    fxrux = -0.5*(fxr-ux)*(fxr-ux)
                    #_fxrux = -0.5*(fx-ux)*(fx-ux)
                    #fxrux = _fxrux.reshape(oo
                    #ttc1e = _tm.time()
                    #xI_f    = (xt0t1 - fxr)*(xt0t1-fxr)*0.5
                    #ttc1f = _tm.time()
                    f_intgrd  = _N.exp((fxrux*iiq2))   #  integrand
                    f_exp_px = _N.sum(f_intgrd*px, axis=1) * dSilenceX
                    #  f_exp_px is a function of f
                    s = -(l0[m]*oo.dt/_N.sqrt(twpi*q2[m])) * f_exp_px  #  a function of x

                    funcf   = -0.5*((fx-U)*(fx-U))/FQ2 + s
                    funcf   -= _N.max(funcf)
                    condPosF= _N.exp(funcf)
                    #print _N.sum(condPosF)

                    norm    = 1./_N.sum(condPosF)
                    f_u_    = norm*_N.sum(fx*condPosF)
                    f_q2_   = norm*_N.sum(condPosF*(fx-f_u_)*(fx-f_u_))
                    f[m]    = _N.sqrt(f_q2_)*_N.random.randn() + f_u_
                    smp_sp_prms[oo.ky_p_f, iter, m] = f[m]
                    smp_sp_hyps[oo.ky_h_f_u, iter, m] = f_u_
                    smp_sp_hyps[oo.ky_h_f_q2, iter, m] = f_q2_

                    #ttc1g = _tm.time()
                    #############  VARIANCE, COVARIANCE
                    if nSpksM >= K:
                        ##  dof of posterior distribution of cluster covariance
                        Sg_nu_[m] = _Sg_nu[m, 0] + nSpksM
                        ##  dof of posterior distribution of cluster covariance
                        ur = u[m].reshape((1, K))
                        Sg_PSI_[m] = _Sg_PSI[m] + _N.dot((clstx - ur).T, (clstx-ur))
                        Sg[m] = s_u.sample_invwishart(Sg_PSI_[m], Sg_nu_[m, 0])
                        # if (_N.sum(_N.isnan(Sg[m])) > 0) or (_N.sum(_N.isinf(Sg[m])) > 0):
                        #     print "Problem  %d" % m
                        #     print Sg_PSI_[m]
                        #     print Sg_nu_[m, 0]
                        #     print nSpksM

                    # #print Sg_PSI_
                    ##############  SAMPLE COVARIANCES

                    ##  dof of posterior distribution of cluster covariance

                    smp_mk_prms[oo.ky_p_Sg][:, :, iter, m] = Sg[m]
                    smp_mk_hyps[oo.ky_h_Sg_nu][0, iter, m] = Sg_nu_[m]
                    smp_mk_hyps[oo.ky_h_Sg_PSI][:, :, iter, m] = Sg_PSI_[m]

                    # ###############  CONDITIONAL q2
                    #xI = (xt0t1-f)*(xt0t1-f)*0.5*iq2xr
                    q2_intgrd   = _N.exp(-0.5*(f[m] - ux)*(f[m]-ux) * iq2xr)  
                    q2_exp_px   = _N.sum(q2_intgrd*px, axis=1) * dSilenceX

                    s = -((l0[m]*oo.dt)/sqrt_2pi_q2x)*q2_exp_px     #  function of q2

                    _Dq2_a = _q2_a[m]# if _q2_a[m] < 200 else 200
                    _Dq2_B = _q2_B[m]#(_q2_B[m]/(_q2_a[m]+1))*(_Dq2_a+1)

                    if nSpksM > 0:
                        #print  _N.sum((xt0t1[sts]-f)*(xt0t1[sts]-f))/(nSpks-1)

                        ##  (1/sqrt(sg2))^S
                        ##  (1/x)^(S/2)   = (1/x)-(a+1)
                        ##  -S/2 = -a - 1     -a = -S/2 + 1    a = S/2-1
                        xI = (xt0t1[sts-t0]-f[m])*(xt0t1[sts-t0]-f[m])*0.5
                        SL_a = 0.5*nSpksM - 1   #  spiking part of likelihood
                        SL_B = _N.sum(xI)  #  spiking part of likelihood
                        #  spiking prior x prior
                        sLLkPr = -(_q2_a[m] + SL_a + 2)*lq2x - iq2x*(_q2_B[m] + SL_B)
                    else:
                        sLLkPr = -(_q2_a[m] + 1)*lq2x - iq2x*_q2_B[m]

                    sat = sLLkPr + s
                    sat -= _N.max(sat)
                    condPos = _N.exp(sat)
                    q2_a_, q2_B_ = ig_prmsUV(q2x, condPos, d_q2x, q2x_m1, ITER=1)
                    q2[m] = _ss.invgamma.rvs(q2_a_ + 1, scale=q2_B_)  #  check
                    #print ((1./nSpks)*_N.sum((xt0t1[sts]-f)*(xt0t1[sts]-f)))

                    smp_sp_prms[oo.ky_p_q2, iter, m]   = q2[m]
                    smp_sp_hyps[oo.ky_h_q2_a, iter, m] = q2_a_
                    smp_sp_hyps[oo.ky_h_q2_B, iter, m] = q2_B_
                    
                    #ttc1h = _tm.time()
                    ###############  CONDITIONAL l0

                    #  _ss.gamma.rvs.  uses k, theta    k is 1/B  (B is our thing)
                    iiq2 = 1./q2[m]
                    # xI = (xt0t1-f)*(xt0t1-f)*0.5*iiq2
                    # BL  = (oo.dt/_N.sqrt(twpi*q2))*_N.sum(_N.exp(-xI))

                    l0_intgrd   = _N.exp(-0.5*(f[m] - ux)*(f[m]-ux) * iiq2)  
                    l0_exp_px   = _N.sum(l0_intgrd*px) * dSilenceX
                    BL  = (oo.dt/_N.sqrt(twpi*q2[m]))*l0_exp_px

                    _Dl0_a = _l0_a[m]# if _l0_a[m] < 400 else 400
                    #_Dl0_a = _l0_a[m] if _l0_a[m] < 25 else 25
                    _Dl0_B = _l0_B[m]#(_l0_B[m]/_l0_a[m]) * _Dl0_a
                    
                    #  a'/B' = a/B
                    #  B' = (B/a)a'
                    aL  = nSpksM
                    l0_a_ = aL + _Dl0_a
                    l0_B_ = BL + _Dl0_B

                    #print "l0_a_ %(a).3e   l0_B_ %(B).3e" % {"a" : l0_a_, "B" : l0_B_}

                    #if (l0_B_ > 0) and (l0_a_ > 1):
                    l0[m] = _ss.gamma.rvs(l0_a_ - 1, scale=(1/l0_B_))  #  check
                    #else:
                    #    print "cluster %(c)d   nSpksM %(n)d   %(B).4f  %(a).4f" % {"c" : m, "n" : nSpksM, "B" : l0_B_, "a" : l0_a_}

                    ###  l0 / _N.sqrt(twpi*q2) is f*dt used in createData2

                    smp_sp_prms[oo.ky_p_l0, iter, m] = l0[m]

                    smp_sp_hyps[oo.ky_h_l0_a, iter, m] = l0_a_
                    smp_sp_hyps[oo.ky_h_l0_B, iter, m] = l0_B_
                    ttc2 = _tm.time()
                    #print "cls %(c)d  1) %(dt1).2f  2) %(dt2).2f  3) %(dt3).2f   4) %(dt4).2f  5) %(dt5).2f   6) %(dt6).2f  7) %(dt7).2f  8) %(dt8).2f  9) %(dt8).2f" % {"c" : m, "dt1" : (ttc1a-ttc1), "dt2" : (ttc1b-ttc1a),  "dt3" : (ttc1c-ttc1b), "dt4" : (ttc1d-ttc1c), "dt5" : (ttc1e-ttc1d), "dt6" : (ttc1f-ttc1e), "dt7" : (ttc1g-ttc1f), "dt8" : (ttc1g-ttc1f), "dt8" : (ttc2-ttc1g)}
                    #print "cls %(c)d  %(dt).3f" % {"c" : m, "dt" : (ttc2-ttc1)}

            tt2 = _tm.time()
            # print "iter time %.1f" % (tt2-tt1)
            # print "l0 is"
            # print l0
            # print "f is" 
            # print f
            # print "q2 is" 
            # print q2
            # print "Sg is" 
            # print Sg
            
            frm   = int(0.6*ITERS)  #  have to test for stationarity

            if nSpks > 0:
                #  ITERS x nSpks x M   
                occ   = _N.mean(_N.mean(gz[frm:ITERS-1], axis=0), axis=0)

            oo.smp_sp_hyps = smp_sp_hyps
            oo.smp_sp_prms = smp_sp_prms
            oo.smp_mk_hyps = smp_mk_hyps
            oo.smp_mk_prms = smp_mk_prms

            l_trlsNearMAP = []
            MAPvalues2(epc, smp_sp_prms, oo.sp_prmPstMd, frm, ITERS, M, 3, occ, gkMAP, l_trlsNearMAP)
            l0[:]         = oo.sp_prmPstMd[epc, oo.ky_p_l0::3]
            f[:]          = oo.sp_prmPstMd[epc, oo.ky_p_f::3]
            q2[:]         = oo.sp_prmPstMd[epc, oo.ky_p_q2::3]
            MAPvalues2(epc, smp_sp_hyps, oo.sp_hypPstMd, frm, ITERS, M, 6, occ, gkMAP, None)
            _f_u[:]       = oo.sp_hypPstMd[epc, oo.ky_h_f_u::6]
            _f_q2[:]      = oo.sp_hypPstMd[epc, oo.ky_h_f_q2::6]
            _q2_a[:]      = oo.sp_hypPstMd[epc, oo.ky_h_q2_a::6]
            _q2_B[:]      = oo.sp_hypPstMd[epc, oo.ky_h_q2_B::6]
            _l0_a[:]      = oo.sp_hypPstMd[epc, oo.ky_h_l0_a::6]
            _l0_B[:]      = oo.sp_hypPstMd[epc, oo.ky_h_l0_B::6]
            # print _f_u
            # print "_f_q2"
            # print _f_q2
            # print "_f_q2"
            # print _q2_a
            # print _q2_B
            # print _l0_a
            # print _l0_B

            #print l_trlsNearMAP
            
            #pcklme["cp%d" % epc] = _N.array(smp_sp_prms)
            #trlsNearMAP = _N.array(list(set(trlsNearMAP_D)))+frm   #  use these trials to pick out posterior params for MARK part

            #oo.mk_prmPstMd = [ epochs, M, K
            #                      epochs, M, K, K ]

            #oo.mk_hypPstMd  = [ epochs, M, K
            #                    epochs, M, K, K
            #                    epochs, M, 1
            #                    epochs, M, K, K

            #smp_mk_prms = [   K, ITERS, M
            #                  K, K, ITERS, M
            #smp_mk_hyps = [   K, ITERS, M
            #                  K, K, ITERS, M
            #                  1, ITERS, M
            #                  K, K, ITERS, M


            ##  params and hyper parms for mark
            for m in xrange(M):
                MAPtrls = l_trlsNearMAP[m]
                if len(MAPtrls) == 0:  #  none of them.  causes nan in mean
                    MAPtrls = _N.arange(frm, ITERS, 10)
                #print MAPtrls
                u[m] = _N.mean(smp_mk_prms[0][:, MAPtrls, m], axis=1)
                Sg[m] = _N.mean(smp_mk_prms[1][:, :, MAPtrls, m], axis=2)
                oo.mk_prmPstMd[oo.ky_p_u][epc, m] = u[m]
                oo.mk_prmPstMd[oo.ky_p_Sg][epc, m]= Sg[m]
                _u_u[m]    = _N.mean(smp_mk_hyps[oo.ky_h_u_u][:, MAPtrls, m], axis=1)
                _u_Sg[m]   = _N.mean(smp_mk_hyps[oo.ky_h_u_Sg][:, :, MAPtrls, m], axis=2)
                _Sg_nu[m]  = _N.mean(smp_mk_hyps[oo.ky_h_Sg_nu][0, MAPtrls, m], axis=0)
                _Sg_PSI[m] = _N.mean(smp_mk_hyps[oo.ky_h_Sg_PSI][:, :, MAPtrls, m], axis=2)
                oo.mk_hypPstMd[oo.ky_h_u_u][epc, m]   = _u_u[m]
                oo.mk_hypPstMd[oo.ky_h_u_Sg][epc, m]  = _u_Sg[m]
                oo.mk_hypPstMd[oo.ky_h_Sg_nu][epc, m] = _Sg_nu[m]
                oo.mk_hypPstMd[oo.ky_h_Sg_PSI][epc, m]= _Sg_PSI[m]
                #print _u_Sg[m]
            u[:]         = oo.mk_prmPstMd[oo.ky_p_u][epc]
            Sg[:]        = oo.mk_prmPstMd[oo.ky_p_Sg][epc]

            ###  hack here.  If we don't reset the prior for 
            ###  what happens when a cluster is unused?
            ###  l0 -> 0, and at the same time, the variance increases.
            ###  the prior then gets pushed to large values, but
            ###  then it becomes difficult to bring it back to small
            ###  values once that cluster becomes used again.  So
            ###  we would like unused clusters to have l0->0, but keep the
            ###  variance small.  That's why we will reset a cluster

            sq25  = 5*_N.sqrt(q2)
            occ = _N.mean(_N.sum(gz[frm:], axis=1), axis=0)  # avg. # of marks assigned to this cluster
            socc = _N.sort(occ)
            minAss = (0.5*(socc[-2]+socc[-1])*0.01)  #  if we're 100 times smaller than the average of the top 2, let's consider it empty

            print occ
            
            if oo.resetClus:
                for m in xrange(M):
                    #  Sg and q2 are treated differently.  Even if no spikes are
                    #  observed, q2 is updated, while Sg is not.  
                    #  This is because NO spikes in physical space AND trajectory
                    #  information contains information about the place field.
                    #  However, in mark space, not observing any marks tells you
                    #  nothing about the mark distribution.  That is why f, q2
                    #  are updated when there are no spikes, but u and Sg are not.

                    if ((occ[m] < minAss) and (l0[m] / _N.sqrt(twpi*q2[m]) < 1)) or \
                                          (f[m] < oo.xLo-sq25[m]) or \
                                          (f[m] > oo.xHi+sq25[m]):
                        print "resetting  cluster %(m)d   %(l0).3f  %(f).3f" % {"m" : m, "l0" : (l0[m] / _N.sqrt(twpi*q2[m])), "f" : f[m]}

                        _q2_a[m] = 1e-4
                        _q2_B[m] = 1e-3
                        _f_q2[m] = 4
                        _u_Sg[m] = _N.identity(K)*9
                        freeClstr[m] = True
                    else:
                        freeClstr[m] = False


            rsmp_sp_prms = smp_sp_prms.swapaxes(1, 0).reshape(ITERS, 3*M, order="F")

            _N.savetxt(resFN("posParams_%d.dat" % epc, dir=oo.outdir), rsmp_sp_prms, fmt=("%.4f %.4f %.4f " * M))
            #_N.savetxt(resFN("posHypParams.dat", dir=oo.outdir), smp_sp_hyps[:, :, 0].T, fmt="%.4f %.4f %.4f %.4f %.4f %.4f")


        pcklme["sp_prmPstMd"] = oo.sp_prmPstMd
        pcklme["mk_prmPstMd"] = oo.mk_prmPstMd
        pcklme["intvs"]       = oo.intvs
        dmp = open(resFN("posteriors.dmp", dir=oo.outdir), "wb")
        pickle.dump(pcklme, dmp, -1)
        dmp.close()

        #_N.savetxt(resFN("posModes.dat", dir=oo.outdir), oo.sp_prmPstMd, fmt=("%.4f %.4f %.4f " * M))
        #_N.savetxt(resFN("hypModes.dat", dir=oo.outdir), oo.sp_hypPstMd, fmt=("%.4f %.4f %.4f %.4f %.4f %.4f" * M))

        
    def figs(self, ep1=0, ep2=None):
        oo  = self
        ep2 = oo.epochs if (ep2 == None) else ep2

        fig = _plt.figure(figsize=(8, 9))
        mnUs   = _N.empty(ep2-ep1)
        mnL0s  = _N.empty(ep2-ep1)
        mnSq2s = _N.empty(ep2-ep1)

        for epc in xrange(ep1, ep2):
            t0 = oo.intvs[epc]
            t1 = oo.intvs[epc+1]
            sts    = _N.where(oo.dat[t0:t1, 1] == 1)[0]

            mnUs[epc-ep1]   = _N.mean(oo.dat[t0:t1, 2])
            mnSq2s[epc-ep1] = _N.mean(oo.dat[t0:t1, 3])
            mnL0s[epc-ep1]  = _N.mean(oo.dat[t0:t1, 4])

        fig.add_subplot(3, 1, 1)
        _plt.plot(mnUs)
        _plt.plot(oo.sp_prmPstMd[:, oo.ky_p_f])
        fig.add_subplot(3, 1, 2)
        _plt.plot(mnL0s)
        _plt.plot(oo.sp_prmPstMd[:, oo.ky_p_l0])
        fig.add_subplot(3, 1, 3)
        _plt.plot(mnSq2s)
        _plt.plot(oo.sp_prmPstMd[:, oo.ky_p_q2])
            
        _plt.savefig(resFN("cmpModesGT", dir=oo.outdir))
