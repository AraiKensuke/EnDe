"""
V1.2   use adaptive range for integrating over f
variance   0.0001 
"""
import stats_util as s_u
import scipy.stats as _ss
import os
import time as _tm
import py_cdf_smp as _pcs
import numpy as _N
import matplotlib.pyplot as _plt
from EnDedirs import resFN, datFN
import pickle
import posteriorUtil as _pU
from filter import gauKer
import gibbsApprMxMutil as gAMxMu
import stochasticAssignment as _sA
#import cdf_smp_tbl as _cdfs
import cdf_smp_sumx_2d as _cdfs
import ig_from_cdf_pkg as _ifcp
import fastnum as _fm
import clrs 
import compress_gz as c_gz
import conv_gau as _gct

class MarkAndRF:
    ky_p_l0 = 0;    ky_p_fx  = 1;    ky_p_fy  = 2;    ky_p_q2x = 3
    ky_p_q2y = 4; 
    ky_h_l0_a = 0;  ky_h_l0_B=1;
    ky_h_f_u  = 2;  ky_h_f_q2=3;
    ky_h_q2_a = 4;  ky_h_q2_B=5;

    dt      = 0.001
    #  position dependent firing rate
    ######################################  PRIORS
    twpi = 2*_N.pi
    #NExtrClstr = 5
    NExtrClstr = 3
    earliest   = 20000      #   min # of gibbs samples

    #  sizes of arrays
    NposHistBins = 200      #   # points to sample position with  (uniform lam(x)p(x))
    
    intvs = None    #  
    dat   = None

    resetClus = True

    diffPerMin = 1.  #  diffusion per minute
    epochs   = None
    adapt    = False

    outdir   = None

    Nupx      = 200

    #  l0, q2      Sig    f, u
    t_hlf_l0 = int(1000*60*5)   # 10minutes
    t_hlf_q2 = int(1000*60*5)   # 10minutes

    diffusePerMin = 0.05    #  diffusion of certainty

    nz_q2               = 500
    nz_f                = 0

    Bx                  = 0    #  noise in xpos

    #  px and spatial integration limits.  Don't make larger accessible space
    xLo      = -6;    xHi      = 6   

    #  limits of conditional probability sampling of f and q2
    f_L   = -12;     f_H = 12   
    q2_L = 1e-6;    q2_H = 1e4

    oneCluster = False

    q2_lvls    = _N.array([0.02**2, 0.05**2, 0.1**2, 0.2**2, 0.5**2, 1**2, 6**2, 100000**2])
    Nupx_lvls  = _N.array([1000, 600, 200, 100, 60, 20, 12, 8])

    priors     = None
    
    def __init__(self, outdir, fn, intvfn, xLo=0, xHi=3, seed=1041, adapt=True, t_hlf_l0_mins=None, t_hlf_q2_mins=None, oneCluster=False, rotate=False):
        oo     = self
        oo.oneCluster = oneCluster
        oo.adapt = adapt
        _N.random.seed(seed)

        ######################################  DATA input, define intervals
        # bFN = fn[0:-4]
        oo.outdir = outdir

        oo.dat    = _N.loadtxt("%s.dat" % datFN(fn, create=False))
        intvs     = _N.loadtxt("%s.dat" % datFN(intvfn, create=False))

        oo.intvs  = _N.array(intvs*oo.dat.shape[0], dtype=_N.int)
        oo.epochs    = oo.intvs.shape[0] - 1
        
        NT     = oo.dat.shape[0]
        oo.xLo = xLo
        oo.xHi = xHi

    def gibbs(self, ITERS, K, priors, ep1=0, ep2=None, saveSamps=True, saveOcc=True, doSepHash=True, nz_pth=0., smth_pth_ker=0, f_STEPS=13, q2_STEPS=13, f_SMALL=10, q2_SMALL=10, f_cldz=10, q2_cldz=10, minSmps=20, diag_cov=False, earliest=20000, ):
        """
        gtdiffusion:  use ground truth center of place field in calculating variance of center.  Meaning of diffPerMin different
        """
        print "gibbs   %.5f" % _N.random.rand()
        oo = self
        oo.earliest=earliest
        twpi     = 2*_N.pi
        pcklme   = {}

        oo.priors = priors

        ep2 = oo.epochs if (ep2 == None) else ep2
        oo.epochs = ep2-ep1

        ######################################  GRID for calculating
        ####  #  points in sum.  
        ####  #  points in uniform sampling of exp(x)p(x)   (non-spike interals)
        ####  #  points in sampling of f  for conditional posterior distribution
        ####  #  points in sampling of q2 for conditional posterior distribution
        ####  NSexp, Nupx, fss, q2ss

        #  numerical grid
        #ux = _N.linspace(oo.xLo, oo.xHi, oo.Nupx, endpoint=False)   # grid over which spatial integrals are calculated
        #uxr = ux.reshape((1, oo.Nupx))

        #  grid over which conditional probability of q2 adaptively sampled
        #  grid over which conditional probability of q2 adaptively sampled

        x      = oo.dat[:, 0]
        y      = oo.dat[:, 1]

        f_q2_rate = (oo.diffusePerMin**2)/60000.  #  unit of minutes  
        
        ######################################  PRECOMPUTED

        _ifcp.init()
        tau_l0 = oo.t_hlf_l0/_N.log(2)
        tau_q2 = oo.t_hlf_q2/_N.log(2)

        _cdfs.init(oo.dt, oo.f_L, oo.f_H, oo.q2_L, oo.q2_H, f_STEPS, q2_STEPS, f_SMALL, q2_SMALL, f_cldz, q2_cldz, minSmps)
        #        _cdfs.init_occ_resolutions(oo.xLo, oo.xHi, oo.q2_lvls, oo.Nupx_lvls)

        M_max    = 1
        M_use    = 1
        s01col   = 2

        for epc in xrange(ep1, ep2):
            t0 = oo.intvs[epc]
            t1 = oo.intvs[epc+1]
            if epc > 0:
                tm1= oo.intvs[epc-1]
                #  0 10 30     20 - 5  = 15    0.5*((10+30) - (10+0)) = 15
                DT = t0-tm1

            posbins  = _N.linspace(oo.xLo, oo.xHi, oo.Nupx+1)
            
            xt0t1 = _N.array(x[t0:t1])#smthd_pos
            yt0t1 = _N.array(y[t0:t1])#smthd_pos
            Nt0t1 = t1-t0

            #_cdfs.change_occ_px(xt0t1, oo.xLo, oo.xHi)

            Asts    = _N.where(oo.dat[t0:t1, s01col] == 1)[0]   #  based at 0

            #######   containers for GIBBS samples iterations
            smp_sp_prms = _N.zeros((5, ITERS, M_use))  
            oo.smp_sp_prms = smp_sp_prms

            #####  MODES  - find from the sampling
            oo.sp_prmPstMd = _N.zeros(3*M_use)   # mode params

            l0       = _N.array([11.,])
            q2x       = _N.array([0.04])
            q2y       = _N.array([0.04])
            fx        = _N.empty(M_use)
            fy        = _N.empty(M_use)

            l0_exp_px = _N.empty(M_use)
            _l0_a    = _N.array([priors._l0_a[0]])
            _l0_B    = _N.array([priors._l0_B[0]])
            _f_u    = _N.array([priors._f_u[0]])
            _f_q2    = _N.array([priors._f_q2[0]])
            _q2_a    = _N.array([priors._q2_a[0]])
            _q2_B    = _N.array([priors._q2_B[0]])

            #fr = f.reshape((M_use, 1))

            ######  the hyperparameters for f, q2, u, Sg, l0 during Gibbs
            #  f_u_, f_q2_, q2_a_, q2_B_, u_u_, u_Sg_, Sg_nu, Sg_PSI_, l0_a_, l0_B_
            NSexp   = t1-t0    #  length of position data  #  # of no spike positions to sum

            nSpks    = len(Asts)
            v_sts = _N.empty(len(Asts), dtype=_N.int)
            cls_str_ind = _N.zeros(M_use+1, dtype=_N.int)
            #cls_len      = _N.zeros(M_use, dtype=_N.int)
            clstsz = _N.zeros(M_use, dtype=_N.int)

            gz   = _N.ones((ITERS, nSpks, M_use), dtype=_N.uint8)
            cls_str_ind[0] = 0
            cls_str_ind[1] = nSpks
            v_sts = Asts
            clstsz[0] = nSpks

            _cdfs.dSilenceX = (NSexp/float(oo.Nupx))*(oo.xHi-oo.xLo)

            ################################  GIBBS ITERS ITERS ITERS
            _Dl0_a = _N.empty(M_use);            _Dl0_B = _N.empty(M_use)
            _Dq2_a = _N.empty(M_use);            _Dq2_B = _N.empty(M_use)

            #iiq2 = 1./q2
            #iiq2r= iiq2.reshape((M_use, 1))

            BLK        = 1000
            iterBLOCKs = ITERS/BLK

            clstsz_rr  = clstsz.reshape(M_use, 1, 1)
            clstsz_r  = clstsz.reshape(M_use, 1)

            ###########  BEGIN GIBBS SAMPLING ##############################
            #for itr in xrange(ITERS):
            for itrB in xrange(iterBLOCKs):
                for itr in xrange(itrB*BLK, (itrB+1)*BLK):
                    if (itr % 500) == 0:    
                        print "-------itr  %(i)d" % {"i" : itr}

                    ###############
                    ###############  Conditional fx
                    ###############
                    q2pr = _f_q2

                    m_rnds = _N.random.rand(M_use)

                    wgt_y  = _N.exp(-0.5*(yt0t1-fy)*(yt0t1-fy)/q2y)
                    _cdfs.smp_f_2d(M_use, clstsz, cls_str_ind, v_sts, xt0t1, t0, fx, q2x, l0, wgt_y, _f_u, q2pr, m_rnds)

                    #f   = _N.array([6.33])
                    smp_sp_prms[oo.ky_p_fx, itr] = fx
                    wgt_x  = _N.exp(-0.5*(xt0t1-fx)*(xt0t1-fx)/q2x)
                    _cdfs.smp_f_2d(M_use, clstsz, cls_str_ind, v_sts, yt0t1, t0, fy, q2y, l0, wgt_x, _f_u, q2pr, m_rnds)
                    smp_sp_prms[oo.ky_p_fy, itr] = fy

                    ##############
                    ##############  SAMPLE SPATIAL VARIANCE
                    ##############
                    #  B' / (a' - 1) = MODE   #keep mode the same after discount

                    # m_rnds = _N.random.rand(M_use)
                    # #  B' = MODE * (a' - 1)
                    _Dq2_a = _q2_a
                    _Dq2_B = _q2_B

                    wgt_y  = _N.exp(-0.5*(yt0t1-fy)*(yt0t1-fy)/q2y)
                    _cdfs.smp_q2_2d(M_use, clstsz, cls_str_ind, v_sts, xt0t1, t0, fx, q2x, l0, wgt_y, _Dq2_a, _Dq2_B, m_rnds, epc)
                    smp_sp_prms[oo.ky_p_q2x, itr]   = q2x
                    # q2x = _N.array([1.])
                    # q2y = _N.array([0.05])
                    # smp_sp_prms[oo.ky_p_q2x, itr] = q2x
                    # smp_sp_prms[oo.ky_p_q2y, itr] = q2y

                    wgt_x  = _N.exp(-0.5*(xt0t1-fx)*(xt0t1-fx)/q2x)
                    _cdfs.smp_q2_2d(M_use, clstsz, cls_str_ind, v_sts, yt0t1, t0, fy, q2y, l0, wgt_x, _Dq2_a, _Dq2_B, m_rnds, epc)
                    smp_sp_prms[oo.ky_p_q2y, itr]   = q2y


                    ###############
                    ###############  CONDITIONAL l0
                    ###############
                    # _ss.gamma.rvs.  uses k, theta  k is 1/B (B is our thing)
                    _cdfs.l0_spatial(xt0t1, yt0t1, Nt0t1, M_use, oo.dt, fx, q2x, fy, q2y, l0_exp_px)

                    BL  = l0_exp_px    #  dim M

                    _Dl0_a = _l0_a
                    _Dl0_B = _l0_B

                    aL  = clstsz + 1
                    l0_a_ = aL + _Dl0_a
                    l0_B_ = BL + _Dl0_B

                    l0 = _ss.gamma.rvs(l0_a_, scale=(1/l0_B_), size=M_use)  #  check

                    #l0  = _N.array([800.])
                    smp_sp_prms[oo.ky_p_l0, itr] = l0
