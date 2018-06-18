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
import stochasticAssignment2d as _sA
#import cdf_smp_tbl as _cdfs
import cdf_smp_2d as _cdfs2d
import ig_from_cdf_pkg as _ifcp
import fastnum as _fm
import clrs 
import compress_gz as c_gz
import conv_gau as _gct

#import conv_px_tbl as _cpt

class MarkAndRF:
    ky_p_l0 = 0;    ky_p_fx  = 1;    ky_p_fy  = 2;    ky_p_q2x = 3
    ky_p_q2y = 4; 
    ky_h_l0_a = 0;  ky_h_l0_B=1;
    ky_h_f_u  = 2;  ky_h_f_q2=3;
    ky_h_q2_a = 4;  ky_h_q2_B=5;

    ky_h_l0_a = 0;  ky_h_l0_B=1;
    ky_h_f_u  = 2;  ky_h_f_q2=3;
    ky_h_q2_a = 4;  ky_h_q2_B=5;

    ky_p_u = 0;       ky_p_Sg = 1;
    ky_h_u_u = 0;     ky_h_u_Sg=1;
    ky_h_Sg_nu = 2;   ky_h_Sg_PSI=3;
    #  posIndx 0, spk01Indx  mksIndx

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
    use_conv_tabl = True

    outdir   = None
    polyFit  = True

    rotate   = False

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

    #q2_lvls    = _N.array([0.02**2, 0.05**2, 0.1**2, 0.2**2, 0.5**2, 1**2, 6**2, 100000**2])
    #Nupx_lvls  = _N.array([1000, 600, 200, 100, 60, 20, 12, 8])

    priors     = None

    def setup_spatial_sum_params(self, q2x=None, fx=None, n_q2_lvls=12, q_mlt_steps=2, q_min=0.01, bins_per_sd=5): 
        """
        *q_mlt_steps=2  means spatial bins [0.01, 0.02, 0.04, 0.08]...
        *if my q2 is bigger than max level, i'll just end up using more bins than neccessary. no prob
        *bins_per_sd=5   heuristically, this is a good setting.
        *q2x, fx     grid 
        """
        oo = self
        oo.q2_lvls = _N.empty(n_q2_lvls)
        oo.Nupx_lvls = _N.empty(n_q2_lvls, dtype=_N.int)
        oo.q2_lvls[0] = q_min**2
        q2_mlt_steps = q_mlt_steps**2
        oo.Nupx_lvls[0] = int(_N.ceil(((oo.xHi-oo.xLo)/_N.sqrt(oo.q2_lvls[0]))*bins_per_sd))

        for i in xrange(1, n_q2_lvls):
            oo.q2_lvls[i] = q2_mlt_steps*oo.q2_lvls[i-1]
            oo.Nupx_lvls[i] = int(_N.ceil(((oo.xHi-oo.xLo)/_N.sqrt(oo.q2_lvls[i]))*bins_per_sd))
            #  Nupx_lvls should not be too small.  how finely 
            oo.Nupx_lvls[i] = 40 if oo.Nupx_lvls[i] < 40 else oo.Nupx_lvls[i]

        oo.binszs     = float(oo.xHi-oo.xLo)/oo.Nupx_lvls
        oo.q2_L, oo.q2_H = q2x
        oo.f_L,  oo.f_H  = fx
    
    def __init__(self, outdir, fn, intvfn, xLo=0, xHi=3, yLo=0, yHi=3, seed=1041, adapt=True, t_hlf_l0_mins=None, t_hlf_q2_mins=None, oneCluster=False, rotate=False, ):
        oo     = self
        oo.oneCluster = oneCluster
        oo.adapt = adapt
        _N.random.seed(seed)

        ######################################  DATA input, define intervals
        # bFN = fn[0:-4]
        oo.outdir = outdir


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

        # if not os.access(bFN, os.F_OK):
        #     os.mkdir(bFN)

        _dat    = _N.loadtxt("%s.dat" % datFN(fn, create=False))
        oo.rotate = rotate
        if not rotate:
            oo.dat  = _dat
        else:
            K = _dat.shape[1] - 3  #  2 spatial dim + 1 spk dim
            oo.dat  = _N.empty((_dat.shape[0], 3+2*K))
            oo.dat[:, 0:3+K] = _dat
    
            sts = _N.where(oo.dat[:, 1] == 1)[0]
            for n in sts:
                oo.dat[n, 3+K:] = _N.dot(rot3, _N.dot(rot4, oo.dat[n, 3:3+K]))

        #oo.datprms= _N.loadtxt("%s_prms.dat" % datFN(fn, create=False))

        intvs     = _N.loadtxt("%s.dat" % datFN(intvfn, create=False))

        oo.intvs  = _N.array(intvs*oo.dat.shape[0], dtype=_N.int)
        oo.epochs    = oo.intvs.shape[0] - 1
        
        NT     = oo.dat.shape[0]
        oo.xLo = xLo   #  this limit used for spatial path sum
        oo.xHi = xHi   #  this limit used for spatial path sum
        oo.yLo = yLo   #  this limit used for spatial path sum
        oo.yHi = yHi   #  this limit used for spatial path sum

    def setup_spatial_sum_params(self, q2x=None, fx=None, n_q2_lvls=12, q_mlt_steps=2, q_min=0.01, bins_per_sd=5): 
        """
        *q_mlt_steps=2  means spatial bins [0.01, 0.02, 0.04, 0.08]...
        *if my q2 is bigger than max level, i'll just end up using more bins than neccessary. no prob
        *bins_per_sd=5   heuristically, this is a good setting.
        *q2x, fx     grid 
        """
        oo = self
        oo.q2_lvls = _N.empty(n_q2_lvls)
        oo.Nupx_lvls = _N.empty(n_q2_lvls, dtype=_N.int)
        oo.Nupy_lvls = _N.empty(n_q2_lvls, dtype=_N.int)
        oo.q2_lvls[0] = q_min**2
        q2_mlt_steps = q_mlt_steps**2
        oo.Nupx_lvls[0] = int(_N.ceil(((oo.xHi-oo.xLo)/_N.sqrt(oo.q2_lvls[0]))*bins_per_sd))
        oo.Nupy_lvls[0] = int(_N.ceil(((oo.yHi-oo.yLo)/_N.sqrt(oo.q2_lvls[0]))*bins_per_sd))

        for i in xrange(1, n_q2_lvls):
            oo.q2_lvls[i] = q2_mlt_steps*oo.q2_lvls[i-1]
            oo.Nupx_lvls[i] = int(_N.ceil(((oo.xHi-oo.xLo)/_N.sqrt(oo.q2_lvls[i]))*bins_per_sd))
            oo.Nupy_lvls[i] = int(_N.ceil(((oo.yHi-oo.yLo)/_N.sqrt(oo.q2_lvls[i]))*bins_per_sd))
            #  Nupx_lvls should not be too small.  how finely 
            oo.Nupx_lvls[i] = 40 if oo.Nupx_lvls[i] < 40 else oo.Nupx_lvls[i]
            oo.Nupy_lvls[i] = 40 if oo.Nupy_lvls[i] < 40 else oo.Nupy_lvls[i]

        oo.binszs_x     = float(oo.xHi-oo.xLo)/oo.Nupx_lvls
        oo.binszs_y     = float(oo.yHi-oo.yLo)/oo.Nupy_lvls
        oo.q2_L, oo.q2_H = q2x   #  grid lims for calc of conditional posterior
        oo.f_L,  oo.f_H  = fx

    def gibbs(self, ITERS, K, priors, ep1=0, ep2=None, saveSamps=True, saveOcc=True, doSepHash=True, nz_pth=0., smth_pth_ker=0, f_STEPS=13, q2_STEPS=13, f_SMALL=10, q2_SMALL=10, f_cldz=10, q2_cldz=10, minSmps=20, diag_cov=False, earliest=20000, cmprs=1):
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

        freeClstr = None
        # if smth_pth_ker > 0:
        #     gk     = gauKer(smth_pth_ker) # 0.1s  smoothing of motion
        #     gk     /= _N.sum(gk)
        #     xf     = _N.convolve(oo.dat[:, 0], gk, mode="same")
        #     oo.dat[:, 0] = xf + nz_pth*_N.random.randn(len(oo.dat[:, 0]))
        # else:
        #     oo.dat[:, 0] += nz_pth*_N.random.randn(len(oo.dat[:, 0]))
        xy      = oo.dat[:, 0:2]

        if oo.rotate:
            init_mks    = _N.array(oo.dat[:, 3:3+K])  #  init using non-rot
            fit_mks    = _N.array(oo.dat[:, 3+K:])  #  fit to rotated
        else:
            init_mks    = _N.array(oo.dat[:, 3:3+K])  #  init using non-rot
            fit_mks    = init_mks

        f_q2_rate = (oo.diffusePerMin**2)/60000.  #  unit of minutes  
        
        ######################################  PRECOMPUTED

        _ifcp.init()
        tau_l0 = oo.t_hlf_l0/_N.log(2)
        tau_q2 = oo.t_hlf_q2/_N.log(2)

        _cdfs2d.init(oo.dt, oo.f_L, oo.f_H, oo.q2_L, oo.q2_H, f_STEPS, q2_STEPS, f_SMALL, q2_SMALL, f_cldz, q2_cldz, minSmps)

        _cdfs2d.init_occ_resolutions(oo.xLo, oo.xHi, oo.yLo, oo.yHi, oo.q2_lvls, oo.Nupx_lvls, oo.Nupy_lvls)
        print "done occ_resolu"

        M_max   = 50   #  100 clusters, max
        M_use    = 0     #  number of non-free + 5 free clusters

        for epc in xrange(ep1, ep2):
            print "^^^^^^^^^^^^^^^^^^^^^^^^    epoch %d" % epc

            t0 = oo.intvs[epc]
            t1 = oo.intvs[epc+1]

            if epc > 0:
                tm1= oo.intvs[epc-1]
                #  0 10 30     20 - 5  = 15    0.5*((10+30) - (10+0)) = 15
                DT = t0-tm1

            #  _N.sum(px)*(xbns[1]-xbns[0]) = 1

            # ##  smooth the positions
            # smthd_pos = x[t0:t1] + oo.Bx*_N.random.randn(t1-t0)
            # ltL = _N.where(smthd_pos < oo.xLo)[0]
            # smthd_pos[ltL] += 2*(oo.xLo - smthd_pos[ltL])
            # gtR = _N.where(smthd_pos > oo.xHi)[0]
            # smthd_pos[gtR] += 2*(oo.xHi - smthd_pos[gtR])


            xt0t1 = _N.array(xy[t0:t1, 0])#smthd_pos
            yt0t1 = _N.array(xy[t0:t1, 1])#smthd_pos

            Asts    = _N.where(oo.dat[t0:t1, 2] == 1)[0]   #  based at 0
            _cdfs2d.change_occ_hist(xt0t1, yt0t1, oo.xLo, oo.xHi, oo.yLo, oo.yHi)


            if epc == ep1:   ###  initialize
                print "len(Asts)  %d" % len(Asts)
                print  "t0   %(0)d     t1  %(1)d" % {"0" : t0, "1" : t1}
                
                labS, labH, flatlabels, M_use, hashthresh, nHSclusters = gAMxMu.initClusters(oo, M_max, K, xy, init_mks, t0, t1, Asts, doSepHash=doSepHash, xLo=oo.xLo, xHi=oo.xHi, oneCluster=oo.oneCluster, spcdim=2)

                m1stSignalClstr = 0 if oo.oneCluster else nHSclusters[0]

                #  hyperparams of posterior. Not needed for all params
                u_u_  = _N.empty((M_use, K))
                u_Sg_ = _N.empty((M_use, K, K))
                Sg_nu_ = _N.empty(M_use)
                Sg_PSI_ = _N.zeros((M_use, K, K))
                Sg_PSI_diag = _N.zeros((M_use, K))
                #Sg_     = _N.empty((M_max, K, K))   # sampled value
                #uptriinds = _N.triu_indices_from(Sg[0],1)

                #######   containers for GIBBS samples iterations
                smp_sp_prms = _N.zeros((5, ITERS, M_use))  
                smp_mk_prms = [_N.zeros((K, ITERS, M_use)), 
                               _N.zeros((K, K, ITERS, M_use))]
                #  need mark hyp params cuz I calculate prior hyp from sampled hyps, unlike where I fit a distribution to sampled parameters and find best hyps from there.  Is there a better way?
                smp_mk_hyps = [_N.zeros((K, ITERS, M_use)),   
                               _N.zeros((K, K, ITERS, M_use)),
                               _N.zeros((ITERS, M_use)), 
                               _N.zeros((K, K, ITERS, M_use))]
                oo.smp_sp_prms = smp_sp_prms
                oo.smp_mk_prms = smp_mk_prms
                oo.smp_mk_hyps = smp_mk_hyps

                #####  MODES  - find from the sampling
                oo.sp_prmPstMd = _N.zeros(5*M_use)   # mode params
                oo.mk_prmPstMd = [_N.zeros((M_use, K)),
                                  _N.zeros((M_use, K, K))]
                      # mode of params


                #  list of freeClstrs
                freeClstr = _N.empty(M_max, dtype=_N.bool)   #  Actual cluster
                freeClstr[:] = True#False

                
                l0_M, fx_M, fy_M, q2x_M, q2y_M, u_M, Sg_M = gAMxMu.declare_params(M_max, K, spcdim=2)   #  nzclstr not inited  # sized to include noise cluster if needed
                l0_exp_hist_M = _N.empty(M_max)

                _l0_a_M, _l0_B_M, _fx_u_M, _fy_u_M, _fx_q2_M, _fy_q2_M, _q2x_a_M, _q2y_a_M, _q2x_B_M, _q2y_B_M, _u_u_M, \
                    _u_Sg_M, _Sg_nu_M, _Sg_PSI_M = gAMxMu.declare_prior_hyp_params(M_max, nHSclusters, K, xy, fit_mks, Asts, t0, priors, labS, labH, spcdim=2)

                l0, fx, fy, q2x, q2y, u, Sg        = gAMxMu.copy_slice_params_2d(M_use, l0_M, fx_M, fy_M, q2x_M, q2y_M, u_M, Sg_M)
                _l0_a, _l0_B, _fx_u, _fy_u, _fx_q2, _fy_q2, _q2x_a, _q2y_a, _q2x_B, _q2y_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI        = gAMxMu.copy_slice_hyp_params_2d(M_use, _l0_a_M, _l0_B_M, _fx_u_M, _fy_u_M, _fx_q2_M, _fy_q2_M, _q2x_a_M, _q2y_a_M, _q2x_B_M, _q2y_B_M, _u_u_M, _u_Sg_M, _Sg_nu_M, _Sg_PSI_M)

                l0_exp_hist = _N.array(l0_exp_hist_M[0:M_use], copy=True)

                gAMxMu.init_params_hyps_2d(oo, M_use, K, l0, fx, fy, q2x, q2y, u, Sg, _l0_a, _l0_B, _fx_u, _fy_u, _fx_q2, _fy_q2, _q2x_a, _q2y_a, _q2x_B, _q2y_B, _u_u, _u_Sg, _Sg_nu, \
                                        _Sg_PSI, Asts, t0, xy, fit_mks, flatlabels, nHSclusters)

                U   = _N.empty(M_use)

                ######  the hyperparameters for f, q2, u, Sg, l0 during Gibbs
                #  f_u_, f_q2_, q2_a_, q2_B_, u_u_, u_Sg_, Sg_nu, Sg_PSI_, l0_a_, l0_B_
            else:
                #  later epochs

                freeInds = _N.where(freeClstr[0:M_use] == True)[0]
                n_fClstrs = len(freeInds)

                print "!!!!!!  %d" % n_fClstrs
                print "bef M_use %d" % M_use
                #  
                if n_fClstrs < oo.NExtrClstr:  #  
                    old_M = M_use
                    M_use  = M_use + (oo.NExtrClstr - n_fClstrs)
                    M_use = M_use if M_use < M_max else M_max
                    #new_M = M_use
                elif n_fClstrs > oo.NExtrClstr:
                    old_M = M_use
                    M_use  = M_use + (oo.NExtrClstr - n_fClstrs)
                    #new_M = M_use

                print "aft M_use %d" % M_use

                l0, f, q2, u, Sg        = gAMxMu.copy_slice_params(M_use, l0_M, f_M, q2_M, u_M, Sg_M)
                _l0_a, _l0_B, _f_u, _f_q2, _q2_a, _q2_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI        = gAMxMu.copy_slice_hyp_params(M_use, _l0_a_M, _l0_B_M, _f_u_M, _f_q2_M, _q2_a_M, _q2_B_M, _u_u_M, _u_Sg_M, _Sg_nu_M, _Sg_PSI_M)

                l0_exp_hist = _N.array(l0_exp_hist_M[0:M_use], copy=True)

                #  hyperparams of posterior. Not needed for all params
                u_u_  = _N.empty((M_use, K))
                u_Sg_ = _N.empty((M_use, K, K))
                Sg_nu_ = _N.empty(M_use)
                Sg_PSI_ = _N.empty((M_use, K, K))

                smp_sp_prms = _N.zeros((3, ITERS, M_use))  
                smp_mk_prms = [_N.zeros((K, ITERS, M_use)), 
                               _N.zeros((K, K, ITERS, M_use))]
                smp_mk_hyps = [_N.zeros((K, ITERS, M_use)),   
                               _N.zeros((K, K, ITERS, M_use)),
                               _N.zeros((ITERS, M_use)), 
                               _N.zeros((K, K, ITERS, M_use))]

                oo.smp_sp_prms = smp_sp_prms
                oo.smp_mk_prms = smp_mk_prms
                oo.smp_mk_hyps = smp_mk_hyps

                #####  MODES  - find from the sampling
                oo.sp_prmPstMd = _N.zeros(3*M_use)   # mode params
                oo.mk_prmPstMd = [_N.zeros((M_use, K)),
                                  _N.zeros((M_use, K, K))]

            NSexp   = t1-t0    #  length of position data  #  # of no spike positions to sum

            xt0t1 = _N.array(xy[t0:t1, 0])#smthd_pos
            yt0t1 = _N.array(xy[t0:t1, 1])#smthd_pos
            cmp_xt0t1 = xt0t1#smthd_pos
            cmp_yt0t1 = yt0t1#smthd_pos

            Nt0t1 = t1-t0



            print "printing priors._fy_u"
            print priors._fy_u
            l0_exp_px = _N.empty(M_use)

            nSpks    = len(Asts)
            v_sts = _N.empty(len(Asts), dtype=_N.int)
            cls_str_ind = _N.zeros(M_use+1, dtype=_N.int)
            #cls_len      = _N.zeros(M_use, dtype=_N.int)
            clstsz = _N.zeros(M_use, dtype=_N.int)

            if M_use > 1:
                gz   = _N.zeros((ITERS, nSpks, M_use), dtype=_N.uint8)
            else:
                gz   = _N.ones((ITERS, nSpks, M_use), dtype=_N.uint8)
                cls_str_ind[0] = 0
                cls_str_ind[1] = nSpks
                #cls_len[0] = nSpks
                v_sts = Asts
                clstsz[0] = nSpks
            oo.gz=gz

            xAS  = _N.array(xy[Asts + t0, 0])   #  position @ spikes.  creates new copy
            yAS  = _N.array(xy[Asts + t0, 1])   #  position @ spikes.  creates new copy
            mAS  = fit_mks[Asts + t0]   #  position @ spikes

            econt = _N.empty((M_use, nSpks))
            rat   = _N.zeros((M_use+1, nSpks))

            qdrMKS = _N.empty((M_use, nSpks))
            ################################  GIBBS ITERS ITERS ITERS

            _iu_Sg = _N.array(_u_Sg)
            for m in xrange(M_use):
                _iu_Sg[m] = _N.linalg.inv(_u_Sg[m])

            ttA = _tm.time()


            # l0_a_is0    = _N.where(_l0_a == 0)[0]
            # l0_a_Init   = _N.where(_l0_a >  0)[0]
            # b_l0_a_is0  = len(l0_a_is0) > 0
            # q2_a_is_m1  = _N.where(_q2_a == -1)[0]
            # q2_a_Init   = _N.where(_q2_a > 0)[0]
            # b_q2_a_is_m1= len(q2_a_is_m1) > 0

            _Dl0_a = _N.empty(M_use);            _Dl0_B = _N.empty(M_use)
            _Dq2_a = _N.empty(M_use);            _Dq2_B = _N.empty(M_use)

            iiq2x = 1./q2x
            iiq2y = 1./q2y

            mcs = _N.empty((M_use, K))   # cluster sample means
            mcsT = _N.empty((M_use, K))   # cluster sample means
            outs1 = _N.empty((M_use, K))
            outs2 = _N.empty((M_use, K))

            BLK        = 200
            iterBLOCKs = ITERS/BLK

            q2x = _N.array([0.02]*M_use)
            q2y = _N.array([0.02]*M_use)

            ###########  BEGIN GIBBS SAMPLING ##############################
            #for itr in xrange(ITERS):
            for itrB in xrange(iterBLOCKs):
                print "printing priors._fy_u"
                print priors._fy_u

                for itr in xrange(itrB*BLK, (itrB+1)*BLK):
                    #ttsw1 = _tm.time()
                    iSg = _N.linalg.inv(Sg)
                    #ttsw2 = _tm.time()
                    if (itr % 100) == 0:    
                        print "-------itr  %(i)d" % {"i" : itr}

                    if M_use > 1:
                        _sA.stochasticAssignment2d(oo, epc, itr, M_use, K, l0, fx, fy, q2x, q2y, u, Sg, iSg, _fx_u, _fy_u, _u_u, _fx_q2, _fy_q2, _u_Sg, Asts, t0, mAS, xAS, yAS, rat, econt, gz, qdrMKS, freeClstr, hashthresh, m1stSignalClstr, ((epc > 0) and (itr == 0)), diag_cov, clstsz)
                        _fm.cluster_bounds2(clstsz, Asts, cls_str_ind, v_sts, gz[itr], t0, M_use, nSpks)    # _fm.cluser_bounds provides no improvement
                    #ttsw3 = _tm.time()

                    #  Asts are all spike times in order they appear
                    #  v_sts[cls_str_ind[mmm]:cls_str_ind[mmm+1]] - spk times for mmm-th cluster
                    clstsz_rr  = clstsz.reshape(M_use, 1, 1)
                    clstsz_r  = clstsz.reshape(M_use, 1)

                    #ttsw4 = _tm.time()

                    ###############
                    ###############     u
                    ###############
                    _N.copyto(u_Sg_, _N.linalg.inv(_iu_Sg + clstsz_rr*iSg))

                    _fm.find_mcs(clstsz, v_sts, cls_str_ind, fit_mks, mcs, M_use, K)
                    _fm.multiple_mat_dot_v(_iu_Sg, _u_u, outs1, M_use, K)
                    _fm.multiple_mat_dot_v(iSg, mcs, outs2, M_use, K)
                    _fm.multiple_mat_dot_v(u_Sg_, outs1 + clstsz_r*outs2, u_u_, M_use, K)

                    #ttsw5 = _tm.time()
                    ucmvnrms= _N.random.randn(M_use, K)

                    try:
                        C       = _N.linalg.cholesky(u_Sg_)
                    except _N.linalg.linalg.LinAlgError:
                        dmp = open("cholesky.dmp", "wb")
                        pickle.dump([u_Sg_, _iu_Sg, clstsz, iSg, _u_Sg, _u_u], dmp, -1)
                        dmp.close()

                        raise
                    u       = _N.einsum("njk,nk->nj", C, ucmvnrms) + u_u_

                    smp_mk_prms[oo.ky_p_u][:, itr] = u.T  # dim of u wrong
                    smp_mk_hyps[oo.ky_h_u_u][:, itr] = u_u_.T
                    smp_mk_hyps[oo.ky_h_u_Sg][:, :, itr] = u_Sg_.T

                    #ttsw6 = _tm.time()
                    ###############
                    ###############  Conditional f
                    ###############
                    if (epc > 0) and oo.adapt:
                        q2xpr = _fx_q2 + f_q2_rate * DT
                        q2ypr = _fy_q2 + f_q2_rate * DT
                    else:
                        q2xpr = _fx_q2
                        q2ypr = _fy_q2



                    m_rnds_x = _N.random.rand(M_use)
                    m_rnds_y = _N.random.rand(M_use)

                    _cdfs2d.smp_f(0, M_use, clstsz, cls_str_ind, v_sts, xt0t1, yt0t1, t0, fx, fy, q2x, q2y, l0, _fx_u, _fx_q2, _fy_u, _fy_q2, m_rnds_x)
                    smp_sp_prms[oo.ky_p_fx, itr] = fx

                    _cdfs2d.smp_f(1, M_use, clstsz, cls_str_ind, v_sts, xt0t1, yt0t1, t0, fx, fy, q2x, q2y, l0, _fx_u, _fx_q2, _fy_u, _fy_q2, m_rnds_y)
                    smp_sp_prms[oo.ky_p_fy, itr] = fy

                    #ttsw7 = _tm.time()
                    ##############
                    ##############  VARIANCE, COVARIANCE
                    ##############

                    # tt6a = 0
                    # tt6b = 0
                    # tt6c = 0

                    Sg_nu_ = _Sg_nu + clstsz                
                    if diag_cov:
                        _fm.Sg_PSI(cls_str_ind, clstsz, v_sts, fit_mks, _Sg_PSI, Sg_PSI_, u, M_use, K)
                        # print "*************************"
                        # print Sg_PSI_
                        # print "-------------------------"
                        for m in xrange(M_use):
                            Sg[m] = 0

                            for ik in xrange(K):
                                Sg[m, ik, ik] = _ifcp.sampIG_single(Sg_nu_[m]*0.5, Sg_PSI_[m, ik, ik])
                        #print "^^^^^^^^^^^^^^^"
                    else:
                        for m in xrange(M_use):
                            #infor = _tm.time()
                            #if clstsz[m] > K:
                            ##  dof of posterior distribution of cluster covariance
                            #Sg_nu_[m] = _Sg_nu[m, 0] + clstsz[m]
                            ##  dof of posterior distribution of cluster covariance
                            ur = u[m].reshape((1, K))
                            #tt6a += (_tm.time()-infor)
                            #infor = _tm.time()
                            clstx    = fit_mks[v_sts[cls_str_ind[m]:cls_str_ind[m+1]]]
                            #  dot((clstx-ur).T, (clstx-ur))==ZERO(K) when clstsz ==0
                            Sg_PSI_[m] = _Sg_PSI[m] + _N.dot((clstx - ur).T, (clstx-ur))
                            #tt6b += (_tm.time()-infor)
                            #infor = _tm.time()
                            # else:
                            #     Sg_nu_ = _Sg_nu[m, 0] 
                            #     ##  dof of posterior distribution of cluster covariance
                            #     ur = u[m].reshape((1, K))
                            #     Sg_PSI_ = _Sg_PSI[m]
                            Sg[m] = _ss.invwishart.rvs(df=Sg_nu_[m], scale=Sg_PSI_[m])
                            #tt6c += (_tm.time()-infor)

                    smp_mk_prms[oo.ky_p_Sg][:, :, itr] = Sg.T
                    smp_mk_hyps[oo.ky_h_Sg_nu][itr] = Sg_nu_
                    smp_mk_hyps[oo.ky_h_Sg_PSI][:, :, itr] = Sg_PSI_.T

                    #ttsw8 = _tm.time()
                    ##############
                    ##############  SAMPLE SPATIAL VARIANCE
                    ##############
                    #  B' / (a' - 1) = MODE   #keep mode the same after discount

                    mx_rnds = _N.random.rand(M_use)
                    my_rnds = _N.random.rand(M_use)
                    #  B' = MODE * (a' - 1)
                    if (epc > 0) and oo.adapt:
                        _mdx_nd= _q2x_B / (_q2x_a + 1)
                        _Dq2x_a = _q2x_a * _N.exp(-DT/tau_q2)
                        _Dq2x_B = _mdx_nd * (_Dq2_a + 1)
                        _mdy_nd= _q2y_B / (_q2y_a + 1)
                        _Dq2y_a = _q2y_a * _N.exp(-DT/tau_q2)
                        _Dq2y_B = _mdy_nd * (_Dq2_a + 1)
                    else:
                        _Dq2x_a = _q2x_a
                        _Dq2x_B = _q2x_B
                        _Dq2y_a = _q2y_a
                        _Dq2y_B = _q2y_B

                    #ttsw9 = _tm.time()



                    m_rnds_x = _N.random.rand(M_use)
                    m_rnds_y = _N.random.rand(M_use)

                    #  B' = MODE * (a' - 1)
                    _cdfs2d.smp_q2(0, M_use, clstsz, cls_str_ind, v_sts, xt0t1, yt0t1, t0, fx, fy, q2x, q2y, l0, _Dq2_a, _Dq2_B, m_rnds_x)
                    #print "-------   sampled q2x is %.4e" % q2x[0]
                    smp_sp_prms[oo.ky_p_q2x, itr]   = q2x

                    _cdfs2d.smp_q2(1, M_use, clstsz, cls_str_ind, v_sts, xt0t1, yt0t1, t0, fx, fy, q2x, q2y, l0, _Dq2_a, _Dq2_B, m_rnds_y)
                    smp_sp_prms[oo.ky_p_q2y, itr]   = q2y

                    #ttsw10 = _tm.time()

                    ###############
                    ###############  CONDITIONAL l0
                    ###############
                    #  _ss.gamma.rvs.  uses k, theta  k is 1/B (B is our thing)

                    _cdfs2d.l0_spatial(M_use, oo.dt, fx, fy, q2x, q2y, l0_exp_px)
                    
                    BL  = l0_exp_px    #  dim M

                    if (epc > 0) and oo.adapt:
                        _mn_nd= _l0_a / _l0_B
                        #  variance is a/b^2
                        #  a/2 / B/2    variance is a/2 / B^2/4 = 2a^2 / B^2  
                        #  variance increases by 2

                        _Dl0_a = _l0_a * _N.exp(-DT/tau_l0)
                        _Dl0_B = _Dl0_a / _mn_nd
                    else:
                        _Dl0_a = _l0_a
                        _Dl0_B = _l0_B

                    aL  = clstsz + 1
                    l0_a_ = aL + _Dl0_a
                    l0_B_ = BL + _Dl0_B

                    try:   #  if there is no prior, if a cluster 
                        l0 = _ss.gamma.rvs(l0_a_, scale=(1/l0_B_), size=M_use)  #  check
                        #l0 = _N.array([900.])
                    except ValueError:
                        print "problem with l0    %d" % itr
                        print l0_exp_hist
                        print l0_a_
                        print l0_B_
                        raise

                    smp_sp_prms[oo.ky_p_l0, itr] = l0

                    #ttsw11 = _tm.time()
                    # print "#timing start"
                    # print "nt+= 1"
                    # print "t2t1+=%.4e" % (#ttsw2-#ttsw1)
                    # print "t3t2+=%.4e" % (#ttsw3-#ttsw2)
                    # print "t4t3+=%.4e" % (#ttsw4-#ttsw3)
                    # print "t5t4+=%.4e" % (#ttsw5-#ttsw4)
                    # print "t6t5+=%.4e" % (#ttsw6-#ttsw5)
                    # print "t7t6+=%.4e" % (#ttsw7-#ttsw6)  # slow
                    # print "t8t7+=%.4e" % (#ttsw8-#ttsw7)
                    # print "t9t8+=%.4e" % (#ttsw9-#ttsw8)
                    # print "t10t9+=%.4e" % (#ttsw10-#ttsw9)
                    # print "t11t10+=%.4e" % (#ttsw11-#ttsw10)
                    # print "#timing end  %.5f" % (#ttsw10-#ttsw1)

                #frms = _pU.find_good_clstrs_and_stationary_from(M_use, smp_sp_prms[:, 0:itr+1])
                #if (itr >= oo.earliest) and (len(_N.where(frms - 4000 < 0)[0]) == M_use):
                #    break

            ttB = _tm.time()
            print (ttB-ttA)


            print "itr is %d" % itr
            gAMxMu.finish_epoch2_2d(oo, nSpks, epc, itr+1, gz, l0, fx, fy, q2x, q2y, u, Sg, _fx_u, _fy_u, _fx_q2, _fy_q2, _q2x_a, _q2y_a, _q2x_B, _q2y_B, _l0_a, _l0_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI, smp_sp_prms, smp_mk_prms, smp_mk_hyps, freeClstr, M_use, K, priors, m1stSignalClstr)
            #  _l0_a is a copy of a subset of _l0_a_M
            #  we need to copy back the values _l0_a back into _l0_a_M
            gAMxMu.contiguous_inuse_2d(M_use, M_max, K, freeClstr, l0, fx, fy, q2x, q2y, u, Sg, _l0_a, _l0_B, _fx_u, _fy_u, _fx_q2, _fy_q2, _q2x_a, _q2y_a, _q2x_B, _q2y_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI, smp_sp_prms, smp_mk_prms, oo.sp_prmPstMd, oo.mk_prmPstMd, gz, priors)
            gAMxMu.copy_back_params_2d(M_use, l0, fx, fy, q2x, q2y, u, Sg, M_max, l0_M, fx_M, fy_M, q2x_M, q2y_M, u_M, Sg_M)
            gAMxMu.copy_back_hyp_params_2d(M_use, _l0_a, _l0_B, _fx_u, _fy_u, _fx_q2, _fy_q2, _q2x_a, _q2y_a, _q2x_B, _q2y_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI, M_max, _l0_a_M, _l0_B_M, _fx_u_M, _fy_u_M, _fx_q2_M, _fy_q2_M, _q2x_a_M, _q2y_a_M, _q2x_B_M, _q2y_B_M, _u_u_M, _u_Sg_M, _Sg_nu_M, _Sg_PSI_M)
            
            #  MAP of nzclstr
            if saveSamps:
                pcklme["smp_sp_prms"] = smp_sp_prms[:, 0:itr+1]
                pcklme["smp_mk_prms"] = [smp_mk_prms[0][:, 0:itr+1], smp_mk_prms[1][:, :, 0:itr+1]]
            pcklme["sp_prmPstMd"] = oo.sp_prmPstMd
            pcklme["mk_prmPstMd"] = oo.mk_prmPstMd
            pcklme["intvs"]       = oo.intvs
            if saveOcc:
                pcklme["occ"]         = c_gz.gz2cgz(gz[0:itr+1])
                pcklme["freeClstr"]           = freeClstr  #  next time
            pcklme["nz_pth"]         = nz_pth
            pcklme["M"]           = M_use
            pcklme["rotate"]      = oo.rotate

                
            dmp = open(resFN("posteriors_%d.dmp" % epc, dir=oo.outdir), "wb")
            pickle.dump(pcklme, dmp, -1)
            dmp.close()
