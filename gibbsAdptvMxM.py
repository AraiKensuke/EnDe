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
from posteriorUtil import MAPvalues2
from filter import gauKer
import gibbsApprMxMutil as gAMxMu
import stochasticAssignment as _sA
import cdf_smp as _cdfs

import clrs 
import compress_gz as c_gz

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
    NExtrClstr = 5

    #  sizes of arrays
    NposHistBins = 200      #   # points to sample position with  (uniform lam(x)p(x))
    
    intvs = None    #  
    dat   = None

    resetClus = True

    diffPerMin = 1.  #  diffusion per minute
    epochs   = None
    adapt    = False

    outdir   = None
    polyFit  = True

    Nupx      = 200

    #  l0, q2      Sig    f, u
    t_hlf_l0 = int(1000*60*5)   # 10minutes
    t_hlf_q2 = int(1000*60*5)   # 10minutes

    nThrds = 2    #  if use_omp
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
    
    def __init__(self, outdir, fn, intvfn, xLo=0, xHi=3, seed=1041, adapt=True, t_hlf_l0_mins=None, t_hlf_q2_mins=None, oneCluster=False):
        oo     = self
        oo.oneCluster = oneCluster
        oo.adapt = adapt
        _N.random.seed(seed)

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

    def gibbs(self, ITERS, K, priors, ep1=0, ep2=None, savePosterior=True, gtdiffusion=False, doSepHash=True, use_spc=True, nz_pth=0., smth_pth_ker=100, ignoresilence=False, use_omp=False, nThrds=2, f_STEPS=13, q2_STEPS=13, f_SMALL=10, q2_SMALL=10, f_cldz=10, q2_cldz=10, minSmps=20):
        """
        gtdiffusion:  use ground truth center of place field in calculating variance of center.  Meaning of diffPerMin different
        """
        print "gibbs   %.5f" % _N.random.rand()
        oo = self
        oo.nThrds = nThrds
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
        if smth_pth_ker > 0:
            gk     = gauKer(smth_pth_ker) # 0.1s  smoothing of motion
            gk     /= _N.sum(gk)
            xf     = _N.convolve(oo.dat[:, 0], gk, mode="same")
            oo.dat[:, 0] = xf + nz_pth*_N.random.randn(len(oo.dat[:, 0]))
        else:
            oo.dat[:, 0] += nz_pth*_N.random.randn(len(oo.dat[:, 0]))
        x      = oo.dat[:, 0]
        mks    = oo.dat[:, 2:]
        if nz_pth > 0:
            _N.savetxt(resFN("nzyx.txt", dir=oo.outdir), x, fmt="%.4f")

        f_q2_rate = (oo.diffusePerMin**2)/60000.  #  unit of minutes  
        
        ######################################  PRECOMPUTED

        tau_l0 = oo.t_hlf_l0/_N.log(2)
        tau_q2 = oo.t_hlf_q2/_N.log(2)

        _cdfs.init(oo.dt, oo.f_L, oo.f_H, oo.q2_L, oo.q2_H, f_STEPS, q2_STEPS, f_SMALL, q2_SMALL, f_cldz, q2_cldz, minSmps)
        _cdfs.init_occ_resolutions(oo.xLo, oo.xHi, oo.q2_lvls, oo.Nupx_lvls)

        M_max   = 100   #  100 clusters, max
        M_use    = 0     #  number of non-free + 5 free clusters

        for epc in xrange(ep1, ep2):
            print "^^^^^^^^^^^^^^^^^^^^^^^^    epoch %d" % epc

            t0 = oo.intvs[epc]
            t1 = oo.intvs[epc+1]
            if epc > 0:
                tm1= oo.intvs[epc-1]
                #  0 10 30     20 - 5  = 15    0.5*((10+30) - (10+0)) = 15
                DT = t0-tm1

            posbins  = _N.linspace(oo.xLo, oo.xHi, oo.Nupx+1)
            #  _N.sum(px)*(xbns[1]-xbns[0]) = 1

            ##  smooth the positions
            smthd_pos = x[t0:t1] + oo.Bx*_N.random.randn(t1-t0)
            ltL = _N.where(smthd_pos < oo.xLo)[0]
            smthd_pos[ltL] += 2*(oo.xLo - smthd_pos[ltL])
            gtR = _N.where(smthd_pos > oo.xHi)[0]
            smthd_pos[gtR] += 2*(oo.xHi - smthd_pos[gtR])
            
            xt0t1 = smthd_pos

            _cdfs.change_occ_px(xt0t1, oo.xLo, oo.xHi)
            #px, xbns = _N.histogram(xt0t1, bins=posbins, normed=True)
            
            #pxr      = px.reshape((1, oo.Nupx))

            Asts    = _N.where(oo.dat[t0:t1, 1] == 1)[0]   #  based at 0

            if epc == ep1:   ###  initialize
                labS, labH, flatlabels, M_use, hashthresh, nHSclusters = gAMxMu.initClusters(oo, M_max, K, x, mks, t0, t1, Asts, doSepHash=doSepHash, xLo=oo.xLo, xHi=oo.xHi, oneCluster=oo.oneCluster)

                m1stSignalClstr = 0 if oo.oneCluster else nHSclusters[0]

                #  hyperparams of posterior. Not needed for all params
                u_u_  = _N.empty((M_use, K))
                u_Sg_ = _N.empty((M_use, K, K))
                Sg_nu_ = _N.empty(M_use)
                Sg_PSI_ = _N.empty((M_use, K, K))
                #Sg_     = _N.empty((M_max, K, K))   # sampled value
                #uptriinds = _N.triu_indices_from(Sg[0],1)

                #######   containers for GIBBS samples iterations
                smp_sp_prms = _N.zeros((3, ITERS, M_use))  
                smp_mk_prms = [_N.zeros((K, ITERS, M_use)), 
                               _N.zeros((K, K, ITERS, M_use))]
                #  need mark hyp params cuz I calculate prior hyp from sampled hyps, unlike where I fit a distribution to sampled parameters and find best hyps from there.  Is there a better way?
                smp_mk_hyps = [_N.zeros((K, ITERS, M_use)),   
                               _N.zeros((K, K, ITERS, M_use)),
                               _N.zeros((1, ITERS, M_use)), 
                               _N.zeros((K, K, ITERS, M_use))]
                oo.smp_sp_prms = smp_sp_prms
                oo.smp_mk_prms = smp_mk_prms
                oo.smp_mk_hyps = smp_mk_hyps

                #####  MODES  - find from the sampling
                oo.sp_prmPstMd = _N.zeros(3*M_use)   # mode params
                oo.mk_prmPstMd = [_N.zeros((M_use, K)),
                                  _N.zeros((M_use, K, K))]
                      # mode of params


                #  list of freeClstrs
                freeClstr = _N.empty(M_max, dtype=_N.bool)   #  Actual cluster
                freeClstr[:] = False

                
                l0_M, f_M, q2_M, u_M, Sg_M = gAMxMu.declare_params(M_max, K)   #  nzclstr not inited  # sized to include noise cluster if needed
                l0_exp_px_M = _N.empty(M_max)
                _l0_a_M, _l0_B_M, _f_u_M, _f_q2_M, _q2_a_M, _q2_B_M, _u_u_M, \
                    _u_Sg_M, _Sg_nu_M, _Sg_PSI_M = gAMxMu.declare_prior_hyp_params(M_max, nHSclusters, K, x, mks, Asts, t0, priors, labS, labH)

                l0, f, q2, u, Sg        = gAMxMu.copy_slice_params(M_use, l0_M, f_M, q2_M, u_M, Sg_M)
                _l0_a, _l0_B, _f_u, _f_q2, _q2_a, _q2_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI        = gAMxMu.copy_slice_hyp_params(M_use, _l0_a_M, _l0_B_M, _f_u_M, _f_q2_M, _q2_a_M, _q2_B_M, _u_u_M, _u_Sg_M, _Sg_nu_M, _Sg_PSI_M)
                l0_exp_px = _N.array(l0_exp_px_M[0:M_use], copy=True)

                fr = f.reshape((M_use, 1))
                gAMxMu.init_params_hyps(oo, M_use, K, l0, f, q2, u, Sg, _l0_a, _l0_B, _f_u, _f_q2, _q2_a, _q2_B, _u_u, _u_Sg, _Sg_nu, \
                    _Sg_PSI, Asts, t0, x, mks, flatlabels, nHSclusters)

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
                if n_fClstrs < oo.NExtrClstr:
                    old_M = M_use
                    M_use  = M_use + (oo.NExtrClstr - n_fClstrs)
                    new_M = M_use
                elif n_fClstrs > oo.NExtrClstr:
                    old_M = M_use
                    M_use  = M_use + (oo.NExtrClstr - n_fClstrs)
                    new_M = M_use

                print "aft M_use %d" % M_use

                l0, f, q2, u, Sg        = gAMxMu.copy_slice_params(M_use, l0_M, f_M, q2_M, u_M, Sg_M)
                _l0_a, _l0_B, _f_u, _f_q2, _q2_a, _q2_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI        = gAMxMu.copy_slice_hyp_params(M_use, _l0_a_M, _l0_B_M, _f_u_M, _f_q2_M, _q2_a_M, _q2_B_M, _u_u_M, _u_Sg_M, _Sg_nu_M, _Sg_PSI_M)

                l0_exp_px = _N.array(l0_exp_px_M[0:M_use], copy=True)

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
                               _N.zeros((1, ITERS, M_use)), 
                               _N.zeros((K, K, ITERS, M_use))]

                oo.smp_sp_prms = smp_sp_prms
                oo.smp_mk_prms = smp_mk_prms
                oo.smp_mk_hyps = smp_mk_hyps

                #####  MODES  - find from the sampling
                oo.sp_prmPstMd = _N.zeros(3*M_use)   # mode params
                oo.mk_prmPstMd = [_N.zeros((M_use, K)),
                                  _N.zeros((M_use, K, K))]

                ###  pack cluster
                """
                if n_fClstrs < oo.NExtrClstr:  #  adding clusters
                    for m in xrange(old_M, new_M):
                        gAMxMu.reset_cluster(epc, m, l0, f, q2, freeClstr, _q2_a, _q2_B, _f_u, _f_q2, _l0_a, _l0_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI, oo, priors, m1stSignalClstr)
                      # mode of params

                    #  resize l0, f, q2, u, Sg
                    #  initialize priors for new free clusters
                    #  init to signal cluster _l0_a[M_prev:], _l0_B[M_prev:]
                """

            NSexp   = t1-t0    #  length of position data  #  # of no spike positions to sum
            xt0t1 = _N.array(x[t0:t1])

            nSpks    = len(Asts)
            gz   = _N.zeros((ITERS, nSpks, M_use), dtype=_N.bool)
            oo.gz=gz

            _cdfs.dSilenceX = (NSexp/float(oo.Nupx))*(oo.xHi-oo.xLo)

            xAS  = x[Asts + t0]   #  position @ spikes.  creates new copy
            mAS  = mks[Asts + t0]   #  position @ spikes
            xASr = xAS.reshape((1, nSpks))
            #mASr = mAS.reshape((1, nSpks, K))
            econt = _N.empty((M_use, nSpks))
            rat   = _N.zeros((M_use+1, nSpks))

            qdrMKS = _N.empty((M_use, nSpks))
            ################################  GIBBS ITERS ITERS ITERS

            clstsz = _N.zeros(M_use, dtype=_N.int)

            _iu_Sg = _N.array(_u_Sg)
            for m in xrange(M_use):
                _iu_Sg[m] = _N.linalg.inv(_u_Sg[m])

            ttA = _tm.time()

            v_sts = _N.empty(len(Asts), dtype=_N.int)
            cls_str_ind = _N.zeros(M_use+1, dtype=_N.int)
            cls_len      = _N.zeros(M_use, dtype=_N.int)

            l0_a_is0    = _N.where(_l0_a == 0)[0]
            l0_a_Init   = _N.where(_l0_a >  0)[0]
            b_l0_a_is0  = len(l0_a_is0) > 0
            q2_a_is_m1  = _N.where(_q2_a == -1)[0]
            q2_a_Init   = _N.where(_q2_a > 0)[0]
            b_q2_a_is_m1= len(q2_a_is_m1) > 0

            _Dl0_a = _N.empty(M_use);            _Dl0_B = _N.empty(M_use)
            _Dq2_a = _N.empty(M_use);            _Dq2_B = _N.empty(M_use)

            iiq2 = 1./q2
            iiq2r= iiq2.reshape((M_use, 1))

            for itr in xrange(ITERS):
                #tt1 = _tm.time()
                iSg = _N.linalg.inv(Sg)

                if (itr % 100) == 0:    
                    print "-------itr  %(i)d" % {"i" : itr}

                _sA.stochasticAssignment(oo, epc, itr, M_use, K, l0, f, q2, u, Sg, iSg, _f_u, _u_u, _f_q2, _u_Sg, Asts, t0, mAS, xASr, rat, econt, gz, qdrMKS, freeClstr, hashthresh, m1stSignalClstr, ((epc > 0) and (itr == 0)), nthrds=oo.nThrds)
                #tt2 = _tm.time()
                ###############  FOR EACH CLUSTER
                for m in xrange(M_use):   #  get the minds
                    minds = _N.where(gz[itr, :, m] == 1)[0]  
                    sts  = Asts[minds] + t0   #  sts is in absolute time
                    L    = len(sts)
                    cls_str_ind[m+1] = L + cls_str_ind[m]
                    clstsz[m]        = L
                    v_sts[cls_str_ind[m]:cls_str_ind[m+1]] = sts

                #tt3 = _tm.time()
                mcs = _N.empty((M_use, K))   # cluster sample means

                #tt4 = _tm.time()

                ###############
                ###############     u
                ###############
                for m in xrange(M_use):
                    if clstsz[m] > 0:   # >= K causes Cholesky to fail at times.
                        u_Sg_[m] = _N.linalg.inv(_iu_Sg[m] + clstsz[m]*iSg[m])
                        clstx    = mks[v_sts[cls_str_ind[m]:cls_str_ind[m+1]]]

                        mcs[m]       = _N.mean(clstx, axis=0)
                        u_u_[m] = _N.einsum("jk,k->j", u_Sg_[m], _N.dot(_iu_Sg[m], _u_u[m]) + clstsz[m]*_N.dot(iSg[m], mcs[m]))
                        ########  POSITION
                        ##  mean of posterior distribution of cluster means
                        #  sigma^2 and mu are the current Gibbs-sampled values
                        ##  mean of posterior distribution of cluster means
                        # print "for cluster %(m)d with size %(sz)d" % {"m" : m, "sz" : clstsz[m]}
                    else:
                        u_Sg_[m] = _N.array(_u_Sg[m])
                        u_u_[m] = _N.array(_u_u[m])
                        # if (epc == 1) and (itr < 10):
                        #     print "u_Sg_[%d]" % m
                        #     print u_Sg_[m]
                        #     print "u_u_[%d]" % m
                        #     print u_u_[m]

                ucmvnrms= _N.random.randn(M_use, K)
                try:
                    C       = _N.linalg.cholesky(u_Sg_)
                except _N.linalg.linalg.LinAlgError:
                    dmp = open("cholesky.dmp", "wb")
                    pickle.dump([u_Sg_, _iu_Sg, clstsz, iSg, _u_Sg, _u_u], dmp, -1)
                    dmp.close()

                    raise
                u       = _N.einsum("njk,nk->nj", C, ucmvnrms) + u_u_
                # if epc == 1:
                #     u[5] = _N.array([ 4.9,  4.7,  2.8,  3.8])

                smp_mk_prms[oo.ky_p_u][:, itr] = u.T  # dim of u wrong
                smp_mk_hyps[oo.ky_h_u_u][:, itr] = u_u_.T
                smp_mk_hyps[oo.ky_h_u_Sg][:, :, itr] = u_Sg_.T


                #tt5 = _tm.time()
                ###############
                ###############  Conditional f
                ###############
                if (epc > 0) and oo.adapt:
                    q2pr = _f_q2 + f_q2_rate * DT
                else:
                    q2pr = _f_q2

                m_rnds = _N.random.rand(M_use)
                
                _cdfs.smp_f(M_use, clstsz, cls_str_ind, v_sts, xt0t1, t0, f, q2, l0, _f_u, q2pr, m_rnds)
                # if epc == 1:
                #     f[:] = -400
                #     f[5] = 3

                #f[0] = 0
                smp_sp_prms[oo.ky_p_f, itr] = f

                #tt6 = _tm.time()
                ##############
                ##############  VARIANCE, COVARIANCE
                ##############

                # tt6a = 0
                # tt6b = 0
                # tt6c = 0
                for m in xrange(M_use):
                    #infor = _tm.time()
                    #if clstsz[m] > K:
                    ##  dof of posterior distribution of cluster covariance
                    Sg_nu_[m] = _Sg_nu[m, 0] + clstsz[m]
                    ##  dof of posterior distribution of cluster covariance
                    ur = u[m].reshape((1, K))
                    clstx    = mks[v_sts[cls_str_ind[m]:cls_str_ind[m+1]]]
                    #tt6a += (_tm.time()-infor)
                    #infor = _tm.time()
                    #clstx    = l_sts[m]]
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

                # print "-----------"
                # print tt6a      0.0008
                # print tt6b      0.0009
                # print tt6c      0.012    invwishart.rvs  very slow
                # print "-----------"
                #s_u.m_sample_invwishart(Sg_PSI_, Sg_nu_, K, uptriinds, Sg)
                
                ##  dof of posterior distribution of cluster covariance

                smp_mk_prms[oo.ky_p_Sg][:, :, itr] = Sg.T
                smp_mk_hyps[oo.ky_h_Sg_nu][0, itr] = Sg_nu_
                smp_mk_hyps[oo.ky_h_Sg_PSI][:, :, itr] = Sg_PSI_.T

                #tt7 = _tm.time()
                ##############
                ##############  SAMPLE SPATIAL VARIANCE
                ##############
                #  B' / (a' - 1) = MODE   #keep mode the same after discount

                m_rnds = _N.random.rand(M_use)
                #  B' = MODE * (a' - 1)
                if (epc > 0) and oo.adapt:
                    _md_nd= _q2_B / (_q2_a + 1)
                    _Dq2_a = _q2_a * _N.exp(-DT/tau_q2)
                    _Dq2_B = _md_nd * (_Dq2_a + 1)
                else:
                    _Dq2_a = _q2_a
                    _Dq2_B = _q2_B

                #tt8 = _tm.time()

                _cdfs.smp_q2(M_use, clstsz, cls_str_ind, v_sts, xt0t1, t0, f, q2, l0, _Dq2_a, _Dq2_B, m_rnds, epc)
                # if epc == 1:
                #     #q2[:] = 0.0001
                #     q2[5] = 0.5

                smp_sp_prms[oo.ky_p_q2, itr]   = q2
                # print "-----------------"
                # print q2

                #tt9 = _tm.time()



                ###############
                ###############  CONDITIONAL l0
                ###############
                #  _ss.gamma.rvs.  uses k, theta  k is 1/B (B is our thing)
                _cdfs.l0_spatial(M_use, oo.dt, f, q2, l0_exp_px)

                #iiq2 = 1./q2
                #iiq2r= iiq2.reshape((M, 1))
                #fr = f.reshape((M, 1))
                #l0_intgrd   = _N.exp(-0.5*(fr - uxr)*(fr-uxr) * iiq2r)  

                #l0_exp_px   = _N.sum(l0_intgrd*pxr, axis=1) * _cdfs.dSilenceX
                #BL  = ((oo.dt)/_N.sqrt(twpi*q2))*l0_exp_px    #  dim M
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

                aL  = clstsz
                l0_a_ = aL + _Dl0_a
                l0_B_ = BL + _Dl0_B


                # _ss.gamma.rvs(20+20, scale=(1/(50+50.))) ~ 0.4
                # _ss.gamma.rvs(20+ 0, scale=(1/(50+50.))) ~ 0.2
                #  if no spikes observed, sampled l0 is smaller.  
                try:   #  if there is no prior, if a cluster 
                    l0 = _ss.gamma.rvs(l0_a_, scale=(1/l0_B_), size=M_use)  #  check
                    # if epc == 1:
                    #     l0[:] = 1e-200
                    #     l0[5] = 300.

                    #l0 = _N.ones(M)#_ss.gamma.rvs(l0_a_, scale=(1/l0_B_))  #  check
                except ValueError:
                    print "problem with l0"
                    print l0_exp_px
                    #_N.savetxt("fxux", (fr - ux)*(fr-ux))
                    #_N.savetxt("fr", fr)
                    #_N.savetxt("iiq2", iiq2)
                    #_N.savetxt("l0_intgrd", l0_intgrd)
                    raise

                smp_sp_prms[oo.ky_p_l0, itr] = l0

                # tt10 = _tm.time()
                # print "#timing start"
                # print "nt+= 1"
                # print "t2t1+=%.4e" % (tt2-tt1)
                # print "t3t2+=%.4e" % (tt3-tt2)
                # print "t4t3+=%.4e" % (tt4-tt3)
                # print "t5t4+=%.4e" % (tt5-tt4)
                # print "t6t5+=%.4e" % (tt6-tt5)
                # print "t7t6+=%.4e" % (tt7-tt6)  # slow
                # print "t8t7+=%.4e" % (tt8-tt7)
                # print "t9t8+=%.4e" % (tt9-tt8)
                # print "t10t9+=%.4e" % (tt10-tt9)
                # print "#timing end"
                # print (tt10-tt1)


            ttB = _tm.time()
            print (ttB-ttA)
            
            gAMxMu.finish_epoch2(oo, nSpks, epc, ITERS, gz, l0, f, q2, u, Sg, _f_u, _f_q2, _q2_a, _q2_B, _l0_a, _l0_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI, smp_sp_prms, smp_mk_prms, smp_mk_hyps, freeClstr, M_use, K, priors, m1stSignalClstr)
            #  _l0_a is a copy of a subset of _l0_a_M
            #  we need to copy back the values _l0_a back into _l0_a_M
            gAMxMu.contiguous_inuse(M_use, M_max, K, freeClstr, l0, f, q2, u, Sg, _l0_a, _l0_B, _f_u, _f_q2, _q2_a, _q2_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI, smp_sp_prms, smp_mk_prms, oo.sp_prmPstMd, oo.mk_prmPstMd, gz, priors)
            gAMxMu.copy_back_params(M_use, l0, f, q2, u, Sg, M_max, l0_M, f_M, q2_M, u_M, Sg_M)
            gAMxMu.copy_back_hyp_params(M_use, _l0_a, _l0_B, _f_u, _f_q2, _q2_a, _q2_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI, M_max, _l0_a_M, _l0_B_M, _f_u_M, _f_q2_M, _q2_a_M, _q2_B_M, _u_u_M, _u_Sg_M, _Sg_nu_M, _Sg_PSI_M)
            #  MAP of nzclstr
            pcklme["smp_sp_prms"] = smp_sp_prms
            pcklme["smp_mk_prms"] = smp_mk_prms
            pcklme["sp_prmPstMd"] = oo.sp_prmPstMd
            pcklme["mk_prmPstMd"] = oo.mk_prmPstMd
            pcklme["intvs"]       = oo.intvs
            pcklme["occ"]         = c_gz.gz2cgz(gz)
            pcklme["nz_pth"]         = nz_pth
            pcklme["M"]           = M_use
            pcklme["freeClstr"]           = freeClstr
                
            dmp = open(resFN("posteriors_%d.dmp" % epc, dir=oo.outdir), "wb")
            pickle.dump(pcklme, dmp, -1)
            dmp.close()

######  Hi Eric
