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

        for epc in xrange(ep1, ep2):
            print "^^^^^^^^^^^^^^^^^^^^^^^^    epoch %d" % epc

            t0 = oo.intvs[epc]
            t1 = oo.intvs[epc+1]
            if epc > 0:
                tm1= oo.intvs[epc-1]
                #  0 10 30     20 - 5  = 15    0.5*((10+30) - (10+0)) = 15
                dt = 0.5*((t1+t0) - (t0+tm1))

            dt = (t1-t0)*0.5

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
                labS, labH, flatlabels, M, MF, hashthresh, nHSclusters = gAMxMu.initClusters(oo, K, x, mks, t0, t1, Asts, doSepHash=doSepHash, xLo=oo.xLo, xHi=oo.xHi, oneCluster=oo.oneCluster)
                m1stHashClstr = 1 if oo.oneCluster else _N.min(_N.unique(labH)) 

                #nHSclusters.append(M - nHSclusters[0]-nHSclusters[1])   #  last are free clusters that are not the noise cluster

                #  posteriors
                u_u_  = _N.empty((M, K))
                u_Sg_ = _N.empty((M, K, K))
                Sg_nu_ = _N.empty(M)
                Sg_PSI_ = _N.empty((M, K, K))
                Sg     = _N.empty((M, K, K))   # sampled value
                uptriinds = _N.triu_indices_from(Sg[0],1)

                #######   containers for GIBBS samples iterations
                smp_sp_prms = _N.zeros((3, ITERS, M))  
                smp_mk_prms = [_N.zeros((K, ITERS, M)), 
                               _N.zeros((K, K, ITERS, M))]
                smp_sp_hyps = _N.zeros((6, ITERS, M))
                smp_mk_hyps = [_N.zeros((K, ITERS, M)), 
                               _N.zeros((K, K, ITERS, M)),
                               _N.zeros((1, ITERS, M)), 
                               _N.zeros((K, K, ITERS, M))]
                oo.smp_sp_prms = smp_sp_prms
                oo.smp_mk_prms = smp_mk_prms
                oo.smp_sp_hyps = smp_sp_hyps
                oo.smp_mk_hyps = smp_mk_hyps

                #  list of freeClstrs
                freeClstr = _N.empty(M, dtype=_N.bool)   #  Actual cluster
                freeClstr[:] = False

                l0, f, q2, u, Sg = gAMxMu.declare_params(M, K)   #  nzclstr not inited  # sized to include noise cluster if needed
                l0_exp_px = _N.empty(M)
                _l0_a, _l0_B, _f_u, _f_q2, _q2_a, _q2_B, _u_u, _u_Sg, _Sg_nu, \
                    _Sg_PSI = gAMxMu.declare_prior_hyp_params(M, MF, K, x, mks, Asts, t0, priors, labS, labH)
                fr = f.reshape((M, 1))
                gAMxMu.init_params_hyps(oo, M, MF, K, l0, f, q2, u, Sg, _l0_a, _l0_B, _f_u, _f_q2, _q2_a, _q2_B, _u_u, _u_Sg, _Sg_nu, \
                    _Sg_PSI, Asts, t0, x, mks, flatlabels, nHSclusters)

                U   = _N.empty(M)
                FQ2 = _N.empty(M)

                ######  the hyperparameters for f, q2, u, Sg, l0 during Gibbs
                #  f_u_, f_q2_, q2_a_, q2_B_, u_u_, u_Sg_, Sg_nu, Sg_PSI_, l0_a_, l0_B_

            NSexp   = t1-t0    #  length of position data  #  # of no spike positions to sum
            xt0t1 = _N.array(x[t0:t1])

            nSpks    = len(Asts)
            gz   = _N.zeros((ITERS, nSpks, M), dtype=_N.bool)
            oo.gz=gz

            _cdfs.dSilenceX = (NSexp/float(oo.Nupx))*(oo.xHi-oo.xLo)

            xAS  = x[Asts + t0]   #  position @ spikes
            mAS  = mks[Asts + t0]   #  position @ spikes
            xASr = xAS.reshape((1, nSpks))
            mASr = mAS.reshape((1, nSpks, K))
            econt = _N.empty((M, nSpks))
            rat   = _N.zeros((M+1, nSpks))

            qdrMKS = _N.empty((M, nSpks))
            ################################  GIBBS ITERS ITERS ITERS

            clstsz = _N.zeros(M, dtype=_N.int)

            _iu_Sg = _N.array(_u_Sg)
            for m in xrange(M):
                _iu_Sg[m] = _N.linalg.inv(_u_Sg[m])

            ttA = _tm.time()

            v_sts = _N.empty(len(Asts), dtype=_N.int)
            cls_str_ind = _N.zeros(M+1, dtype=_N.int)
            cls_len      = _N.zeros(M, dtype=_N.int)

            l0_a_is0    = _N.where(_l0_a == 0)[0]
            l0_a_Init   = _N.where(_l0_a >  0)[0]
            b_l0_a_is0  = len(l0_a_is0) > 0
            q2_a_is_m1  = _N.where(_q2_a == -1)[0]
            q2_a_Init   = _N.where(_q2_a > 0)[0]
            b_q2_a_is_m1= len(q2_a_is_m1) > 0

            _Dl0_a = _N.empty(M);            _Dl0_B = _N.empty(M)
            _Dq2_a = _N.empty(M);            _Dq2_B = _N.empty(M)

            iiq2 = 1./q2
            iiq2r= iiq2.reshape((M, 1))
            #sLLkPr      = _N.empty((M, oo.q2ss))

            for iter in xrange(ITERS):
                #tt1 = _tm.time()
                iSg = _N.linalg.inv(Sg)

                if (iter % 100) == 0:    
                    print "-------iter  %(i)d" % {"i" : iter}

                _sA.stochasticAssignment(oo, epc, iter, M, K, l0, f, q2, u, Sg, iSg, _f_u, _u_u, _f_q2, _u_Sg, Asts, t0, mASr, xASr, rat, econt, gz, qdrMKS, freeClstr, hashthresh, m1stHashClstr, ((epc > 0) and (iter == 0)), nthrds=oo.nThrds)
                #tt2 = _tm.time()
                ###############  FOR EACH CLUSTER
                for m in xrange(M):   #  get the minds
                    minds = _N.where(gz[iter, :, m] == 1)[0]  
                    sts  = Asts[minds] + t0   #  sts is in absolute time
                    L    = len(sts)
                    cls_str_ind[m+1] = L + cls_str_ind[m]
                    clstsz[m]        = L
                    v_sts[cls_str_ind[m]:cls_str_ind[m+1]] = sts

                #tt3 = _tm.time()
                mcs = _N.empty((M, K))   # cluster sample means

                #tt4 = _tm.time()

                ###############
                ###############     u
                ###############
                for m in xrange(M):
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

                ucmvnrms= _N.random.randn(M, K)
                try:
                    C       = _N.linalg.cholesky(u_Sg_)
                except _N.linalg.linalg.LinAlgError:
                    print "linalg error in u_Sg_, iter %d" % iter
                    print u_Sg_

                    dmp = open("cholesky.dmp", "wb")
                    pickle.dump([u_Sg_, _iu_Sg, clstsz, iSg, _u_Sg, _u_u], dmp, -1)
                    dmp.close()

                    raise
                u       = _N.einsum("njk,nk->nj", C, ucmvnrms) + u_u_

                smp_mk_prms[oo.ky_p_u][:, iter] = u.T  # dim of u wrong
                smp_mk_hyps[oo.ky_h_u_u][:, iter] = u_u_.T
                smp_mk_hyps[oo.ky_h_u_Sg][:, :, iter] = u_Sg_.T

                #tt5 = _tm.time()
                ###############
                ###############  Conditional f
                ###############
                if (epc > 0) and oo.adapt:
                    q2pr = _f_q2 + f_q2_rate * dt
                else:
                    q2pr = _f_q2

                m_rnds = _N.random.rand(M)
                _cdfs.smp_f(M, clstsz, cls_str_ind, v_sts, xt0t1, t0, f, q2, l0, _f_u, q2pr, m_rnds)
                #f[0] = 0
                smp_sp_prms[oo.ky_p_f, iter] = f

                #tt6 = _tm.time()
                ##############
                ##############  VARIANCE, COVARIANCE
                ##############

                for m in xrange(M):
                    #if clstsz[m] > K:
                    ##  dof of posterior distribution of cluster covariance
                    Sg_nu_[m] = _Sg_nu[m, 0] + clstsz[m]
                    ##  dof of posterior distribution of cluster covariance
                    ur = u[m].reshape((1, K))
                    clstx    = mks[v_sts[cls_str_ind[m]:cls_str_ind[m+1]]]
                    #clstx    = l_sts[m]]
                    #  dot((clstx-ur).T, (clstx-ur))==ZERO(K) when clstsz ==0
                    Sg_PSI_[m] = _Sg_PSI[m] + _N.dot((clstx - ur).T, (clstx-ur))
                    # else:
                    #     Sg_nu_ = _Sg_nu[m, 0] 
                    #     ##  dof of posterior distribution of cluster covariance
                    #     ur = u[m].reshape((1, K))
                    #     Sg_PSI_ = _Sg_PSI[m]
                smp_mk_hyps[oo.ky_h_Sg_nu][0, iter] = Sg_nu_
                smp_mk_hyps[oo.ky_h_Sg_PSI][:, :, iter] = Sg_PSI_.T
                s_u.m_sample_invwishart(Sg_PSI_, Sg_nu_, K, uptriinds, Sg)
                
                ##  dof of posterior distribution of cluster covariance

                smp_mk_prms[oo.ky_p_Sg][:, :, iter] = Sg.T

                #tt7 = _tm.time()
                ##############
                ##############  SAMPLE SPATIAL VARIANCE
                ##############
                #  B' / (a' - 1) = MODE   #keep mode the same after discount

                m_rnds = _N.random.rand(M)
                #  B' = MODE * (a' - 1)
                if (epc > 0) and oo.adapt:
                    _md_nd= _q2_B / (_q2_a + 1)
                    _Dq2_a = _q2_a * _N.exp(-dt/tau_q2)
                    _Dq2_B = _Dq2_a / _md_nd
                else:
                    _Dq2_a = _q2_a
                    _Dq2_B = _q2_B

                #tt8 = _tm.time()

                _cdfs.smp_q2(M, clstsz, cls_str_ind, v_sts, xt0t1, t0, f, q2, l0, _Dq2_a, _Dq2_B, m_rnds)
                smp_sp_prms[oo.ky_p_q2, iter]   = q2
                # print "-----------------"
                # print q2

                #tt9 = _tm.time()



                ###############
                ###############  CONDITIONAL l0
                ###############
                #  _ss.gamma.rvs.  uses k, theta  k is 1/B (B is our thing)
                _cdfs.l0_spatial(M, oo.dt, f, q2, l0_exp_px)

                #iiq2 = 1./q2
                #iiq2r= iiq2.reshape((M, 1))
                #fr = f.reshape((M, 1))
                #l0_intgrd   = _N.exp(-0.5*(fr - uxr)*(fr-uxr) * iiq2r)  

                #l0_exp_px   = _N.sum(l0_intgrd*pxr, axis=1) * _cdfs.dSilenceX
                #BL  = ((oo.dt)/_N.sqrt(twpi*q2))*l0_exp_px    #  dim M
                BL  = l0_exp_px    #  dim M

                if (epc > 0) and oo.adapt:
                    _md_nd= _l0_a / _l0_B

                    _Dl0_a = _l0_a * _N.exp(-dt/tau_l0)
                    _Dl0_B = _Dl0_a / _md_nd
                else:
                    _Dl0_a = _l0_a
                    _Dl0_B = _l0_B

                aL  = clstsz
                l0_a_ = aL + _Dl0_a
                l0_B_ = BL + _Dl0_B
                
                try:   #  if there is no prior, if a cluster 
                    l0 = _ss.gamma.rvs(l0_a_, scale=(1/l0_B_))  #  check
                    #l0 = _N.ones(M)#_ss.gamma.rvs(l0_a_, scale=(1/l0_B_))  #  check
                except ValueError:
                    print "problem with l0"
                    print l0_exp_px
                    #_N.savetxt("fxux", (fr - ux)*(fr-ux))
                    #_N.savetxt("fr", fr)
                    #_N.savetxt("iiq2", iiq2)
                    #_N.savetxt("l0_intgrd", l0_intgrd)
                    raise

                smp_sp_prms[oo.ky_p_l0, iter] = l0
                smp_sp_hyps[oo.ky_h_l0_a, iter] = l0_a_
                smp_sp_hyps[oo.ky_h_l0_B, iter] = l0_B_

                #tt10 = _tm.time()
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


            ttB = _tm.time()
            print (ttB-ttA)
            
            gAMxMu.finish_epoch2(oo, nSpks, epc, ITERS, gz, l0, f, q2, u, Sg, _f_u, _f_q2, _q2_a, _q2_B, _l0_a, _l0_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI, smp_sp_hyps, smp_sp_prms, smp_mk_hyps, smp_mk_prms, freeClstr, M, K, priors, m1stHashClstr)
            #  MAP of nzclstr
            pcklme["smp_sp_hyps"] = smp_sp_hyps
            pcklme["smp_mk_hyps"] = smp_mk_hyps
            pcklme["smp_sp_prms"] = smp_sp_prms
            pcklme["smp_mk_prms"] = smp_mk_prms
            pcklme["sp_prmPstMd"] = oo.sp_prmPstMd
            pcklme["mk_prmPstMd"] = oo.mk_prmPstMd
            pcklme["intvs"]       = oo.intvs
            pcklme["occ"]         = gz
            pcklme["nz_pth"]         = nz_pth
            pcklme["M"]           = M
                
            dmp = open(resFN("posteriors_%d.dmp" % epc, dir=oo.outdir), "wb")
            pickle.dump(pcklme, dmp, -1)
            dmp.close()

######  Hi Eric
