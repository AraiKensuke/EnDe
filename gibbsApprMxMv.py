"""
V1.2   use adaptive range for integrating over f
variance   0.0001 
"""
import stats_util as s_u
import scipy.stats as _ss
import os
import time as _tm
#from ig_prmLib import mltpl_ig_prmsUV
import py_cdf_smp as _pcs
import numpy as _N
import matplotlib.pyplot as _plt
from EnDedirs import resFN, datFN
import pickle
from posteriorUtil import MAPvalues2
from filter import gauKer
import gibbsApprMxMutil as gAMxMu
from par_intgrls_f  import M_times_N_f_intgrls_raw
from par_intgrls_q2 import M_times_N_q2_intgrls_raw
import raw_random_access as _rra

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
    Nupx = 200      #   # points to sample position with  (uniform lam(x)p(x))
    fss = 200       #  sampling at various values of f
    q2ss = 150      #  sampling at various values of q2

    intvs = None    #  
    dat   = None

    resetClus = True

    diffPerMin = 1.  #  diffusion per minute
    epochs   = None
    adapt    = False

    outdir   = None
    polyFit  = True
    xLo      = -6
    xHi      = 6

    #  l0, q2      Sig    f, u
    t_hlf_l0 = int(1000*60*2.5)   # 10minutes
    t_hlf_q2 = int(1000*60*2.5)   # 10minutes

    nzclstr  = False
    nThrds = 2    #  if use_omp
    diffusePerMin = 0.05    #  diffusion of certainty

    nz_q2               = 500
    nz_f                = 0

    q2x_L = 1e-7
    q2x_H = 1e2

    q2_min = 5e-5
    q2_max = 1e5

    q2_dec_wgts = None   #  weights to place on bin size for q2 numerical  [wgtL, wgtH].  Equally sized bins in logscale by default.
    
    oneCluster = False
    
    def __init__(self, outdir, fn, intvfn, xLo=0, xHi=3, seed=1041, adapt=True, nzclstr=False, t_hlf_l0_mins=None, t_hlf_q2_mins=None, oneCluster=False):
        oo     = self
        oo.oneCluster = oneCluster
        oo.adapt = adapt
        _N.random.seed(seed)
        oo.nzclstr = nzclstr

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

    def gibbs(self, ITERS, K, priors, ep1=0, ep2=None, savePosterior=True, gtdiffusion=False, doSepHash=True, use_spc=True, nz_pth=0., smth_pth_ker=100, ignoresilence=False, use_omp=False, nThrds=2):
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
        ux = _N.linspace(oo.xLo, oo.xHi, oo.Nupx, endpoint=False)   # uniform x position   #  grid over 
        uxr = ux.reshape((1, oo.Nupx))
        uxrr= ux.reshape((1, 1, oo.Nupx))
        #q2x    = _N.exp(_N.linspace(_N.log(1e-7), _N.log(100), oo.q2ss))  #  5 orders of
        q2x    = _N.exp(_N.linspace(_N.log(oo.q2x_L), _N.log(oo.q2x_H), oo.q2ss))  #  5 orders of
        q2x_z  = _N.exp(_N.linspace(_N.log(oo.q2x_L), _N.log(oo.q2x_H), oo.q2ss*10))  #  5 orders of
        oo.q2ss_z = 10*oo.q2ss

        if oo.q2_dec_wgts is not None:
            q2x_m    = _N.log(q2x)
            d_c_log= _N.diff(q2x_m)   #  <=  constant
            d_c_log *= _N.linspace(oo.q2_dec_wgts[0], oo.q2_dec_wgts[1], oo.q2ss-1)

            for ii in xrange(oo.q2ss - 1):
                q2x_m[ii+1] = q2x_m[ii] + d_c_log[ii]

            #print q2x
            #print q2x_m
            q2x = _N.exp(q2x_m)   # re-weighted bin sizes
            
        d_q2x  = _N.diff(q2x)
        d_q2x_z  = _N.diff(q2x_z)
        q2x_m1 = _N.array(q2x[0:-1])
        q2x_m1_z = _N.array(q2x_z[0:-1])
        lq2x    = _N.log(q2x)
        iq2x    = 1./q2x
        q2xr     = q2x.reshape((oo.q2ss, 1))
        q2xr_z     = q2x_z.reshape((oo.q2ss*10, 1))

        iq2xr     = 1./q2xr
        q2xrr     = q2x.reshape((1, oo.q2ss, 1))
        iq2xrr     = 1./q2xrr
        d_q2xr  =  d_q2x.reshape((oo.q2ss - 1, 1))
        d_q2xr_z  =  d_q2x_z.reshape((oo.q2ss*10 - 1, 1))
        q2x_m1  = _N.array(q2x[0:-1])
        q2x_m1r = q2x_m1.reshape((oo.q2ss-1, 1))
        q2x_m1r_z = q2x_m1_z.reshape((oo.q2ss*10-1, 1))

        sqrt_2pi_q2x   = _N.sqrt(twpi*q2x)
        l_sqrt_2pi_q2x = _N.log(sqrt_2pi_q2x)

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

        for epc in xrange(ep1, ep2):
            print "^^^^^^^^^^^^^^^^^^^^^^^^    epoch %d" % epc

            t0 = oo.intvs[epc]
            t1 = oo.intvs[epc+1]
            if epc > 0:
                tm1= oo.intvs[epc-1]
                #  0 10 30     20 - 5  = 15    0.5*((10+30) - (10+0)) = 15
                dt = 0.5*((t1+t0) - (t0+tm1))

            dt = (t1-t0)*0.5
            xt0t1 = _N.array(x[t0:t1])
            posbins  = _N.linspace(oo.xLo, oo.xHi, oo.Nupx+1)
            #  _N.sum(px)*(xbns[1]-xbns[0]) = 1
            px, xbns = _N.histogram(xt0t1, bins=posbins, normed=True)
            pxr      = px.reshape((1, oo.Nupx))
            pxrr     = px.reshape((1, 1, oo.Nupx))

            Asts    = _N.where(oo.dat[t0:t1, 1] == 1)[0]   #  based at 0

            if epc == ep1:   ###  initialize
                labS, labH, flatlabels, M, MF, hashthresh, nHSclusters = gAMxMu.initClusters(oo, K, x, mks, t0, t1, Asts, doSepHash=doSepHash, xLo=oo.xLo, xHi=oo.xHi, oneCluster=oo.oneCluster, nzclstr=oo.nzclstr)
 
                Mwowonz       = M+1 if oo.nzclstr else M
                #nHSclusters.append(M - nHSclusters[0]-nHSclusters[1])   #  last are free clusters that are not the noise cluster

                u_u_  = _N.empty((M, K))
                u_Sg_ = _N.empty((M, K, K))
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

                if oo.nzclstr:
                    smp_nz_l0     = _N.zeros(ITERS)
                    smp_nz_hyps = _N.zeros((2, ITERS))

                #  list of freeClstrs
                freeClstr = _N.empty(M, dtype=_N.bool)   #  Actual cluster
                freeClstr[:] = False

                l0, f, q2, u, Sg = gAMxMu.declare_params(M, K, nzclstr=oo.nzclstr)   #  nzclstr not inited  # sized to include noise cluster if needed
                _l0_a, _l0_B, _f_u, _f_q2, _q2_a, _q2_B, _u_u, _u_Sg, _Sg_nu, \
                    _Sg_PSI = gAMxMu.declare_prior_hyp_params(M, MF, K, x, mks, Asts, t0, priors, labS, labH)
                fr = f[0:M].reshape((M, 1))
                gAMxMu.init_params_hyps(oo, M, MF, K, l0, f, q2, u, Sg, _l0_a, _l0_B, _f_u, _f_q2, _q2_a, _q2_B, _u_u, _u_Sg, _Sg_nu, \
                    _Sg_PSI, Asts, t0, x, mks, flatlabels, nHSclusters, nzclstr=oo.nzclstr)
                # print "-----------------"
                # print _l0_a
                # print _l0_B
                # print _q2_a                
                # print _q2_B
                # print "-----------------"
                U   = _N.empty(M)
                FQ2 = _N.empty(M)
                _fxs0 = _N.tile(_N.linspace(0, 1, oo.fss), M).reshape(M, oo.fss)

                f_exp_px = _N.empty((M, oo.fss))
                q2_exp_px= _N.empty((M, oo.q2ss))

                if oo.nzclstr:
                    nz_l0_intgrd   = _N.exp(-0.5*ux*ux / q2[Mwowonz-1])
                    _nz_l0_a       = 0.001
                    _nz_l0_B       = 0.1

                ######  the hyperparameters for f, q2, u, Sg, l0 during Gibbs
                #  f_u_, f_q2_, q2_a_, q2_B_, u_u_, u_Sg_, Sg_nu, Sg_PSI_, l0_a_, l0_B_


            NSexp   = t1-t0    #  length of position data  #  # of no spike positions to sum
            xt0t1 = _N.array(x[t0:t1])

            nSpks    = len(Asts)
            gz   = _N.zeros((ITERS, nSpks, Mwowonz), dtype=_N.bool)
            oo.gz=gz
            print "spikes %d" % nSpks

            dSilenceX1 = (NSexp/float(oo.Nupx))*(oo.xHi-oo.xLo)
            dSilenceX2 = NSexp*(xbns[1]-xbns[0])  # dx of histogram
            print "-------------------------- %(1).4f  %(2).4f" % {"1" : dSilenceX1, "2" : dSilenceX2}

            dSilenceX  = dSilenceX1

            xAS  = x[Asts + t0]   #  position @ spikes
            mAS  = mks[Asts + t0]   #  position @ spikes
            xASr = xAS.reshape((1, nSpks))
            mASr = mAS.reshape((1, nSpks, K))
            econt = _N.empty((Mwowonz, nSpks))
            rat   = _N.zeros((Mwowonz+1, nSpks))

            qdrMKS = _N.empty((Mwowonz, nSpks))
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

            iiq2 = 1./q2[0:M]
            iiq2r= iiq2.reshape((M, 1))
            iiq2rr= iiq2.reshape((M, 1, 1))
            sLLkPr      = _N.empty((M, oo.q2ss))
            sLLk_f_Pr      = _N.empty((M, oo.fss))

            for iter in xrange(ITERS):
                #tt1 = _tm.time()
                iSg = _N.linalg.inv(Sg)

                if (iter % 100) == 0:    
                    #print "-------iter  %(i)d   %(r).5f" % {"i" : iter, "r" : _N.random.rand()}
                    print "-------iter  %(i)d" % {"i" : iter}

                gAMxMu.stochasticAssignment(oo, epc, iter, M, Mwowonz, K, l0, f, q2, u, Sg, _f_u, _u_u, _f_q2, _u_Sg, Asts, t0, mASr, xASr, rat, econt, gz, qdrMKS, freeClstr, hashthresh, ((epc > 0) and (iter == 0)), nthrds=oo.nThrds)
                #gAMxMu.stochasticAssignment(oo, iter, M, Mwowonz, K, l0, f, q2, u, Sg, _f_u, _u_u, Asts, t0, mASr, xASr, rat, econt, gz, qdrMKS, freeClstr, hashthresh, iter==0, nthrds=oo.nThrds)

                ###############  FOR EACH CLUSTER

                #l_sts = []

                for m in xrange(M):   #  get the minds
                    minds = _N.where(gz[iter, :, m] == 1)[0]  
                    sts  = Asts[minds] + t0   #  sts is in absolute time
                    L    = len(sts)
                    cls_str_ind[m+1] = L + cls_str_ind[m]
                    clstsz[m]        = L
                    v_sts[cls_str_ind[m]:cls_str_ind[m+1]] = sts
                    #clstsz[m] = len(sts)
                    #l_sts.append(sts)
                # for m in xrange(Mwowonz):   #  get the minds
                #     minds = _N.where(gz[iter, :, m] == 1)[0]  
                #     print "cluster %(m)d   len %(l)d    " % {"m" : m, "l" : len(minds)}
                #     print u[m]
                #     print f[m]

                #tt2 = _tm.time()
                ###############
                ###############  CONDITIONAL l0
                ###############
                mcs = _N.empty((M, K))   # cluster sample means

                #tt3 = _tm.time()

                ###############
                ###############     u
                ###############
                for m in xrange(M):
                    if clstsz[m] >= K:
                        u_Sg_[m] = _N.linalg.inv(_iu_Sg[m] + clstsz[m]*iSg[m])
                        #clstx    = mks[l_sts[m]]
                        clstx    = mks[v_sts[cls_str_ind[m]:cls_str_ind[m+1]]]

                        mcs[m]       = _N.mean(clstx, axis=0)
                        #u_u_[m] = _N.dot(u_Sg_[m], _N.dot(_iu_Sg[m], _u_u[m]) + clstsz[m]*_N.dot(iSg[m], mcs[m]))
                        u_u_[m] = _N.einsum("jk,k->j", u_Sg_[m], _N.dot(_iu_Sg[m], _u_u[m]) + clstsz[m]*_N.dot(iSg[m], mcs[m]))
                        # print "mean of cluster %d" % m
                        # print mcs[m]
                        # print u_u_[m]
                        # hyp
                        ########  POSITION
                        ##  mean of posterior distribution of cluster means
                        #  sigma^2 and mu are the current Gibbs-sampled values

                        ##  mean of posterior distribution of cluster means
                        # print "for cluster %(m)d with size %(sz)d" % {"m" : m, "sz" : clstsz[m]}
                        # print mcs[m]
                        # print u_u_[m]
                        # print _u_u[m]
                    else:
                        u_Sg_[m] = _N.array(_u_Sg[m])
                        u_u_[m] = _N.array(_u_u[m])

                ucmvnrms= _N.random.randn(M, K)
                try:
                    C       = _N.linalg.cholesky(u_Sg_)
                except _N.linalg.linalg.LinAlgError:
                    print "linalg error in u_Sg_"
                    #_N.savetxt("nonPD_u_Sg_", u_Sg_)
                    raise
                u[0:M]       = _N.einsum("njk,nk->nj", C, ucmvnrms) + u_u_

                smp_mk_prms[oo.ky_p_u][:, iter] = u[0:M].T  # dim of u wrong
                smp_mk_hyps[oo.ky_h_u_u][:, iter] = u_u_.T
                smp_mk_hyps[oo.ky_h_u_Sg][:, :, iter] = u_Sg_.T

                #tt4 = _tm.time()
                ###############
                ###############  Conditional f
                ###############

                if (epc > 0) and oo.adapt:
                    q2pr = _f_q2 + f_q2_rate * dt
                else:
                    q2pr = _f_q2

                _rra.f_spiking_portion(xt0t1, t0, v_sts, cls_str_ind, 
                                    clstsz, q2, _f_u, q2pr, M, U, FQ2)

                """
                for m in xrange(M):
                    #sts = l_sts[m]
                    sts  = v_sts[cls_str_ind[m]:cls_str_ind[m+1]]
                    if clstsz[m] > 0:
                        fs  = (1./clstsz[m])*_N.sum(xt0t1[sts-t0])
                        fq2 = q2[m]/clstsz[m]
                        U[m]   = (fs*q2pr[m] + _f_u[m]*fq2) / (q2pr[m] + fq2)
                        FQ2[m] = (q2pr[m]*fq2) / (q2pr[m] + fq2)
                    else:
                        U[m]   = _f_u[m]
                        FQ2[m] = q2pr[m]
                """

                FQ    = _N.sqrt(FQ2)
                Ur    = U.reshape((M, 1))  # mean of spikes assigned to clstr
                FQr   = FQ.reshape((M, 1))
                FQ2r  = FQ2.reshape((M, 1))

                if use_spc:
                    fxs  = _N.copy(_fxs0)
                    fxs *= (FQr*80)   #  40   0.1  
                    fxs -= (FQr*40)
                    fxs += Ur
                    d_fxsr = _N.diff(fxs, axis=1)
                    fxsr     = fxs.reshape((M, oo.fss, 1))

                    if use_omp:
                        M_times_N_f_intgrls_raw(fxs, ux, iiq2, dSilenceX, px, f_exp_px, M, oo.fss, oo.Nupx, oo.nThrds)
                    else:
                        fxsr     = fxs.reshape((M, oo.fss, 1))
                        fxrux = -0.5*(fxsr-uxrr)*(fxsr-uxrr)
                        #  f_intgrd    is M x fss x Nupx
                        f_intgrd  = _N.exp(fxrux*iiq2rr)   #  integrand
                        f_exp_px = _N.sum(f_intgrd*pxrr, axis=2) * dSilenceX
                        #  f_exp_px   is M x fss
                    l0r = l0[0:M].reshape((M, 1))
                    q2r = q2[0:M].reshape((M, 1))
                     #  s   is (M x fss)
                    s = -(l0r*oo.dt/_N.sqrt(twpi*q2r)) * f_exp_px  #  a function of x
                    #if (iter > ITERS - 40) and (iter % 5 == 0):
                    #    print f_exp_px
                    #    _plt.plot(fxs[0], _N.s)
                else:
                    s = _N.zeros(M)

                #  U, FQ2 is   dim(M)   
                #  fxs is M x fss
                sLLk_f_pr = -0.5*((fxs-Ur)*(fxs-Ur))/FQ2r
                # funcf   =  + s
                # maxes   = _N.max(funcf, axis=1)   
                # maxesr  = maxes.reshape((M, 1))
                # funcf   -= maxesr
                # condPosF= _N.exp(funcf)   #  condPosF is M x fss
                # ttB = _tm.time()

                #  fxs   M x fss
                #  fxs            M x fss
                #  condPosF       M x fss
                # norm    = 1./_N.sum(condPosF, axis=1)  #  sz M
                # f_u_    = norm*_N.sum(fxs*condPosF, axis=1)  #  sz M
                # f_u_r   = f_u_.reshape((M, 1))
                # f_q2_   = norm*_N.sum(condPosF*(fxs-f_u_r)*(fxs-f_u_r), axis=1)
                # f[0:M]       = _N.sqrt(f_q2_)*_N.random.randn() + f_u_

                f[0:M] = _pcs.smp_from_cdf_interp_ind_x(fxsr, sLLk_f_pr.T, s.T, d_fxsr, None)

                smp_sp_prms[oo.ky_p_f, iter] = f[0:M]
                #smp_sp_hyps[oo.ky_h_f_u, iter] = f_u_
                #smp_sp_hyps[oo.ky_h_f_q2, iter] = f_q2_

                #tt5 = _tm.time()
                ##############
                ##############  VARIANCE, COVARIANCE
                ##############
                for m in xrange(M):
                    if clstsz[m] >= K:
                        ##  dof of posterior distribution of cluster covariance
                        Sg_nu_ = _Sg_nu[m, 0] + clstsz[m]
                        ##  dof of posterior distribution of cluster covariance
                        ur = u[m].reshape((1, K))
                        clstx    = mks[v_sts[cls_str_ind[m]:cls_str_ind[m+1]]]
                        #clstx    = l_sts[m]]
                        Sg_PSI_ = _Sg_PSI[m] + _N.dot((clstx - ur).T, (clstx-ur))
                    else:
                        Sg_nu_ = _Sg_nu[m, 0] 
                        ##  dof of posterior distribution of cluster covariance
                        ur = u[m].reshape((1, K))
                        Sg_PSI_ = _Sg_PSI[m]
                    Sg[m] = s_u.sample_invwishart(Sg_PSI_, Sg_nu_)
                    smp_mk_hyps[oo.ky_h_Sg_nu][0, iter, m] = Sg_nu_
                    smp_mk_hyps[oo.ky_h_Sg_PSI][:, :, iter, m] = Sg_PSI_

                
                ##  dof of posterior distribution of cluster covariance

                smp_mk_prms[oo.ky_p_Sg][:, :, iter] = Sg[0:M].T

                #tt6 = _tm.time()
                ##############
                ##############  SAMPLE SPATIAL VARIANCE
                ##############
                if use_spc:
                    #  M x q2ss x Nupx  
                    #  f        M x 1    x 1
                    #  iq2xrr   1 x q2ss x 1
                    #  uxrr     1 x 1    x Nupx

                    if use_omp:  #ux variable held fixed
                        M_times_N_q2_intgrls_raw(f, ux, iq2x, dSilenceX, px, q2_exp_px, M, oo.q2ss, oo.Nupx, oo.nThrds)
                    else:
                        frr       = f.reshape((M, 1, 1))
                        q2_intgrd = _N.exp(-0.5*(frr - uxrr)*(frr-uxrr) * iq2xrr)
                        q2_exp_px = _N.sum(q2_intgrd*pxrr, axis=2) * dSilenceX

                    # function of q2

                    s = -((l0r*oo.dt)/sqrt_2pi_q2x)*q2_exp_px
                else:
                    s = _N.zeros((oo.q2ss, M))
                #  B' / (a' - 1) = MODE   #keep mode the same after discount
                #  B' = MODE * (a' - 1)
                if (epc > 0) and oo.adapt:
                    _md_nd= _q2_B[q2_a_Init] / (_q2_a[q2_a_Init] + 1)
                    _Dq2_a[q2_a_Init] = _q2_a[q2_a_Init] * _N.exp(-dt/tau_q2)
                    _Dq2_B[q2_a_Init] = _Dq2_a[q2_a_Init] / _md_nd

                    if b_q2_a_is_m1:    #  uninitialized cluster
                        _Dq2_a[q2_a_is_m1] = _q2_a[q2_a_is_m1]
                        _Dq2_B[q2_a_is_m1] = _q2_B[q2_a_is_m1]
                else:
                    _Dq2_a = _q2_a
                    _Dq2_B = _q2_B

                SL_Bs = _N.empty(M)
                SL_as = _N.empty(M)
                #tt7 = _tm.time()
                for m in xrange(M):
                    if clstsz[m] > 0:
                        #sts = l_sts[m]
                        sts = v_sts[cls_str_ind[m]:cls_str_ind[m+1]]
                        xI = (xt0t1[sts-t0]-f[m])*(xt0t1[sts-t0]-f[m])*0.5
                        SL_a = 0.5*clstsz[m] - 1   #  spiking part of likelihood
                        SL_B = _N.sum(xI)  #  spiking part of likelihood
                        SL_Bs[m] = SL_B
                        SL_as[m] = SL_a
                        #  spiking prior x prior
                        #sLLkPr[m] = -(SL_a + 1)*lq2x - iq2x*SL_B
                        #sLLkPr[m] = -(_q2_a[m] + SL_a + 2)*lq2x - iq2x*(_q2_B[m] + SL_B)
                        sLLkPr[m] = -(_q2_a[m] + SL_a - 1)*lq2x - iq2x*(_q2_B[m] + SL_B)
                    else:
                        sLLkPr[m] = -(_q2_a[m] + 1)*lq2x - iq2x*_q2_B[m]

                #tt8 = _tm.time()
                #q2_a_, q2_B_ = mltpl_ig_prmsUV(q2xr, sLLkPr.T, s.T, d_q2xr, q2x_m1r, clstsz, iter, mks, t0, xt0t1, gz, SL_as, SL_Bs, _q2_a, _q2_B, oo.q2_min, oo.q2_max)

                #q2[0:M] = _ss.invgamma.rvs(q2_a_ + 1, scale=q2_B_)  #  check

                #q2[0:M] = _pcs.smp_from_cdf_interp(q2xr_z, sLLkPr_z.T, s_z.T, d_q2xr_z, q2x_m1r_z)
                q2[0:M] = _pcs.smp_from_cdf_interp(q2xr, sLLkPr.T, s.T, d_q2xr, q2x_m1r)

                #tt9 = _tm.time()

                smp_sp_prms[oo.ky_p_q2, iter]   = q2[0:M]
                #smp_sp_hyps[oo.ky_h_q2_a, iter] = q2_a_
                #smp_sp_hyps[oo.ky_h_q2_B, iter] = q2_B_

                # print "timing start"
                # print (tt2-tt1)
                # print (tt3-tt2)
                # print (tt4-tt3)
                # print (tt5-tt4)
                # print (tt6-tt5)
                # print (tt7-tt6)  # slow
                # print (tt8-tt7)
                # print (tt9-tt8)
                # print "timing end"

                #  _ss.gamma.rvs.  uses k, theta  k is 1/B (B is our thing)
                iiq2 = 1./q2[0:M]
                iiq2r= iiq2.reshape((M, 1))
                iiq2rr= iiq2.reshape((M, 1, 1))

                fr = f[0:M].reshape((M, 1))
                l0_intgrd   = _N.exp(-0.5*(fr - ux)*(fr-ux) * iiq2r)  

                l0_exp_px   = _N.sum(l0_intgrd*pxr, axis=1) * dSilenceX
                BL  = ((oo.dt)/_N.sqrt(twpi*q2[0:M]))*l0_exp_px    #  dim M

                if (epc > 0) and oo.adapt:
                    _md_nd= _l0_a[l0_a_Init] / _l0_B[l0_a_Init]

                    _Dl0_a[l0_a_Init] = _l0_a[l0_a_Init] * _N.exp(-dt/tau_l0)
                    _Dl0_B[l0_a_Init] = _Dl0_a[l0_a_Init] / _md_nd

                    if b_l0_a_is0:    #  uninitialized cluster
                        _Dl0_a[l0_a_is0] = _l0_a[l0_a_is0]
                        _Dl0_B[l0_a_is0] = _l0_B[l0_a_is0]
                    #  do something special for when _l0_a is 0.  this causes _Dl0_B to be nan.
                else:
                    _Dl0_a = _l0_a
                    _Dl0_B = _l0_B

                aL  = clstsz + 1
                l0_a_ = aL + _Dl0_a - 1   #  if no prior, l0_a_ = aL
                l0_B_ = BL + _Dl0_B
                
                l0 = _ss.gamma.rvs(l0_a_, scale=(1/l0_B_))  #  check

                smp_sp_prms[oo.ky_p_l0, iter] = l0[0:M]
                smp_sp_hyps[oo.ky_h_l0_a, iter] = l0_a_
                smp_sp_hyps[oo.ky_h_l0_B, iter] = l0_B_
                    
                #  nz clstr.  fixed width
                if oo.nzclstr:
                    nz_l0_exp_px   = _N.sum(nz_l0_intgrd*px) * dSilenceX
                    BL  = (oo.dt/_N.sqrt(twpi*q2[Mwowonz-1]))*nz_l0_exp_px

                    minds = len(_N.where(gz[iter, :, Mwowonz-1] == 1)[0])
                    l0_a_ = minds + 1 + _nz_l0_a - 1
                    l0_B_ = BL    + _nz_l0_B

                    l0[Mwowonz-1]  = _ss.gamma.rvs(l0_a_, scale=(1/l0_B_)) 
                    smp_nz_l0[iter]       = l0[Mwowonz-1]
                    smp_nz_hyps[0, iter]  = l0_a_
                    smp_nz_hyps[1, iter]  = l0_B_

            ttB = _tm.time()
            print (ttB-ttA)


            gAMxMu.finish_epoch2(oo, nSpks, epc, ITERS, gz, l0, f, q2, u, Sg, _f_u, _f_q2, _q2_a, _q2_B, _l0_a, _l0_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI, smp_sp_hyps, smp_sp_prms, smp_mk_hyps, smp_mk_prms, freeClstr, M, K, priors)

            #  MAP of nzclstr
            if oo.nzclstr:
                frm = 1000
                _nz_l0_a              = _N.median(smp_nz_hyps[0, frm:])
                _nz_l0_B              = _N.median(smp_nz_hyps[1, frm:])
            #  SAMPLED parameters and hyps
            pcklme["smp_sp_hyps"] = smp_sp_hyps  # 
            pcklme["smp_mk_hyps"] = smp_mk_hyps
            pcklme["smp_sp_prms"] = smp_sp_prms
            pcklme["smp_mk_prms"] = smp_mk_prms
            #  MODE values - point estimates for params for decode
            pcklme["sp_prmPstMd"] = oo.sp_prmPstMd
            pcklme["mk_prmPstMd"] = oo.mk_prmPstMd
            #  _u_u etc etc.  are passed directly to finish_epoch()
            pcklme["intvs"]       = oo.intvs
            pcklme["occ"]         = gz
            pcklme["nz_pth"]         = nz_pth
            pcklme["M"]           = M
            pcklme["Mwowonz"]           = Mwowonz
            if Mwowonz > M:  # or oo.nzclstr == True
                pcklme["nz_fs"]       = f[M]
                pcklme["nz_q2"]       = q2[M]
                pcklme["nz_Sg"]       = Sg[M]
                pcklme["nz_u"]        = u[M]
                pcklme["smp_nz_l0"]  = smp_nz_l0
                pcklme["smp_nz_hyps"]= smp_nz_hyps
                
            dmp = open(resFN("posteriors_%d.dmp" % epc, dir=oo.outdir), "wb")
            pickle.dump(pcklme, dmp, -1)
            dmp.close()

######  Hi Eric
