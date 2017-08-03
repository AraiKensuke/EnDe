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
import gibbsApprMxMutil as gAMxMu

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
    q2ss = 300      #  sampling at various values of q2

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
        _N.random.seed(1041523)
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
        q2x    = _N.exp(_N.linspace(_N.log(0.0001), _N.log(1000), oo.q2ss))  #  5 orders of
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

        for epc in xrange(ep1, ep2):
            print "^^^^^^^^^^^^^^^^^^^^^^^^    epoch %d" % epc

            t0 = oo.intvs[epc]
            t1 = oo.intvs[epc+1]

            xt0t1 = _N.array(x[t0:t1])
            posbins  = _N.linspace(oo.xLo, oo.xHi, oo.Nupx+1)
            px, xbns = _N.histogram(xt0t1, bins=posbins, normed=True)

            Asts    = _N.where(oo.dat[t0:t1, 1] == 1)[0]   #  based at 0
            Ants    = _N.where(oo.dat[t0:t1, 1] == 0)[0]

            if epc == ep1:   ###  initialize
                labS, labH, lab, flatlabels, M, MF, hashthresh = gAMxMu.initClusters(oo, K, x, mks, t0, t1, Asts, doSepHash=doSepHash)

                #######   containers for GIBBS samples iterations
                smp_sp_prms = _N.zeros((3, ITERS, M))  
                smp_mk_prms = [_N.zeros((K, ITERS, M)), 
                               _N.zeros((K, K, ITERS, M))]
                smp_sp_hyps = _N.zeros((6, ITERS, M))
                smp_mk_hyps = [_N.zeros((K, ITERS, M)), 
                               _N.zeros((K, K, ITERS, M)),
                               _N.zeros((1, ITERS, M)), 
                               _N.zeros((K, K, ITERS, M))]

                #  list of freeClstrs
                freeClstr = _N.empty(M, dtype=_N.bool)   #  Actual cluster
                freeClstr[:] = False

                l0, f, q2, u, Sg = gAMxMu.declare_params(M, K)
                _l0_a, _l0_B, _f_u, _f_q2, _q2_a, _q2_B, _u_u, _u_Sg, _Sg_nu, \
                    _Sg_PSI = gAMxMu.declare_prior_hyp_params(M, MF, K, x, mks, Asts, t0)
                gAMxMu.init_params_hyps(oo, M, MF, K, l0, f, q2, u, Sg, Asts, t0, x, mks, flatlabels)
                ######  the hyperparameters for f, q2, u, Sg, l0 during Gibbs
                #  f_u_, f_q2_, q2_a_, q2_B_, u_u_, u_Sg_, Sg_nu, Sg_PSI_, l0_a_, l0_B_

            NSexp   = t1-t0    #  length of position data  #  # of no spike positions to sum
            xt0t1 = _N.array(x[t0:t1])

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

            for iter in xrange(ITERS):
                if iter == 0:
                    print _N.random.rand()
                iSg = _N.linalg.inv(Sg)
                tt1 = _tm.time()
                #if (iter % 100) == 0:    print "iter  %d" % iter
                print "iter  %d" % iter

                gAMxMu.stochasticAssignment(oo, iter, hashthresh, x, mks, M, K, l0, f, q2, u, Sg, _f_u, _u_u, Asts, t0, mASr, xASr, rat, econt, gz, qdrMKS, freeClstr, ((epc > 1) and (iter == 0)), usenzclstr)
                ################  FOR EACH CLUSTER

                for m in xrange(M):
                    ttc1 = _tm.time()
                    iiq2 = 1./q2[m]

                    minds = _N.where(gz[iter, :, m] == 1)[0]
                    sts  = Asts[minds] + t0
                    nSpksM   = len(sts)
                    clusSz[m] = nSpksM

                    mcs = _N.empty((M, K))   # cluster sample means

                    ###############  mark mean
                    if nSpksM > 0:
                        #try:
                        iiSg = _N.linalg.inv(Sg[m])
                        #u_Sg_ = _N.linalg.inv(_N.linalg.inv(_u_Sg[m]) + nSpksM*iSg[m])
                        u_Sg_ = _N.linalg.inv(_N.linalg.inv(_u_Sg[m]) + nSpksM*iiSg)
                        clstx    = mks[sts]

                        mcs[m]       = _N.mean(clstx, axis=0)
                        #u_u_ = _N.einsum("jk,k->j", u_Sg_, _N.dot(_N.linalg.inv(_u_Sg[m]), _u_u[m]) + nSpksM*_N.dot(iSg[m], mcs[m]))
                        u_u_ = _N.einsum("jk,k->j", u_Sg_, _N.dot(_N.linalg.inv(_u_Sg[m]), _u_u[m]) + nSpksM*_N.dot(iiSg, mcs[m]))
                        u[m] = _N.random.multivariate_normal(u_u_, u_Sg_)

                        # hyp
                        ########  POSITION
                        ##  mean of posterior distribution of cluster means
                        #  sigma^2 and mu are the current Gibbs-sampled values

                        ##  mean of posterior distribution of cluster means

                    smp_mk_prms[oo.ky_p_u][:, iter, m] = u[m]
                    smp_mk_hyps[oo.ky_h_u_u][:, iter, m] = u_u_
                    smp_mk_hyps[oo.ky_h_u_Sg][:, :, iter, m] = u_Sg_

                    ###############  CONDITIONAL f
                    q2pr = _f_q2[m] if (_f_q2[m] > q2rate) else q2rate
                    if nSpksM > 0:  #  spiking portion likelihood x prior
                        fs  = (1./nSpksM)*_N.sum(xt0t1[sts-t0])
                        fq2 = q2[m]/nSpksM
                        U   = (fs*q2pr + _f_u[m]*fq2) / (q2pr + fq2)
                        FQ2 = (q2pr*fq2) / (q2pr + fq2)
                    else:
                        U   = _f_u[m]
                        FQ2 = q2pr
                    FQ    = _N.sqrt(FQ2)
                    fx    = _N.linspace(U - FQ*60, U + FQ*60, oo.fss)
                    fxr     = fx.reshape((oo.fss, 1))
                    fxrux = -0.5*(fxr-ux)*(fxr-ux)
                    #f_intgrd  = _N.exp((fxrux*iiq2))   #  integrand
                    # f_intgrd  = _N.exp((fxrux/q2[m]))   #  integrand
                    # f_exp_px = _N.sum(f_intgrd*px, axis=1) * dSilenceX
                    s = 0#-(l0[m]*oo.dt/_N.sqrt(twpi*q2[m])) * f_exp_px  #  a function of x

                    funcf   = -0.5*((fx-U)*(fx-U))/FQ2 + s
                    funcf   -= _N.max(funcf)
                    condPosF= _N.exp(funcf)

                    norm    = 1./_N.sum(condPosF)
                    f_u_    = norm*_N.sum(fx*condPosF)
                    f_q2_   = norm*_N.sum(condPosF*(fx-f_u_)*(fx-f_u_))
                    f[m]    = _N.sqrt(f_q2_)*_N.random.randn() + f_u_
                    smp_sp_prms[oo.ky_p_f, iter, m] = f[m]
                    smp_sp_hyps[oo.ky_h_f_u, iter, m] = f_u_
                    smp_sp_hyps[oo.ky_h_f_q2, iter, m] = f_q2_

                    #ttc1g = _tm.time()
                    #############  conditinoal COVARIANCE
                    if nSpksM >= K:
                        ##  dof of posterior distribution of cluster covariance
                        Sg_nu_ = _Sg_nu[m, 0] + nSpksM
                        ##  dof of posterior distribution of cluster covariance
                        ur = u[m].reshape((1, K))
                        Sg_PSI_ = _Sg_PSI[m] + _N.dot((clstx - ur).T, (clstx-ur))
                        Sg[m] = s_u.sample_invwishart(Sg_PSI_, Sg_nu_)
                    ##  dof of posterior distribution of cluster covariance

                    smp_mk_prms[oo.ky_p_Sg][:, :, iter, m] = Sg[m]
                    smp_mk_hyps[oo.ky_h_Sg_nu][0, iter, m] = Sg_nu_
                    smp_mk_hyps[oo.ky_h_Sg_PSI][:, :, iter, m] = Sg_PSI_

                    # ###############  CONDITIONAL q2
                    #xI = (xt0t1-f)*(xt0t1-f)*0.5*iq2xr
                    # q2_intgrd   = _N.exp(-0.5*(f[m] - ux)*(f[m]-ux) * iq2xr)  
                    # q2_exp_px   = _N.sum(q2_intgrd*px, axis=1) * dSilenceX
                    #s = -((l0[m]*oo.dt)/sqrt_2pi_q2x)*q2_exp_px     #  function of q2

                    _Dq2_a = _q2_a[m]# if _q2_a[m] < 200 else 200
                    _Dq2_B = _q2_B[m]#(_q2_B[m]/(_q2_a[m]+1))*(_Dq2_a+1)

                    if nSpksM > 0:
                        ##  (1/sqrt(sg2))^S
                        ##  (1/x)^(S/2)   = (1/x)-(a+1)
                        ##  -S/2 = -a - 1     -a = -S/2 + 1    a = S/2-1
                        xI = (xt0t1[sts-t0]-f[m])*(xt0t1[sts-t0]-f[m])*0.5
                        SL_a = 0.5*nSpksM - 1   #  spiking part of likelihood
                        SL_B = _N.sum(xI)  #  spiking part of likelihood
                        #assert SL_B > 0.00001, "SL_B is too small   %.3e" % SL_B
                        #  spiking prior x prior
                        sLLkPr = -(_q2_a[m] + SL_a + 2)*lq2x - (_q2_B[m] + SL_B)*iq2x
                        #print "------------   here %(m)d   %(a).3f  %(B).3f" % {"m" : m, "a" : (_q2_a[m] + SL_a + 2), "B" : (_q2_B[m] + SL_B)}
                    else:
                        sLLkPr = -(_q2_a[m] + 1)*lq2x - _q2_B[m]*iq2x
                        #print "------------   or here %d" % m
                        #_plt.grid()
                        #_plt.xscale("log")

                    sat = sLLkPr# + s
                    sat -= _N.max(sat)
                    condPos = _N.exp(sat)
                    #q2_a_, q2_B_ = ig_prmsUV(q2x, sLLkPr, s, condPos, d_q2x, q2x_m1, ITER=1, nSpksM=nSpksM)
                    # _plt.figure()
                    # _plt.plot(q2x, _N.exp(sLLkPr))
                    # _plt.xscale("log")

                    q2_a_, q2_B_ = ig_prmsUV(q2x, sLLkPr, s, d_q2x, q2x_m1, ITER=1, nSpksM=nSpksM)
                    # _plt.savefig("llk%d" % m)
                    # _plt.close()

                    q2[m] = _ss.invgamma.rvs(q2_a_ + 1, scale=q2_B_)  #  check

                    #print ((1./nSpks)*_N.sum((xt0t1[sts]-f)*(xt0t1[sts]-f)))

                    if q2[m] < 0:
                        print "********  q2[%(m)d] = %(q2).3f" % {"m" : m, "q2" : q2[m]}

                    smp_sp_prms[oo.ky_p_q2, iter, m]   = q2[m]
                    smp_sp_hyps[oo.ky_h_q2_a, iter, m] = q2_a_
                    smp_sp_hyps[oo.ky_h_q2_B, iter, m] = q2_B_
                    
                    if q2[m] < 0:
                        print "^^^^^^^^  q2[%(m)d] = %(q2).3f" % {"m" : m, "q2" : q2[m]}
                        print q2[m]
                        print smp_sp_prms[oo.ky_p_q2, 0:iter+1, m]


                    iiq2 = 1./q2[m]
                    #ttc1h = _tm.time()
                    ###############  CONDITIONAL l0
                    #  _ss.gamma.rvs.  uses k, theta  k is 1/B (B is our thing)

                    # xI = (xt0t1-f)*(xt0t1-f)*0.5*iiq2
                    # BL  = (oo.dt/_N.sqrt(twpi*q2))*_N.sum(_N.exp(-xI))

                    #l0_intgrd   = _N.exp(-0.5*(f[m] - ux)*(f[m]-ux) * iiq2)  
                    l0_intgrd   = _N.exp(-0.5*(f[m] - ux)*(f[m]-ux) / q2[m])  
                    l0_exp_px   = _N.sum(l0_intgrd*px) * dSilenceX
                    BL  = (oo.dt/_N.sqrt(twpi*q2[m]))*l0_exp_px

                    _Dl0_a = _l0_a[m]# if _l0_a[m] < 400 else 400
                    _Dl0_B = _l0_B[m]#(_l0_B[m]/_l0_a[m]) * _Dl0_a
                    
                    #  a'/B' = a/B
                    #  B' = (B/a)a'
                    aL  = nSpksM
                    l0_a_ = aL + _Dl0_a
                    l0_B_ = BL + _Dl0_B

                    l0[m] = _ss.gamma.rvs(l0_a_ - 1, scale=(1/l0_B_))  #  check
                    ###  l0 / _N.sqrt(twpi*q2) is f*dt used in createData2

                    smp_sp_prms[oo.ky_p_l0, iter, m] = l0[m]

                    smp_sp_hyps[oo.ky_h_l0_a, iter, m] = l0_a_
                    smp_sp_hyps[oo.ky_h_l0_B, iter, m] = l0_B_
                    ttc2 = _tm.time()



            ###  THIS LEVEL:  Finished Gibbs iters
            gAMxMu.finish_epoch(oo, epc, ITERS, gz, l0, f, q2, u, Sg, _f_u, _f_q2, _q2_a, _q2_B, _l0_a, _l0_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI, smp_sp_hyps, smp_sp_prms, smp_mk_hyps, smp_mk_prms, freeClstr, M, K)
        ####  finished all epochs    
        pcklme["smp_sp_hyps"] = smp_sp_hyps
        pcklme["smp_mk_hyps"] = smp_mk_hyps
        pcklme["smp_sp_prms"] = oo.smp_sp_prms
        pcklme["smp_mk_prms"] = oo.smp_mk_prms
        pcklme["sp_prmPstMd"] = oo.sp_prmPstMd
        pcklme["mk_prmPstMd"] = oo.mk_prmPstMd
        pcklme["intvs"]       = oo.intvs
        pcklme["occ"]         = gz
        dmp = open(resFN("posteriors.dmp", dir=oo.outdir), "wb")
        pickle.dump(pcklme, dmp, -1)
        dmp.close()
