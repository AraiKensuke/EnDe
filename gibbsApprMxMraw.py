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
from posteriorUtil import MAPvalues2
from filter import gauKer
import gibbsApprMxMutil as gAMxMu
from par_intgrls  import M_times_N_f_intgrls_raw

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

    ##  Int p(x) exp[-0.5 (fc - x)^2 / (2 sig_c)^2]   
    #   do this for fss values ofss values of fc.  Riemann sum at Nupx points
    Nupx = 200      #   # points to sample position with  (uniform lam(x)p(x))
    fss = 30       #  sampling at various values of f
    ##  Int p(x) exp[-0.5 (fc - x)^2 / (2 sig_c)^2]   
    #   do this for q2ss values ofss values of sig_c.  Riemann sum at Nupx point
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
    
    diffusePerMin = 0.05    #  diffusion of certainty

    nz_q2               = 500
    nz_f                = 0

    def __init__(self, outdir, fn, intvfn, xLo=0, xHi=3, seed=1041, adapt=True, nzclstr=False, t_hlf_l0_mins=None, t_hlf_q2_mins=None):
        oo     = self
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
        oo.t_hlf_l0 = int(1000*60*t_hlf_l0_mins) if (t_hlf_l0_mins is not None) else oo.t_hlf_l0
        oo.t_hlf_q2 = int(1000*60*t_hlf_q2_mins) if (t_hlf_q2_mins is not None) else oo.t_hlf_q2

    def gibbs(self, ITERS, K, ep1=0, ep2=None, savePosterior=True, gtdiffusion=False, Mdbg=None, doSepHash=True, use_spc=True, nz_pth=0., ignoresilence=False):
        """
        gtdiffusion:  use ground truth center of place field in calculating variance of center.  Meaning of diffPerMin different
        """
        print "RAW gibbs"
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
        q2x    = _N.exp(_N.linspace(_N.log(1e-7), _N.log(100), oo.q2ss))  #  5 orders of
        d_q2x  = _N.diff(q2x)
        q2x_m1 = _N.array(q2x[0:-1])
        lq2x    = _N.log(q2x)
        iq2x    = 1./q2x
        q2xr     = q2x.reshape((oo.q2ss, 1))
        iq2xr     = 1./q2xr
        sqrt_2pi_q2x   = _N.sqrt(twpi*q2x)
        l_sqrt_2pi_q2x = _N.log(sqrt_2pi_q2x)

        freeClstr = None
        gk     = gauKer(100) # 0.1s  smoothing of motion
        gk     /= _N.sum(gk)
        xf     = _N.convolve(oo.dat[:, 0], gk, mode="same")
        oo.dat[:, 0] = xf + nz_pth*_N.random.randn(len(oo.dat[:, 0]))
        x      = oo.dat[:, 0]
        mks    = oo.dat[:, 2:]

        f_q2_rate = (oo.diffusePerMin**2)/60000.  #  unit of minutes  

        ######################################  PRECOMPUTED

        tau_l0 = oo.t_hlf_l0/_N.log(2)
        tau_q2 = oo.t_hlf_q2/_N.log(2)

        for epc in xrange(ep1, ep2):
            t0 = oo.intvs[epc]
            t1 = oo.intvs[epc+1]
            if epc > 0:
                tm1= oo.intvs[epc-1]
                #  0  10 30     20 - 5  = 15    0.5*((10+30) - (10+0)) = 15
                dt = 0.5*((t1+t0) - (t0+tm1))

            dt = (t1-t0)*0.5
            xt0t1 = _N.array(x[t0:t1])
            posbins  = _N.linspace(oo.xLo, oo.xHi, oo.Nupx+1)
            #  _N.sum(px)*(xbns[1]-xbns[0]) = 1
            px, xbns = _N.histogram(xt0t1, bins=posbins, normed=True)   

            Asts    = _N.where(oo.dat[t0:t1, 1] == 1)[0]   #  based at 0
            Ants    = _N.where(oo.dat[t0:t1, 1] == 0)[0]

            if epc == ep1:   ###  initialize
                labS, labH, lab, flatlabels, M, MF, hashthresh, nHSclusters = gAMxMu.initClusters(oo, K, x, mks, t0, t1, Asts, doSepHash=doSepHash, xLo=oo.xLo, xHi=oo.xHi)  # nHSclusters  is # of clusters in hash and signal 

                signalClusters = _N.where(flatlabels < nHSclusters[0])[0]
                Mwowonz = M if not oo.nzclstr else M + 1
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


                U   = _N.empty(M)
                FQ2 = _N.empty(M)
                _fxs0 = _N.tile(_N.linspace(0, 1, oo.fss), M).reshape(M, oo.fss)

                f_exp_px = _N.empty((M, oo.fss))
                q2_exp_px= _N.empty((M, oo.q2ss))

                if oo.nzclstr:
                    smp_nz_l0     = _N.zeros(ITERS)
                    smp_nz_hyps = _N.zeros((2, ITERS))

                #  list of freeClstrs
                freeClstr = _N.empty(M, dtype=_N.bool)   #  Actual cluster
                freeClstr[:] = False

                l0, f, q2, u, Sg = gAMxMu.declare_params(M, K, nzclstr=oo.nzclstr)   #  nzclstr not INITED, sized to include noise cluster if needed
                _l0_a, _l0_B, _f_u, _f_q2, _q2_a, _q2_B, _u_u, _u_Sg, _Sg_nu, \
                    _Sg_PSI = gAMxMu.declare_prior_hyp_params(M, MF, K, x, mks, Asts, t0)    #  hyper params don't include noise cluster
                gAMxMu.init_params_hyps(oo, M, MF, K, l0, f, q2, u, Sg, Asts, t0, x, mks, flatlabels, nzclstr=oo.nzclstr, signalClusters=signalClusters)

                ######  the hyperparameters for f, q2, u, Sg, l0 during Gibbs
                #  f_u_, f_q2_, q2_a_, q2_B_, u_u_, u_Sg_, Sg_nu, Sg_PSI_, l0_a_, l0_B_

                if oo.nzclstr:
                    nz_l0_intgrd   = _N.exp(-0.5*ux*ux / q2[Mwowonz-1])
                    _nz_l0_a       = 0.001
                    _nz_l0_B       = 0.1

            NSexp   = t1-t0    #  length of position data  #  # of no spike positions to sum
            xt0t1 = _N.array(x[t0:t1])

            nSpks    = len(Asts)
            gz   = _N.zeros((ITERS, nSpks, Mwowonz), dtype=_N.bool)
            oo.gz=gz
            print "spikes %d" % nSpks

            #dSilenceX = (NSexp/float(oo.Nupx))*(oo.xHi-oo.xLo)
            dSilenceX = NSexp*(xbns[1]-xbns[0])  # dx of histogram

            xAS  = x[Asts + t0]   #  position @ spikes
            mAS  = mks[Asts + t0]   #  position @ spikes
            xASr = xAS.reshape((1, nSpks))
            mASr = mAS.reshape((nSpks, 1, K))
            econt = _N.empty((Mwowonz, nSpks))
            rat   = _N.zeros((Mwowonz+1, nSpks))

            qdrMKS = _N.empty((Mwowonz, nSpks))
            ################################  GIBBS ITERS ITERS ITERS

            #  linalgerror
            #_iSg_Mu = _N.einsum("mjk,mk->mj", _N.linalg.inv(_u_Sg), _u_u)

            clusSz = _N.zeros(M, dtype=_N.int)
            clstsz = _N.zeros(M, dtype=_N.int)

            _iu_Sg = _N.array(_u_Sg)

            for m in xrange(M):
                _iu_Sg[m] = _N.linalg.inv(_u_Sg[m])

            for iter in xrange(ITERS):
                ttA = _tm.time()
                iSg = _N.linalg.inv(Sg)
                tt1 = _tm.time()
                if (iter % 5) == 0:    
                    print "iter  %d" % iter

                gAMxMu.stochasticAssignment(oo, iter, M, Mwowonz, K, l0, f, q2, u, Sg, _f_u, _u_u, Asts, t0, mASr, xASr, rat, econt, gz, qdrMKS, freeClstr, hashthresh, ((epc > 0) and (iter == 0)))
                ttSA = _tm.time()

        #         ###############  FOR EACH CLUSTER

                l_sts = []
                for m in xrange(M):
                    minds = _N.where(gz[iter, :, m] == 1)[0]
                    sts  = Asts[minds] + t0
                    nSpksM   = len(sts)
                    clusSz[m] = nSpksM
                    clstsz[m] = len(sts)
                    l_sts.append(sts)

                    ###############
                    ###############  CONDITIONAL l0
                    ###############

                    #  _ss.gamma.rvs.  uses k, theta  k is 1/B (B is our thing)
                    iiq2 = 1./q2[m]
                    # xI = (xt0t1-f[m])*(xt0t1-f[m])*0.5*iiq2
                    # BL  = (oo.dt/_N.sqrt(twpi*q2[m]))*_N.sum(_N.exp(-xI))

                    #  l0_intgrd   (M x Nupx)
                    l0_intgrd   = _N.exp(-0.5*(f[m] - ux)*(f[m]-ux) * iiq2)  
                    l0_exp_px   = _N.sum(l0_intgrd*px) * dSilenceX
                    BL  = (oo.dt/_N.sqrt(twpi*q2[m]))*l0_exp_px


                    #    #  keep mode same after discount
                    #  a' - 1 / B' = MODE  # mode is a - 1 / B
                    #  B' = (a' - 1) / MODE
                    #  discount a
                    #if (epc > 0) and oo.adapt and (_l0_a[m] > 1.1):
                    if (epc > 0) and oo.adapt:
                        _md_nd= _l0_a[m] / _l0_B[m]
                        _Dl0_a = _l0_a[m] * _N.exp(-dt/tau_l0)
                        _Dl0_B = _Dl0_a / _md_nd
                    else:
                        _Dl0_a = _l0_a[m]
                        _Dl0_B = _l0_B[m]

                    #  a'/B' = a/B
                    #  B' = (B/a)a'
                    aL  = nSpksM
                    l0_a_ = aL + _Dl0_a
                    l0_B_ = BL + _Dl0_B


                    # print "------------------"
                    # print "liklhd  BL   %(B).3f     f   %(f).3f   a %(a)d    B/a  %(ba).3f" % {"B" : BL, "f" : f[m], "ba" : (aL/ BL), "a" : aL}
                    # print "prior   BL   %(B).3f     f   %(f).3f   a %(a)d    B/a  %(ba).3f" % {"B" : l0_B_, "f" : f[m], "ba" : (l0_a_/ l0_B_), "a" : l0_a_}
                    # print (len(xt0t1)*oo.dt)
                    # print "******************"
                    
                    #print "%(1).5f   %(2).5f" % {"1" : l0_a_, "2" : l0_B_}

                    try:
                        l0[m] = _ss.gamma.rvs(l0_a_, scale=(1/l0_B_))  #  check
                    except ValueError:
                        print "fail"
                        print "M:        %d" % M
                        print "_l0_a[m]  %.3f" % _l0_a[m]
                        print "_l0_B[m]  %.3f" % _l0_B[m]
                        print "l0_a_     %.3f" % l0_a_
                        print "l0_B_     %.3f" % l0_B_
                        print "aL        %.3f" % aL
                        print "BL        %.3f" % BL
                        print "_Dl0_a    %.3f" % _Dl0_a
                        print "_Dl0_B    %.3f" % _Dl0_B
                        raise

                    ###  l0 / _N.sqrt(twpi*q2) is f*dt used in createData2
                    smp_sp_prms[oo.ky_p_l0, iter, m] = l0[m]
                    smp_sp_hyps[oo.ky_h_l0_a, iter, m] = l0_a_
                    smp_sp_hyps[oo.ky_h_l0_B, iter, m] = l0_B_
                    mcs = _N.empty((M, K))   # cluster sample means

                    if nSpksM >= K:
                        u_Sg_ = _N.linalg.inv(_iu_Sg[m] + nSpksM*iSg[m])
                        clstx    = mks[sts]

                        mcs[m]       = _N.mean(clstx, axis=0)
                        #u_u_ = _N.einsum("jk,k->j", u_Sg_, _N.dot(_N.linalg.inv(_u_Sg[m]), _u_u[m]) + nSpksM*_N.dot(iSg[m], mcs[m]))
                        u_u_ = _N.einsum("jk,k->j", u_Sg_, _N.dot(_iu_Sg[m], _u_u[m]) + nSpksM*_N.dot(iSg[m], mcs[m]))
                        # hyp
                        ########  POSITION
                        ##  mean of posterior distribution of cluster means
                        #  sigma^2 and mu are the current Gibbs-sampled values

                        ##  mean of posterior distribution of cluster means
                    else:
                        u_Sg_ = _N.array(_u_Sg[m])

                        u_u_ = _N.array(_u_u[m])
                    u[m] = _N.random.multivariate_normal(u_u_, u_Sg_)

                    smp_mk_prms[oo.ky_p_u][:, iter, m] = u[m]
                    smp_mk_hyps[oo.ky_h_u_u][:, iter, m] = u_u_
                    smp_mk_hyps[oo.ky_h_u_Sg][:, :, iter, m] = u_Sg_

                #####################
                #####################   f
                #####################
                if (epc > 0) and oo.adapt:
                    q2pr = _f_q2 + f_q2_rate * dt
                else:
                    q2pr = _f_q2
                for m in xrange(M):
                    sts = l_sts[m]
                    if clstsz[m] > 0:
                        fs  = (1./clstsz[m])*_N.sum(xt0t1[sts-t0])
                        fq2 = q2[m]/clstsz[m]
                        U[m]   = (fs*q2pr[m] + _f_u[m]*fq2) / (q2pr[m] + fq2)
                        FQ2[m] = (q2pr[m]*fq2) / (q2pr[m] + fq2)
                    else:
                        U[m]   = _f_u[m]
                        FQ2[m] = q2pr[m]


                FQ    = _N.sqrt(FQ2)
                Ur    = U.reshape((M, 1))
                FQr   = FQ.reshape((M, 1))
                FQ2r  = FQ2.reshape((M, 1))

                if use_spc:
                    fxs  = _N.copy(_fxs0)
                    fxs *= (FQr*30)
                    fxs -= (FQr*15)
                    fxs += Ur

                    tt1     = _tm.time()
                    iiq2_v = 1./q2
                    M_times_N_f_intgrls_raw(fxs, ux, iiq2_v, dSilenceX, px, f_exp_px, M, oo.fss, oo.Nupx, 8)
                    tt2     = _tm.time()
                    l0r = l0[0:M].reshape((M, 1))  # working with sig clus
                    q2r = q2[0:M].reshape((M, 1))
                     #  s   is (M x fss)
                    s = -(l0r*oo.dt/_N.sqrt(twpi*q2r)) * f_exp_px  #  a function of x
                else:
                    s = _N.zeros(M)

                #  U, FQ2 is   dim(M)   
                #  fxs is M x fss
                funcf   = -0.5*((fxs-Ur)*(fxs-Ur))/FQ2r + s
                maxes   = _N.max(funcf, axis=1)   
                maxesr  = maxes.reshape((M, 1))
                funcf   -= maxesr
                condPosF= _N.exp(funcf)   #  condPosF is M x fss
                ttB = _tm.time()
                # print (ttB-ttA)
                # print (tt2-tt1)
                # print "time vector"

                #  fxs   M x fss
                #  fxs            M x fss
                #  condPosF       M x fss
                norm    = 1./_N.sum(condPosF, axis=1)  #  sz M
                f_u_    = norm*_N.sum(fxs*condPosF, axis=1)  #  sz M
                f_u_r   = f_u_.reshape((M, 1))
                f_q2_   = norm*_N.sum(condPosF*(fxs-f_u_r)*(fxs-f_u_r), axis=1)
                #f       = _N.sqrt(f_q2_)*_N.random.randn() + f_u_
                f       = f_u_
                smp_sp_prms[oo.ky_p_f, iter] = f
                smp_sp_hyps[oo.ky_h_f_u, iter] = f_u_
                smp_sp_hyps[oo.ky_h_f_q2, iter] = f_q2_

                for m in xrange(M):
                    #ttc1g = _tm.time()
                    #############  VARIANCE, COVARIANCE
                    if nSpksM >= K:
                        ##  dof of posterior distribution of cluster covariance
                        Sg_nu_ = _Sg_nu[m, 0] + nSpksM
                        ##  dof of posterior distribution of cluster covariance
                        ur = u[m].reshape((1, K))
                        Sg_PSI_ = _Sg_PSI[m] + _N.dot((clstx - ur).T, (clstx-ur))
                        Sg[m] = s_u.sample_invwishart(Sg_PSI_, Sg_nu_)
                    else:
                        Sg_nu_ = _Sg_nu[m, 0] 
                        ##  dof of posterior distribution of cluster covariance
                        ur = u[m].reshape((1, K))
                        Sg_PSI_ = _Sg_PSI[m]
                        Sg[m] = s_u.sample_invwishart(Sg_PSI_, Sg_nu_)

                    ##############  SAMPLE COVARIANCES

                    ##  dof of posterior distribution of cluster covariance

                    smp_mk_prms[oo.ky_p_Sg][:, :, iter, m] = Sg[m]
                    smp_mk_hyps[oo.ky_h_Sg_nu][0, iter, m] = Sg_nu_
                    smp_mk_hyps[oo.ky_h_Sg_PSI][:, :, iter, m] = Sg_PSI_
                

            #         # ###############  CONDITIONAL q2
            #         #xI = (xt0t1-f)*(xt0t1-f)*0.5*iq2xr

            #         if use_spc:
            #             q2_intgrd = _N.exp(-0.5*(f[m] - ux)*(f[m]-ux) * iq2xr)
            #             q2_exp_px = _N.sum(q2_intgrd*px, axis=1) * dSilenceX

            #             # function of q2
            #             s = -((l0[m]*oo.dt)/sqrt_2pi_q2x)*q2_exp_px
            #         else:
            #             s = 0
            #         #  B' / (a' - 1) = MODE   #keep mode the same after discount
            #         #  B' = MODE * (a' - 1)
            #         if (epc > 0) and oo.adapt:
            #             _md_nd= _q2_B[m] / (_q2_a[m] + 1)
            #             _Dq2_a = _q2_a[m] * _N.exp(-dt/tau_q2)
            #             _Dq2_B = _Dq2_a / _md_nd
            #         else:
            #             _Dq2_a = _q2_a[m]
            #             _Dq2_B = _q2_B[m]

            #         if nSpksM > 0:
            #             ##  (1/sqrt(sg2))^S
            #             ##  (1/x)^(S/2)   = (1/x)-(a+1)
            #             ##  -S/2 = -a - 1     -a = -S/2 + 1    a = S/2-1
            #             xI = (xt0t1[sts-t0]-f[m])*(xt0t1[sts-t0]-f[m])*0.5
            #             SL_a = 0.5*nSpksM - 1   #  spiking part of likelihood
            #             SL_B = _N.sum(xI)  #  spiking part of likelihood
            #             #  spiking prior x prior
            #             sLLkPr = -(_q2_a[m] + SL_a + 2)*lq2x - iq2x*(_q2_B[m] + SL_B)
            #         else:
            #             sLLkPr = -(_q2_a[m] + 1)*lq2x - iq2x*_q2_B[m]


            #         sat = sLLkPr + s
            #         sat -= _N.max(sat)
            #         condPos = _N.exp(sat)
            #         q2_a_, q2_B_ = ig_prmsUV(q2x, sLLkPr, s, d_q2x, q2x_m1, ITER=1, nSpksM=nSpksM, clstr=m, l0=l0[m])

            #         # sat = sLLkPr + s
            #         # sat -= _N.max(sat)
            #         # condPos = _N.exp(sat)
            #         # q2_a_, q2_B_ = ig_prmsUV(q2x, condPos, d_q2x, q2x_m1, ITER=1)
            #         q2[m] = _ss.invgamma.rvs(q2_a_ + 1, scale=q2_B_)  #  check


            #         #q2[m] = 1.1**2

            #         #print ((1./nSpks)*_N.sum((xt0t1[sts]-f)*(xt0t1[sts]-f)))

            #         if q2[m] < 0:
            #             print "********  q2[%(m)d] = %(q2).3f" % {"m" : m, "q2" : q2[m]}

            #         smp_sp_prms[oo.ky_p_q2, iter, m]   = q2[m]
            #         smp_sp_hyps[oo.ky_h_q2_a, iter, m] = q2_a_
            #         smp_sp_hyps[oo.ky_h_q2_B, iter, m] = q2_B_
                    
            #         if q2[m] < 0:
            #             print "^^^^^^^^  q2[%(m)d] = %(q2).3f" % {"m" : m, "q2" : q2[m]}
            #             print q2[m]
            #             print smp_sp_prms[oo.ky_p_q2, 0:iter+1, m]
            #         iiq2 = 1./q2[m]

            #         #ttc1h = _tm.time()
                    

            #     #  nz clstr.  fixed width
            #     if oo.nzclstr:
            #         nz_l0_exp_px   = _N.sum(nz_l0_intgrd*px) * dSilenceX
            #         BL  = (oo.dt/_N.sqrt(twpi*q2[Mwowonz-1]))*nz_l0_exp_px

            #         minds = len(_N.where(gz[iter, :, Mwowonz-1] == 1)[0])
            #         l0_a_ = minds + _nz_l0_a
            #         l0_B_ = BL    + _nz_l0_B

            #         l0[Mwowonz-1]  = _ss.gamma.rvs(l0_a_, scale=(1/l0_B_)) 
            #         smp_nz_l0[iter]       = l0[Mwowonz-1]
            #         smp_nz_hyps[0, iter]  = l0_a_
            #         smp_nz_hyps[1, iter]  = l0_B_
                    

            # ###  THIS LEVEL:  Finished Gibbs iters for epoch
            # gAMxMu.finish_epoch(oo, nSpks, epc, ITERS, gz, l0, f, q2, u, Sg, _f_u, _f_q2, _q2_a, _q2_B, _l0_a, _l0_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI, smp_sp_hyps, smp_sp_prms, smp_mk_hyps, smp_mk_prms, freeClstr, M, K)
            # #  MAP of nzclstr
            # if oo.nzclstr:
            #     frm = int(0.7*ITERS)
            #     _nz_l0_a              = _N.median(smp_nz_hyps[0, frm:])
            #     _nz_l0_B              = _N.median(smp_nz_hyps[1, frm:])
            # pcklme["smp_sp_hyps"] = smp_sp_hyps
            # pcklme["smp_mk_hyps"] = smp_mk_hyps
            # pcklme["smp_sp_prms"] = smp_sp_prms
            # pcklme["smp_mk_prms"] = smp_mk_prms
            # pcklme["sp_prmPstMd"] = oo.sp_prmPstMd
            # pcklme["mk_prmPstMd"] = oo.mk_prmPstMd
            # pcklme["intvs"]       = oo.intvs
            # pcklme["occ"]         = gz
            # pcklme["nz_pth"]         = nz_pth
            # pcklme["M"]           = M
            # pcklme["Mwowonz"]           = Mwowonz
            # if Mwowonz > M:  # or oo.nzclstr == True
            #     pcklme["smp_nz_l0"]  = smp_nz_l0
            #     pcklme["smp_nz_hyps"]= smp_nz_hyps
                
            # dmp = open(resFN("posteriors_%d.dmp" % epc, dir=oo.outdir), "wb")
            # pickle.dump(pcklme, dmp, -1)
            # dmp.close()

