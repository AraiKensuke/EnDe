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
from fitutil import  emMKPOS_sep, sepHashEM

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
    Nupx = 500      #   # points to sample position with  (uniform lam(x)p(x))
    fss = 100       #  sampling at various values of f
    q2ss = 300      #  sampling at various values of q2

    intvs = None    #  
    dat   = None

    diffPerMin = 1.  #  diffusion per minute
    epochs   = None

    outdir   = None
    polyFit  = True
    
    def __init__(self, outdir, fn, intvfn):
        oo     = self
        ######################################  DATA input, define intervals
        # bFN = fn[0:-4]
        oo.outdir = outdir

        # if not os.access(bFN, os.F_OK):
        #     os.mkdir(bFN)

        oo.dat    = _N.loadtxt("%s.dat" % datFN(fn, create=False))
        oo.datprms= _N.loadtxt("%s_prms.dat" % datFN(fn, create=False))

        intvs     = _N.loadtxt("%s.dat" % datFN(intvfn, create=False))

        oo.intvs  = _N.array(intvs*oo.dat.shape[0], dtype=_N.int)
        oo.epochs    = oo.intvs.shape[0] - 1
        
        NT     = oo.dat.shape[0]

    def gibbs(self, ITERS, M, K, ep1=0, ep2=None, savePosterior=True, gtdiffusion=False):
        """
        gtdiffusion:  use ground truth center of place field in calculating variance of center.  Meaning of diffPerMin different
        """
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
        ux = _N.linspace(0, 3, oo.Nupx, endpoint=False)   # uniform x position
        q2x    = _N.exp(_N.linspace(_N.log(0.00001), _N.log(100), oo.q2ss))  #  5 orders of
        d_q2x  = _N.diff(q2x)
        q2x_m1 = _N.array(q2x[0:-1])
        lq2x    = _N.log(q2x)
        iq2x    = 1./q2x
        q2xr     = q2x.reshape((oo.q2ss, 1))
        iq2xr     = 1./q2xr
        sqrt_2pi_q2x   = _N.sqrt(twpi*q2x)
        l_sqrt_2pi_q2x = _N.log(sqrt_2pi_q2x)

        x      = oo.dat[:, 0]
        mks    = oo.dat[:, 2:]

        q2rate = oo.diffPerEpoch**2  #  unit of minutes  
        ######################################  PRECOMPUTED
        posbins  = _N.linspace(0, 3, oo.Nupx+1)

        for epc in xrange(ep1, ep2):
            print "epoch %d" % epc

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

                unonhash, hashsp, gmms = sepHashEM(_x)
                if (len(unonhash) > 0) and (len(hashsp) > 0):
                    labS, labH, clstrs = emMKPOS_sep(_x[unonhash], _x[hashsp])
                elif len(unonhash) == 0:
                    labS, labH, clstrs = emMKPOS_sep(None, _x[hashsp], TR=1)
                else:
                    labS, labH, clstrs = emMKPOS_sep(_x[unonhash], None, TR=1)

                MF     = clstrs[0] + clstrs[1]
                M = int(MF * 1.2) + 1   #  20% more clusters

                #  PRIORS
                #  priors  prefixed w/ _
                _f_u   = _N.zeros(M);    _f_q2  = _N.ones(M) #  wide
                #  inverse gamma
                _q2_a  = _N.ones(M)*1e-4;    _q2_B  = _N.ones(M)*1e-3
                #_plt.plot(q2x, q2x**(-_q2_a-1)*_N.exp(-_q2_B / q2x))
                _l0_a = _N.ones(M);     _l0_B  = _N.zeros(M)*(1/30.)

                #  
                _u_u   = _N.zeros((M, K));  
                _u_Sg = _N.tile(_N.identity(K), M).T.reshape((M, K, K))*0.1
                _Sg_nu = _N.ones((M, 1));  
                _Sg_PSI = _N.tile(_N.identity(K), M).T.reshape((M, K, K))*0.1

                #####  MODES  - find from the sampling
                oo.sp_prmPstMd = _N.zeros((oo.epochs, 3*M))   # mode of params
                oo.sp_hypPstMd  = _N.zeros((oo.epochs, (2+2+2)*M)) # hyperparam
                oo.mk_prmPstMd = [_N.zeros((oo.epochs, M, K)),
                                  _N.zeros((oo.epochs, M, K, K))]
                                  # mode of params
                oo.mk_hypPstMd  = _N.zeros((oo.epochs, (2+2+2)*M)) # hyperparam

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
                u_Sg_    = _N.zeros((M, K, K))
                Sg_nu_   = _N.zeros((M, 1))
                Sg_PSI_  = _N.zeros((M, K, K))

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

                #  now assign the cluster we've found to Gaussian mixtures
                covAll = _N.cov(tmpx.T)
                dcovMag= _N.diagonal(covAll)*0.005

                for im in xrange(M):  #if lab < 0, these marks not used for init
                    if im < MF:
                        kinds = _N.where(lab == im)[0]  #  inds
                        f[im]  = _N.mean(x[Asts[kinds]+t0], axis=0)
                        u[im]  = _N.mean(mks[Asts[kinds]+t0], axis=0)
                        q2[im] = 0.05
                        Sg[im] = _N.identity(K)*0.1
                    else:
                        f[im]  = _N.random.rand()*3
                        u[im]  = _N.random.rand(K)
                        q2[im] = 0.05
                        Sg[im] = _N.identity(K)*0.1

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

            dSilenceX = (NSexp/float(oo.Nupx))*3

            xAS  = x[Asts + t0]   #  position @ spikes
            mAS  = mks[Asts + t0]   #  position @ spikes
            xASr = xAS.reshape((1, nSpks))
            mASr = mAS.reshape((nSpks, 1, K))
            econt = _N.empty((M, nSpks))
            rat   = _N.zeros((M+1, nSpks))

            qdrMKS = _N.empty((M, nSpks))
            ################################  GIBBS ITERS ITERS ITERS

            _iSg_Mu = _N.einsum("mjk,mk->mj", _N.linalg.inv(_u_Sg), _u_u)

            clusSz = _N.zeros(M, dtype=_N.int)
            for iter in xrange(ITERS):
                if (iter % 100) == 0:    print "iter  %d" % iter

                ur         = u.reshape((1, M, K))
                fr         = f.reshape((M, 1))    # centers
                iq2        = 1./q2
                iSg        = _N.linalg.inv(Sg)
                iq2r       = iq2.reshape((M, 1))  
                try:
                    pkFR       = _N.log(l0/_N.sqrt(twpi*q2))
                except Warning:
                    print "WARNING"
                    print l0
                    print q2

                mkNrms = _N.log(1/_N.sqrt(twpi*_N.linalg.det(Sg)))
                mkNrms = mkNrms.reshape((M, 1))

                rnds       = _N.random.rand(nSpks)

                pkFRr      = pkFR.reshape((M, 1))
                dmu        = (mASr - ur)

                _N.einsum("nmj,mjk,nmk->mn", dmu, iSg, dmu, out=qdrMKS)
                cont       = pkFRr + mkNrms - 0.5*((fr - xASr)*(fr - xASr)*iq2r + qdrMKS)

                mcontr     = _N.max(cont, axis=0).reshape((1, nSpks))  
                cont       -= mcontr
                _N.exp(cont, out=econt)

                for m in xrange(M):
                    rat[m+1] = rat[m] + econt[m]

                rat /= rat[M]
                """
                print f
                print u
                print q2
                print Sg
                print l0
                """

                # print rat

                M1 = rat[1:] >= rnds
                M2 = rat[0:-1] <= rnds

                gz[iter] = (M1&M2).T

                ###############  FOR EACH CLUSTER
                print "^^^^^^^^^^^^^^"
                for m in xrange(M):
                    iiq2 = 1./q2[m]
                    minds = _N.where(gz[iter, :, m] == 1)[0]
                    sts  = Asts[minds]
                    nSpksM   = len(sts)
                    clusSz[m] = nSpksM

                    #  prior described by hyper-parameters.
                    #  prior described by function

                    #  likelihood

                    """
                    ############################################
                    """
                    mcs = _N.empty((M, K))   # cluster sample means

                    u_Sg_[m] = _N.linalg.inv(_u_Sg[m] + nSpksM*iSg[m])

                    if nSpksM > 0:
                        clstx    = mks[sts]
                        mcs[m]       = _N.mean(clstx, axis=0)
                        u_u_[m] = _N.einsum("jk,k->j", u_Sg_[m], _iSg_Mu[m] + nSpksM*_N.dot(iSg[m], mcs[m]))
                        u[m] = _N.random.multivariate_normal(u_u_[m], u_Sg_[m])

                        # hyp
                        ########  POSITION
                        ##  mean of posterior distribution of cluster means
                        #  sigma^2 and mu are the current Gibbs-sampled values

                        ##  mean of posterior distribution of cluster means

                    smp_mk_prms[oo.ky_p_u][:, iter, m] = u[m]
                    print ".   %d" % nSpksM
                    smp_mk_hyps[oo.ky_h_u_u][:, iter, m] = u_u_[m]
                    smp_mk_hyps[oo.ky_h_u_Sg][:, :, iter, m] = u_Sg_[m]

                    # dot(MATRIX, vector)   
                    """
                    ############################################
                    """
                    ###############  CONDITIONAL f
                    q2pr = _f_q2[m] if (_f_q2[m] > q2rate) else q2rate

                    if nSpksM > 0:  #  spiking portion likelihood x prior
                        fs  = (1./nSpksM)*_N.sum(xt0t1[sts])
                        fq2 = q2[m]/nSpksM
                        U   = (fs*q2pr + _f_u[m]*fq2) / (q2pr + fq2)
                        FQ2 = (q2pr*fq2) / (q2pr + fq2)
                    else:
                        U   = _f_u[m]
                        FQ2 = q2pr

                    FQ    = _N.sqrt(FQ2)
                    fx    = _N.linspace(U - FQ*150, U + FQ*150, oo.fss)
                    fxr     = fx.reshape((oo.fss, 1))

                    fxrux = -0.5*(fxr-ux)**2
                    xI_f    = (xt0t1 - fxr)**2*0.5

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

                    #############  VARIANCE, COVARIANCE
                    if nSpksM >= 1:
                        ##  dof of posterior distribution of cluster covariance
                        Sg_nu_[m] = _Sg_nu[m, 0] + nSpksM
                        ##  dof of posterior distribution of cluster covariance
                        ur = u[m].reshape((1, K))
                        Sg_PSI_[m] = _Sg_PSI[m] + _N.dot((clstx - ur).T, (clstx-ur))
                        Sg[m] = s_u.sample_invwishart(Sg_PSI_[m], Sg_nu_[m])
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

                    _Dq2_a = _q2_a[m] if _q2_a[m] < 200 else 200
                    _Dq2_B = (_q2_B[m]/(_q2_a[m]+1))*(_Dq2_a+1)

                    if nSpksM > 0:
                        #print  _N.sum((xt0t1[sts]-f)*(xt0t1[sts]-f))/(nSpks-1)

                        ##  (1/sqrt(sg2))^S
                        ##  (1/x)^(S/2)   = (1/x)-(a+1)
                        ##  -S/2 = -a - 1     -a = -S/2 + 1    a = S/2-1
                        xI = (xt0t1[sts]-f[m])*(xt0t1[sts]-f[m])*0.5
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

                    ###############  CONDITIONAL l0
                    #  _ss.gamma.rvs.  uses k, theta    k is 1/B  (B is our thing)
                    iiq2 = 1./q2[m]
                    # xI = (xt0t1-f)*(xt0t1-f)*0.5*iiq2
                    # BL  = (oo.dt/_N.sqrt(twpi*q2))*_N.sum(_N.exp(-xI))

                    l0_intgrd   = _N.exp(-0.5*(f[m] - ux)*(f[m]-ux) * iiq2)  
                    l0_exp_px   = _N.sum(l0_intgrd*px) * dSilenceX
                    BL  = (oo.dt/_N.sqrt(twpi*q2[m]))*l0_exp_px

                    #_Dl0_a = _l0_a[m] if _l0_a[m] < 400 else 400
                    _Dl0_a = _l0_a[m] if _l0_a[m] < 25 else 25
                    _Dl0_B = (_l0_B[m]/_l0_a[m]) * _Dl0_a
                    
                    #  a'/B' = a/B
                    #  B' = (B/a)a'

                    aL  = nSpksM
                    l0_a_ = aL + _Dl0_a
                    l0_B_ = BL + _Dl0_B

                    #print "l0_a_ %(a).3e   l0_B_ %(B).3e" % {"a" : l0_a_, "B" : l0_B_}

                    if (l0_B_ > 0) and (l0_a_ > 1):
                        l0[m] = _ss.gamma.rvs(l0_a_ - 1, scale=(1/l0_B_))  #  check

                    ###  l0 / _N.sqrt(twpi*q2) is f*dt used in createData2

                    smp_sp_prms[oo.ky_p_l0, iter, m] = l0[m]
                    smp_sp_hyps[oo.ky_h_l0_a, iter, m] = l0_a_
                    smp_sp_hyps[oo.ky_h_l0_B, iter, m] = l0_B_

            print "^^^^^^^^^^^^^^^"
            for m in xrange(M):
                print "mean %d"% m
                print smp_mk_prms[oo.ky_p_u][:, ITERS-1, m]
                print "cov %d"% m
                print smp_mk_prms[oo.ky_p_Sg][:, :, ITERS-1, m]

            frm   = int(0.6*ITERS)  #  have to test for stationarity

            occ   =  _N.mean(gz[ITERS-1], axis=0)


            l_trlsNearMAP = []
            for m in xrange(M):
                trlsNearMAP = _N.arange(0, ITERS-frm)
                for ip in xrange(3):  # params
                    L     = _N.min(smp_sp_prms[ip, frm:, m]);   H     = _N.max(smp_sp_prms[ip, frm:, m])
                    cnts, bns = _N.histogram(smp_sp_prms[ip, frm:, m], bins=_N.linspace(L, H, 50))
                    ###  take the iters that gave 25% top counts
                    ###  intersect them for each param.
                    ###  
                    col = 3*m+ip

                    if oo.polyFit:
                        xfit = 0.5*(bns[0:-1] + bns[1:])
                        yfit = cnts
                        ac = _N.polyfit(xfit, yfit, 2)  #a[0]*x^2 + a[1]*x + a[2]
                        if ac[0] < 0:  #  found a maximum
                            xMAP = -ac[1] / (2*ac[0])

                            yPnF = ac[0]*xfit**2+ac[1]*xfit+ac[2]
                            yMAP = ac[0]*xMAP**2+ac[1]*xMAP+ac[2]
                            xLo  = xfit[_N.where(yPnF > yMAP*0.75)[0][0]]
                            xHi  = xfit[_N.where(yPnF > yMAP*0.75)[0][-1]]

                            if occ[m] > 0:
                                these=_N.where((smp_sp_prms[ip, frm:, m] > xLo) & (smp_sp_prms[ip, frm:, m] < xHi))[0]
                                trlsNearMAP = _N.intersect1d(these, trlsNearMAP)
                        else:
                            ib  = _N.where(cnts == _N.max(cnts))[0][0]
                            xMAP  = bns[ib]                        
                    else:
                        ib  = _N.where(cnts == _N.max(cnts))[0][0]
                        xMAP  = bns[ib]

                    if   ip == oo.ky_p_l0: 
                        l0[m] = oo.sp_prmPstMd[epc, col] = xMAP
                    elif ip == oo.ky_p_f:  
                        f[m]  = oo.sp_prmPstMd[epc, col] = xMAP
                    elif ip == oo.ky_p_q2: 
                        q2[m] = oo.sp_prmPstMd[epc, col] = xMAP

                trlsNearMAP += frm
                l_trlsNearMAP.append(trlsNearMAP)
                print trlsNearMAP

            for m in xrange(M):
                if occ[m] > 0:
                    print "mth %d cluster" % m
                    print l_trlsNearMAP[m]

            pcklme["cp%d" % epc] = _N.array(smp_sp_prms)
            #trlsNearMAP = _N.array(list(set(trlsNearMAP_D)))+frm   #  use these trials to pick out posterior params for MARK part

            for m in xrange(M):
                for ip in xrange(6):  # hyper params
                    L     = _N.min(smp_sp_hyps[ip, frm:, m]);   H     = _N.max(smp_sp_hyps[ip, frm:, m])
                    cnts, bns = _N.histogram(smp_sp_hyps[ip, frm:, m], bins=_N.linspace(L, H, 50))
                    ib  = _N.where(cnts == _N.max(cnts))[0][0]

                    col = 6*m+ip
                    vl  = bns[ib]

                    if   ip == oo.ky_h_l0_a: 
                        _l0_a[m] = oo.sp_hypPstMd[epc, col] = vl
                    elif ip == oo.ky_h_l0_B: 
                        _l0_B[m] = oo.sp_hypPstMd[epc, col] = vl
                    elif ip == oo.ky_h_f_u:  _f_u[m]  = oo.sp_hypPstMd[epc, col] = vl
                    elif ip == oo.ky_h_f_q2: _f_q2[m] = oo.sp_hypPstMd[epc, col] = vl
                    elif ip == oo.ky_h_q2_a: _q2_a[m] = oo.sp_hypPstMd[epc, col] = vl
                    elif ip == oo.ky_h_q2_B: _q2_B[m] = oo.sp_hypPstMd[epc, col] = vl


            ##  params and hyper parms for mark
            for m in xrange(M):
                u[m] = _N.mean(smp_mk_prms[0][:, l_trlsNearMAP[m], m], axis=1)
                Sg[m] = _N.mean(smp_mk_prms[1][:, :, l_trlsNearMAP[m], m], axis=2)
                if occ[m] > 0:
                    print "for cluster %d" % m
                    print u[m]
                    print Sg[m]

            ###  hack here.  If we don't reset the prior for 
            ###  what happens when a cluster is unused?
            ###  l0 -> 0, and at the same time, the variance increases.
            ###  the prior then gets pushed to large values, but
            ###  then it becomes difficult to bring it back to small
            ###  values once that cluster becomes used again.  So
            ###  we would like unused clusters to have l0->0, but keep the
            ###  variance small.  That's why we will reset a cluster
            print occ

            for m in xrange(M):
                if (occ[m] == 0) and (l0[m] / _N.sqrt(twpi*q2[m]) < 1):
                    print "resetting"
                    _q2_a[m] = 1e-4
                    _q2_B[m] = 1e-3


        if savePosterior:
            _N.savetxt(resFN("posParams.dat", dir=oo.outdir), smp_sp_prms.T, fmt=("%.4f %.4f %.4f " * M))
            #_N.savetxt(resFN("posHypParams.dat", dir=oo.outdir), smp_sp_hyps[:, :, 0].T, fmt="%.4f %.4f %.4f %.4f %.4f %.4f")

        pcklme["md"] = _N.array(oo.sp_prmPstMd)
        dmp = open(resFN("posteriors.dump", dir=oo.outdir), "wb")
        pickle.dump(pcklme, dmp, -1)
        dmp.close()

        _N.savetxt(resFN("posModes.dat", dir=oo.outdir), oo.sp_prmPstMd, fmt=("%.4f %.4f %.4f " * M))
        _N.savetxt(resFN("hypModes.dat", dir=oo.outdir), oo.sp_hypPstMd, fmt=("%.4f %.4f %.4f %.4f %.4f %.4f" * M))

        
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
