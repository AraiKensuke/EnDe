"""
V1.2   use adaptive range for integrating over f
variance   0.0001 
"""

import scipy.stats as _ss
import os
import time as _tm
from ig_prmLib import ig_prmsUV
import numpy as _N
import matplotlib.pyplot as _plt
from EnDedirs import resFN, datFN
import pickle



class multiRecptvFld:
    ky_p_l0 = 0;    ky_p_f  = 1;    ky_p_q2 = 2
    ky_h_l0_a = 0;  ky_h_l0_B=1;
    ky_h_f_u  = 2;  ky_h_f_q2=3;
    ky_h_q2_a = 4;  ky_h_q2_B=5;

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

    diffPerMin = 1.  #  diffusion per minute
    epochs   = None

    outdir   = None
    polyFit  = True
    xLo      = 0
    xHi      = 3

    def __init__(self, outdir, fn, intvfn, xLo=0, xHi=3):
        oo     = self
        oo.xLo = xLo
        oo.xHi = xHi
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

    def gibbs(self, ITERS, M, ep1=0, ep2=None, savePosterior=True, gtdiffusion=False):
        """
        gtdiffusion:  use ground truth center of place field in calculating variance of center.  Meaning of diffPerMin different
        """
        oo = self

        #  PRIORS
        #  priors  prefixed w/ _
        #_f_u   = _N.zeros(M);    _f_q2  = _N.ones(M) #  wide
        _f_u   = _N.linspace(oo.xLo+1, oo.xHi-1, M);    _f_q2  = _N.ones(M) #  wide
        #  inverse gamma
        _q2_a  = _N.ones(M)*2.1;    _q2_B  = _N.ones(M)*1e-2
        #_plt.plot(q2x, q2x**(-_q2_a-1)*_N.exp(-_q2_B / q2x))
        _l0_a = _N.ones(M);     _l0_B  = _N.zeros(M)*(1/30.)

        ep2 = oo.epochs if (ep2 == None) else ep2
        oo.epochs = ep2-ep1
        #oo.prmPstMd = _N.zeros((oo.epochs, 3, M))   # mode of the params
        oo.prmPstMd = _N.zeros((oo.epochs, 3*M))   # mode of the params
        #oo.hypPstMd  = _N.zeros((oo.epochs, (2+2+2), M))   # the hyper params
        oo.hypPstMd  = _N.zeros((oo.epochs, (2+2+2)*M))   # the hyper params
        twpi     = 2*_N.pi

        pcklme   = {}

        #  Gibbs sampling
        #  parameters l0, f, q2

        ######################################  GIBBS samples, need for MAP estimate
        smp_prms = _N.zeros((3, ITERS, M))  
        #
        smp_hyps = _N.zeros((6, ITERS, M))  

        ######################################  INITIAL VALUE OF PARAMS
        l0       = _N.array([1.,]*M)
        q2       = _N.array([0.0144]*M)
        f        = _N.array([1.1]*M)

        oo.f_     = f
        oo.q2_    = q2
        oo.l0_    = l0

        ######################################  GRID for calculating
        ####  #  points in sum.  
        ####  #  points in uniform sampling of exp(x)p(x)   (non-spike interals)
        ####  #  points in sampling of f  for conditional posterior distribution
        ####  #  points in sampling of q2 for conditional posterior distribution
        ####  NSexp, Nupx, fss, q2ss

        #  numerical grid
        ux = _N.linspace(oo.xLo, oo.xHi, oo.Nupx, endpoint=False)   # uniform x position
        q2x    = _N.exp(_N.linspace(_N.log(0.000001), _N.log(400), oo.q2ss))  #  5 orders of
        #q2x    = _N.exp(_N.linspace(_N.log(0.0001), _N.log(100), oo.q2ss))  #  5 orders of
        d_q2x  = _N.diff(q2x)
        q2x_m1 = _N.array(q2x[0:-1])
        lq2x    = _N.log(q2x)
        iq2x    = 1./q2x
        q2xr     = q2x.reshape((oo.q2ss, 1))
        iq2xr     = 1./q2xr
        sqrt_2pi_q2x   = _N.sqrt(twpi*q2x)
        l_sqrt_2pi_q2x = _N.log(sqrt_2pi_q2x)

        x      = oo.dat[:, 0]

        q2rate = oo.diffPerEpoch**2  #  unit of minutes  
        ######################################  PRECOMPUTED
        posbins  = _N.linspace(oo.xLo, oo.xHi, oo.Nupx+1)
        rat      = _N.zeros(M+1)
        pc       = _N.zeros(M)

        tempSlnc = _N.empty((oo.q2ss, 3))
        for epc in xrange(ep1, ep2):
            f = 3*_N.random.rand(M)
            q2 = 3*_N.random.rand(M)*0.1
            l0 = 3*_N.random.rand(M)
            #print q2
            print "epoch %d" % epc

            t0 = oo.intvs[epc]
            t1 = oo.intvs[epc+1]
            Asts    = _N.where(oo.dat[t0:t1, 1] == 1)[0]   #  based at 0
            Ants    = _N.where(oo.dat[t0:t1, 1] == 0)[0]

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
            #dSilenceX = (NSexp/float(oo.Nupx))*3

            xAS  = x[Asts + t0]   #  position @ spikes
            xASr = xAS.reshape((1, nSpks))
            econt = _N.empty((M, nSpks))
            rat   = _N.zeros((M+1, nSpks))

            print "-^---------"
            for iter in xrange(ITERS):
                if (iter % 100) == 0:    print iter

                fr         = f.reshape((M, 1))
                iq2        = 1./q2
                iq2r       = iq2.reshape((M, 1))
                try:
                    pkFR       = _N.log(l0/_N.sqrt(twpi*q2))
                except Warning:
                    print "WARNING"
                    print l0
                    print q2
                pkFRr      = pkFR.reshape((M, 1))
                rnds       = _N.random.rand(nSpks)

                cont       = pkFRr - 0.5*(fr - xASr)*(fr - xASr)*iq2r

                mcontr     = _N.max(cont, axis=0).reshape((1, nSpks))  
                cont       -= mcontr
                _N.exp(cont, out=econt)
                for m in xrange(M):
                    rat[m+1] = rat[m] + econt[m]

                rat /= rat[M]

                M1 = rat[1:] >= rnds
                M2 = rat[0:-1] <= rnds

                gz[iter] = (M1&M2).T
                
                for m in xrange(M):
                    iiq2 = 1./q2[m]
                    sts  = Asts[_N.where(gz[iter, :, m] == 1)[0]]
                    #print sts
                    nSpksM   = len(sts)

                    #  prior described by hyper-parameters.
                    #  prior described by function

                    #  likelihood

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
                    fx    = _N.linspace(U - FQ*60, U + FQ*60, oo.fss)
                    fxr     = fx.reshape((oo.fss, 1))

                    fxrux = -0.5*(fxr-ux)*(fxr-ux)
                    #xI_f    = (xt0t1 - fxr)**2*0.5

                    f_intgrd  = _N.exp((fxrux*iiq2))   #  integrand
                    f_exp_px = _N.sum(f_intgrd*px, axis=1) * dSilenceX
                    #  f_exp_px is a function of f
                    slnc = -(l0[m]*oo.dt/_N.sqrt(twpi*q2[m])) * f_exp_px  #  a function of x

                    funcf   = -0.5*((fx-U)*(fx-U))/FQ2 + slnc
                    funcf   -= _N.max(funcf)
                    condPosF= _N.exp(funcf)
                    #print _N.sum(condPosF)

                    norm    = 1./_N.sum(condPosF)
                    f_u_    = norm*_N.sum(fx*condPosF)
                    f_q2_   = norm*_N.sum(condPosF*(fx-f_u_)*(fx-f_u_))
                    f[m]    = _N.sqrt(f_q2_)*_N.random.randn() + f_u_
                    smp_prms[oo.ky_p_f, iter, m] = f[m]
                    smp_hyps[oo.ky_h_f_u, iter, m] = f_u_
                    smp_hyps[oo.ky_h_f_q2, iter, m] = f_q2_

                    # ###############  CONDITIONAL q2
                    #xI = (xt0t1-f)*(xt0t1-f)*0.5*iq2xr
                    q2_intgrd   = _N.exp(-0.5*(f[m] - ux)*(f[m]-ux) * iq2xr)  
                    q2_exp_px   = _N.sum(q2_intgrd*px, axis=1) * dSilenceX

                    slnc = -((l0[m]*oo.dt)/sqrt_2pi_q2x)*q2_exp_px     #  function of q2
                    #print "s  %.3e" % s

                    _Dq2_a = _q2_a[m]# if _q2_a[m] < 200 else 200
                    _Dq2_B = _q2_B[m]#(_q2_B[m]/(_q2_a[m]+1))*(_Dq2_a+1)

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

                    sat = sLLkPr + slnc
                    sat -= _N.max(sat)
                    condPos = _N.exp(sat)
                    q2_a_, q2_B_ = ig_prmsUV(q2x, condPos, d_q2x, q2x_m1, ITER=1)
                    if q2_a_ > 10000:
                        putit = _N.empty((oo.q2ss, 4))
                        putit[:, 0] = q2x
                        putit[:, 1] = sLLkPr
                        putit[:, 2] = slnc
                        putit[:, 3] = condPos
                        _N.savetxt("putit", putit, fmt="%.4e %.4e %.4e %.4e")
                        print _q2_a[m]
                        print SL_a
                        print _q2_B[m]
                        print SL_B
                        
                        print "------------    %d" % nSpksM
                        assert q2_a_ < 10000, "q2_a_ too big"

                    print "it %(it)d   q2 m: %(m)d  nS: %(ns)d    q2_a_   %(a) .3e, q2_B_ %(B) .3e" % {"a" : q2_a_, "B" : q2_B_, "m" : m, "ns" : nSpksM, "it" : iter}
                    #  domain error when q2_a_ and q2_B_ is nan

                    # if nSpksM == 0:
                    #     tempSlnc[:, 0] = condPos
                    #     tempSlnc[:, 1] = slnc
                    #     tempSlnc[:, 2] = sLLkPr
                    #     _N.savetxt(resFN("slnc_nspks0_%(it)d_%(m)d" % {"it" : iter, "m" : m}, dir=oo.outdir), tempSlnc)
                    # if (q2_a_ > 10e10) or (q2_B_ > 10e10):
                    #     _N.savetxt("badcondposBIG", condPos)
                    #     _N.savetxt("slncBIG", slnc)
                    #     _N.savetxt("sLLkPrBIG", sLLkPr)

                    #     print "BIG  it %(it)d   q2_a_ %(1).3e  q2_B_ %(2).3e" % {"1" : q2_a_, "2" : q2_B_, "it" : iter}
                    #     print "m is %(m)d   nSpksM is %(ns)d" % {"m" : m, "ns" : nSpksM}

                    try:
                        q2[m] = _ss.invgamma.rvs(q2_a_ + 1, scale=q2_B_)  #  check
                    except ValueError:
                        fig = _plt.figure()
                        fig.add_subplot(2, 1, 1)
                        _N.savetxt("badcondpos", condPos)
                        _N.savetxt("slnc", slnc)
                        _N.savetxt("sLLkPr", sLLkPr)
                        _plt.plot(condPos)
                        _plt.ylim(-0.1, 1.1)
                        fig.add_subplot(2, 1, 2)
                        _plt.plot(_N.log(condPos))

                        print "ValueError  it %(it)d   q2_a_ %(1).3e  q2_B_ %(2).3e" % {"1" : q2_a_, "2" : q2_B_, "it" : iter}
                        print "m is %(m)d   nSpksM is %(ns)d" % {"m" : m, "ns" : nSpksM}
                        raise
                    #print ((1./nSpks)*_N.sum((xt0t1[sts]-f)*(xt0t1[sts]-f)))

                    smp_prms[oo.ky_p_q2, iter, m]   = q2[m]
                    smp_hyps[oo.ky_h_q2_a, iter, m] = q2_a_
                    smp_hyps[oo.ky_h_q2_B, iter, m] = q2_B_

                    #print "l0 1"
                    ###############  CONDITIONAL l0
                    #  _ss.gamma.rvs.  uses k, theta    k is 1/B  (B is our thing)
                    iiq2 = 1./q2[m]
                    # xI = (xt0t1-f)*(xt0t1-f)*0.5*iiq2
                    # BL  = (oo.dt/_N.sqrt(twpi*q2))*_N.sum(_N.exp(-xI))

                    l0_intgrd   = _N.exp(-0.5*(f[m] - ux)*(f[m]-ux) * iiq2)  
                    l0_exp_px   = _N.sum(l0_intgrd*px) * dSilenceX
                    BL  = (oo.dt/_N.sqrt(twpi*q2[m]))*l0_exp_px
                    # if iter == 50:
                    #     print "BL  %(BL).2f    BL2  %(BL2).2f" % {"BL" : BL, "BL2" : BL2}

                    #print "l0 2"
                    #_Dl0_a = _l0_a[m] if _l0_a[m] < 400 else 400
                    _Dl0_a = _l0_a[m] if _l0_a[m] < 25 else 25
                    _Dl0_B = (_l0_B[m]/_l0_a[m]) * _Dl0_a
                    
                    #  a'/B' = a/B
                    #  B' = (B/a)a'

                    aL  = nSpksM
                    l0_a_ = aL + _Dl0_a
                    l0_B_ = BL + _Dl0_B

                    #print "l0_a_ %(a).3e   l0_B_ %(B).3e" % {"a" : l0_a_, "B" : l0_B_}
                    #print "l0 3"

                    if (l0_B_ > 0) and (l0_a_ > 1):
                        l0[m] = _ss.gamma.rvs(l0_a_ - 1, scale=(1/l0_B_))  #  check

                    ###  l0 / _N.sqrt(twpi*q2) is f*dt used in createData2

                    smp_prms[oo.ky_p_l0, iter, m] = l0[m]
                    smp_hyps[oo.ky_h_l0_a, iter, m] = l0_a_
                    smp_hyps[oo.ky_h_l0_B, iter, m] = l0_B_

            frm   = int(0.6*ITERS)  #  have to test for stationarity
            #print _N.sum(_N.mean(gz[frm:], axis=0), axis=0)

            #print "f[0]  %(1).3f    f[1]  %(2).3f" % {"1" : f[0], "2" : f[1]}
            #print "here"
            fig = _plt.figure(figsize=(8, 4))
            for m in xrange(M):
                #print smp_prms[oo.ky_p_f, frm:, m]
                for ip in xrange(3):  # params
                    L     = _N.min(smp_prms[ip, frm:, m]);   H     = _N.max(smp_prms[ip, frm:, m])
                    cnts, bns = _N.histogram(smp_prms[ip, frm:, m], bins=_N.linspace(L, H, 50))

                    if oo.polyFit:
                        xfit = 0.5*(bns[0:-1] + bns[1:])
                        yfit = cnts
                        ac = _N.polyfit(xfit, yfit, 2)  #a[0]*x^2 + a[1]*x + a[2]
                        if ac[0] < 0:  #  found a maximum
                            xMAP = -ac[1] / (2*ac[0])
                        else:
                            ib  = _N.where(cnts == _N.max(cnts))[0][0]
                            xMAP  = bns[ib]
                    else:
                        ib  = _N.where(cnts == _N.max(cnts))[0][0]
                        xMAP  = bns[ib]

                    col = 3*m+ip
                    if   ip == oo.ky_p_l0: l0[m] = oo.prmPstMd[epc, col] = xMAP
                    elif ip == oo.ky_p_f:  f[m]  = oo.prmPstMd[epc, col] = xMAP
                    elif ip == oo.ky_p_q2: q2[m] = oo.prmPstMd[epc, col] = xMAP
            pcklme["cp%d" % epc] = _N.array(smp_prms)

            for m in xrange(M):
                for ip in xrange(6):  # hyper params
                    L     = _N.min(smp_hyps[ip, frm:, m]);   H     = _N.max(smp_hyps[ip, frm:, m])
                    cnts, bns = _N.histogram(smp_hyps[ip, frm:, m], bins=_N.linspace(L, H, 50))

                    if oo.polyFit:
                        xfit = 0.5*(bns[0:-1] + bns[1:])
                        yfit = cnts
                        ac = _N.polyfit(xfit, yfit, 2)  #a[0]*x^2 + a[1]*x + a[2]
                        #y  = a[0]x^2 + a[1]x + a[2]
                        #y' = 2a[0]x + a[1]
                        #y''= 2a[0]    if a[0]
                        if ac[0] < 0:  #  found a maximum
                            xMAP = -ac[1] / (2*ac[0])
                        else:
                            ib  = _N.where(cnts == _N.max(cnts))[0][0]
                            xMAP  = bns[ib]
                    else:
                        ib  = _N.where(cnts == _N.max(cnts))[0][0]
                        xMAP  = bns[ib]

                    col = 6*m+ip

                    if   ip == oo.ky_h_l0_a: _l0_a[m] = oo.hypPstMd[epc, col] = xMAP
                    elif ip == oo.ky_h_l0_B: _l0_B[m] = oo.hypPstMd[epc, col] = xMAP
                    elif ip == oo.ky_h_f_u:  _f_u[m]  = oo.hypPstMd[epc, col] = xMAP
                    elif ip == oo.ky_h_f_q2: _f_q2[m] = oo.hypPstMd[epc, col] = xMAP
                    elif ip == oo.ky_h_q2_a: _q2_a[m] = oo.hypPstMd[epc, col] = xMAP
                    elif ip == oo.ky_h_q2_B: _q2_B[m] = oo.hypPstMd[epc, col] = xMAP

            ###  hack here.  If we don't reset the prior for 
            ###  what happens when a cluster is unused?
            ###  l0 -> 0, and at the same time, the variance increases.
            ###  the prior then gets pushed to large values, but
            ###  then it becomes difficult to bring it back to small
            ###  values once that cluster becomes used again.  So
            ###  we would like unused clusters to have l0->0, but keep the
            ###  variance small.  That's why we will reset a cluster
            occ   =  _N.mean(gz[ITERS-1], axis=0)
            print occ

            for m in xrange(M):
                if (occ[m] == 0) and (l0[m] / _N.sqrt(twpi*q2[m]) < 1):
                    print "resetting"
                    _q2_a[m] = 1e-4
                    _q2_B[m] = 1e-3

        pcklme["gz"] = gz
        pcklme["smp_hyps"] = smp_hyps
        pcklme["smp_prms"] = smp_prms
        pcklme["prmPstMd"] = oo.prmPstMd
        pcklme["intvs"]       = oo.intvs
        dmp = open(resFN("posteriors.dmp", dir=oo.outdir), "wb")
        pickle.dump(pcklme, dmp, -1)
        dmp.close()
        
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
        _plt.plot(oo.prmPstMd[:, oo.ky_p_f])
        fig.add_subplot(3, 1, 2)
        _plt.plot(mnL0s)
        _plt.plot(oo.prmPstMd[:, oo.ky_p_l0])
        fig.add_subplot(3, 1, 3)
        _plt.plot(mnSq2s)
        _plt.plot(oo.prmPstMd[:, oo.ky_p_q2])
            
        _plt.savefig(resFN("cmpModesGT", dir=oo.outdir))

