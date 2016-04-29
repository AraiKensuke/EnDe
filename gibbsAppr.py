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

class singleRecptvFld:
    ky_p_l0 = 0;    ky_p_f  = 1;    ky_p_q2 = 2
    ky_h_l0_a = 0;  ky_h_l0_B=1;
    ky_h_f_u  = 2;  ky_h_f_q2=3;
    ky_h_q2_a = 4;  ky_h_q2_B=5;

    dt      = 0.001
    #  position dependent firing rate
    ######################################  PRIORS
    twpi = 2*_N.pi

    #  sizes of arrays
    Nupx = 500      #   # points to sample position with  (uniform lam(x)p(x))
    fss = 100       #  sampling at various values of f
    q2ss = 200      #  sampling at various values of q2

    intvs = None    #  
    dat   = None

    diffPerMin = 1.  #  diffusion per minute
    epochs   = None

    outdir   = None

    def __init__(self, outdir, fn, intvfn):
        oo     = self
        ######################################  DATA input, define intervals
        # bFN = fn[0:-4]
        oo.outdir = outdir

        # if not os.access(bFN, os.F_OK):
        #     os.mkdir(bFN)

        oo.dat    = _N.loadtxt("%s.dat" % datFN(fn, create=False))
        intvs     = _N.loadtxt("%s.dat" % datFN(intvfn, create=False))

        oo.intvs  = _N.array(intvs*oo.dat.shape[0], dtype=_N.int)
        oo.epochs    = oo.intvs.shape[0] - 1
        
        NT     = oo.dat.shape[0]

    def gibbs(self, ITERS, ep1=0, ep2=None, savePosterior=True):
        oo = self

        #  PRIORS
        #  priors  prefixed w/ _
        _f_u   = 0;    _f_q2  = 0
        #  inverse gamma
        _q2_a  = 0;    _q2_B  = 0
        #_plt.plot(q2x, q2x**(-_q2_a-1)*_N.exp(-_q2_B / q2x))
        _l0_a = 0;     _l0_B = 0

        ep2 = oo.epochs if (ep2 == None) else ep2
        oo.epochs = ep2-ep1
        oo.posSmpls = _N.zeros((oo.epochs, 3))   # mode of the params
        oo.posHyps  = _N.zeros((oo.epochs, 2+2+2))   # the hyper params
        twpi     = 2*_N.pi

        #  Gibbs sampling
        #  parameters l0, f, q2

        ######################################  GIBBS samples, need for MAP estimate
        smp_prms = _N.zeros((3, ITERS, 1))  
        #
        smp_hyps = _N.zeros((6, ITERS, 1))  

        ######################################  INITIAL VALUE OF PARAMS
        l0       = 50
        q2       = 0.0144
        f        = 1.1

        ######################################  GRID for calculating
        ####  #  points in sum.  
        ####  #  points in uniform sampling of exp(x)p(x)   (non-spike interals)
        ####  #  points in sampling of f  for conditional posterior distribution
        ####  #  points in sampling of q2 for conditional posterior distribution
        ####  NSexp, Nupx, fss, q2ss

        #  numerical grid
        ux = _N.linspace(0, 3, oo.Nupx, endpoint=False)   # uniform x position

        q2x    = _N.exp(_N.linspace(_N.log(0.00005), _N.log(10), oo.q2ss))  #  5 orders of
        d_q2x  = _N.diff(q2x)
        q2x_m1 = _N.array(q2x[0:-1])
        lq2x    = _N.log(q2x)
        iq2x    = 1./q2x
        q2xr     = q2x.reshape((oo.q2ss, 1))
        iq2xr     = 1./q2xr
        sqrt_2pi_q2x   = _N.sqrt(twpi*q2x)
        l_sqrt_2pi_q2x = _N.log(sqrt_2pi_q2x)

        x      = oo.dat[:, 0]

        q2rate = (oo.diffPerMin/(60.*1000))**2  #  unit of minutes  
        Tepch      = ((oo.dat.shape[0])/float(oo.epochs)) 
        q2rate  *=  Tepch
        ######################################  PRECOMPUTED
        posbins  = _N.linspace(0, 3, oo.Nupx+1)

        for epc in xrange(ep1, ep2):
            # if i > 0:
            #     q2x     = _N.linspace(0.001, 4, q2ss)
            #     q2xr     = q2x.reshape((q2ss, 1))
            #     iq2xr     = 1./q2xr

            #print q2
            print "epoch %d" % epc

            t0 = oo.intvs[epc]
            t1 = oo.intvs[epc+1]
            sts    = _N.where(oo.dat[t0:t1, 1] == 1)[0]
            nts    = _N.where(oo.dat[t0:t1, 1] == 0)[0]

            NSexp   = t1-t0    #  length of position data  #  # of no spike positions to sum
            xt0t1 = _N.array(x[t0:t1])
            px, xbns = _N.histogram(xt0t1, bins=posbins, normed=True)

            nSpks    = len(sts)
            print "spikes %d" % nSpks

            dSilenceX = (NSexp/float(oo.Nupx))*3

            for iter in xrange(ITERS):
                #print "iter   %d" % iter
                iiq2 = 1./q2

                #  prior described by hyper-parameters.
                #  prior described by function

                #  likelihood

                ###############  CONDITIONAL f
                if epc == 0:
                    fs  = (1./nSpks)*_N.sum(xt0t1[sts])
                    fq2 = q2/nSpks
                    M   = fs
                    Sg2 = fq2
                else:
                    q2pr = _f_q2 + q2rate
                    if nSpks > 0:  #  spiking portion likelihood x prior
                        fs  = (1./nSpks)*_N.sum(xt0t1[sts])
                        fq2 = q2/nSpks
                        M   = (fs*q2pr +  + _f_u*fq2) / (q2pr + fq2)
                        Sg2 = (q2pr*fq2) / (q2pr + fq2)
                    else:
                        M   = _f_u
                        Sg2 = q2pr

                Sg    = _N.sqrt(Sg2)
                fx    = _N.linspace(M - Sg*50, M + Sg*50, oo.fss)
                fxr     = fx.reshape((oo.fss, 1))

                fxrux = -0.5*(fxr-ux)**2
                xI_f    = (xt0t1 - fxr)**2*0.5

                f_intgrd  = _N.exp((fxrux*iiq2))   #  integrand
                f_exp_px = _N.sum(f_intgrd*px, axis=1) * dSilenceX
                #  f_exp_px is a function of f
                s = -(l0*oo.dt/_N.sqrt(twpi*q2)) * f_exp_px  #  a function of x

                #print Sg2
                #print M
                
                funcf   = -0.5*((fx-M)*(fx-M))/Sg2 + s
                funcf   -= _N.max(funcf)
                condPosF= _N.exp(funcf)
                #print _N.sum(condPosF)

                """
                if iter == 0:
                    fig = _plt.figure()
                    _plt.plot(fx, condPosF)
                    _plt.xlim(0.8, 1.3)
                    _plt.savefig("%(dir)s/condposF%(i)d" % {"dir" : outdir, "i" : i})
                    _plt.close()
                """

                norm    = 1./_N.sum(condPosF)
                f_u_    = norm*_N.sum(fx*condPosF)
                f_q2_   = norm*_N.sum(condPosF*(fx-f_u_)*(fx-f_u_))
                f       = _N.sqrt(f_q2_)*_N.random.randn() + f_u_
                smp_prms[oo.ky_p_f, iter, 0] = f
                smp_hyps[oo.ky_h_f_u, iter, 0] = f_u_
                smp_hyps[oo.ky_h_f_q2, iter, 0] = f_q2_
                #ax1.plot(fx, L_f, color="black")

                # ###############  CONDITIONAL q2
                #xI = (xt0t1-f)*(xt0t1-f)*0.5*iq2xr
                q2_intgrd   = _N.exp(-0.5*(f - ux)*(f-ux) * iq2xr)  
                q2_exp_px   = _N.sum(q2_intgrd*px, axis=1) * dSilenceX
                s = -((l0*oo.dt)/sqrt_2pi_q2x)*q2_exp_px     #  function of q2

                if nSpks > 0:
                    #print  _N.sum((xt0t1[sts]-f)*(xt0t1[sts]-f))/(nSpks-1)

                    ##  (1/sqrt(sg2))^S
                    ##  (1/x)^(S/2)   = (1/x)-(a+1)
                    ##  -S/2 = -a - 1     -a = -S/2 + 1    a = S/2-1
                    xI = (xt0t1[sts]-f)*(xt0t1[sts]-f)*0.5
                    SL_a = 0.5*nSpks - 1   #  spiking part of likelihood
                    SL_B = _N.sum(xI)  #  spiking part of likelihood
                    #  spiking prior x prior
                    if epc > 0:
                        sLLkPr = -(_q2_a + SL_a + 2)*lq2x - iq2x*(_q2_B + SL_B)
                    else:
                        sLLkPr = -(SL_a + 1)*lq2x - iq2x*SL_B
                else:
                    sLLkPr = -(_q2_a + 1)*lq2x - iq2x*(_q2_B)

                sat = sLLkPr + s
                sat -= _N.max(sat)
                condPos = _N.exp(sat)
                """
                if iter == 10:
                    fig = _plt.figure()
                    _plt.plot(q2x, condPos)
                    _plt.xlim(0, 0.5)
                    _plt.savefig("%(dir)s/condpos%(i)d" % {"dir" : outdir, "i" : i})
                    _plt.close()
                """
                q2_a_, q2_B_ = ig_prmsUV(q2x, condPos, d_q2x, q2x_m1, ITER=1)

                #print condPos
                _plt.plot(q2x, condPos)
                q2 = _ss.invgamma.rvs(q2_a_ + 1, scale=q2_B_)  #  check
                #print ((1./nSpks)*_N.sum((xt0t1[sts]-f)*(xt0t1[sts]-f)))

                smp_prms[oo.ky_p_q2, iter, 0] = q2
                smp_hyps[oo.ky_h_q2_a, iter, 0] = q2_a_
                smp_hyps[oo.ky_h_q2_B, iter, 0] = q2_B_

                ###############  CONDITIONAL l0
                #  _ss.gamma.rvs.  uses k, theta    k is 1/B  (B is our thing)
                iiq2 = 1./q2
                # xI = (xt0t1-f)*(xt0t1-f)*0.5*iiq2
                # BL  = (oo.dt/_N.sqrt(twpi*q2))*_N.sum(_N.exp(-xI))

                l0_intgrd   = _N.exp(-0.5*(f - ux)*(f-ux) * iiq2)  
                l0_exp_px   = _N.sum(l0_intgrd*px) * dSilenceX
                BL  = (oo.dt/_N.sqrt(twpi*q2))*l0_exp_px
                # if iter == 50:
                #     print "BL  %(BL).2f    BL2  %(BL2).2f" % {"BL" : BL, "BL2" : BL2}

                aL  = nSpks
                if epc == 0:
                    _l0_B = 0
                    _l0_a = 0
                l0_a_ = aL + _l0_a
                l0_B_ = BL + _l0_B

                l0 = _ss.gamma.rvs(l0_a_ - 1, scale=(1/l0_B_))  #  check
                ###  l0 / _N.sqrt(twpi*q2) is f*dt used in createData2

                smp_prms[oo.ky_p_l0, iter, 0] = l0
                smp_hyps[oo.ky_h_l0_a, iter, 0] = l0_a_
                smp_hyps[oo.ky_h_l0_B, iter, 0] = l0_B_

            #tt2 = _tm.time()
            #print "time %.3f" % (tt2-tt1)
            #_plt.savefig("%(dir)s/likelihood%(i)d" % {"dir" : outdir, "i" : i})
            #_plt.close()

            frm   = 30

            for ip in xrange(3):  # params
                L     = _N.min(smp_prms[ip, frm:, 0]);   H     = _N.max(smp_prms[ip, frm:, 0])
                cnts, bns = _N.histogram(smp_prms[ip, frm:, 0], bins=_N.linspace(L, H, 50))
                ib  = _N.where(cnts == _N.max(cnts))[0][0]
                
                if   ip == oo.ky_p_l0: l0 = oo.posSmpls[epc, ip] = bns[ib]
                elif ip == oo.ky_p_f:  f  = oo.posSmpls[epc, ip] = bns[ib]
                elif ip == oo.ky_p_q2: q2 = oo.posSmpls[epc, ip] = bns[ib]

            for ip in xrange(6):  # hyper params
                L     = _N.min(smp_hyps[ip, frm:, 0]);   H     = _N.max(smp_hyps[ip, frm:, 0])
                cnts, bns = _N.histogram(smp_hyps[ip, frm:, 0], bins=_N.linspace(L, H, 50))
                ib  = _N.where(cnts == _N.max(cnts))[0][0]
                
                if   ip == oo.ky_h_l0_a: _l0_a = oo.posHyps[epc, ip] = bns[ib]
                elif ip == oo.ky_h_l0_B: _l0_B = oo.posHyps[epc, ip] = bns[ib]
                elif ip == oo.ky_h_f_u:  _f_u  = oo.posHyps[epc, ip] = bns[ib]
                elif ip == oo.ky_h_f_q2: _f_q2 = oo.posHyps[epc, ip] = bns[ib]
                elif ip == oo.ky_h_q2_a: _q2_a = oo.posHyps[epc, ip] = bns[ib]
                elif ip == oo.ky_h_q2_B: _q2_B = oo.posHyps[epc, ip] = bns[ib]
        if savePosterior:
            _N.savetxt(resFN("posParams.dat", dir=oo.outdir), smp_prms[:, :, 0].T, fmt="%.4f %.4f %.4f")
            _N.savetxt(resFN("posHypParams.dat", dir=oo.outdir), smp_hyps[:, :, 0].T, fmt="%.4f %.4f %.4f %.4f %.4f %.4f")

        
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
        _plt.plot(oo.posSmpls[:, oo.ky_p_f])
        fig.add_subplot(3, 1, 2)
        _plt.plot(mnL0s)
        _plt.plot(oo.posSmpls[:, oo.ky_p_l0])
        fig.add_subplot(3, 1, 3)
        _plt.plot(mnSq2s)
        _plt.plot(oo.posSmpls[:, oo.ky_p_q2])
            
