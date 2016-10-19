import numpy as _N
import matplotlib.pyplot as _plt
import utilities as _U

def ig_prmsUV_gt(a_gt, B_gt, sg2s, d_sg2s, sg2s_m1, ITER=1):
    ###  does well when a is large
    #y     = sg2s**(-a_gt-1)*_N.exp(-B_gt/sg2s)
    ly     = (-a_gt-1)*_N.log(sg2s) -B_gt/sg2s
    mly    = _N.max(ly)
    ly     -= mly
    y      = _N.exp(ly)

    for i in xrange(ITER):
        p     = _N.array(y[0:-1]*d_sg2s)   #  p weighted by bin size
        p     /= _N.sum(p)
        u = _N.sum(sg2s_m1 * p)
        vr= _N.sum((sg2s_m1-u)*(sg2s_m1-u) * p)

        print "u %(u).4f   vr %(vr).4f" % {"u" : u, "vr" : vr}
        a_ = u*u /vr + 2
        B_ = (a_ - 1)*u
    #_plt.plot(sg2s, y)

    return a_, B_

#def ig_prmsUV(sg2s, y, d_sg2s, sg2s_m1, ITER=1):
def ig_prmsUV(sg2s, sLLkPr, s, d_sg2s, sg2s_m1, ITER=1, nSpksM=0, clstr=-1, l0=None):
    sat = sLLkPr + s
    sat -= _N.max(sat)
    y = _N.exp(sat)


    ###  does well when a is large

    for i in xrange(ITER):
        p     = _N.array(y[0:-1]*d_sg2s)   #  p weighted by bin size
        p     /= _N.sum(p)
        u = _N.sum(sg2s_m1 * p)
        vr= _N.sum((sg2s_m1-u)*(sg2s_m1-u) * p)

        # fig = _plt.figure()
        # _plt.plot(sg2s, y)
        # _plt.xscale("log")
        #print "u %(u).4f   vr %(vr).4f" % {"u" : u, "vr" : vr}
        a_ = u*u /vr + 2
        B_ = (a_ - 1)*u

    # if clstr == 2:
    #     ufn = _U.uniqFN("p", serial=True)
    #     sv = _N.empty((len(sg2s), 4))
    #     sv[:, 0] = sg2s
    #     sv[:, 1] = sLLkPr
    #     sv[:, 2] = s
    #     sv[:, 3] = sat
    #     _U.savetxtWCom(ufn, sv, fmt="%.4e %.4e %.4e %.4e", com=("# nSpks=%(n)d    l0=%(l0).3e" % {"n" : nSpksM, "l0" : l0}))
    #_plt.plot(sg2s, y)

    
    if a_ > 10000:
        print "oops   nSpksM is %(n)d, cl %(c)d" % {"n" : nSpksM, "c" : clstr}
        if nSpksM > 0:
            fig = _plt.figure(figsize=(8, 4))
            sLLkPr -= _N.max(sLLkPr)
            s -= _N.max(sLLkPr)

            plths = _N.empty((len(sLLkPr), 4))
            plths[:, 0] = sg2s
            plths[:, 1] = sLLkPr
            plths[:, 2] = sat
            plths[:, 3] = s
            _N.savetxt("plths.dat", plths, fmt="%.5e %.5f %.5f %.5f")
            fig.add_subplot(2, 1, 1)

            _plt.plot(sg2s, sLLkPr, lw=2)
            _plt.plot(sg2s, s)
            _plt.xscale("log")
            fig.add_subplot(2, 1, 2)
            _plt.plot(sg2s, sat, lw=2)
            _plt.xscale("log")
            _plt.savefig("%(it)d,%(cl)d" % {"it" : ITER, "cl" : clstr})
            _plt.close()

        if u > 1.5:  # wide  B / (a+1)   
            print "u > 1.5    clstr %(c)d   nSpksM %(n)d" % {"c" : clstr, "n" : nSpksM}
            return 0.1+0.2*_N.random.rand(), 1+20*_N.random.rand()   #  just wide
        
    #assert a_ < 10000, "clstr %(cl)d   a_ too big in ig_prmsUV u %(u).3e  vr %(vr).3e  a_ %(a_).3e" % {"u" : u, "vr" : vr, "a_" : a_, "cl" : clstr}
    return a_, B_

#  sg2s is the x-axis over which posterior is numerically evaluated
#  sLLkPr  spiking likelihood + prior
#  s is the spatial modulation due to silence
def mltpl_ig_prmsUV(sg2s, sLLkPr, s, d_sg2s, sg2s_m1, clstsz, l0=None):
    sat = sLLkPr + s
    sat -= _N.max(sat, axis=0)
    y = _N.exp(sat)

    # print y.shape
    # print d_sg2s.shape
    # ###  does well when a is large

    p     = _N.array(y[0:-1]*d_sg2s)   #  p weighted by bin size
    p     /= _N.sum(p, axis=0)

    u = _N.sum(sg2s_m1 * p, axis=0)
    vr= _N.sum((sg2s_m1-u)*(sg2s_m1-u) * p, axis=0)

    a_ = u*u /vr + 2
    B_ = (a_ - 1)*u

    agt = _N.where(a_ > 10000)[0]

    for im in agt:
        print "oops   nSpksM is %(n)d, cl %(c)d" % {"n" : clstsz[im], "c" : clstr}
        if u[im] > 1.5:  # wide  B / (a+1)   
            print "u > 1.5    clstr %(c)d   nSpksM %(n)d" % {"c" : im, "n" : clstsz[im]}
            a_[agt] = 0.1+0.2*_N.random.rand()
            B_[agt] = 1+20*_N.random.rand()
            #  just wide
        
    return a_, B_


def ig_prmsMU(sg2s, y, d_sg2s, sg2s_m1, ITER=1):
    mode  = sg2s[_N.where(y == _N.max(y))[0][0]]  #  find mode

    p     = _N.array(y[0:-1]*d_sg2s)   #  p weighted by bin size
    p     /= _N.sum(p)

    u= _N.sum(sg2s[0:-1] * p)

    a_ = (mode+u)/(u-mode)
    B_ = (2*mode*u) / (u-mode)

    return a_, B_


