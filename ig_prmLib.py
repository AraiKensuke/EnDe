import numpy as _N

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

def ig_prmsUV(sg2s, y, d_sg2s, sg2s_m1, ITER=1):
    ###  does well when a is large

    for i in xrange(ITER):
        p     = _N.array(y[0:-1]*d_sg2s)   #  p weighted by bin size
        p     /= _N.sum(p)
        u = _N.sum(sg2s_m1 * p)
        vr= _N.sum((sg2s_m1-u)*(sg2s_m1-u) * p)

        #print "u %(u).4f   vr %(vr).4f" % {"u" : u, "vr" : vr}
        a_ = u*u /vr + 2
        B_ = (a_ - 1)*u
    #_plt.plot(sg2s, y)

    return a_, B_


def ig_prmsMU(a_gt, B_gt, sg2s, d_sg2s, sg2s_m1, ITER=1):
    y     = sg2s**(-a_gt-1)*_N.exp(-B_gt/sg2s)

    for i in xrange(ITER):
        mode  = sg2s[_N.where(y == _N.max(y))[0][0]]

        p     = _N.array(y[0:-1]*d_sg2s)   #  p weighted by bin size
        p     /= _N.sum(p)

        u= _N.sum(sg2s[0:-1] * p)

        a_ = (mode+u)/(u-mode)
        B_ = (2*mode*u) / (u-mode)

    return a_, B_
