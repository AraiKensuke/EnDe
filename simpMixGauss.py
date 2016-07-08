import stats_util as s_u
import numpy as _N
import pickle
import matplotlib.pyplot as _plt
import scipy.cluster.vq as scv
import scipy.stats as _ss
import fitutil as _fu
import time as _tm

mvn    = _N.random.multivariate_normal
class simpMixGauss:
    ##################  hyper parameters - do not change during Gibbs
    # HYPER PARAMS for prior covariance: nu, PSI

    #  how many clusters do I think there are
    
    def fit(self, ITERS, M, x, _f_u=None, _f_q2=None, _q2_a=None, _q2_B=None, _ms_alp=None, f_0=None, q2_0=None, ms_0=None):
        """
        Fit, with the inverting done in blocks
        """
        oo = self
        nSpks  = x.shape[0]

        xLo    = _N.min(x)
        xHi    = _N.max(x)

        f      = _N.empty((ITERS, M, 1))
        q2     = _N.empty((ITERS, M, 1))
        ms     = _N.empty((ITERS, M, 1))
        xs = _N.sort(x)

        if ms_0 is None:
            rats = _N.ones(M)/M
            rats += (1./M)*0.1*_N.random.randn(M)
            rats /= _N.sum(rats)
            ms[0, :, 0] = rats
        else:
            ms[0, :, 0] = ms_0
        if f_0 is None:
            ii = 0
            for m in xrange(M):
                f[0, m, 0]  = _N.mean(xs[ii:ii+int(ms[0, m, 0]*nSpks)])
                ii += int(ms[0, m, 0]*nSpks)
        else:
            f[0, :, 0] = f_0
        if q2_0 is None:
            ii = 0
            for m in xrange(M):
                q2[0, m, 0]  = _N.std(xs[ii:ii+int(ms[0, m, 0]*nSpks)])**2
                ii += int(ms[0, m, 0]*nSpks)
        else:
            q2[0, :, 0] = q2_0

        xr = x.reshape((1, nSpks))
        gz   = _N.zeros((ITERS, nSpks, M), dtype=_N.int)

        if _f_u is None:
            _f_u = _N.empty(M)
            ii = 0
            for m in xrange(M):
                _f_u[m]  = _N.mean(xs[ii:ii+int(ms[0, m, 0]*nSpks)])
                ii += int(ms[0, m, 0]*nSpks)
        if _f_q2 is None:
            _f_q2 = _N.ones(M)*(10**2)
        if _q2_a is None:
            _q2_a = _N.ones(M)*1e-4
        if _q2_B is None:
            _q2_B  = _N.ones(M)*1e-3
        if _ms_alp is None:
            _ms_alp = _N.ones(M)*(1. / M)

        # #  termporary containers
        econt = _N.empty((M, nSpks))
        rat   = _N.zeros((M+1, nSpks))
        alp_  = _N.empty(M)

        # print ms[0, :, 0]
        # print f[0, :, 0]
        # print q2[0, :, 0]
        # print "^^^^^^^^"

        oo.f  = f
        oo.q2 = q2
        oo.ms = ms
        oo.gz = gz

        #  initial values given for it == 0
        for it in xrange(1, ITERS):
            ####################  STOCHASTIC ASSIGNMENT
            norms = 1/_N.sqrt(2*_N.pi*q2[it-1])
            zrs   = _N.where(ms[it-1] == 0)[0]
            ms[it-1, zrs, 0] = 1e-30
            lms   = _N.log(ms[it-1])

            iq2 = 1./q2[it-1]
            rnds       = _N.random.rand(nSpks)
            qdrSPC     = (f[it-1] - xr)*(f[it-1] - xr)*iq2  #  M x nSpks
            cont       = lms + norms - 0.5*qdrSPC

            _N.max(cont, axis=0)
            mcontr     = _N.max(cont, axis=0).reshape((1, nSpks))  
            cont       -= mcontr
            _N.exp(cont, out=econt)

            for m in xrange(M):
                rat[m+1] = rat[m] + econt[m]

            rat /= rat[M]

            # print rat

            M1 = rat[1:] >= rnds
            M2 = rat[0:-1] <= rnds

            gz[it] = (M1&M2).T
            #  prior for weights + likelihood used to sample
            #  _alp (prior) and alp_ posterior hyper

            _N.add(_ms_alp, _N.sum(gz[it], axis=0), out=alp_)

            ##############  SAMPLE WEIGHTS
            ms[it, :, 0] = _N.random.dirichlet(alp_)

            for m in xrange(M):
                thisgr = _N.where(gz[it, :, m] == 1)[0]
                nSpksC  = len(thisgr)
                if nSpksC > 0:

                    ####  sample component MEANS
                    xbar   = _N.mean(x[thisgr])

                    q2_nc  = q2[it-1,m,0]/nSpksC
                    Mu     =   (_f_q2[m] * xbar + _f_u[m]*q2_nc) / (_f_q2[m] + q2_nc)
                    S      =   (_f_q2[m] * q2_nc) / (_f_q2[m] + q2_nc)
                    f[it, m, 0] = Mu + _N.sqrt(S)*_N.random.randn()
                    # print "----"
                    # print xbar
                    # print Mu
                    # print S

                    ####  sample component variancs
                    a_ = 0.5*(nSpksC + 2*_q2_a[m])
                    B_ = 0.5*_N.sum((x[thisgr] - f[it, m, 0])*(x[thisgr] - f[it, m, 0])) + _q2_B[m]
                    q2[it, m, 0] = _ss.invgamma.rvs(a_, scale=B_)
                else:
                    f[it, m, 0] = f[it-1, m, 0]
                    q2[it, m, 0] = q2[it-1, m, 0]
            # print ms[it, :, 0]
            # print f[it, :, 0]
            # print q2[it, :, 0]
            # print "^^^^^^^^"
