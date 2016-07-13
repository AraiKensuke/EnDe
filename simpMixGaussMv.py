import stats_util as s_u
import numpy as _N
import pickle
import matplotlib.pyplot as _plt
import scipy.cluster.vq as scv
import scipy.stats as _ss
import fitutil as _fu
import time as _tm

mvn    = _N.random.multivariate_normal
twpi   = 2*_N.pi
class simpMixGaussMv:
    ##################  hyper parameters - do not change during Gibbs
    # HYPER PARAMS for prior covariance: nu, PSI

    #  how many clusters do I think there are
    
    def fit(self, ITERS, M, x, _u_u=None, _u_Sg=None, _Sg_nu=None, _Sg_PSI=None, _ms_alp=None, u_0=None, Sg_0=None, ms_0=None):
        """
        Fit, with the inverting done in blocks
        """
        oo = self
        mdim   = x.shape[1]
        nSpks  = x.shape[0]

        u      = _N.empty((ITERS, M, mdim))
        Sg     = _N.empty((ITERS, M, mdim, mdim))
        ms     = _N.empty((ITERS, M, 1))
        xs     = _N.sort(x)

        if ms_0 is None:
            rats = _N.ones(M)/M
            rats += (1./M)*0.1*_N.random.randn(M)
            rats /= _N.sum(rats)
            ms[0, :, 0] = rats
        # else:
        #     ms[0, :, 0] = ms_0
        if u_0 is None:
            for m in xrange(M):
                u[0, m] = _N.mean(x, axis=0)
        # else:
        #     u[0, :, 0] = f_0
        if Sg_0 is None:
            for m in xrange(M):
                Sg[0, m]  = _N.identity(mdim)
        # else:
        #     q2[0, :, 0] = q2_0

        mAS  = x   #  position @ spikes
        mASr = mAS.reshape((nSpks, 1, mdim))

        gz   = _N.zeros((ITERS, nSpks, M), dtype=_N.bool)

        if _u_u is None:
            _u_u = _N.empty((M, mdim))
            for m in xrange(M):
                _u_u[m] = _N.array(u[0, m])
        if _u_Sg is None:
            _u_Sg = _N.empty((M, mdim, mdim))
            for m in xrange(M):
                _u_Sg[m] = _N.cov(x, rowvar=0)
        if _Sg_nu is None:
            _Sg_nu = _N.ones((M, 1));  
        if _Sg_PSI is None:
            _Sg_PSI = _N.tile(_N.identity(mdim), M).T.reshape((M, mdim, mdim))*0.1
        if _ms_alp is None:
            _ms_alp = _N.ones(M)*(1. / M)

        # #  termporary containers
        econt = _N.empty((M, nSpks))
        rat   = _N.zeros((M+1, nSpks))
        alp_  = _N.empty(M)
        qdrMKS = _N.empty((M, nSpks))

        # print ms[0, :, 0]
        # print f[0, :, 0]
        # print q2[0, :, 0]
        # print "^^^^^^^^"

        oo.u  = u
        oo.Sg = Sg
        oo.ms = ms
        oo.gz = gz

        #  initial values given for it == 0

        for it in xrange(1, ITERS):
            ur         = u[it-1].reshape((1, M, mdim))
            iSg        = _N.linalg.inv(Sg[it-1])

            zrs   = _N.where(ms[it-1] == 0)[0]
            ms[it-1, zrs, 0] = 1e-30
            lms   = _N.log(ms[it-1])

            mkNrms = _N.log(1/_N.sqrt(twpi*_N.linalg.det(Sg[it-1])))
            mkNrms = mkNrms.reshape((M, 1))

            rnds       = _N.random.rand(nSpks)
            dmu        = (mASr - ur)

            _N.einsum("nmj,mjk,nmk->mn", dmu, iSg, dmu, out=qdrMKS)

            cont       = lms + mkNrms - 0.5*qdrMKS

            mcontr     = _N.max(cont, axis=0).reshape((1, nSpks))  
            cont       -= mcontr
            _N.exp(cont, out=econt)

            for m in xrange(M):
                rat[m+1] = rat[m] + econt[m]

            rat /= rat[M]

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
                if nSpksC > mdim:
                    u_Sg_ = _N.linalg.inv(_N.linalg.inv(_u_Sg[m]) + nSpksC*iSg[m])
                    clstx    = mAS[thisgr]

                    mcs       = _N.mean(clstx, axis=0)
                    u_u_ = _N.einsum("jk,k->j", u_Sg_, _N.dot(_N.linalg.inv(_u_Sg[m]), _u_u[m]) + nSpksC*_N.dot(iSg[m], mcs))
                    u[it, m] = _N.random.multivariate_normal(u_u_, u_Sg_)

                    ####  sample component variancs
                    Sg_nu_ = _Sg_nu[m, 0] + nSpksC
                    ##  dof of posterior distribution of cluster covariance
                    ur = u[it, m].reshape((1, mdim))
                    Sg_PSI_ = _Sg_PSI[m] + _N.dot((clstx - ur).T, (clstx-ur))
                    Sg[it, m] = s_u.sample_invwishart(Sg_PSI_, Sg_nu_)
                else:
                    u[it, m]  = u[it-1, m]
                    Sg[it, m] = Sg[it-1, m]

            
        
