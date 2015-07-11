import stats_util as s_u
import numpy as _N
import pickle
import matplotlib.pyplot as _plt
import scipy.cluster.vq as scv

mvn    = _N.random.multivariate_normal
class fitMvNorm:
    # PARAMS for prior covariance: nu, PSI
    PR_cov_nu    = 3
    PR_cov_PSI   = None
    # HYPER PARAMS mean: nu, PSI
    PR_mu_mu = None
    PR_mu_sg = None
    iPR_mu_sg= None

    #  HYPER PARAMS mixture coeff
    PR_m_alp   = None
    #####

    #  how many clusters do I think there are
    M        = 10

    ITERS = 1

    #  samples of mu, cov
    scov = None
    smu  = None
    mnd  = None
    sm      = None   #  cluster weight

    #  augmented variables
    gz   = None

    def load(self):
        with open("mND.dump", "rb") as f:
            self.mnd = pickle.load(f)

    def init0(self, n1, n2):
        oo = self
        mnd = oo.mnd
        x   = mnd.x[n1:n2]
        N   = n2-n1
        k   = mnd.k
        M   = oo.M

        #  sampled variables
        oo.scov = _N.empty((oo.ITERS, M, k, k))
        oo.smu  = _N.empty((oo.ITERS, M, k))
        oo.gz   = _N.zeros((oo.ITERS, N, M), dtype=_N.int)
        oo.sm   = _N.ones((oo.ITERS, M, 1))/M

        ###  
        oo.PR_cov_PSI = _N.tile(_N.eye(k)*0.5, M).T.reshape(M, k, k)
        oo.PR_cov_nu  = _N.ones(M, dtype=_N.int)

        oo.PR_mu_mu = _N.zeros((M, k))
        oo.PR_mu_sg = _N.tile(_N.eye(k)*0.5, M).T.reshape(M, k, k)
        oo.iPR_mu_sg= _N.linalg.inv(oo.PR_mu_sg)

        oo.PR_m_alp = _N.ones(M) * (N/M)

        #  Gibbs sampling 
        ################  init cluster centers
        scr, lab = scv.kmeans2(x, mnd.M)

        SI = N / M
        covAll = _N.cov(x.T)
        dcovMag= _N.diagonal(covAll)*0.125

        for im in xrange(M):
            kinds = _N.where(lab == im)[0]
            oo.scov[0, im] = covAll*0.125
            
            if len(kinds) > 0:
                oo.smu[0, im]  = _N.mean(x[kinds], axis=0)
            else:
                oo.smu[0, im]  = _N.mean(x[SI*im:SI*(im+1)], axis=0)

    #  I need to know initial values smu, scov, sm.  Generate gz 
    def fit(self, n1, n2):
        oo = self
        mnd    = oo.mnd
        x   = mnd.x[n1:n2]
        N   = n2-n1
        k   = mnd.k
        M   = oo.M

        covAll = _N.cov(x.T)
        dcovMag= _N.diagonal(covAll)*0.125

        #  termporary containers
        expTrm = _N.empty((M, N))
        expArg = _N.empty((M, N))
        crats = _N.zeros((M+1, N))
        rands = _N.random.rand(N, 1)
        dirArgs = _N.empty(M, dtype=_N.int)

        rsum = _N.empty((1, N))
        skpM   = _N.arange(0, N)*M
        
        for it in xrange(oo.ITERS-1):
            iscov = _N.linalg.inv(oo.scov[it])
            norms = 1/_N.sqrt(2*_N.pi*_N.linalg.det(oo.scov[it]))
            norms = norms.reshape(M, 1)

            for im in xrange(M):
                expArg[im] = -0.5*_N.sum(_N.multiply((x-oo.smu[it, im]), _N.dot(x-oo.smu[it, im], iscov[im])), axis=1)   #  expArg[im] is size N
            
            rexpArg = expArg.T.reshape(M*N)
            lrgInM = expArg.argmax(axis=0)
            lrgstArgs = rexpArg[skpM+lrgInM]
            expArg0 = expArg - lrgstArgs

            expTrm = _N.exp(expArg0)
            rats = oo.sm[it]*expTrm*norms  #  shape is M x oo.N
            _N.sum(rats, axis=0, out=rsum[0, :])

            rats /= rsum   #  each column of "rats" sums to 1

            for im in xrange(M):
                crats[im+1] = rats[im] + crats[im]

            rands = _N.random.rand(N)
            rrands = _N.tile(rands, M).reshape(M, N)
            ###  THIS once broke because we had an empty cluster
            irw, icl = _N.where((rrands >= crats[:-1]) & (rrands <= crats[1:]))

            oo.gz[it+1, icl, irw] = 1   #  we must clean out gz

            #  _N.sum(oo.gz...) sz M   its vec of num. of obs of each state 'm'
            _N.add(oo.PR_m_alp, _N.sum(oo.gz[it+1], axis=0), out=dirArgs)
            oo.sm[it+1, :, 0] = _N.random.dirichlet(dirArgs)
            
            for im in xrange(M):
                minds = _N.where(oo.gz[it+1, :, im] == 1)[0]

                if len(minds) > 0:
                    clstx    = x[minds]
                    mc       = _N.mean(clstx, axis=0)
                    Nm       = clstx.shape[0]
                    ##  cov of posterior distribution of cluster means
                    po_mu_sg = _N.linalg.inv(oo.iPR_mu_sg[im] + Nm*iscov[im])
                    ##  mean of posterior distribution of cluster means
                    po_mu_mu  = _N.dot(po_mu_sg, _N.dot(oo.iPR_mu_sg[im], oo.PR_mu_mu[im]) + Nm*_N.dot(iscov[im], mc))
                    oo.smu[it+1, im] = mvn(po_mu_mu, po_mu_sg)

                    ##  dof of posterior distribution of cluster covariance
                    po_sg_dof = oo.PR_cov_nu[im] + Nm
                    ##  dof of posterior distribution of cluster covariance
                    po_sg_PSI = oo.PR_cov_PSI[im] + _N.dot((clstx - oo.smu[it+1, im]).T, (clstx-oo.smu[it+1, im]))

                    oo.scov[it+1, im] = s_u.sample_invwishart(po_sg_PSI, po_sg_dof)
                    dgl = _N.diagonal(oo.scov[it + 1, im])
                    rat = dgl / dcovMag
                    bgr = _N.where(rat > 1)[0]
                    if len(bgr) > 0:
                        #print "making smaller"
                        scl = rat[_N.argmax(rat)]
                        oo.scov[it+1, im] /= scl
                else:  #  no marks assigned to this cluster 
                    oo.scov[it+1, im] = oo.scov[it, im]
                    oo.smu[it+1, im]  = oo.smu[it, im]

    def set_priors_and_initial_values(self):
        """
        after a first run, 
        """
        oo = self
        mid = oo.ITERS/2
        #  the posteriors are now priors
        oo.PR_m_alp[:] = _N.sum(_N.mean(oo.gz[mid:oo.ITERS-1], axis=0), axis=0)
        oo.PR_mu_mu[:] = _N.mean(oo.smu[mid:oo.ITERS-1], axis=0)
        oo.PR_mu_sg[:] = _N.mean(oo.scov[mid:oo.ITERS-1], axis=0)
        oo.PR_cov_PSI[:] = _N.mean(oo.scov[mid:oo.ITERS-1], axis=0)
        oo.PR_cov_nu[:] = _N.sum(oo.gz[oo.ITERS-1], axis=0)

        # last sampled values will be starting values

        oo.sm[0]   = oo.sm[oo.ITERS-1]
        oo.smu[0]  = oo.smu[oo.ITERS-1]
        oo.scov[0] = oo.scov[oo.ITERS-1]

        #oo.gz   = _N.zeros((oo.ITERS, oo.mnd.N, oo.M), dtype=_N.int)
        oo.gz[:,:,:] = 0
