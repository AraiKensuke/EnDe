import stats_util as s_u
import numpy as _N
import pickle
import matplotlib.pyplot as _plt

class fitMvNorm:
    # HYPER PARAMS covariance: nu, PSI
    hy_nu    = 5
    hy_PSI   = None
    # HYPER PARAMS mean: nu, PSI
    hy_mu_mu = None
    hy_mu_sg = None
    ihy_mu_sg= None

    #  HYPER PARAMS mixture coeff
    hy_alp   = None
    #####

    #  how many clusters do I think there are
    M        = 10

    ITERS = 1

    #  samples of mu, cov
    scov = None
    smu  = None
    mnd  = None

    mu   = None    #  centers of the M clusters

    sm      = None   #  cluster weight

    def load_init(self):
        oo = self
        with open("mND.dump", "rb") as f:
            oo.mnd = pickle.load(f)
        mnd = oo.mnd
        
        oo.hy_PSI = _N.empty((oo.M, mnd.k, mnd.k))

        for im in xrange(oo.M):
            oo.hy_PSI[im] = _N.eye(mnd.k)*0.5

        oo.hy_mu_mu = _N.zeros(mnd.k)
        oo.hy_mu_sg = _N.eye(mnd.k)*0.5
        oo.ihy_mu_sg= _N.linalg.inv(oo.hy_mu_sg)

    def fit(self):
        oo = self
        mnd    = oo.mnd

        #  Gibbs sampling 
        oo.scov = _N.empty((oo.ITERS, oo.M, mnd.k, mnd.k))
        oo.smu  = _N.empty((oo.ITERS, oo.M, mnd.k))
        
        oo.gz   = _N.zeros((oo.ITERS, mnd.N, oo.M), dtype=_N.int)

        for im in xrange(oo.M):
            oo.scov[0, im] = _N.cov(mnd.x.T)
            oo.smu[0, im]  = mnd.u[im]#_N.random.randn(mnd.k)
            #oo.smu[0, im]  = _N.random.randn(mnd.k)

        expTrm = _N.empty(oo.M)
        oo.sm   = _N.ones((oo.ITERS, oo.M))/oo.M

        crats = _N.zeros(oo.M+1)
        rands = _N.random.rand(mnd.N, 1)

        dirArgs = _N.empty(oo.M, dtype=_N.int)

        ##  Initialize
        #oo.smu
        #oo.scov 
        #oo.sm
        ##  Generate

        oo.hy_alp = _N.ones(oo.M) * (mnd.N/oo.M)
        for it in xrange(oo.ITERS-1):
            iscov = _N.linalg.inv(oo.scov[it])
            norms = 1/_N.sqrt(2*_N.pi*_N.linalg.det(oo.scov[it]))

            for n in xrange(mnd.N):
                #print "n %d" % n
                for im in xrange(oo.M):
                    expTrm[im] = _N.exp(-0.5*_N.dot(mnd.x[n] - oo.smu[it, im], _N.dot(iscov[im], mnd.x[n] - oo.smu[it, im])))
                rats = oo.sm[it]*expTrm*norms

                rats /= _N.sum(rats)

                for im in xrange(oo.M):
                    crats[im+1] = rats[im] + crats[im]
                inds = _N.where((rands[n] >= crats[:-1]) & (rands[n] <= crats[1:]))[0]
                try:
                    oo.gz[it+1, n, inds[0]] = 1
                except IndexError:
                    print "Index Error   it=%(it)d    n=%(n)d" % {"it" : it, "n" : n}
                    print expTrm
                    print norms
                    print oo.sm[it]
                    print rats
                    print crats
                    print "Index Error    %d" % inds

            _N.add(oo.hy_alp, _N.sum(oo.gz[it+1], axis=0), out=dirArgs)
            oo.sm[it+1] = _N.random.dirichlet(dirArgs)
                

            #  update oo.ms
            #  oo.ms[it+1]
            clr = ["blue", "black", "red"]
            for im in xrange(oo.M):
                #print "^^^^^^^^^"
                minds = _N.where(oo.gz[it+1, :, im] == 1)[0]

                clstx    = mnd.x[minds]
                mc       = _N.mean(clstx, axis=0)
                Nm       = clstx.shape[0]

                po_mu_sg = _N.linalg.inv(oo.ihy_mu_sg + Nm*iscov[im])
                po_mu_mu  = _N.dot(po_mu_sg, _N.dot(oo.ihy_mu_sg, oo.hy_mu_mu) + Nm*_N.dot(iscov[im], mc))
                oo.smu[it+1, im] = _N.random.multivariate_normal(po_mu_mu, po_mu_sg)

                po_sg_dof = oo.hy_nu + Nm
                po_sg_PSI = oo.hy_PSI[im] + _N.dot((clstx - oo.smu[it+1, im]).T, (clstx-oo.smu[it+1, im]))

                oo.scov[it+1, im] = s_u.sample_invwishart(po_sg_PSI, po_sg_dof)

