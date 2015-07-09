import stats_util as s_u
import numpy as _N
import pickle
import matplotlib.pyplot as _plt
import scipy.cluster.vq as scv
import warnings
#warnings.filterwarnings("error")

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

        #try:
        scr, lab = scv.kmeans2(mnd.x, mnd.M)
        #except Warning:
        #    pass
        sinds = [i[0] for i in sorted(enumerate(mnd.x[:, 0]), key=lambda x:x[1])] 
        SI = mnd.N / oo.M
        covAll = _N.cov(mnd.x.T)
        dcovMag= _N.diagonal(covAll)*0.125
        print dcovMag


        for im in xrange(oo.M):
            kinds = _N.where(lab == im)[0]
            oo.scov[0, im] = covAll*0.125
            
            if len(kinds) > 0:
                oo.smu[0, im]  = _N.mean(mnd.x[kinds], axis=0)
            else:
                oo.smu[0, im]  = _N.mean(mnd.x[SI*im:SI*(im+1)], axis=0)

        expTrm = _N.empty((oo.M, mnd.N))
        expArg = _N.empty((oo.M, mnd.N))
        oo.sm   = _N.ones((oo.ITERS, oo.M, 1))/oo.M

        crats = _N.zeros((oo.M+1, mnd.N))
        rands = _N.random.rand(mnd.N, 1)

        dirArgs = _N.empty(oo.M, dtype=_N.int)

        ##  Initialize
        #oo.smu
        #oo.scov 
        #oo.sm
        ##  Generate

        oo.hy_alp = _N.ones(oo.M) * (mnd.N/oo.M)
        rsum = _N.empty((1, mnd.N))
        skpM   = _N.arange(0, mnd.N)*oo.M

        for it in xrange(oo.ITERS-1):
            iscov = _N.linalg.inv(oo.scov[it])
            norms = 1/_N.sqrt(2*_N.pi*_N.linalg.det(oo.scov[it]))
            norms = norms.reshape(oo.M, 1)

            for im in xrange(oo.M):
                #expTrm[im] = _N.exp(-0.5*_N.sum(_N.multiply((mnd.x-oo.smu[it, im]), _N.dot(mnd.x-oo.smu[it, im], iscov[im])), axis=1))
                expArg[im] = -0.5*_N.sum(_N.multiply((mnd.x-oo.smu[it, im]), _N.dot(mnd.x-oo.smu[it, im], iscov[im])), axis=1)   #  expArg[im] is size N

            
            rexpArg = expArg.T.reshape(oo.M*mnd.N)
            lrgInM = expArg.argmax(axis=0)
            lrgstArgs = rexpArg[skpM+lrgInM]
            expArg0 = expArg - lrgstArgs

            # expArg -= expArg[:, 0]
            # print expArg[:, 0].shape
            # print expArg[:, 0]
            expTrm = _N.exp(expArg0)
            rats = oo.sm[it]*expTrm*norms  #  shape is oo.M x oo.N
            _N.sum(rats, axis=0, out=rsum[0, :])

            rats /= rsum   #  each column of "rats" sums to 1

            for im in xrange(oo.M):
                crats[im+1] = rats[im] + crats[im]

            rands = _N.random.rand(mnd.N)
            rrands = _N.tile(rands, oo.M).reshape(oo.M, mnd.N)
            irow, icol = _N.where((rrands >= crats[:-1]) & (rrands <= crats[1:]))
            if len(irow) == 0:
                print "^^^^^^^^^^^^^^^^^^^^^^   %d"  % it
                print irow
                print icol
                print crats
                print expArg
                
                return crats, expArg
            #print rats
            #print rrands

            try:
                oo.gz[it+1, icol, irow] = 1
                #print "total particles %d" % _N.sum(_N.sum(oo.gz[it+1], axis=0))
            except IndexError:
                print "Index Error   it=%(it)d    n=%(n)d" % {"it" : it, "n" : n}

            _N.add(oo.hy_alp, _N.sum(oo.gz[it+1], axis=0), out=dirArgs)
            oo.sm[it+1, :, 0] = _N.random.dirichlet(dirArgs)
                

            #  update oo.ms
            #  oo.ms[it+1]

            for im in xrange(oo.M):
                minds = _N.where(oo.gz[it+1, :, im] == 1)[0]

                if len(minds) > 0:
                    clstx    = mnd.x[minds]
                    mc       = _N.mean(clstx, axis=0)
                    Nm       = clstx.shape[0]

                    po_mu_sg = _N.linalg.inv(oo.ihy_mu_sg + Nm*iscov[im])
                    po_mu_mu  = _N.dot(po_mu_sg, _N.dot(oo.ihy_mu_sg, oo.hy_mu_mu) + Nm*_N.dot(iscov[im], mc))
                    oo.smu[it+1, im] = _N.random.multivariate_normal(po_mu_mu, po_mu_sg)

                    po_sg_dof = oo.hy_nu + Nm
                    po_sg_PSI = oo.hy_PSI[im] + _N.dot((clstx - oo.smu[it+1, im]).T, (clstx-oo.smu[it+1, im]))

                    #print po_sg_PSI.shape
                    try:
                        oo.scov[it+1, im] = s_u.sample_invwishart(po_sg_PSI, po_sg_dof)
                        dgl = _N.diagonal(oo.scov[it + 1, im])
                        rat = dgl / dcovMag
                        bgr = _N.where(rat > 1)[0]
                        if len(bgr) > 0:
                            #print "making smaller"
                            scl = rat[_N.argmax(rat)]
                            oo.scov[it+1, im] /= scl
                    except ValueError:
                        print "^^^^^^^^"
                        #print po_sg_PSI
                else:  #  no marks assigned to this cluster 
                    oo.scov[it+1, im] = oo.scov[it, im]
                    oo.smu[it+1, im]  = oo.smu[it, im]
