import numpy as _N
import pickle
import EnDedirs as _edd

class mixNormDat:
    k      = 2      #  data dimensionality
    N      = 500    #  no of datas
    M      = 3      #  no components

    u      = None
    Cv     = None
    x      = None
    clstr  = None   #  which cluster?

    m      = None   #  cluster weight
    md     = 5      #  How far are clusters separated, compared to acov
    acov   = 0.5

    def create(self, setname):
        oo  = self
        k   = oo.k
        N   = oo.N
        M   = oo.M
        
        oo.u   = _N.empty((M, k))
        oo.Cov = _N.empty((M, k, k))

        m   = _N.random.rand(M)
        m   /= _N.sum(m)
        crat= _N.zeros(M+1)

        for im in xrange(M):
            crat[im+1] = m[im] + crat[im]

        mr  = _N.random.rand(N)
        oo.clstr= _N.ones(N, dtype=_N.int)*-1
        for im in xrange(M):
            inds = _N.where((mr >= crat[im]) & (mr <= crat[im+1]))[0]
            oo.clstr[inds] = im

        oo.x   = _N.empty((N, k))


        #  set up cov. matrices
        for im in xrange(M):
            oo.u[im]   = _N.random.randn(k)*oo.md   
            for ik in xrange(k):
                oo.Cov[im, ik, ik] = oo.acov + _N.random.rand()
            for ik1 in xrange(k):
                for ik2 in xrange(ik1 + 1, k):
                    oo.Cov[im, ik1, ik2] = 0.03*_N.random.randn()
                    oo.Cov[im, ik2, ik1] = oo.Cov[im, ik1, ik2]

        for n in xrange(N):
            ##### create data
            oo.x[n]   = _N.random.multivariate_normal(oo.u[oo.clstr[n]], oo.Cov[oo.clstr[n]])


        dmp = open(_edd.resFN("mND.dump", dir=setname, create=True), "wb")
        pickle.dump(oo, dmp)
        dmp.close()
