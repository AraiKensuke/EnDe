import numpy as _N
import pickle
import EnDedirs as _edd
mvn = _N.random.multivariate_normal

class mixNormDat:
    k      = 2      #  data dimensionality
    N      = 500    #  no of datas
    M      = 3      #  no components

    u      = None
    Cv     = None

    x      = None   #  position + marks
    clstr  = None   #  which cluster?

    m      = None   #  cluster weight
    md     = 3.4      #  How far are clusters separated, compared to acov
    acov   = 0.2

    def create(self, setname):
        #  time series to model ratio of the states
        oo  = self
        k   = oo.k
        N   = oo.N
        M   = oo.M
        
        oo.u   = _N.empty((M, N, k))     #  I want 
        oo.Cov = _N.empty((M, k, k))
        oo.clstr= _N.ones(N, dtype=_N.int)*-1

        oo.m   = _N.empty((M, N))
        oo.m[:, 0]   = _N.random.rand(M)
        oo.m[:, 0]   /= _N.sum(oo.m[:, 0])

        mr  = _N.random.rand(N)
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

        crat= _N.zeros(M+1)
        cstab = 0.*_N.random.rand(M)   #  how stable is mixture coeff
        _mstab = 0.*oo.md*_N.random.rand(M)   #  how stable is mark?
        mstab = _N.tile(_mstab, k).reshape(M, k)

        c_a  = 0.9999999*_N.ones(M)           # AR coeff.
        c_ls = _N.zeros((M, N))
        _m_a  = 0.9999999*_N.ones(M)           # AR coeff.  k component ls
        m_a = _N.tile(_m_a, k).reshape(M, k)

        m_ls = _N.zeros((M, N, k))

        for n in xrange(N):
            if n > 0:
                c_ls[:, n] += c_a*c_ls[:, n-1] + cstab*_N.random.randn(M)
                m_ls[:, n] += m_a*m_ls[:, n-1] + mstab*_N.random.randn(M, k)
            oo.m[:, n] = oo.m[:, 0] + c_ls[:, n]
            oo.m[(oo.m[:, n] < 0), n] = 0
            oo.u[:, n] = oo.u[:,0] + m_ls[:, n]

            oo.m[:, n] /= _N.sum(oo.m[:, n])
            for im in xrange(M):
                crat[im+1] = oo.m[im, n] + crat[im]

            for im in xrange(M):
                inds = _N.where((mr[n] >= crat[0:-1]) & (mr[n] <= crat[1:]))[0]
                oo.clstr[n] = inds[0]
                
            oo.x[n]   = mvn(oo.u[oo.clstr[n], n], oo.Cov[oo.clstr[n]])

        dmp = open(_edd.resFN("mND.dump", dir=setname, create=True), "wb")
        pickle.dump(oo, dmp)
        dmp.close()
