import scipy.stats as _ss
import time as _tm
K   = 4
Sg0 = _N.array([[1.2,    0.3,   -0.4,   -0.1],
                [0.3,    1.8,   -0.2,   -0.3],
                [-0.4,  -0.2,   0.9,    -0.3],
                [-0.1,  -0.3,   -0.3,   1.7]])
u0  = _N.array([1.1, -0.3, 0.3, 1.2])

N   = 200
X = _N.random.multivariate_normal(u0, Sg0, size=N)

ITERS = 1000

us     = _N.empty((ITERS, K))
Sgs    = _N.empty((ITERS, K, K))

iSgs    = _N.empty((ITERS, K, K))

Sgs[0] = _N.eye(K)

mnx    = _N.mean(X, axis=0)

iSg_n_ = N

tWs    = 0
tiWs    = 0
for it in xrange(1, ITERS):
    #  hyper params for center
    iSgs = Sgs[it-1]
    u_Sg_ = _N.linalg.inv(iSgs*N)
    iu_Sg_= _N.linalg.inv(u_Sg_)
    iSg   = _N.linalg.inv(Sgs[it-1])
    u_u_ = _N.einsum("jk,k->j", u_Sg_, N*_N.dot(iSgs, mnx))

    us[it]    = _N.random.multivariate_normal(u_u_, u_Sg_)
    #  hyper params for 
    u         = us[it]
    # ##  dof of posterior distribution of cluster covariance
    ur = u.reshape((1, K))
    # #  dot((clstx-ur).T, (clstx-ur))==ZERO(K) when clstsz ==0

    M  = _N.dot((X - ur).T, (X-ur))
    iSg_V_ = _N.linalg.inv(M)

    t1 = _tm.time()
    iSg = _ss.wishart.rvs(df=iSg_n_, scale=iSg_V_)
    _N.linalg.inv(iSg)
    t2 = _tm.time()
    iSg = _ss.invwishart.rvs(df=iSg_n_, scale=M)

    t3 = _tm.time()

    tWs += (t2-t1)
    tiWs += (t3-t2)
    Sgs[it] = _N.linalg.inv(iSg)
    # #tt6b += (_tm.time()-infor)
