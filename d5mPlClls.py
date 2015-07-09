import numpy as _N
import pickle
import EnDedirs as _edd
import time as _tm
import d5mkde as _kde

class SimPlaceCell:
    setname       = None

    #  Tuning 
    Nc  = 0
    ux  = None     #  place fields
    wx  = None     #  width place fields

    um  = None     #  mean in mark space
    cm  = None     #  width in mark space

    alp = None     #  max Hz 

    Nx  = None     #  used for representing lambda
    Nm  = None

    kde = None

    #  Create behavior
    a   = 0.993
    sd  = 0.1
    N   = 10000   #time points.  every 1ms
    Ntr = 10000   #  training
    xtr = None
    x   = None
    xA  = None
    marks = None
    dt  = 0.001

    kde = None    #  kernel density etimate
    mdim= None    #  mark dimension
    #  kernel bandwidths.  In units of std. dev.
    cBm  = None;   Bx  = None;   bx = None
    #  Bandwidth of mark is   B = cBm . I, I is mdim
    

    def create(self, setname, xc=None):
        t1 = _tm.time()
        isqrt2pi = 1/_N.sqrt(2*_N.pi)
        oo = self
        oo.setname = setname
        oo.x      = _N.empty(oo.N)
        oo.xtr    = _N.empty(oo.Ntr)

        ###  training data
        oo.xtr[0]= _N.random.randn()*0.3

        rd  = oo.sd*_N.random.randn(oo.Ntr)
        for n in xrange(1, oo.Ntr):
            oo.xtr[n] = oo.a*oo.xtr[n-1] + rd[n]

        ###  behavioral data
        if xc is None:
            oo.x[0]= _N.random.randn()*0.3

            rd  = oo.sd*_N.random.randn(oo.N)
            for n in xrange(1, oo.N):
                oo.x[n] = oo.a*oo.x[n-1] + rd[n]
        else:
            oo.x[:] = xc
        qdx = _N.empty((oo.Nc, oo.Nx))

        # 4 mark vectors
        # ([m11, m12, m13], [m21, m22, m23], [m31, m32, m33], [m41, m42, m43])
        #  B is I . B^2, B scalara  M-
        cnard = [oo.Nc, oo.Nx]
        nard = [oo.Nx]
        for d in xrange(oo.mdim):
            nard.append(oo.Nm)
        cnard.append(oo.Nm**oo.mdim)
        qdm = _N.empty((oo.Nc, oo.Nm**oo.mdim))
        
        oo.lmd  = _N.zeros(nard) # nard=(Nx, Nm, Nm, Nm...)
        lmdC    = _N.zeros(cnard) # nard=(Nx, Nm, Nm, Nm...).  lmd for each cell

        xR  = _N.linspace(-oo.xA, oo.xA, oo.Nx)
        mR  = _N.linspace(0, oo.mA, oo.Nm)
        oo.zrspksx = []

        mkpts = oo.Nm**oo.mdim
        args = []
        for d in xrange(oo.mdim):
            args.append(mR)

        #  # pts of mesh   x   mark dim    is size of axyz
        axyz = _N.array(_N.meshgrid(*args, indexing="ij")) 
        axyz = axyz.reshape(oo.mdim, mkpts).T
        oo.lmd = oo.lmd.reshape((oo.Nx, mkpts))

        qdm = _N.empty((oo.Nc, mkpts))

        for nc in xrange(oo.Nc):   #  cell identity
            wdx     = 2*oo.wx[nc]**2
            qdx[nc] = (xR - oo.ux[nc])**2 / wdx
            for n in xrange(mkpts):  #  sample domain of the marks
                qdm[nc, n] = _N.dot(axyz[n]-oo.um[nc], _N.dot(oo.iCMm[nc], axyz[n]-oo.um[nc]))

        t2 = _tm.time()
        m = _N.empty(oo.mdim)
        for nc in xrange(oo.Nc):
            for ix in xrange(oo.Nx):  #qdm[nc, ...]
                lmdC[nc, ix] = _N.exp(oo.alp[nc] - qdx[nc, ix] - qdm[nc])

        _N.sum(lmdC, axis=0, out=oo.lmd)  #  sum over Nc neurons
        oo.lmd = oo.lmd.reshape(nard)
        
        t3 = _tm.time()

        ###  now create spikes

        marks    = _N.empty(oo.N, dtype=list)
        markstr  = _N.empty(oo.Ntr, dtype=list)

        for nc in xrange(oo.Nc):        
            wdx     = 2*oo.wx[nc]**2

            fr   = _N.exp(oo.alp[nc] - (oo.x - oo.ux[nc])**2 / wdx)
            frtr = _N.exp(oo.alp[nc] - (oo.xtr - oo.ux[nc])**2 / wdx)

            rands = _N.random.rand(oo.N)
            randstr = _N.random.rand(oo.Ntr)
            thrX  = _N.where(rands < fr*oo.dt)[0]

            thrXtr  = _N.where(randstr < frtr*oo.dt)[0]
            mk    = _N.random.multivariate_normal(oo.um[nc], oo.CMm[nc], size=len(thrX))
            mktr    = _N.random.multivariate_normal(oo.um[nc], oo.CMm[nc], size=len(thrXtr))

            for n in xrange(len(thrX)):   # iterate over time
                tm = thrX[n]
                if marks[tm] is None:
                    marks[tm] = [tm, mk[n]]
                else:
                    marks[tm].append(mk[n])
            for n in xrange(len(thrXtr)):   # iterate over time
                tm = thrXtr[n]
                if markstr[tm] is None:
                    markstr[tm] = [tm, mktr[n]]
                else:
                    markstr[tm].append(mktr[n])

        oo.marks = []
        oo.markstr = []
        for n in xrange(oo.N):
            if marks[n] is not None:
                oo.marks.append(marks[n])
        for n in xrange(oo.Ntr):
            if markstr[n] is not None:        #  spike here
                oo.markstr.append(markstr[n])
            else:
                oo.zrspksx.append(oo.xtr[n])

        # oo.kde1 = _kde.kde(setname)
        # oo.kde1.est(oo.xtr, oo.mdim, markstr, oo.xA, oo.mA, oo.Ntr, oo.Nx, oo.Nm, oo.Bx, oo.cBm, oo.bx, t0=0, t1=10000, filename="kde1.dump")
        # oo.kde2 = _kde.kde(setname)
        # oo.kde2.est(oo.xtr, oo.mdim, markstr, oo.xA, oo.mA, oo.Ntr, oo.Nx, oo.Nm, oo.Bx, oo.cBm, oo.bx, t0=5000, t1=10000, filename="kde2.dump")
        
        dmp = open(_edd.resFN("marks.dump", dir=oo.setname, create=True), "wb")
        pickle.dump(oo, dmp)
        dmp.close()
