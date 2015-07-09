import numpy as _N
import pickle
import EnDedirs as _edd

class SimPlaceCell:
    setname       = None

    #  Tuning 
    Nc  = 0
    ux  = None     #  place fields
    wx  = None     #  width place fields

    um  = None     #  mean in mark space
    wm  = None     #  width in mark space

    alp = None     #  max Hz 

    Nx  = None     #  used for representing lambda
    Nm  = None

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

    def create(self, setname, xc=None):
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
        qdm = _N.empty((oo.Nc, oo.Nm))
        oo.lmd = _N.zeros((oo.Nx, oo.Nm))
        oo.kde = _N.zeros((oo.Nx, oo.Nm))

        xR  = _N.linspace(-oo.xA, oo.xA, oo.Nx)
        mR  = _N.linspace(0, oo.mA, oo.Nm)

        for i in xrange(oo.Nc):
            wdx     = 2*oo.wx[i]**2
            wdm     = 2*oo.wm[i]**2
            qdx[i] = (xR - oo.ux[i])**2 / wdx
            qdm[i] = (mR - oo.um[i])**2 / wdm

        for n in xrange(oo.Nc):
            for ix in xrange(oo.Nx):
                for im in xrange(oo.Nm):
                    oo.lmd[ix, im] += _N.exp(oo.alp[n] - qdx[n, ix] - qdm[n, im])

        """
        ###  now create spikes

        oo.marks    = _N.empty(oo.N, dtype=list)
        oo.markstr  = _N.empty(oo.Ntr, dtype=list)

        for nc in xrange(oo.Nc):        
            wdx     = 2*oo.wx[nc]**2
            wdm     = 2*oo.wm[nc]**2

            fr   = _N.exp(oo.alp[nc] - (oo.x - oo.ux[nc])**2 / wdx)
            frtr = _N.exp(oo.alp[nc] - (oo.xtr - oo.ux[nc])**2 / wdx)

            rands = _N.random.rand(oo.N)
            randstr = _N.random.rand(oo.Ntr)
            thrX  = _N.where(rands < fr*oo.dt)[0]
            thrXtr  = _N.where(randstr < frtr*oo.dt)[0]
            mk    = oo.um[nc] + oo.wm[nc]*_N.random.randn(len(thrX))
            mktr    = oo.um[nc] + oo.wm[nc]*_N.random.randn(len(thrXtr))

            for n in xrange(len(thrX)):   # iterate over time
                if oo.marks[thrX[n]] is None:
                    oo.marks[thrX[n]] = [mk[n]]
                else:
                    oo.marks[thrX[n]].append(mk[n])            
            for n in xrange(len(thrXtr)):   # iterate over time, training
                ui = thrXtr[n]   #  time and location of spike
                if oo.markstr[ui] is None:
                    oo.markstr[ui] = [mktr[n]]
                else:
                    oo.markstr[ui].append(mktr[n])

        dmp = open(_edd.resFN("marks.dump", dir=oo.setname, create=True), "wb")
        pickle.dump(oo, dmp)
        dmp.close()
        """
