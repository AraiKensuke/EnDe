import numpy as _N
import matplotlib.pyplot as _plt
import pickle

class trueLikelihoods:
    #  From 
    #  Cov
    #  stdP
    #  um
    #  uP
    
    twpi = _N.sqrt(2*_N.pi)
    mND  = None
    Nx   = None
    xA   = None

    jm   = None   #  just the marks, no Nones
    jp   = None
    mt   = None

    def __init__(self, mndfile):
        oo = self
        with open(mndfile, "rb") as f:
            oo.mND = pickle.load(f)

    def lklhd(self, fxdMks, t):
        oo = self
        Lam = oo.LAM(t)
        iSgs= _N.linalg.inv(oo.mND.Cov)

        cmps= _N.empty((oo.mND.M, oo.Nx))
        rhs = _N.empty(oo.mND.k)

        #  prob at each value of xA
        for m in xrange(oo.mND.M):
            lnp = (oo.xA - oo.mND.uP[m, 0, t])**2 / (2*oo.mND.stdP[m, 0, t]**2)
            _N.dot(iSgs[m], (fxdMks - oo.mND.um[m, t]).T, out=rhs)
            _N.dot(fxdMks-oo.mND.um[m, t], rhs)
            cmps[m] = oo.mND.alp[m, 0, t] * _N.exp(-0.5*_N.dot(fxdMks-oo.mND.um[m, t], rhs) - 0.5 * lnp)

        zs = _N.sum(cmps, axis=0)

        return zs*0.001*Lam

    def LAM(self, t):
        oo = self
        #  For our case, 
        cmps= _N.empty((oo.mND.M, 51))
        iCov = _N.linalg.inv(oo.mND.Cov)


        for m in xrange(oo.mND.M):
            lnp = (oo.xA - oo.mND.uP[m, 0, t])**2 / (2*oo.mND.stdP[m, 0, t]**2)
            cmps[m] = oo.mND.alp[m, 0, t]*_N.sqrt(oo.twpi*_N.linalg.det(oo.mND.Cov[m])) * _N.exp(-0.5*lnp)

        zs = _N.sum(cmps, axis=0)
        return zs

    def theseMarks(self, t0, t1):
        oo = self
        
        nons    = _N.equal(oo.mND.marks[t0:t1], None)
        mInds  = _N.where(nons == False)[0] + t0

        print mInds

        oo.mND.marks[mInds]
        oo.mND.pos[mInds]

        marks = []
        pos   = []
        ts    = []

        iii  = -1

        #  oo.mND.marks      --  array
        #  oo.mND.marks[0]   --  list
        #  oo.mND.marks]

        for mkl in oo.mND.marks[mInds]:
            iii += 1

            for i in xrange(len(mkl[0])):
                marks.append(mkl[0][i])
                pos.append(oo.mND.pos[mInds[iii]])
                ts.append(mInds[iii])

        oo.jm   = _N.array(marks)
        oo.jp   = _N.array(pos)
        oo.ts   = _N.array(ts)



        oo.Nx     = 51
        oo.xA     = _N.linspace(-6, 6, oo.Nx)

        N      = len(mInds)

        #  oo.mND.alp   (M x Npf x T)
        #  oo.mND.um    (M x T x k)
        #  oo.mND.uP    (M x Npf x T)
        #  oo.mND.Cov   (M x k x k)
        #  oo.mND.stdP  (M x Npf x T)


        scale = 1000.
        it0 = int(t0)
        it1 = int(t1)

        pg   = 0
        onPg = 0

        for n in xrange(len(ts)):
            t = ts[n]
            if onPg == 0:
                fig = _plt.figure(figsize=(13, 8))        
            fig.add_subplot(4, 6, onPg + 1)
            _plt.plot(oo.xA, oo.lklhd(oo.jm[n], oo.ts[n]), color="black")
            _plt.axvline(x=oo.jp[n], color="red", lw=2)
            _plt.yticks([])
            _plt.xticks([-6, -3, 0, 3, 6])
            _plt.title("t = %.3f" % (float(t) / scale))
            onPg += 1

            if onPg >= 24:
                fig.subplots_adjust(wspace=0.35, hspace=0.35, left=0.08, right=0.92, top=0.92, bottom=0.08)
                _plt.savefig("tLklhd_pg=%(pg)d" % {"pg" : pg})
                _plt.close()
                pg += 1
                onPg = 0

        if onPg > 0:
            fig.subplots_adjust(wspace=0.15, hspace=0.15, left=0.08, right=0.92, top=0.92, bottom=0.08)
            _plt.savefig("tLklhd_pg=%(pg)d" % {"pg" : pg})
            _plt.close()




