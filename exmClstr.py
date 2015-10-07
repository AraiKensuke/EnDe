import mvn
from EnDedirs import resFN
import numpy as _N
import matplotlib.pyplot as _plt
import scipy.cluster.vq as scv

def show_posmarks(dec, setname):
    for nt in xrange(dec.nTets):
        fig = _plt.figure(figsize=(10, 8))
        for k in xrange(1, dec.mdim+1):
            fig.add_subplot(2, 2, k)
            x = []
            y = []

            for l in xrange(dec.tt0, dec.tt1):
                if (dec.marks[l, nt] is not None):
                    x.append(dec.pos[l])
                    y.append(dec.marks[l, nt][0][k-1])

            _plt.scatter(x, y, color="black", s=2)
            _plt.scatter(dec.mvNrm[nt].us[:, 0], dec.mvNrm[nt].us[:, k], color="red", s=30)

            for m in xrange(dec.M):
                ux   = dec.mvNrm[nt].us[m, 0]  #  position
                uy   = dec.mvNrm[nt].us[m, k]
                ex_x = _N.sqrt(dec.mvNrm[nt].covs[m, 0, 0])
                ex_y = _N.sqrt(dec.mvNrm[nt].covs[m, k, k])
                _plt.plot([ux-ex_x, ux+ex_x], [uy, uy], color="red", lw=2)
                _plt.plot([ux, ux], [uy-ex_y, uy+ex_y], color="red", lw=2)

            _plt.xlim(-6, 6)

        fn= "look" if (dec.usetets is None) else "look_tet%s" % dec.usetets[nt]
        _plt.savefig(resFN(fn, dir=setname))
        _plt.close()

def showMarginalMarkDistributions(dec, setname, mklim=[-6, 8], dk=0.1):
    for tet in xrange(dec.nTets):
        ###    marginalize tetrode marks
        mrgidx = _N.array([1, 2, 3, 4])
        xp     = _N.linspace(-6, 6, 121)
        fig    = _plt.figure(figsize=(13, 12))
        fig.add_subplot(3, 2, 1)
        p  = _N.zeros(121)
        for m in xrange(dec.M):
            mn, mcov = mvn.marginalPDF(dec.mvNrm[tet].us[m], dec.mvNrm[tet].covs[m], mrgidx)
            p  += dec.mvNrm[tet].ms[m]/_N.sqrt(2*_N.pi*mcov[0,0]) *_N.exp(-0.5*(xp - mn[0])**2 / mcov[0, 0])

        x =_plt.hist(dec.tr_pos[tet], bins=_N.linspace(-6, 6, 121), normed=True, color="black")
        _plt.plot(xp, (p/_N.sum(p))*10, color="red", lw=2)


        ###    marginalize position + 3 tetrode marks
        allinds = _N.arange(5)

        bins   = _N.linspace(mklim[0], mklim[1], (mklim[1]-mklim[0])*(1./dk)+1)
        for shk in xrange(1, 5):
            fig.add_subplot(3, 2, shk+2)
            mrgidx = _N.setdiff1d(allinds, _N.array([shk]))

            p  = _N.zeros(len(bins))

            for m in xrange(dec.M):
                mn, mcov = mvn.marginalPDF(dec.mvNrm[tet].us[m], dec.mvNrm[tet].covs[m], mrgidx)
                p  += dec.mvNrm[tet].ms[m]/_N.sqrt(2*_N.pi*mcov[0,0]) *_N.exp(-0.5*(bins - mn[0])**2 / mcov[0, 0])
            x =_plt.hist(dec.tr_marks[tet][:, shk-1], bins=bins, normed=True, color="black")
            _plt.plot(bins, (p/_N.sum(p))*(1./dk), color="red", lw=2)

        fn= "margDists" if (dec.usetets is None) else "margDists%s" % dec.usetets[tet]
        _plt.savefig(resFN(fn, dir=setname))
        _plt.close()
