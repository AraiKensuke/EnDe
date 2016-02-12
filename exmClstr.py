import mvn
from EnDedirs import resFN
import numpy as _N
import matplotlib.pyplot as _plt
import scipy.cluster.vq as scv
import mcmcFigs as mF

def show_posmarks(dec, setname, ylim=None, win=None, singles=False):
    MTHR = 0.001   #  how much smaller is mixture compared to maximum

    for nt in xrange(dec.nTets):
        if not singles:
            fig = _plt.figure(figsize=(7, 5))
        for k in xrange(1, dec.mdim+1):
            if singles:
                fig = _plt.figure(figsize=(4, 3))
                ax  = fig.add_subplot(1, 1, 1)
            else:
                ax  = fig.add_subplot(2, 2, k)

            """
            for l in xrange(dec.tt0, dec.tt1):
                if (dec.marks[l, nt] is not None):
                    x.append(dec.pos[l])
                    y.append(dec.marks[l, nt][0][k-1])
            """

            _plt.scatter(dec.tr_pos[nt], dec.tr_marks[nt][:, k-1], color="black", s=2)
            #_plt.scatter(dec.mvNrm[nt].us[:, 0], dec.mvNrm[nt].us[:, k], color="red", s=30)
            mThr = MTHR * _N.max(dec.mvNrm[nt].ms)

            for m in xrange(dec.M):
                if dec.mvNrm[nt].ms[m, 0] >= mThr:
                    ux   = dec.mvNrm[nt].us[m, 0]  #  position
                    uy   = dec.mvNrm[nt].us[m, k]
                    ex_x = _N.sqrt(dec.mvNrm[nt].covs[m, 0, 0])
                    ex_y = _N.sqrt(dec.mvNrm[nt].covs[m, k, k])
                    _plt.plot([ux-ex_x, ux+ex_x], [uy, uy], color="red", lw=2)
                    _plt.plot([ux, ux], [uy-ex_y, uy+ex_y], color="red", lw=2)

                    _plt.scatter(dec.mvNrm[nt].us[m, 0], dec.mvNrm[nt].us[m, k], color="red", s=30)

            _plt.xlim(-6, 6)
            if ylim is not None:
                _plt.ylim(ylim[0], ylim[1])

            if singles:
                _plt.suptitle("k=%(k)d  t0=%(2).2fs : t1=%(3).2fs" % {"2" : (dec.tt0/1000.), "3" : (dec.tt1/1000.), "k" : k})
                fn= "look" if (dec.usetets is None) else "look_tet%s" % dec.usetets[nt]

                mF.arbitraryAxes(ax)
                mF.setLabelTicks(_plt, xlabel="position", ylabel="mark", xtickFntSz=14, ytickFntSz=14, xlabFntSz=16, ylabFntSz=16)
                fig.subplots_adjust(left=0.2, bottom=0.2, top=0.85)
                _plt.savefig(resFN("%(k)d_%(1)s_win=%(w)d.eps" % {"1" : fn, "w" : win, "k" : k}, dir=setname), transparent=True)
                _plt.close()


        if not singles:
            _plt.suptitle("t0=%(2)d,t1=%(3)d" % {"2" : dec.tt0, "3" : dec.tt1})
            fn= "look" if (dec.usetets is None) else "look_tet%s" % dec.usetets[nt]
            _plt.savefig(resFN("%(1)s_win=%(w)d.png" % {"1" : fn, "w" : win}, dir=setname, create=True), transparent=True)
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


def showTrajectory(dec, t0, t1, ep, setname, dir):
    fig = _plt.figure(figsize=(14, 7))
    ax  = fig.add_subplot(1, 1, 1)
    _plt.imshow(dec.pX_Nm[t0:t1].T, aspect=(0.5*(t1-t0)/50.), cmap=_plt.get_cmap("Reds"))
    _plt.plot(_N.linspace(t0-t0, t1-t0, t1-t0), (dec.xA+dec.pos[t0:t1])/dec.dxp, color="grey", lw=3, ls="--")
    #_plt.plot(_N.linspace(float(t0)/1000., float(t1)/1000., t1-t0), (dec.xA+dec.pos[t0:t1])/dec.dxp, color="red", lw=2)
    #print (float(t0)/1000)
    #print (float(t1)/1000)
    _plt.xlim(0, t1-t0)
    _plt.ylim(-(dec.nTets*4), 50)
    #_plt.xticks(_N.arange(0, t1-t0, 2000), _N.arange(t0, t1, 2000, dtype=_N.float)/1000)
    dt = int((((int(t1/1000.)*1000) - (int(t0/1000.)*1000))/4.)/1000.)*1000

    stT0 = t0 - int(t0/1000.)*1000
    enT1 = t1 - int(t1/1000.)*1000
    #_plt.xticks(_N.arange(0, t1-t0, dt), _N.arange(t0, t1, dt, dtype=_N.float)/1000)
    _plt.xticks(_N.arange(stT0, t1-t0, dt), _N.arange(int(t0/1000.)*1000, int(t1/1000.)*1000, dt, dtype=_N.int)/1000)
    #_plt.locator_params(nbins=6, axis="x")
    _plt.yticks(_N.linspace(0, 50, 5), [-6, -3, 0, 3, 6])
    mF.arbitaryAxes(ax, axesVis=[False, False, False, False], x_tick_positions="bottom", y_tick_positions="left")
    mF.setLabelTicks(_plt, xlabel="Time (sec.)", ylabel="Position", xtickFntSz=30, ytickFntSz=30, xlabFntSz=32, ylabFntSz=32)

    x = []
    y = []
    for nt in xrange(dec.nTets):
        x.append([])
        y.append([])

    for t in xrange(t0, t1):
        for nt in xrange(dec.nTets):
            if dec.marks[t, nt] is not None:
                x[nt].append(t-t0)
                y[nt].append(-1.5 - 3*nt)

    for nt in xrange(dec.nTets):
        _plt.plot(x[nt], y[nt], ls="", marker="|", ms=15, color="black")

    fig.subplots_adjust(bottom=0.15, left=0.15)
    _plt.savefig(resFN("decode_%(uts)s_%(mth)s_win=%(e)d.eps" % {"e" : (ep/2), "mth" : dec.decmth, "uts" : dec.utets_str, "dir" : dir}, dir=setname, create=True))
    _plt.close()
