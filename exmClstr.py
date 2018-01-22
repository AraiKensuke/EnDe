import mvn
import mkdecoder as _mkd
from EnDedirs import resFN, datFN
import numpy as _N
import matplotlib.pyplot as _plt
import scipy.cluster.vq as scv
import mcmcFigs as mF
from fitutil import  sepHash, channelMins

def show_posmarks(dec, setname, ylim=None, win=None, singles=False, baseFN=None):
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

            if dec.marksObserved[nt] > 0:
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
                fn= baseFN if (dec.usetets is None) else "%(bf)s_tet%(t)s" % {"bf" : baseFN, "t" : dec.usetets[nt]}

                mF.arbitraryAxes(ax)
                mF.setLabelTicks(_plt, xlabel="position", ylabel="mark", xtickFntSz=14, ytickFntSz=14, xlabFntSz=16, ylabFntSz=16)
                fig.subplots_adjust(left=0.2, bottom=0.2, top=0.85)
                _plt.savefig(resFN("%(1)s_win=%(w)d.png" % {"1" : fn, "w" : win}, dir=setname), transparent=True)
                _plt.close()


        if not singles:
            _plt.suptitle("t0=%(2)d,t1=%(3)d" % {"2" : dec.tt0, "3" : dec.tt1})
            fn= baseFN if (dec.usetets is None) else "%(bf)s_tet%(t)s" % {"bf" : baseFN, "t" : dec.usetets[nt]}
            _plt.savefig(resFN("%(1)s_win=%(w)d.png" % {"1" : fn, "w" : win}, dir=setname, create=True), transparent=True)
            _plt.close()

def show_posmarksCNTR(dec, setname, mvNrm, ylim=None, win=None, singles=False, showScatter=True, baseFN="look", scatskip=1):
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


            _plt.xlim(-6, 6)
            if ylim is not None:
                _plt.ylim(ylim[0], ylim[1])
            else:
                ylim = _N.empty(2)
                ylim[0] = _N.min(dec.tr_marks[nt][:, k-1])
                ylim[1] = _N.max(dec.tr_marks[nt][:, k-1])
                yAMP    = ylim[1] - ylim[0]
                ylim[0] -= 0.1*yAMP
                ylim[1] += 0.1*yAMP


            if showScatter and dec.marksObserved[nt] > 0:
                _plt.scatter(dec.tr_pos[nt][::scatskip], dec.tr_marks[nt][::scatskip, k-1], color="grey", s=1)
            img = mvNrm.evalAll(1000, k-1, ylim=ylim)
            _plt.imshow(img, origin="lower", extent=(-6, 6, ylim[0], ylim[1]), cmap=_plt.get_cmap("Reds"))
            if singles:
                _plt.suptitle("k=%(k)d  t0=%(2).2fs : t1=%(3).2fs" % {"2" : (dec.tt0/1000.), "3" : (dec.tt1/1000.), "k" : k})
                fn= baseFN if (dec.usetets is None) else "%(fn)s_tet%(tets)s" % {"fn" : baseFN, "tets" : dec.usetets[nt]}

                mF.arbitraryAxes(ax)
                mF.setLabelTicks(_plt, xlabel="position", ylabel="mark", xtickFntSz=14, ytickFntSz=14, xlabFntSz=16, ylabFntSz=16)
                fig.subplots_adjust(left=0.2, bottom=0.2, top=0.85)
                _plt.savefig(resFN("%(1)s_win=%(w)d.png" % {"1" : fn, "w" : win}, dir=setname), transparent=True)
                _plt.close()

        if not singles:
            _plt.suptitle("t0=%(2)d,t1=%(3)d" % {"2" : dec.tt0, "3" : dec.tt1})
            fn= baseFN if (dec.usetets is None) else "%(fn)s_tet%(tets)s" % {"fn" : baseFN, "tets" : dec.usetets[nt]}
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


def timeline(bfn, datfn, itvfn, outfn="timeline", ch1=0, ch2=1, xL=0, xH=3, yticks=[0, 1, 2, 3], thin=1, allFR=None):   #  25Hz
    d = _N.loadtxt(datFN("%s.dat" % datfn))   #  marks
    itv = _N.loadtxt(datFN("%s.dat" % itvfn))
    N = d.shape[0]          #  ALL spikes
    epochs = itv.shape[0]-1
    ch1 += 2   #  because this is data col
    ch2 += 2

    wvfmMin = _N.min(d[:, 2:], axis=0)
    wvfmMax = _N.max(d[:, 2:], axis=0)

    Aint    = _N.array(itv*N, dtype=_N.int)
    Asts = _N.where(d[Aint[0]:Aint[1], 1] == 1)[0]
    N1      = len(Asts)   # num spikes in 1st epoch

    _x = _N.empty((N1, 5))
    _x[:, 0] = d[Asts, 0]
    _x[:, 1:] = d[Asts, 2:]
    T1  = Aint[1] - Aint[0]
    hz = _x.shape[0] / float(T1*0.001)


    if allFR is not None:
        unonhash, hashsp, hashthresh = sepHash(_x, BINS=20, blksz=5, xlo=xL, xhi=xH)

        rh  = len(hashsp) / float(N1)
        rnh = len(unonhash) / float(N1)

        #(m x rh + rnh)*hz = allFR
        #allFR  - rnh x hz = m x rh x hz
        #  m = (rnh x hz - allFR) / (rh x hz)

        m = (allFR - rnh * hz) / (rh * hz)

        #  ((m*len(hashsp) + len(unonhash)) / (T1*0.001))  == allFR  (target FR)
        #  how much of hashsp should we remove?
        #  Find lowest (1 - m)

        _x[:, 1:].sort(axis=0)

        chmins  = channelMins(_x, 100, 4, int((1-m)*len(hashsp)))

        #spk_n, chs = _N.where(_x[:, 1:] > chmins)
        #unique, counts = _N.unique(spk_n, return_counts=True)

        #print len(_N.where(counts > 0)[0])  # at least 1 channel above
        #print (len(_N.where(counts > 0)[0]) / float(T1*0.001))

        _sts = _N.where(d[:, 1] == 1)[0]   #  ALL spikes in ALL epochs
        spk_n, chs = _N.where(d[_sts, 2:] > chmins)

        unique_spk_IDs, counts = _N.unique(spk_n, return_counts=True)
        sts  = _sts[unique_spk_IDs[_N.where(counts > 0)[0]]]

    if allFR is None:
        _sts = _N.where(d[:, 1] == 1)[0]
        if thin == 1:
            sts = _sts
        else:
            sts = _sts[::thin]

    fig = _plt.figure(figsize=(10, 12))
    #######################
    ax =_plt.subplot2grid((4, 3), (0, 0), colspan=3)
    _plt.scatter(sts/1000., d[sts, 0], s=2, color="black")
    mF.arbitraryAxes(ax, axesVis=[True, True, False, False], xtpos="bottom", ytpos="left")
    mF.setTicksAndLims(xlabel="time (s)", ylabel="position", xticks=None, yticks=yticks, xticksD=None, yticksD=None, xlim=[0, N/1000.], ylim=[xL-0.3, xH+0.3], tickFS=15, labelFS=18)
    for ep in xrange(epochs):
        _plt.axvline(x=(itv[ep+1]*N/1000.), color="red", ls="--")
    #######################
    ax = _plt.subplot2grid((4, 3), (1, 0), colspan=3)
    print len(sts)
    _plt.scatter(sts/1000., d[sts, ch1], s=2, color="black")
    mF.arbitraryAxes(ax, axesVis=[True, True, False, False], xtpos="bottom", ytpos="left")
    mF.setTicksAndLims(xlabel="time (s)", ylabel=("mk elctrd %d" % (ch1-1)), xticks=None, yticks=[0, 3, 6], xticksD=None, yticksD=None, xlim=[0, N/1000.], ylim=[wvfmMin[0], wvfmMax[0]], tickFS=15, labelFS=18)
    for ep in xrange(epochs):
        _plt.axvline(x=(itv[ep+1]*N/1000.), color="red", ls="--")
    #######################
    ax = _plt.subplot2grid((4, 3), (2, 0), colspan=3)
    _plt.scatter(sts/1000., d[sts, ch2], s=2, color="black")
    mF.arbitraryAxes(ax, axesVis=[True, True, False, False], xtpos="bottom", ytpos="left")
    mF.setTicksAndLims(xlabel="time (s)", ylabel=("mk elctrd %d" % (ch2-1)), xticks=None, yticks=[0, 3, 6], xticksD=None, yticksD=None, xlim=[0, N/1000.], ylim=[wvfmMin[1], wvfmMax[1]], tickFS=15, labelFS=18)
    for ep in xrange(epochs):
        _plt.axvline(x=(itv[ep+1]*N/1000.), color="red", ls="--")
    ##############
    ax = _plt.subplot2grid((4, 3), (3, 0), colspan=1)
    _plt.scatter(d[sts, ch1], d[sts, ch2], s=2, color="black")
    mF.arbitraryAxes(ax, axesVis=[True, True, False, False], xtpos="bottom", ytpos="left")
    mF.setTicksAndLims(xlabel=("mk elctrd %d" % (ch1-1)), ylabel=("mk elctrd %d" % (ch2-1)), xticks=[0, 3, 6], yticks=[0, 3, 6], xticksD=None, yticksD=None, xlim=[wvfmMin[0], wvfmMax[0]], ylim=[wvfmMin[1], wvfmMax[1]], tickFS=15, labelFS=18)
    ##############
    ax = _plt.subplot2grid((4, 3), (3, 1), colspan=1)
    _plt.scatter(d[sts, 0], d[sts, ch1], s=2, color="black")
    mF.arbitraryAxes(ax, axesVis=[True, True, False, False], xtpos="bottom", ytpos="left")
    mF.setTicksAndLims(xlabel="pos", ylabel=("mk elctrd %d" % (ch1-1)), xticks=_N.linspace(xL, xH, 3), yticks=[0, 3, 6], xticksD=None, yticksD=None, xlim=[xL, xH], ylim=None, tickFS=15, labelFS=18)
    ##############
    ax = _plt.subplot2grid((4, 3), (3, 2), colspan=1)
    _plt.scatter(d[sts, 0], d[sts, ch2], s=2, color="black")
    mF.arbitraryAxes(ax, axesVis=[True, True, False, False], xtpos="bottom", ytpos="left")
    mF.setTicksAndLims(xlabel="pos", ylabel=("mk elctrd %d" % (ch2-1)), xticks=_N.linspace(xL, xH, 3), yticks=[0, 3, 6], xticksD=None, yticksD=None, xlim=[xL, xH], ylim=None, tickFS=15, labelFS=18)
    ##############

    fig.subplots_adjust(left=0.15, bottom=0.15, wspace=0.38, hspace=0.38)
    epochs = len(itv)-1
    choutfn = "%(of)s_%(1)d,%(2)d" % {"of" : outfn, "1" : (ch1-1), "2" : (ch2-1)}
    _plt.savefig(resFN(choutfn, dir=bfn), transparent=True)
    _plt.close()


def timeline_v2(bfn, datfn, itvfn, outfn="timeline", ch1=0, ch2=1, xL=0, xH=3, yticks=[0, 1, 2, 3], thin=1, allFR=None, tcksz=19, lblsz=21, t0=None, t1=None, figw=10, figh=10, ytpos="left"):   #  25Hz
    d = _N.loadtxt(datFN("%s.dat" % datfn))   #  marks
    N = d.shape[0]          #  ALL spikes
    itv = _N.loadtxt(datFN("%s.dat" % itvfn))

    epochs = itv.shape[0]-1
    ch1 += 2   #  because this is data col
    ch2 += 2

    wvfmMin = _N.min(d[:, 2:], axis=0)
    wvfmMax = _N.max(d[:, 2:], axis=0)

    Aint    = _N.array(itv*N, dtype=_N.int)
    Asts = _N.where(d[Aint[0]:Aint[1], 1] == 1)[0]
    N1      = len(Asts)   # num spikes in 1st epoch

    _x = _N.empty((N1, 5))
    _x[:, 0] = d[Asts, 0]
    _x[:, 1:] = d[Asts, 2:]
    T1  = Aint[1] - Aint[0]
    hz = _x.shape[0] / float(T1*0.001)


    if allFR is not None:
        unonhash, hashsp, hashthresh = sepHash(_x, BINS=20, blksz=5, xlo=xL, xhi=xH)

        rh  = len(hashsp) / float(N1)
        rnh = len(unonhash) / float(N1)

        #(m x rh + rnh)*hz = allFR
        #allFR  - rnh x hz = m x rh x hz
        #  m = (rnh x hz - allFR) / (rh x hz)

        m = (allFR - rnh * hz) / (rh * hz)

        #  ((m*len(hashsp) + len(unonhash)) / (T1*0.001))  == allFR  (target FR)
        #  how much of hashsp should we remove?
        #  Find lowest (1 - m)

        _x[:, 1:].sort(axis=0)

        chmins  = channelMins(_x, 100, 4, int((1-m)*len(hashsp)))

        #spk_n, chs = _N.where(_x[:, 1:] > chmins)
        #unique, counts = _N.unique(spk_n, return_counts=True)

        #print len(_N.where(counts > 0)[0])  # at least 1 channel above
        #print (len(_N.where(counts > 0)[0]) / float(T1*0.001))

        _sts = _N.where(d[:, 1] == 1)[0]   #  ALL spikes in ALL epochs
        spk_n, chs = _N.where(d[_sts, 2:] > chmins)

        unique_spk_IDs, counts = _N.unique(spk_n, return_counts=True)
        sts  = _sts[unique_spk_IDs[_N.where(counts > 0)[0]]]

    if allFR is None:
        _sts = _N.where(d[:, 1] == 1)[0]
        if thin == 1:
            sts = _sts
        else:
            sts = _sts[::thin]

    fig = _plt.figure(figsize=(figw, figh))
    #######################
    #ax =_plt.subplot2grid((5, 3), (0, 0), colspan=3)
    ax =_plt.subplot2grid((33, 3), (0, 0), colspan=3, rowspan=5)
    _plt.scatter(sts/1000., d[sts, 0], s=2, color="black")
    if ytpos == "left":
        mF.arbitraryAxes(ax, axesVis=[True, True, False, False], xtpos="bottom", ytpos="left")
        mF.setTicksAndLims(xlabel="time (s)", ylabel="position", yticks=[-6, -5, -3, -1, 0, 1, 3, 5, 6], xticks=None, yticksD=["H     ", "C", "L     ", "C", "H     ", "C", "R     ", "C", "H     "], xticksD=None, xlim=[0, N/1000.], ylim=[xL-0.3, xH+0.3], tickFS=tcksz, labelFS=lblsz)
    else:
        mF.arbitraryAxes(ax, axesVis=[False, True, True, False], xtpos="bottom", ytpos="right")
        mF.setTicksAndLims(xlabel="time (s)", ylabel="position", yticks=[-6, -5, -3, -1, 0, 1, 3, 5, 6], xticks=None, yticksD=["H", "   C", "L", "   C", "H", "   C", "R", "   C", "H"], xticksD=None, xlim=[0, N/1000.], ylim=[xL-0.3, xH+0.3], tickFS=tcksz, labelFS=lblsz)

    for ep in xrange(epochs):
        _plt.axvline(x=(itv[ep+1]*N/1000.), color="red", ls="--")
    #######################
    #ax = _plt.subplot2grid((5, 3), (1, 0), colspan=3)
    ax = _plt.subplot2grid((33, 3), (10, 0), colspan=3, rowspan=5)
    print len(sts)
    _plt.scatter(sts/1000., d[sts, 2], s=2, color="black")
    if ytpos == "left":
        mF.arbitraryAxes(ax, axesVis=[True, True, False, False], xtpos="bottom", ytpos="left")
        mF.setTicksAndLims(xlabel=None, ylabel="channel 1\nmark (a.u.)", xticks=None, yticks=[], xticksD=None, yticksD=None, xlim=[0, N/1000.], ylim=[wvfmMin[0], wvfmMax[0]], tickFS=tcksz, labelFS=lblsz)
    else:
        mF.arbitraryAxes(ax, axesVis=[False, True, True, False], xtpos="bottom", ytpos="right")
        mF.setTicksAndLims(xlabel=None, ylabel="channel 1\nmark (a.u.)", xticks=None, yticks=[], xticksD=None, yticksD=None, xlim=[0, N/1000.], ylim=[wvfmMin[0], wvfmMax[0]], tickFS=tcksz, labelFS=lblsz)

    for ep in xrange(epochs):
        _plt.axvline(x=(itv[ep+1]*N/1000.), color="red", ls="--")
    #######################
    ax = _plt.subplot2grid((33, 3), (16, 0), colspan=3, rowspan=5)
    #ax = _plt.subplot2grid((5, 3), (2, 0), colspan=3)
    _plt.scatter(sts/1000., d[sts, 3], s=2, color="black")
    if ytpos == "left":
        mF.arbitraryAxes(ax, axesVis=[True, True, False, False], xtpos="bottom", ytpos="left")
        mF.setTicksAndLims(xlabel=None, ylabel="channel 2\nmark (a.u.)", xticks=None, yticks=[], xticksD=None, yticksD=None, xlim=[0, N/1000.], ylim=[wvfmMin[0], wvfmMax[0]], tickFS=tcksz, labelFS=lblsz)
    else:
        mF.arbitraryAxes(ax, axesVis=[False, True, True, False], xtpos="bottom", ytpos="right")
        mF.setTicksAndLims(xlabel=None, ylabel="channel 2\nmark (a.u.)", xticks=None, yticks=[], xticksD=None, yticksD=None, xlim=[0, N/1000.], ylim=[wvfmMin[0], wvfmMax[0]], tickFS=tcksz, labelFS=lblsz)
    for ep in xrange(epochs):
        _plt.axvline(x=(itv[ep+1]*N/1000.), color="red", ls="--")
    #######################
    #ax = _plt.subplot2grid((5, 3), (3, 0), colspan=3)
    ax = _plt.subplot2grid((33, 3), (22, 0), colspan=3, rowspan=5)
    print len(sts)
    _plt.scatter(sts/1000., d[sts, 4], s=2, color="black")
    if ytpos == "left":
        mF.arbitraryAxes(ax, axesVis=[True, True, False, False], xtpos="bottom", ytpos="left")
        mF.setTicksAndLims(xlabel=None, ylabel="channel 3\nmark (a.u.)", xticks=None, yticks=[], xticksD=None, yticksD=None, xlim=[0, N/1000.], ylim=[wvfmMin[0], wvfmMax[0]], tickFS=tcksz, labelFS=lblsz)
    else:
        mF.arbitraryAxes(ax, axesVis=[False, True, True, False], xtpos="bottom", ytpos="right")
        mF.setTicksAndLims(xlabel=None, ylabel="channel 3\nmark (a.u.)", xticks=None, yticks=[], xticksD=None, yticksD=None, xlim=[0, N/1000.], ylim=[wvfmMin[0], wvfmMax[0]], tickFS=tcksz, labelFS=lblsz)
    for ep in xrange(epochs):
        _plt.axvline(x=(itv[ep+1]*N/1000.), color="red", ls="--")
    #######################
    ax = _plt.subplot2grid((33, 3), (28, 0), colspan=3, rowspan=5)
    #ax = _plt.subplot2grid((5, 3), (4, 0), colspan=3)
    _plt.scatter(sts/1000., d[sts, 5], s=2, color="black")
    if ytpos == "left":
        mF.arbitraryAxes(ax, axesVis=[True, True, False, False], xtpos="bottom", ytpos="left")
        mF.setTicksAndLims(xlabel="time (s)", ylabel="channel 4\nmark (a.u.)", xticks=None, yticks=[], xticksD=None, yticksD=None, xlim=[0, N/1000.], ylim=[wvfmMin[0], wvfmMax[0]], tickFS=tcksz, labelFS=lblsz)
    else:
        mF.arbitraryAxes(ax, axesVis=[False, True, True, False], xtpos="bottom", ytpos="right")
        mF.setTicksAndLims(xlabel="time (s)", ylabel="channel 4\nmark (a.u.)", xticks=None, yticks=[], xticksD=None, yticksD=None, xlim=[0, N/1000.], ylim=[wvfmMin[0], wvfmMax[0]], tickFS=tcksz, labelFS=lblsz)
    for ep in xrange(epochs):
        _plt.axvline(x=(itv[ep+1]*N/1000.), color="red", ls="--")




    #fig.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.95, wspace=0.9, hspace=0.9)
    if ytpos == "left":
        fig.subplots_adjust(left=0.16, bottom=0.08, right=0.98, top=0.95, wspace=0.4, hspace=0.4)
    if ytpos == "right":
        fig.subplots_adjust(left=0.05, bottom=0.08, right=0.84, top=0.95, wspace=0.4, hspace=0.4)
    epochs = len(itv)-1
    for fmt in ["png", "eps"]:
        choutfn = "%(of)s.%(fmt)s" % {"of" : outfn, "fmt" : fmt}
        _plt.savefig(resFN(choutfn, dir=bfn), transparent=True)
    _plt.close()


def pos_timeline(bfn, datfn, itvfn, outfn="timeline", ch1=0, ch2=1, xL=0, xH=3, yticks=[0, 1, 2, 3], yticksD=[0, 1, 2, 3], thin=1, t0=None, t1=None, skp=1, maze=_mkd.mz_CRCL):
    d = _N.loadtxt(datFN("%s.dat" % datfn))   #  marks
    t0 = 0 if (t0 is None) else t0
    t1 = d.shape[0] if (t1 is None) else t1
    itv = _N.loadtxt(datFN("%s.dat" % itvfn))
    N = d.shape[0]
    epochs = itv.shape[0]-1
    ch1 += 2   #  because this is data col
    ch2 += 2

    _sts = _N.where(d[t0:t1, 1] == 1)[0] + t0
    if thin == 1:
        sts = _sts
    else:
        sts = _sts[::thin]

    wvfmMin = _N.min(d[t0:t1, 2:], axis=0)
    wvfmMax = _N.max(d[t0:t1, 2:], axis=0)

    #fig = _plt.figure(figsize=(4, 2.2))
    fig = _plt.figure(figsize=(10, 2.2))
    #######################
    ax =_plt.subplot2grid((1, 3), (0, 0), colspan=3)
    _plt.scatter(_N.arange(d[t0:t1:100].shape[0], dtype=_N.float)/10., d[t0:t1:100, 0], s=9, color="black")
    _plt.scatter(sts[::skp]/1000., d[sts[::skp], 0], s=1, color="orange")

    if maze == _mkd.mz_W:
        _plt.axhline(y=-6, ls="--", lw=1, color="black")
        _plt.axhline(y=-3, ls=":", lw=1, color="black")
        _plt.axhline(y=0,  ls="--", lw=1, color="black")
        _plt.axhline(y=3,  ls=":", lw=1, color="black")
        _plt.axhline(y=6,  ls="--", lw=1, color="black")
    mF.arbitraryAxes(ax, axesVis=[True, True, False, False], xtpos="bottom", ytpos="left")
    #itv[-1] = 0.97
    for ep in xrange(epochs):
        _plt.axvline(x=(itv[ep+1]*N/1000.), color="red", ls="-.")
    mF.setTicksAndLims(xlabel="time (s)", ylabel="position", xticks=None, yticks=yticks, xticksD=None, yticksD=yticksD, xlim=[t0/1000., t1/1000.], ylim=[xL-0.3, xH+0.3], tickFS=15, labelFS=18)

    choutfn = "%(of)s_%(1)d,%(2)d" % {"of" : outfn, "1" : (ch1-1), "2" : (ch2-1)}
    fig.subplots_adjust(bottom=0.28, left=0.1, top=0.96, right=0.99)
    _plt.savefig(resFN("%s.pdf" % choutfn, dir=bfn), transparent=True)
    _plt.close()

