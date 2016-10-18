#  position dependent firing rate
#  discontinuous change of place field params
import os
import utilities as _U
from utils import createSmoothedPath, createSmoothedPathK
import numpy as _N
import matplotlib.pyplot as _plt
import pickle
import time as _tm

UNIF  = 0
NUNIF = 1

APPEAR=0
DISAPPEAR=1
NEITHER=-1
#_N.array([[0, 0.02], [0.1, 0.03], [0.2, 0.04], [0.5, 0.05], [0.7, 0.06]]
def chpts(npts, yL, yH, t0=0, ad=NEITHER):
    #  [tn, dy]
    cpts = _N.empty((npts, 2))
    ts   = _N.random.rand(npts)
    ts[0] = 0
    for i in xrange(1, npts):
        ts[i] += ts[i-1]
    ts /= _N.sum(ts[-1])
    cpts[:, 0] = ts
    F = 0.9999
    x = _N.random.randn()
    cpts[0, 1] = x
    for i in xrange(1, npts):
        x = F*x + 0.1*_N.random.randn()
        cpts[i, 1] = x
    miny = _N.min(cpts[:, 1])
    cpts[:, 1] -= miny   # min value is now 0
    maxy = _N.max(cpts[:, 1])
    cpts[:, 1] /= maxy   # max value is now 1
    cpts[:, 1] *= (yH-yL)   # max value is now 1
    cpts[:, 1] += yL
    if ad != NEITHER:
        if ad == APPEAR:
            cpts[0:t0, 1] = 0
        else:
            cpts[t0:, 1]  = 0
    return cpts

        
def makeCovs(nNrns, K, LoHisMk):
    Covs = _N.empty((nNrns, K, K))
    for n in xrange(nNrns):
        for k1 in xrange(K):
            #Covs[n, k1, k1] = (LoHisMk[n, k1, 1] - LoHisMk[n, k1, 0])*(0.1+0.1*_N.random.rand())
            Covs[n, k1, k1] = (LoHisMk[n, k1, 1] - LoHisMk[n, k1, 0])*0.2
        for k1 in xrange(K):
            for k2 in xrange(k1+1, K):
                Covs[n, k1, k2] = 0.8 * _N.sqrt(Covs[n, k1, k1]*Covs[n, k2, k2])#(0.5 + 0.3*_N.random.rand()) * _N.sqrt(Covs[n, k1, k1]*Covs[n, k2, k2])
                Covs[n, k2, k1] = Covs[n, k1, k2]
    return Covs

def create(Lx, Hx, N, mvPat, RTs, frqmx, Amx, pT, l_sx_chpts, sxts, l_l0_chpts, l0ts, l_ctr_chpts, ctrts, mk_chpts, mkts, Covs, LoHis, km, bckgrdLam=None, script="no info", addShortStops=False, stops=10, stopDur=500, thresh=None):
    """
    km  tells me neuron N gives rise to clusters km[N]  (list)
    bckgrd is background spike rate  (Hz)
    """
    global UNIF, NUNIF
    #####  First check that the number of neurons and PFs all consistent.
    nNrnsSX = len(l_sx_chpts)
    nNrnsL0 = len(l_l0_chpts)
    nNrnsCT = len(l_ctr_chpts)
    nNrnsMK = len(mk_chpts)
    nNrnsMKA= LoHis.shape[0]
    nNrnsMKC= Covs.shape[0]

    if not (nNrnsSX == nNrnsL0 == nNrnsCT == nNrnsMK == nNrnsMKA == nNrnsMKC):
        print "Number of neurons not consistent"
        return None
    nNrns = nNrnsSX

    if not (LoHis.shape[1] == Covs.shape[1] == Covs.shape[2]):
        print "Covariance of LoHis not correct"
        return None
    K = LoHis.shape[1]
    
    PFsPerNrn = _N.zeros(nNrns, dtype=_N.int)

    sx_chpts  = []
    l0_chpts  = []
    ctr_chpts = []
    M         = 0
    nrnNum    = []
    for nrn in xrange(nNrns):
        #  # of place fields for neuron nrn
        nPFsSX = len(l_sx_chpts[nrn])
        nPFsL0 = len(l_l0_chpts[nrn])
        nPFsCT = len(l_ctr_chpts[nrn])
        sx_chpts.extend(l_sx_chpts[nrn])
        l0_chpts.extend(l_l0_chpts[nrn])
        ctr_chpts.extend(l_ctr_chpts[nrn])

        if not (nPFsSX == nPFsL0 == nPFsCT):
            print "Number of PFs for neuron %d not consistent" % nrn
            return None
        M += len(l_ctr_chpts[nrn])
        nrnNum += [nrn]*nPFsSX
        PFsPerNrn[nrn] = nPFsSX

    #  M = # of clusters  (in mark + pos space)  
    #  nNrns = # of neurons
    ####  build data
    Ns     = _N.empty(RTs, dtype=_N.int)
    if mvPat == NUNIF:

        for rt in xrange(RTs):
            Ns[rt] = N*((1-pT) + pT*_N.random.rand())
    else:
        Ns[:] = N

    NT     = _N.sum(Ns)
    pths    = _N.empty(NT)

    x01    = _N.linspace(0, 1, len(pths))
    x01    = x01.reshape((1, NT))
    plastic = False

    ##########  nonstationary center width
    #  sxt  should be (M x NT)
    sxt   = _N.empty((M, NT))
    for m in xrange(M):  # sxts time scale
        sxt[m] = createSmoothedPath(sx_chpts[m], NT, sxts)
        if len(sx_chpts[m]) > 1:  plastic = True

    sx    = sxt**2     #  var of firing rate function

    ##########  nonstationary center height l0
    #  f is NT x M
    l0   = _N.empty((M, NT))
    for m in xrange(M):
        l0[m] = createSmoothedPath(l0_chpts[m], NT, l0ts)
        if len(l0_chpts[m]) > 1:  plastic = True

    f     = l0/_N.sqrt(2*_N.pi*sx)   #  f*dt

    ##########  nonstationary center location
    ctr  = _N.empty((M, NT))
    for m in xrange(M):
        ctr[m] = createSmoothedPath(ctr_chpts[m], NT, ctrts)
        if len(ctr_chpts[m]) > 1:  plastic = True

    if K > 0:
        ##########  nonstationary marks
        mk_MU  = _N.empty((nNrns, NT, K))
        for n in xrange(nNrns):
            mk_MU[n] = createSmoothedPathK(mk_chpts[n], NT, K, mkts, LoHis[n])
            if len(mk_chpts[n]) > 1:  plastic = True

    if mvPat == NUNIF:
        now = 0
        for rt in xrange(RTs):
            N = Ns[rt]    #  each traverse slightly different duration
            rp  = _N.random.rand(N/100)
            x     = _N.linspace(Lx, Hx, N)
            xp     = _N.linspace(Lx, Hx, N/100)

            r   = _N.interp(x, xp, rp)       #  creates a velocity vector
            #  create movement without regard for place field
            r += Amx*(1.1+_N.sin(2*_N.pi*_N.linspace(0, 1, N, endpoint=False)*frqmx))
            pth = _N.zeros(N+1)
            for n in xrange(1, N+1):
                pth[n] = pth[n-1] + r[n-1]

            pth   /= (pth[-1] - pth[0])
            pth   *= (Hx-Lx)
            pth   += Lx

            pths[now:now+N]     = pth[0:N]
            now += N
    else:
        now = 0
        x = _N.linspace(Lx, Hx, N)
        for rt in xrange(RTs):
            N = Ns[rt]
            pths[now:now+N]     = x
            now += N

    if addShortStops:
        for ist in xrange(stops):
            done   = False
            while not done:
                t0 = int(_N.random.rand()*NT)
                t1 = t0 + int(stopDur*(1+0.1*_N.random.randn()))
                if _N.abs(_N.max(_N.diff(pths[t0:t1]))) < 0.05*(Hx-Lx):
                    done = True   #  not crossing origin
                
            pths[t0:t1] = _N.mean(pths[t0:t1])

    ###  now calculate firing rates
    dt   = 0.001
    fdt  = f*dt
    #  change place field location
    Lam   = f*dt*_N.exp(-0.5*(pths-ctr)**2 / sx)

    rnds = _N.random.rand(M, NT)

    #dat = _N.zeros((NT, 2 + K))
    dat = _N.zeros((NT, 2 + K))
    dat[:, 0] = pths

    for m in xrange(M):
        sts  = _N.where(rnds[m] < Lam[m])[0]
        dat[sts, 1] = 1

        nrn = nrnNum[m]
        if K > 0:
            for t in xrange(len(sts)):
                dat[sts[t], 2:] = _N.random.multivariate_normal(mk_MU[nrn, t], Covs[nrn], size=1)

        #  now noise spikes
        if bckgrdLam is not None:
            nzsts  = _N.where(rnds[m] < (bckgrdLam*dt)/float(M))[0]
            dat[nzsts, 1] = 1
            nrn = nrnNum[m]
            if K > 0:
                for t in xrange(len(nzsts)):
                    dat[nzsts[t], 2:] = _N.random.multivariate_normal(mk_MU[nrn, t], Covs[nrn], size=1)

    if thresh is not None:
        sts = _N.where(dat[:, 1] == 1)[0]
        nID, nC = _N.where(dat[sts, 2:] < thresh)

        swtchs  = _N.zeros((len(sts), K))
        swtchs[nID, nC] = 1    #  for all cells, all components below hash == 1

        swtchsK = _N.sum(swtchs, axis=1)
        
        blw_thrsh_all_chs = _N.where(swtchsK == K)[0]
        abv_thr = _N.setdiff1d(_N.arange(len(sts)), blw_thrsh_all_chs)
        print "below thresh in all channels  %(1)d / %(2)d" % {"1" : len(blw_thrsh_all_chs), "2" : len(sts)}
        dat[sts[blw_thrsh_all_chs], 1:] = 0

    bFnd  = False

    ##  us un   uniform sampling of space, stationary or non-stationary place field
    ##  ns nn   non-uni sampling of space, stationary or non-stationary place field
    ##  bs bb   biased and non-uni sampling of space

    bfn     = "" if (M == 1) else ("%d" % M)

    if mvPat == UNIF:
        bfn += "u"
    else:
        bfn += "b" if (Amx > 0) else "n"

    bfn += "n" if plastic else "s"

    iInd = 0
    while not bFnd:
        iInd += 1
        dd   = os.getenv("__EnDeDataDir__")
        fn = "%(dd)s/%(bfn)s%(iI)d.dat" % {"bfn" : bfn, "iI" : iInd, "dd" : dd}
        fnocc="%(dd)s/%(bfn)s%(iI)docc.png" % {"bfn" : bfn, "iI" : iInd, "dd" : dd}
        fnprm = "%(dd)s/%(bfn)s%(iI)d_prms.pkl" % {"bfn" : bfn, "iI" : iInd, "dd" : dd}

        if not os.access(fn, os.F_OK):  # file exists
            bFnd = True

    smk = " %.4f" * K
    _U.savetxtWCom("%s" % fn, dat, fmt=("%.4f %d" + smk), delimiter=" ", com="#  script=%s.py" % script)

    pcklme = {}

    pcklme["l0"]  = l0[:, ::100]
    pcklme["f"]   = ctr[:, ::100]
    pcklme["sq2"] = sx[:, ::100]
    pcklme["u"]   = mk_MU[:, ::100]
    pcklme["covs"]= Covs
    pcklme["intv"]= 100
    pcklme["km"]  = km

    dmp = open(fnprm, "wb")
    pickle.dump(pcklme, dmp, -1)
    dmp.close()

    print "created %s" % fn

    fig = _plt.figure()
    _plt.hist(dat[:, 0], bins=_N.linspace(Lx, Hx, 101), color="black")
    _plt.savefig(fnocc)
    _plt.close()
