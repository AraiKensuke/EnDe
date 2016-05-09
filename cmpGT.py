#  plot the receptive field
import pickle
import EnDedirs as _ed
import numpy as _N
import matplotlib.pyplot as _plt
import mcmcFigs as mF

#  l0, f, sd2     posModes.dat for each epoch    mode
#  get distribution
#  dat_param.dat

def cmpGTtoFIT(basefn, datfn, itvfn, M, epc, xticks=None, yticks=None):
    dmp = open(_ed.resFN("posteriors.dump", dir=basefn), "rb")
    gt  = _N.loadtxt(_ed.datFN("%s_prms.dat" % datfn))
    _intvs  = _N.loadtxt(_ed.datFN("%s.dat" % itvfn))
    intvs = _N.array(gt.shape[0] * _intvs, dtype=_N.int)
    pckl = pickle.load(dmp)
    dmp.close()

    N     = 300
    x     = _N.linspace(0, 3, N)


    smpls = pckl["cp%d" % epc]
    mds   = pckl["md"]
    l0s   = smpls[0]   #  GIBBS ITER  x  M
    fs    = smpls[1]
    q2s   = smpls[2]

    frm   = 200
    Nsmp  = l0s.shape[0] - frm

    SMPS  = 1000

    rfsmps= _N.empty((SMPS, N, M))  #  samples

    rs    = _N.random.rand(SMPS, 3, M)
    for m in xrange(M):
        for ss in xrange(SMPS):
            i_l0 = int((Nsmp-frm)*rs[ss, 0, m])  #  one of the iters
            i_f  = int((Nsmp-frm)*rs[ss, 1, m])
            i_q2 = int((Nsmp-frm)*rs[ss, 2, m])
            l0   = l0s[frm+i_l0, m]
            f    = fs[frm+i_f, m]
            q2   = q2s[frm+i_q2, m]

            rfsmps[ss, :, m] = (l0 / _N.sqrt(2*_N.pi*q2)) * _N.exp(-0.5*(x - f)*(x-f)/q2)

            #_plt.plot(x, rfsmps[ss], color="black")

    Arfsmps = _N.sum(rfsmps, axis=2)
    srtd = _N.sort(Arfsmps, axis=0)

    fig  = _plt.figure(figsize=(5, 4))
    ax   = fig.add_subplot(1, 1, 1)
    #_plt.fill_between(x, srtd[50, :], srtd[950, :], alpha=0.3)
    _plt.fill_between(x, srtd[50, :], srtd[950, :], color="#CCCCFF")

    fr_s = _N.zeros(N)
    fr_e = _N.zeros(N)
    Mgt  = gt.shape[1]/3
    print Mgt
    for m in xrange(Mgt):
        l0_s = gt[intvs[epc], 2+3*m]
        l0_e = gt[intvs[epc+1]-1, 2+3*m]
        f_s = gt[intvs[epc], 3*m]
        f_e = gt[intvs[epc+1]-1, 3*m]
        q2_s = gt[intvs[epc], 1+3*m]
        q2_e = gt[intvs[epc+1]-1, 1+3*m]

        fr_s += (l0_s / _N.sqrt(2*_N.pi*q2_s)) * _N.exp(-0.5*(x - f_s)*(x-f_s)/q2_s)
        fr_e += (l0_e / _N.sqrt(2*_N.pi*q2_e)) * _N.exp(-0.5*(x - f_e)*(x-f_e)/q2_e)
    _plt.plot(x, 0.5*(fr_s+fr_e), lw=3, color="black")

    fr_m = _N.zeros(N)
    for m in xrange(M):
        l0_m = mds[epc, 3*m]
        f_m = mds[epc, 1+3*m]
        q2_m = mds[epc, 2+3*m]
        fr_m += (l0_m / _N.sqrt(2*_N.pi*q2_m)) * _N.exp(-0.5*(x - f_m)*(x-f_m)/q2_m)
    _plt.plot(x, fr_m, lw=3, color="blue")

    mF.setTicksAndLims(xlabel="position", ylabel="Hz", xticks=xticks, yticks=yticks, tickFS=22, labelFS=24)

    mF.arbitaryAxes(ax, axesVis=[True, True, False, False])
    fig.subplots_adjust(left=0.2, bottom=0.2, right=0.95, top=0.95)

    _plt.savefig(_ed.resFN("cmpGT2FIT%d.eps" % epc, dir=basefn))

