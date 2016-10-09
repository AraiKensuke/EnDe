import numpy as _N
import matplotlib.pyplot as _plt
from filter import gauKer
import time as _tm


def createSmoothedPath(cps, N, ts):
    """
    cps = [[t1, dy1], [t2, dy2], [t3, dy3], ...]
    specify points when curve changes value.  
    N   = how many points to use
    ts  = time scale for smoothing
    """
    if cps is None:
        return _N.zeros(N)
    Ne  = 3*N
    pth = _N.zeros(Ne)

    NC  = cps.shape[0]

    gk  = gauKer(int(ts*N))
    gk  /= _N.sum(gk)

    pth[0:N] = cps[0, 1]

    for p in xrange(cps.shape[0]):
        t0 = cps[p, 0]
        t1 = cps[p+1, 0] if (p < NC - 1) else 2

        pth[N+int(N*t0):N+int(N*t1)] = cps[p, 1]

    spth = _N.convolve(pth, gk, mode="same")

    return _N.array(spth[N:2*N])

def createSmoothedPathK(cps, N, K, ts, LoHis):
    """
    cps = [[x1, y1], [x2, y2], [x3, y3], ...], yi in [0, 1]
    specify points when curve changes value.  
    N   = how many points to use
    K   = dims
    LoHis= upper and lower lim for each channel   K x 2
    ts  = time scale for smoothing
    """
    if cps is None:
        return _N.zeros(N)
    Ne  = 3*N
    pth = _N.zeros((Ne, K))

    gk  = gauKer(int(ts*N))
    gk  /= _N.sum(gk)

    Amps = _N.diff(LoHis, axis=1).reshape(K)

    for k in xrange(K):
        NC  = cps[k].shape[0]
        pth[0:N, k] = LoHis[k, 0] + Amps[k]*cps[k, 0, 1]   #  
        for p in xrange(1, cps[k].shape[0]-1):
            cps[k, p, 0] += _N.random.randn()*0.02
        for p in xrange(cps[k].shape[0]):
            t0 = cps[k, p, 0]
            t1 = cps[k, p+1, 0] if (p < NC - 1) else 2

            pth[N+int(N*t0):N+int(N*t1), k] = LoHis[k, 0] + Amps[k]*cps[k, p, 1]

        spth = _N.convolve(pth[:, k], gk, mode="same")
        pth[:, k] = spth
    return _N.array(pth[N:2*N])

def generateMvt(N, vAmp=1, constV=False, pLR=0.5, nLf=None, nRh=None, Fv=0.9995, sv=0.00007):
    """
    vAmp  is speed amplitude.  Different meaning when constV==True

    specify
    
    pLR
    --or--
    nLf and nRh
    """
    x = 0
    pos = _N.empty(N)

    dirc = -1 if _N.random.rand() < 0.5 else 1
    done = False
    v    = 0

    bFxdLRsq = True if (nLf is not None) and (nRh is not None) else False
    fLRdir   = 1 if (_N.random.rand() < 0.5) else -1
    if bFxdLRsq:
        dirc = fLRdir
        
    nL       = 0;    nR       = 0
    for n in xrange(N):
        if constV:
            x += vAmp*dirc            
        else:
            v = Fv*v + sv*_N.random.randn() 
            x += vAmp*_N.abs(v)*dirc

        if (x > 6) and (dirc > 0):
            done = True
        elif (x < -6) and (dirc < 0):
            done = True

        if done:         ######  PASSED BOUNDARY
            if not bFxdLRsq:
                dirc = -1 if _N.random.rand() < pLR else 1
                pLR = (pLR - 0.25) if (dirc == -1) else pLR + 0.25
            else:
                if fLRdir == 1:  #  
                    if nR < nRh-1:
                        nR += 1
                        dirc = 1
                    else:
                        nR = 0
                        dirc = -1
                        fLRdir = -1
                else:
                    if nL < nLf-1:
                        nL += 1
                        dirc = -1
                    else:
                        nL = 0
                        dirc = 1
                        fLRdir = 1

            if dirc < 0:     #  dirc is direction to go next
                if x <= -6:  #  -6.1 -> -0.1   #    prev. dirc was -1
                    x = x + 6
                else:  #  6.1 -> -0.1     prev. dirc was 1
                    x = -1*(x - 6)
            if dirc > 0:
                if x <= -6:  #  -6.1 -> 0.1
                    x = -1*(x + 6)
                else:  #  6.1 -> 0.1
                    x = x - 6
        pos[n] = x
        done = False

    return pos

def cLklhds(dec, t0, t1, tet=0, scale=1., onlyLklhd=False):
    it0 = int(t0*scale)
    it1 = int(t1*scale)

    pg   = 0
    onPg = 0

    incr = 1 if onlyLklhd else 2
    for t in xrange(it0, it1):
        if dec.marks[t, tet] is not None:
            if onPg == 0:
                fig = _plt.figure(figsize=(13, 8))        
            fig.add_subplot(4, 6, onPg + 1)
            _plt.plot(dec.xp, dec.Lklhd[0, t], color="black")
            _plt.axvline(x=dec.pos[t], color="red", lw=2)
            _plt.yticks([])
            _plt.xticks([-6, -3, 0, 3, 6])
            _plt.title("t = %.3f" % (float(t) / scale))

            if not onlyLklhd:
                fig.add_subplot(4, 6, onPg + 2)
                _plt.plot(dec.xp, dec.pX_Nm[t])        
                _plt.axvline(x=dec.pos[t], color="red", lw=2)
                _plt.yticks([])
                _plt.xticks([-6, -3, 0, 3, 6])
                _plt.title("t = %.3f" % (float(t) / scale))
            onPg += incr

        if onPg >= 24:
            fig.subplots_adjust(wspace=0.35, hspace=0.35, left=0.08, right=0.92, top=0.92, bottom=0.08)
            _plt.savefig("cLklhd_%(mth)s,pg=%(pg)d" % {"pg" : pg, "mth" : dec.decmth})
            _plt.close()
            pg += 1
            onPg = 0

    if onPg > 0:
        fig.subplots_adjust(wspace=0.15, hspace=0.15, left=0.08, right=0.92, top=0.92, bottom=0.08)
        _plt.savefig("cLklhd_%(mth)s,pg=%(pg)d" % {"pg" : pg, "mth" : dec.decmth})
        _plt.close()

def at(dec, t0, t1):
    nons = _N.equal(dec.marks[t0:t1, 0], None)
    inds = _N.where(nons == False)[0]

    nfigs = len(inds)

    pg    = 0
    figsPP = 12

    for nf in xrange(nfigs):
        if (nf % figsPP) == 0:
            fig = _plt.figure(figsize=(12, 8))
            pg  += 1

        fig.add_subplot(6, 4, 2*(nf % figsPP)+1)
        _plt.plot(dec.xp, dec.pX_Nm[t0+inds[nf]], lw=2, color="black")
        _plt.axvline(x=dec.pos[t0+inds[nf]], color="grey", lw=2)
        _plt.yticks([])

        fig.add_subplot(6, 4, 2*(nf % figsPP)+2)
        l = _N.product(dec.Lklhd[:, t0+inds[nf]], axis=0)
        _plt.plot(dec.xp, l, lw=2, color="red")
        _plt.axvline(x=dec.pos[t0+inds[nf]], color="grey", lw=2)
        _plt.yticks([])

        if (nf % figsPP) == figsPP-1:
            fig.subplots_adjust(wspace=0.2, hspace=0.4, bottom=0.1, left=0.1, right=0.9, top=0.9)
            _plt.savefig("lookat_%(pg)d_%(m)s_%(1)d,%(2)d" % {"m" : dec.decmth, "1" : t0, "2" : t1, "pg" : pg})


    if (nf % figsPP) != figsPP-1:
        fig.subplots_adjust(wspace=0.2, hspace=0.4, bottom=0.1, left=0.1, right=0.9, top=0.9)
        _plt.savefig("lookat_%(pg)d_%(m)s_%(1)d,%(2)d" % {"m" : dec.decmth, "1" : t0, "2" : t1, "pg" : pg})
