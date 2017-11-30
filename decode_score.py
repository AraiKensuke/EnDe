import numpy as _N

def scores(pctm1, lagfit, epts, ignr, scrs, xp, pX_Nm, pos, usemaze):
    """
    xp    grid on which posterior position is defined on
    """
    NBins = xp.shape[0]
    for epc in xrange(lagfit, epts.shape[0]-1):
        t0 = epts[epc]
        t1 = epts[epc]
        pX_Nm[t0:t1] /= _N.sum(pX_Nm[t0:t1], axis=1).reshape((t1-t0, 1)) #nrmlz

    t0   = epts[lagfit]
    t1   = epts[epts.shape[0]-1]

    in5 = _N.zeros(t1-t0, dtype=_N.bool)
    sz95= _N.zeros(t1-t0)

    decded= _N.array(pX_Nm[t0:t1])
    sdecded = _N.sort(decded, axis=1)  #  low to high prob
    cdecded = _N.cumsum(sdecded, axis=1)  # cumulative

    for t in xrange(0, t1-t0):

        li5p = _N.where((cdecded[t, 1:] >= pctm1) & (cdecded[t, 0:-1] <= pctm1))[0]
        if len(li5p) == 0:
            i5p = 0
        else:
            i5p = li5p[0]

        thrP=  sdecded[t, i5p+1]   # threshold prob/bin for 95%

        binsGT5 = _N.where(decded[t] > thrP)[0]
        if len(binsGT5) == 0:
            thrP=  sdecded[t, i5p]   # threshold prob/bin for 95%
            binsGT5 = _N.where(decded[t] > thrP)[0]
            if len(binsGT5) == 0:
                print "binsGT5 0 at %(t)d" % {"t" : t}

        ib = _N.where((pos[t+t0] >= xp[0:-1]) & (pos[t+t0] <= xp[1:]))[0][0]

        in5[t] = len(_N.where(binsGT5 == ib)[0])
        sz95[t] = len(binsGT5) / float(NBins)

    maxV = _N.max(pX_Nm, axis=1)
    maxInds = _N.empty(t1-t0, dtype=_N.int)

    for epc in xrange(lagfit, epts.shape[0]-1):
        et0 = epts[epc]
        et1 = epts[epc+1]
        scrs[epc-1, 0] = float(_N.sum(in5[et0+ignr-t0:et1-t0])) / (et1-et0-ignr)
        scrs[epc-1, 1] = _N.mean(sz95[et0+ignr-t0:et1-t0])

        maxInds = _N.empty(et1-et0-ignr, dtype=_N.int)
        for t in xrange(et0+ignr, et1):
            maxInds[t-et0-ignr] = _N.where(pX_Nm[t] == maxV[t])[0][0]

        dist = _N.abs(xp[maxInds] - pos[et0+ignr:et1])
        otherside_dist = _N.where(dist > 6)[0]    #  CORRECT FOR CRCL
        dist[otherside_dist] = 12 - dist[otherside_dist]
        scrs[epc-1, 2] = _N.mean(dist)
