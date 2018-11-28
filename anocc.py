import numpy as _N
from filter import gauKer

def approx_occhist_w_gau(xp, fN):
    """
    approximate occupational histogram with gaussians.  
    fN bins of xp.  then smooth histogram, and find its peaks.
    fit peaks with Gaussian, and return
    """
    N   = xp.shape[0]

    x0 = _N.min(xp)
    x1 = _N.max(xp)

    fN  = 401   #  # of bin boundary points. # of bins is fN-1

    #  spatial histogram
    fs  = _N.linspace(-6, 6, fN, endpoint=True)  # bin edges
    cnts, bins = _N.histogram(xp, bins=fs)   #  give bins= the bin boundaries
    dx = _N.diff(bins)[0]             # bin widths
    x = 0.5*(bins[1:] + bins[0:-1])   # bin centers

    #  smooth the spatial histogram
    smth_krnl = 2
    gk        = gauKer(smth_krnl) 
    gk        /= _N.sum(gk)
    fcnts = _N.convolve(cnts, gk, mode="same")
    dfcnts= _N.diff(fcnts)

    xp_inn = _N.where((dfcnts[0:-1] <= 0) & (dfcnts[1:] > 0))[0]  #  minima of histogram

    pcs    = xp_inn.shape[0] + 1
    xp_bds = _N.zeros(pcs+1, dtype=_N.int)
    xp_bds[1:pcs] = xp_inn
    xp_bds[pcs]   = fN-1

    y  = _N.zeros(fN-1) 
    mns = _N.empty(pcs)
    sds = _N.empty(pcs)

    for t in xrange(pcs):
        mns[t] = _N.dot(cnts[xp_bds[t]:xp_bds[t+1]], x[xp_bds[t]:xp_bds[t+1]]) / _N.sum(cnts[xp_bds[t]:xp_bds[t+1]])

        p      = cnts[xp_bds[t]:xp_bds[t+1]] / float(_N.sum(cnts[xp_bds[t]:xp_bds[t+1]]))
        sds[t] = _N.dot(p, (x[xp_bds[t]:xp_bds[t+1]]-mns[t])*(x[xp_bds[t]:xp_bds[t+1]]-mns[t]))

    isNOTinfORnan = 1 - (_N.isnan(mns) | _N.isinf(mns))

    pcsEff = _N.sum(isNOTinfORnan)
    mnsE = _N.empty(pcsEff)
    sdsE = _N.empty(pcsEff)
    amps = _N.empty(pcsEff)

    ipcE = -1
    for t in xrange(pcs):
        if not (_N.isnan(mns[t]) or _N.isinf(mns[t])):
            ipcE += 1
            areaUnder = dx*(_N.sum(cnts[xp_bds[t]:xp_bds[t+1]]))
            y += (areaUnder / _N.sqrt(2*_N.pi*sds[t])) * _N.exp(-0.5*(x-mns[t])*(x-mns[t])/sds[t])
            amps[ipcE] = areaUnder
            mnsE[ipcE] = mns[t]
            sdsE[ipcE] = sds[t]

    return pcsEff, amps, mnsE, sdsE
