import numpy as _N
from filter import gauKer
import matplotlib.pyplot as _plt
import scipy.special as _ssp
#from scipy.spatial.distance import cdist, euclidean

_GAMMA         = 0
_INV_GAMMA     = 1

def MAPvalues(epc, smp_prms, postMode, frm, ITERS, M, nprms, occ, gk, l_trlsNearMAP):

    for m in xrange(M):
        fig = _plt.figure(figsize=(11, 4))
        trlsNearMAP = _N.arange(0, ITERS-frm)
        for ip in xrange(nprms):  # params
            fig.add_subplot(1, nprms, ip+1)
            L     = _N.min(smp_prms[ip, frm:, m]);   H     = _N.max(smp_prms[ip, frm:, m])
            AMP   = H-L
            L     -= 0.1*AMP
            H     += 0.1*AMP
            cnts, bns = _N.histogram(smp_prms[ip, frm:, m], bins=_N.linspace(L, H, 60))
            ###  take the iters that gave 25% top counts
            ###  intersect them for each param.
            ###  
            col = nprms*m+ip
            
            xfit = 0.5*(bns[0:-1] + bns[1:])
            yfit = cnts

            fcnts = _N.convolve(cnts, gk, mode="same")
            _plt.plot(xfit, fcnts)
            ib  = _N.where(fcnts == _N.max(fcnts))[0][0]

            xLo  = xfit[_N.where(fcnts > fcnts[ib]*0.6)[0][0]]
            xHi  = xfit[_N.where(fcnts > fcnts[ib]*0.6)[0][-1]]

            if occ[m] > 0:
                these=_N.where((smp_prms[ip, frm:, m] > xLo) & (smp_prms[ip, frm:, m] < xHi))[0]
                trlsNearMAP = _N.intersect1d(these, trlsNearMAP)

            xMAP  = bns[ib]                        
            postMode[epc, col] = xMAP

        if l_trlsNearMAP is not None:
            trlsNearMAP += frm
            l_trlsNearMAP.append(trlsNearMAP)

def MAPvalues2(epc, smp_prms, postMode, frm, ITERS, M, nprms, occ, gk, l_trlsNearMAP, alltrials=False):
    print "IN MAPvalues2 - not mean"
    N   = smp_prms.shape[1] - frm
    for m in xrange(M):
        #fig = _plt.figure(figsize=(11, 4))
        trlsNearMAP = _N.arange(0, ITERS-frm)
        if alltrials:
            l_trlsNearMAP.append(_N.arange(ITERS))
        else:
            for ip in xrange(nprms):  # for each param
                col = nprms*m+ip      #  0, 3, 6  for lambda0
                #fig.add_subplot(1, nprms, ip+1)

                smps  = smp_prms[ip, frm:, m]
                #postMode[epc, col] = _N.mean(smps)
                postMode[epc, col] = _N.median(smps)
            if l_trlsNearMAP is not None:
                # trlsNearMAP for each params added to list
                l_trlsNearMAP.append(trlsNearMAP)  


def gam_inv_gam_dist_ML(smps, dist=_GAMMA):
    """
    The a, B hyperparameters for gamma or inverse gamma distributions
    """
    N = len(smps)

    s_x_ix = _N.sum(smps) if dist == _GAMMA else _N.sum(1./smps)
    pm_s_logx = _N.sum(_N.log(smps)) 
    pm_s_logx *= 1 if dist == _GAMMA else -1

    mn  = _N.mean(smps)
    vr  = _N.std(smps)**2
    BBP = (mn / vr) if dist == _GAMMA else mn*(1 + (mn*mn)/vr)

    Bx  = _N.linspace(BBP/50, BBP*50, 1000)
    yx  = _N.empty(1000)
    iB  = -1
    for B in Bx:
        iB += 1
        P0 = _ssp.digamma(B/N * s_x_ix)
        yx[iB] = N*(_N.log(B) - P0) + pm_s_logx

    lst = _N.where((yx[:-1] >= 0) & (yx[1:] <= 0))[0]
    if len(lst) > 0:
        ib4 = lst[0]
        #  decreasing
        mslp  = (yx[ib4+1] - yx[ib4]) 
        rd    = (0-yx[ib4])  / mslp

        Bml = Bx[ib4] + rd*(Bx[ib4+1]-Bx[ib4])
        aml = (Bml/N)*s_x_ix   #  sum xi / N  = B/a
        return aml, Bml
    else:
        return None, None

"""
smps  = _ss.invgamma.rvs(a, scale=B, size=N)
ismps = smps.argsort()
spcng = _N.diff(smps[ismps])   #  
ispcng= spcng.argsort()

hstSpc= _N.mean(spcng[ispcng[0:int(0.1*N)]])

thr_ispc = _N.where(spcng[ispcng] < 100*hstSpc)

L     = _N.min(smps)
H     = _N.max(smps[ismps[thr_ispc[0]]])
A     = H-L
cnts, bns = _N.histogram(smps, bins=_N.linspace(L-0.1*A, H+0.1*A, 60))

gk    = gauKer(3)
xb    = 0.5*(bns[1:] + bns[0:-1])
fcnts = _N.convolve(cnts, gk, mode="same")
iMax  = int(_N.mean(_N.where(fcnts == _N.max(fcnts))[0]))
"""

#  borrowed from
#  http://stackoverflow.com/questions/30299267/geometric-median-of-multidimensional-points
"""
def geometric_median(X, eps=1e-5):
    y = _N.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = _N.sum(Dinv)
        W = Dinv / Dinvs
        T = _N.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - _N.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = _N.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1
"""
