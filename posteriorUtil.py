import numpy as _N
from filter import gauKer
import matplotlib.pyplot as _plt
#from scipy.spatial.distance import cdist, euclidean

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
                postMode[epc, col] = _N.mean(smps)
                #postMode[epc, col] = _N.median(smps)
            if l_trlsNearMAP is not None:
                # trlsNearMAP for each params added to list
                l_trlsNearMAP.append(trlsNearMAP)  




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
