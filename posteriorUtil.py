import numpy as _N
from filter import gauKer#, contiguous_pack2
import matplotlib.pyplot as _plt
import scipy.special as _ssp
import scipy.stats as _ss
#from scipy.spatial.distance import cdist, euclidean

_GAMMA         = 0
_INV_GAMMA     = 1

def sampling_from_stationary_KS(smps, blksz=200):
    SMPS, M   = smps.shape[1:]

    wins      = SMPS/blksz - 1

    pvs       = _N.empty((M, 2, wins))
    ds        = _N.empty((M, 2, wins))
    frms      = _N.empty(M, dtype=_N.int)

    for m in xrange(2, M):
        diffDist = 0
        i = wins
        win1stDffrnt      = wins - 1
        lastWinDffrnt     = wins - 1
        thisWinDffrnt     = False
        while (diffDist <= 6) and (i > 0):
            i -= 1
            it0 = i*blksz
            it1 = (i+1)*blksz

            for d in xrange(2):
                kss, pv = _ss.ks_2samp(smps[d, SMPS-blksz:SMPS:5, m], smps[d, it0:it1:5, m])
                # if d == 0:
                #     fig = _plt.figure()
                #     _plt.plot(smps[d, SMPS-blksz:SMPS, m], color="black")
                #     _plt.plot(smps[d, it0:it1, m], color="blue")
                #     _plt.suptitle("%(w1)d  %(w2)d    %(pv).3e" % {"w1" : SMPS-blksz, "w2" : it0, "pv" : pv})

                pvs[m, d, i] = pv

                if pv < 5e-3:
                    if diffDist == 0:
                        win1stFound = it0

                    lastWinDffrnt     = i                    
                    diffDist += 1
            if lastWinDffrnt - i > 1:
                diffDist = 0   #  reset

        frms[m] = win1stFound

    return pvs, frms+blksz


def stationary_from_Z(smps, blksz=200):
    SMPS, M   = smps.shape[1:]

    wins      = SMPS/blksz - 1

    pvs       = _N.empty((M, 2, wins))
    ds        = _N.empty((M, 2, wins))
    frms      = _N.empty(M, dtype=_N.int)

    rshpd     = smps.reshape((3, wins+1, blksz, M))
    mrshpd    = _N.mean(rshpd, axis=2)
    sdrshpd   = _N.std(rshpd, axis=2)

    mLst                =         mrshpd[:, wins].reshape(3, 1, M)
    sdLst               =         sdrshpd[:, wins].reshape(3, 1, M)
    sdNLst               =         sdrshpd[:, 0:-1].reshape(3, wins, M)

    zL                =         (mrshpd[:, 0:-1] - mLst)/sdLst
    zNL               =         (mrshpd[:, 0:-1] - mLst)/sdNLst

    for m in xrange(2, M):
        win1stFound=0
        diffDist = 0
        i = wins
        win1stDffrnt      = wins - 1
        lastWinDffrnt     = wins - 1
        thisWinDffrnt     = False

        while (diffDist <= 6) and (i > 0):
            i -= 1
            it0 = i*blksz
            it1 = (i+1)*blksz

            for d in xrange(3):
                if ((zL[d, i, m] > 0.75) or (zL[d, i, m] < -0.75)) or \
                   ((zNL[d, i, m] > 0.75) or (zNL[d, i, m] < -0.75)):
                    if diffDist == 0:
                        win1stFound = it0

                    lastWinDffrnt     = i                    
                    diffDist += 1
                    #print "%(i)d stats of block different than earlier  diffDist %(dD)d"  % {"i" : i, "dD" : diffDist}
            #print "lastWinDffrnt - i  %d" % (lastWinDffrnt - i)
            if (lastWinDffrnt - i > 1) and (diffDist <= 6):
                #print "reset  %d" % i
                diffDist = 0   #  reset

        frms[m] = win1stFound

    return frms+blksz

def MAPvalues2(epc, smp_prms, postMode, frms, ITERS, M, nprms, occ, l_trlsNearMAP, alltrials=False):

    for m in xrange(M):
        frm = frms[m]
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

def gam_inv_gam_dist_ML(smps, dist=_GAMMA, clstr=None):

    """
    The a, B hyperparameters for gamma or inverse gamma distributions
    """
    N = len(smps)

    s_x_ix = _N.sum(smps) if dist == _GAMMA else _N.sum(1./smps)

    if len(_N.where(smps == 0)[0]) > 0:
        print "0 found for cluster %(c)d   for dist %(d)s" % {"c" : clstr, "d" : ("gamma" if dist == _GAMMA else "inv gamma")}
        return None, None
    
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
