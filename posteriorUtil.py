import numpy as _N
from filter import gauKer
import matplotlib.pyplot as _plt
import scipy.special as _ssp
#from scipy.spatial.distance import cdist, euclidean

_GAMMA         = 0
_INV_GAMMA     = 1

def findstat(smp_sp_prms, blksz, initBlk):
    """
    find stationary region of samples
    """
    smps   = _N.array(smp_sp_prms)    #  make a copy
    ITRS   = smps.shape[1]
    ITRSa  = ITRS/blksz
    M      = smps.shape[2]
    smps[2:] = _N.sqrt(smps[2:])
    fsmps  = _N.empty((6, ITRSa, M))
    fsmps[0:3] = _N.mean(smps.reshape((3, ITRSa, blksz, M)), axis=2)
    fsmps[3:6] = _N.std(smps.reshape((3, ITRSa, blksz, M)), axis=2)

    big1      = 9
    big2      = 3

    #  ITRSa = 10, initBlk = 3, last 3 used to mn    (7,8,9)   0,1,2,3,4,5,6
    dsts = _N.zeros((6, M, ITRSa-initBlk))

    for n in xrange(0, ITRSa-initBlk):  #  n is
        sd = _N.std(fsmps[:, n+1:, :], axis=1)  #  
        mn = _N.mean(fsmps[:, n+1:, :], axis=1)

        dsts[:, :, n] = (fsmps[:, n+1] - mn)/sd

    statry_frm_here = _N.empty(M, dtype=_N.int)
    for m in xrange(M):
        lstchgpts = []
        for ip in xrange(6):   # parameters
            bigabv  = _N.where(_N.abs(dsts[ip, m]) > big1)[0]   #  single big or 2 smaller cons
            consabv = _N.where((dsts[ip, m, 0:-1] < -big2) & (dsts[ip, m, 1:] < -big2))[0]
            consbel = _N.where((dsts[ip, m, 0:-1] >  big2) & (dsts[ip, m, 1:] >  big2))[0]

            lstchgpts.extend(bigabv)
            lstchgpts.extend(consabv+1)
            lstchgpts.extend(consbel+1)

        statry_frm_here[m] = lstchgpts[-1]+2 if len(lstchgpts) > 0 else 0
    return statry_frm_here

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
