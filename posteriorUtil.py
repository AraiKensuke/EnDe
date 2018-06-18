import numpy as _N
from filter import gauKer#, contiguous_pack2
import matplotlib.pyplot as _plt
import scipy.special as _ssp
import scipy.stats as _ss
#from scipy.spatial.distance import cdist, euclidean

_GAMMA         = 0
_INV_GAMMA     = 1

def find_good_clstrs_and_stationary_from(M, smps, spcdim=1):
    #  treat narrow and wide cluster differently because the correlation
    #  timescale of sequential samples are quite different between slow
    #  and wide clusers
    if spcdim == 1:
        frm_narrow = stationary_from_Z_bckwd(smps, blksz=200)
        frm_wide   = stationary_from_Z_bckwd(smps, blksz=500)
    else:
        frm_narrow = stationary_from_Z_bckwd_2d(smps, blksz=200)
        frm_wide   = stationary_from_Z_bckwd_2d(smps, blksz=500)

    ITERS      = smps.shape[1]

    frms       = _N.empty(M, dtype=_N.int)

    if spcdim == 1:
        q2_mdn     = _N.median(smps[2, ITERS-1000:], axis=0)

        wd_clstrs  = _N.where(q2_mdn > 9)[0]
        nrw_clstrs  = _N.where(q2_mdn <= 9)[0]    
    else:
        q2x_mdn     = _N.median(smps[3, ITERS-1000:], axis=0)
        q2y_mdn     = _N.median(smps[4, ITERS-1000:], axis=0)

        wd_clstrs  = _N.where((q2x_mdn > 9) | (q2y_mdn > 9))[0]
        nrw_clstrs  = _N.where((q2x_mdn <= 9) | (q2y_mdn <= 9))[0]    

    frms[nrw_clstrs] = frm_narrow[nrw_clstrs]
    frms[wd_clstrs] = frm_wide[wd_clstrs]

    return frms

"""    
def stationary_from_Z_bckwd(smps, blksz=200):
    #  Detect when stationarity reached in Gibbs sampling
    #  Also, detect whether cluster is switching between local extremas
    #
    SMPS, M   = smps.shape[1:]

    wins         = SMPS/blksz
    wins_m1      = wins - 1

    frms      = _N.empty(M, dtype=_N.int)  # current start of stationarity

    rshpd     = smps.reshape((3, wins, blksz, M))
    mrshpd    = _N.mean(rshpd, axis=2)   #  3 x wins_m1+1 x M
    sdrshpd   = _N.std(rshpd, axis=2)

    mLst                =         mrshpd[:, wins_m1].reshape(3, 1, M)
    sdLst               =         sdrshpd[:, wins_m1].reshape(3, 1, M)
    sdNLst               =         sdrshpd[:, 0:-1].reshape(3, wins_m1, M)

    zL                =         (mrshpd[:, 0:-1] - mLst)/sdLst
    zNL               =         (mrshpd[:, 0:-1] - mLst)/sdNLst

    #  mn, std in each win
    #  u1=0.3, sd1=0.9      u2=0.4, sd2=0.8
    #  (u2-u1)/sd1  

    #  detect sudden changes

    for m in xrange(M):
        win1stFound=(wins_m1-1)*blksz
        sameDist = 0
        i = 0

        thisWinSame     = False
        lastWinSame     = 0#wins_m1

        #  want 3 consecutive windows where distribution looks different
        while (sameDist <= 3) and (i < wins_m1-1):
            i += 1
            it0 = i*blksz
            it1 = (i+1)*blksz

            thisWinSame = 0

            for d in xrange(3):
                if ((zL[d, i, m] < 0.75) and (zL[d, i, m] > -0.75)) and \
                   ((zNL[d, i, m] < 0.75) and (zNL[d, i, m] > -0.75)):
                    
                    thisWinSame += 1

            if thisWinSame == 3:
                if sameDist == 0:
                    win1stFound = it0
                lastWinSame = i

                sameDist += 1

            if (i - lastWinSame > 1) and (sameDist <= 3):
                #print "reset  %d" % i
                sameDist = 0   #  reset
                win1stFound = (wins_m1-1)*blksz

        frms[m] = win1stFound

    return frms+blksz
"""

def stationary_from_Z_bckwd(smps, blksz=200):
    #  Detect when stationarity reached in Gibbs sampling
    #  Also, detect whether cluster is switching between local extremas
    #
    SMPS, M   = smps.shape[1:]   #  smp_sp_prms = _N.zeros((3, ITERS, M_use))  

    wins         = SMPS/blksz
    wins_m1      = wins - 1

    frms      = _N.empty(M, dtype=_N.int)  # current start of stationarity

    if spcdim == 1:
        reparam     = _N.empty((2, SMPS, M))   #  reparameterized
        reparam[0]  = smps[1]
        reparam[1]  = smps[0] / _N.sqrt(smps[2])

        rshpd     = reparam.reshape((2, wins, blksz, M))
    else:
        reparam     = _N.empty((3, SMPS, M))   #  reparameterized
        reparam[0]  = smps[1]  # fx
        reparam[1]  = smps[2]  # fy
        reparam[2]  = smps[0] / _N.sqrt(smps[3]*smps[4])  #  

        rshpd     = reparam.reshape((3, wins, blksz, M))

    mrshpd    = _n.median(rshpd, axis=2)   #  2 x wins_m1+1 x m
    sdrshpd   = _n.std(rshpd, axis=2)

    mlst                =         mrshpd[:, wins_m1].reshape(2, 1, M)
    sdlst               =         sdrshpd[:, wins_m1].reshape(2, 1, M)
    sdnlst               =         sdrshpd[:, 0:-1].reshape(2, wins_m1, M)

    zL                =         (mrshpd[:, 0:-1] - mLst)/sdLst
    zNL               =         (mrshpd[:, 0:-1] - mLst)/sdNLst

    #  mn, std in each win
    #  u1=0.3, sd1=0.9      u2=0.4, sd2=0.8
    #  (u2-u1)/sd1  

    #  detect sudden changes

    for m in xrange(M):
        win1stFound=(wins_m1-1)*blksz
        sameDist = 0
        i = 0

        thisWinSame     = False
        lastWinSame     = 0#wins_m1

        #  want 3 consecutive windows where distribution looks different
        while (sameDist <= 2) and (i < wins_m1-1):
            i += 1
            it0 = i*blksz
            it1 = (i+1)*blksz

            thisWinSame = 0

            for d in xrange(2):
                if ((zL[d, i, m] < 0.75) and (zL[d, i, m] > -0.75)) and \
                   ((zNL[d, i, m] < 0.75) and (zNL[d, i, m] > -0.75)):
                    
                    thisWinSame += 1

            if thisWinSame == 2:
                if sameDist == 0:
                    win1stFound = it0
                lastWinSame = i

                sameDist += 1

            if (i - lastWinSame > 1) and (sameDist <= 2):
                #print "reset  %d" % i
                sameDist = 0   #  reset
                win1stFound = (wins_m1-1)*blksz

        frms[m] = win1stFound

    return frms+blksz

def stationary_from_Z_bckwd_2d(smps, blksz=200):
    #  Detect when stationarity reached in Gibbs sampling
    #  Also, detect whether cluster is switching between local extremas
    #
    SMPS, M   = smps.shape[1:]   #  smp_sp_prms = _N.zeros((3, ITERS, M_use))  

    wins         = SMPS/blksz
    wins_m1      = wins - 1

    frms      = _N.empty(M, dtype=_N.int)  # current start of stationarity

    reparam     = _N.empty((3, SMPS, M))   #  reparameterized
    reparam[0]  = smps[1]  # fx
    reparam[1]  = smps[2]  # fy
    reparam[2]  = smps[0] / _N.sqrt(smps[3]*smps[4])  #  

    rshpd     = reparam.reshape((3, wins, blksz, M))

    mrshpd    = _N.median(rshpd, axis=2)   #  2 x wins_m1+1 x m
    sdrshpd   = _N.std(rshpd, axis=2)

    mLst                =         mrshpd[:, wins_m1].reshape(3, 1, M)
    sdLst               =         sdrshpd[:, wins_m1].reshape(3, 1, M)
    sdNLst               =         sdrshpd[:, 0:-1].reshape(3, wins_m1, M)

    zL                =         (mrshpd[:, 0:-1] - mLst)/sdLst
    zNL               =         (mrshpd[:, 0:-1] - mLst)/sdNLst

    #  mn, std in each win
    #  u1=0.3, sd1=0.9      u2=0.4, sd2=0.8
    #  (u2-u1)/sd1  

    #  detect sudden changes

    for m in xrange(M):
        win1stFound=(wins_m1-1)*blksz
        sameDist = 0
        i = 0

        thisWinSame     = False
        lastWinSame     = 0#wins_m1

        #  want 3 consecutive windows where distribution looks different
        while (sameDist <= 2) and (i < wins_m1-1):
            i += 1
            it0 = i*blksz
            it1 = (i+1)*blksz

            thisWinSame = 0

            for d in xrange(3):
                if ((zL[d, i, m] < 0.75) and (zL[d, i, m] > -0.75)) and \
                   ((zNL[d, i, m] < 0.75) and (zNL[d, i, m] > -0.75)):
                    
                    thisWinSame += 1

            if thisWinSame == 3:
                if sameDist == 0:
                    win1stFound = it0
                lastWinSame = i

                sameDist += 1

            if (i - lastWinSame > 1) and (sameDist <= 2):
                #print "reset  %d" % i
                sameDist = 0   #  reset
                win1stFound = (wins_m1-1)*blksz

        frms[m] = win1stFound

    return frms+blksz

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

    
    for m in xrange(M):
        print "mmmmmmmmmmmmmmmmmmm  %d" % m
        for d in xrange(3):
            diff_dist          =         _N.where((zL[d, :, m] > 0.75) | (zL[d, :, m] < -0.75))[0]
            print diff_dist
                                           
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

                                        
