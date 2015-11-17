import numpy as _N
import fastdist as _fdst
import scipy.cluster.vq as scv
import matplotlib.pyplot as _plt
import clrs as _clrs
from filter import gauKer
from sklearn import cluster
from scipy.spatial import distance
import sklearn.datasets
from sklearn.preprocessing import StandardScaler

def spClstrs3MkCl(smkpos):
    spbins   = _N.linspace(-6, 6, 121)   # spatial bins
    hst, bns = _N.histogram(smkpos[:, 0], bins=spbins)
    gk = gauKer(2)
    gk /= _N.sum(gk)

    cx = _N.convolve(hst, gk, mode="same")

    dcx = _N.diff(cx)

    #
    _bdrs = _N.where((dcx[0:-1] <= 0) & (dcx[1:] >= 0))[0]
    bdrs  = _N.empty(len(_bdrs) + 2, dtype=_N.int)
    bdrs[1:-1] = _bdrs
    bdrs[0] = 0
    bdrs[-1] = 120

    allLabs = _N.empty(smkpos.shape[0], dtype=_N.int)

    iss = -1

    bestKs = _N.empty(len(bdrs)-1, dtype=_N.int)

    labsAll   = []

    globalInds = _N.ones(smkpos.shape[0], dtype=_N.int)*-1

    for spcl in xrange(len(bdrs)-1):
        iss += 1
        L = spbins[bdrs[spcl]]
        R = spbins[bdrs[spcl+1]]

        clInSp = _N.where((smkpos[:, 0] >= L) & (smkpos[:, 0] <= R))[0]

        TR = 20
        bics = _N.empty((6, TR))

        labsTemp = []
        for K in xrange(1, 7):
            labsTemp.append([])
            for tr in xrange(TR):
                cls, labs = scv.kmeans2(smkpos[clInSp, 1:], K)
                bcs = _N.bincount(labs)
                zrs = _N.where(bcs == 0)[0]

                if (len(bcs) < K) or (len(zrs) > 0):  #  missing a cluster
                    bics[K-1, tr] = -10e50
                else:
                    bics[K-1, tr] = kmBIC(cls, labs, K, smkpos[clInSp, 1:])
                labsTemp[K-1].append(labs)

        mb = _N.max(bics, axis=1)
        bestKs[spcl] = _N.where(mb == _N.max(mb))[0][0] + 1

        coords = _N.where(bics == _N.max(bics))
        bestLab = labsTemp[coords[0][0]][coords[1][0]]

        labsAll.append(bestLab)

        # fig = _plt.figure()
        # _plt.plot(_N.max(bics, axis=1))
        # fig = _plt.figure(figsize=(8, 8))
        # fig.add_subplot(2, 2, 1)
        # aclrs = _N.array(clrs.get_colors(bestKs[spcl]))[bestLab]
        # _plt.scatter(mkpos[nhid[clInSp], 0], mkpos[nhid[clInSp], 1], c=aclrs, s=50)
        # fig.add_subplot(2, 2, 2)
        # _plt.scatter(mkpos[nhid[clInSp], 0], mkpos[nhid[clInSp], 2], c=aclrs, s=50)
        # fig.add_subplot(2, 2, 3)
        # _plt.scatter(mkpos[nhid[clInSp], 0], mkpos[nhid[clInSp], 3], c=aclrs, s=50)
        # fig.add_subplot(2, 2, 4)
        # _plt.scatter(mkpos[nhid[clInSp], 0], mkpos[nhid[clInSp], 4], c=aclrs, s=50)

        globalInds[clInSp] = bestLab + _N.sum(bestKs[0:spcl])

    #aclrs = _N.array(clrs.get_colors(_N.sum(bestKs)))[globalInds]

    # fig = _plt.figure(figsize=(11, 11))
    # for i in xrange(4):
    #     fig.add_subplot(2, 2, i+1)
    #     _plt.scatter(mkpos[nhid, 0], mkpos[nhid, i+1], c=aclrs, s=50)
    return globalInds

    # spbins   = _N.linspace(-6, 6, 121)   # spatial bins
    # hst, bns = _N.histogram(smkpos[:, 0], bins=spbins)
    # gk = gauKer(2)
    # gk /= _N.sum(gk)

    # cx = _N.convolve(hst, gk, mode="same")
    
    # dcx = _N.diff(cx)

    # #
    # _bdrs = _N.where((dcx[0:-1] <= 0) & (dcx[1:] >= 0))[0]
    # bdrs  = _N.empty(len(_bdrs) + 2, dtype=_N.int)
    # bdrs[1:-1] = _bdrs
    # bdrs[0] = 0
    # bdrs[-1] = 120

    # allLabs = _N.empty(smkpos.shape[0], dtype=_N.int)

    # iss = -1
    # for spcl in xrange(len(bdrs)-1):
    #     iss += 1
    #     L = spbins[bdrs[spcl]]
    #     R = spbins[bdrs[spcl+1]]

    #     clInSp = _N.where((smkpos[:, 0] >= L) & (smkpos[:, 0] <= R))[0]
    #     cls, labs = scv.kmeans2(smkpos[clInSp, 1:], 3)
    #     for clInMk in xrange(3):
    #         inds = _N.where(labs == clInMk)[0]

    #         if len(inds) > 0:
    #             allLabs[clInSp[inds]] = clInMk + iss*4
    # return allLabs


def bestcluster(ITERS, smkpos, MS):
    """
    MS  # of clusters
    """
    ICR         = _N.zeros(ITERS)
    minOfMaxICR = _N.zeros(ITERS)
    maxOfMaxICR = _N.zeros(ITERS)
    InterCR     = _N.zeros(ITERS)

    allCtrs = []
    allLabs = []

    for it in xrange(ITERS):
        ctrs, labs = scv.kmeans2(smkpos, MS)  #  big clear
        intraClusRadii = []
        maxIntraClusRadii = []
        tot  = 0
        #interClusRadii = 0

        prs = 0
        for m in xrange(MS):
            inClus = _N.where(labs == m)[0]

            if len(inClus) > 3:
                radii = _N.sqrt(_N.sum((smkpos[inClus] - ctrs[m])**2, axis=1))
                intraClusRadii.append(_N.sum(radii))
                maxIntraClusRadii.append(_N.max(radii))
                tot += len(inClus)
        ICR[it]    = _N.sum(intraClusRadii)/ tot
        #minOfMaxICR[it] = _N.min(maxIntraClusRadii)
        maxOfMaxICR[it] = _N.max(maxIntraClusRadii)
        #InterCR[it]     = interClusRadii

        allCtrs.append(ctrs)
        allLabs.append(labs)


    m1 = _N.mean(ICR)
    s1 = _N.std(ICR)
    z1 = (ICR - m1) / s1

    m3 = _N.mean(maxOfMaxICR)
    s3 = _N.std(maxOfMaxICR)
    z3 = (maxOfMaxICR - m3) / s3
    
    z13 = z1 * z3
    bestlabs = _N.where(z13 == _N.min(z13))[0]

    return allLabs[bestlabs[0]]

def findBestClusterBySplit(mkpos, MS, mdim, ITERS):  #  non-hash
    """
    For the non-hash
    We don't mind if an actual cluster is represented by 2 nearby Gaussians 
    We do mind if only 1 Gaussian is used to represent 2 clusters
    We go about initing the clusters then, as follows:
    
    Rough k-means first.
    Then, for each of the clusters, ask if it is better to split that cluster in 2.
    If so, split that cluster in 2
    MS  # of clusters
    """

    allLabs = []
    opCls   = _N.empty(ITERS, dtype=_N.int)
    maxOfMaxICR = _N.empty(ITERS)
    for it in xrange(ITERS):
        MS0 = int(MS*0.7)
        MS1 = int(MS*0.3)
        #  let's allocate
        ctrs, labs = scv.kmeans2(mkpos, MS0)  #  big clear

        nmks =  mkpos.shape[0]

        zrsFnd = 0
        for m in xrange(MS0-1, -1, -1):
            if len(_N.where(labs == m)[0]) == 0:
                zrsFnd += 1
                #  strictly 0. 
                #  for all labels > m, subtract 1
                labs[_N.where(labs >= m+1)[0]] -= 1

        print "found %d zero clusters" % zrsFnd
        opCl = MS0-zrsFnd

        done = False
        while not done:
            fd2 = 0
            for cl in xrange(len(labs)):
                bigClstr = _N.where(labs == cl)[0]
                if len(bigClstr) > mdim*6:
                    nc, labs01 = oneORtwo(mkpos[bigClstr])
                    if nc > 1:
                        fd2 += 1
                        #  labs01 == 0    keep these in cl-th cluster
                        #  labs01 == 1    move these to opCl-th cluster
                        labs[bigClstr[_N.where(labs01 == 1)[0]]] = opCl
                        opCl += 1
            done = True if (fd2 == 0) else False

        maxIntraClusRadii = []

        for m in xrange(opCl):
            inClus = _N.where(labs == m)[0]

            if len(inClus) > 1:
                ctrs = _N.mean(mkpos[inClus], axis=0)
                #  radii - all distances between all marks and center
                radii = _N.sqrt(_N.sum((mkpos[inClus] - ctrs)**2, axis=1))
                #intraClusRadii.append(_N.sum(radii))
                maxIntraClusRadii.append(_N.max(radii))
                #tot += len(inClus)
        #ICR[it]    = _N.sum(intraClusRadii)/ tot
        #minOfMaxICR[it] = _N.min(maxIntraClusRadii)
        maxOfMaxICR[it] = _N.max(maxIntraClusRadii)
        #InterCR[it]     = interClusRadii
        opCls[it] = opCl

        allLabs.append(labs)
        #colorclusters(mkpos, labs, opCl)

    bestlabs = _N.where(maxOfMaxICR == _N.min(maxOfMaxICR))[0]
    print "bestlab is %d" % bestlabs[0]
    return allLabs[bestlabs[0]], opCls[bestlabs[0]]
        
def bestclusterNoBi(ITERS, smkpos, MS):
    """
    MS  # of clusters
    #  Don't care if single real cluster is fitted with 2 clusters
    #  I DO care that > 2 clusters are fitted with a single
    """
    ICR         = _N.zeros(ITERS)
    minOfMaxICR = _N.zeros(ITERS)
    maxOfMaxICR = _N.zeros(ITERS)
    InterCR     = _N.zeros(ITERS)

    allCtrs = []
    allLabs = []

    for it in xrange(ITERS):
        ctrs, labs = scv.kmeans2(smkpos, MS)  #  big clear
        intraClusRadii = []
        maxIntraClusRadii = []
        tot  = 0
        #interClusRadii = 0

        prs = 0
        for m in xrange(MS):
            inClus = _N.where(labs == m)[0]

            if len(inClus) > 3:
                radii = _N.sqrt(_N.sum((smkpos[inClus] - ctrs[m])**2, axis=1))
                intraClusRadii.append(_N.sum(radii))
                maxIntraClusRadii.append(_N.max(radii))
                tot += len(inClus)
        ICR[it]    = _N.sum(intraClusRadii)/ tot
        #minOfMaxICR[it] = _N.min(maxIntraClusRadii)
        maxOfMaxICR[it] = _N.max(maxIntraClusRadii)
        #InterCR[it]     = interClusRadii

        allCtrs.append(ctrs)
        allLabs.append(labs)

    m1 = _N.mean(ICR)
    s1 = _N.std(ICR)
    z1 = (ICR - m1) / s1

    m3 = _N.mean(maxOfMaxICR)
    s3 = _N.std(maxOfMaxICR)
    z3 = (maxOfMaxICR - m3) / s3
    
    z13 = z1 * z3
    bestlabs = _N.where(z13 == _N.min(z13))[0]

    return allLabs[bestlabs[0]]

def positionalClusters(hsh, bins, MH):
    #histdat = _plt.hist(hsh, bins=bins)
    histdat = _N.histogram(hsh, bins=bins)
    hst     = histdat[0]

    inds  = _N.array([i[0] for i in sorted(enumerate(hst), key=lambda x:x[1], reverse=True)])

    kpths = _N.empty(MH+1, dtype=_N.int)
    kpths[0] = inds[0]
    bDone = False

    i    = 0
    added= 1
    while not bDone:
        i += 1
        if _N.min(_N.abs(inds[i] - kpths[0:added])) > 2:
            kpths[added] = inds[i]
            added += 1

        bDone = True if (added == MH+1) else False

    skpths = _N.sort(kpths)
    skpths[0] = 0
    skpths[-1] = bins.shape[0]-1

    #  put labels on these

    labsH  = _N.empty(hsh.shape[0], dtype=_N.int)

    for i in xrange(hsh.shape[0]):
        n = _N.where((bins[skpths[0:-1]] <= hsh[i]) & (bins[skpths[1:]] >= hsh[i]))[0]
        labsH[i] = n

    return labsH


def colorclusters(smkpos, labs, MS):
    fig = _plt.figure(figsize=(12, 8))

    myclrs = _clrs.get_colors(MS)
    for m in xrange(MS):
        inds = _N.where(labs == m)[0]
        for k in xrange(4):
            fig.add_subplot(2, 2, k+1)
            _plt.scatter(smkpos[inds, 0], smkpos[inds, k+1], color=myclrs[m], s=9)

def oneORtwo(mks):
    ctrs, labs = scv.kmeans2(mks, 2)
    if (len(_N.where(labs == 0)[0]) < 2) or (len(_N.where(labs == 1)[0]) < 2):
        return 1, None
    else:
        return 2, labs
        """
        D0 = _fdst.dist(mks[_N.where(labs == 0)[0]])
        D1 = _fdst.dist(mks[_N.where(labs == 1)[0]])
        DA = _fdst.dist(mks)

        mDA, mD0, mD1 = _N.mean(DA), _N.mean(D0), _N.mean(D1)
        if (mDA > 1.6*mD0) or (mDA > 1.6*mD1):
            return 2, labs
        else:
            return 1, None
        """

def sepHash(_x, BINS=20, blksz=20, xlo=-6, xhi=6):
    ##########################
    bins    = _N.linspace(xlo, xhi, BINS+1)
    
    cumcnts = _N.zeros(BINS)
    #####################   separate hash / nonhash indices
    nonhash = []
    for ch in xrange(1, 5):
        done    = False
        inds  = _N.array([i[0] for i in sorted(enumerate(_x[:, ch]), key=lambda x:x[1], reverse=True)])

        blk = -1
        cumcnts[:] = 0

        while not done:
            blk += 1
            cnts, bns = _N.histogram(_x[inds[blk*blksz:(blk+1)*blksz], 0], bins=bins)
            cumcnts += cnts
            if len(_N.where(cumcnts < 2)[0]) <= 5:
                done = True
                nonhash.extend(inds[0:(blk+1)*blksz])

    unonhash = _N.unique(nonhash)  #  not hash spikes
    hashsp   = _N.setdiff1d(inds, unonhash)  #  inds is contiguous but reordered all

    return unonhash, hashsp


def kmBIC(ctrs, labs, K, X):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    #number of clusters
    m = K
    # size of the clusters
    n = _N.bincount(labs)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    #cl_var = [(1.0 / (n[i] - m)) * sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 'euclidean')**2)  for i in xrange(m)]
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[_N.where(labs == i)], [ctrs[i]], 'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * _N.log(N) * (d+1)

    BIC = _N.sum([n[i] * _N.log(n[i]) -
                  n[i] * _N.log(N) -
                  ((n[i] * d) / 2) * _N.log(2*_N.pi*cl_var) -
                  ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)

