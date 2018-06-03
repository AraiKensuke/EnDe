import numpy as _N
import scipy.cluster.vq as scv
import matplotlib.pyplot as _plt
import clrs as _clrs
from filter import gauKer
from sklearn import cluster
from sklearn import mixture
from scipy.spatial import distance
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
import openTets as _oT
import clrs
import scipy.stats as _ss
import matplotlib.pyplot as _plt

_qrtrd_  = 0
_halved_ = 1
_single_ = 2

def contiguous_pack2(arr, startAt=0):
    """
    arr4 = _N.array([10, 4, 2, 2, 9, 3])
    Contiguous_pack(arr4)
    #  arr4  ->  4 2 0 0 3 1   #  array where _N.unique(entries) is contiguous
    Probably faster previous version
    """
    unqItms = _N.unique(arr)   #  5 uniq items
    nUnqItms= unqItms.shape[0] #  

    contg   = _N.arange(0, len(unqItms)) + unqItms[0]
    nei     = _N.where(unqItms > contg)[0]
    for i in xrange(len(nei)):
        arr[_N.where(arr == unqItms[nei[i]])[0]] = contg[nei[i]]
    arr += (startAt - unqItms[0])
    return nUnqItms

def channelMins(posmarks, steps, K, ignoreN):
    mins = _N.min(posmarks[:, 1:], axis=0)
    maxs = _N.max(posmarks[:, 1:], axis=0)

    print "ignoreN   %d" % ignoreN
    dm   = (maxs - mins) / steps

    im = 0
    bDone= False
    while (not bDone) and (im < steps):
        im += 1
        spk_n, chs = _N.where(posmarks[:, 1:] < (mins + im*dm))
        unique, counts = _N.unique(spk_n, return_counts=True)
        if len(_N.where(counts == K)[0]) >= ignoreN:
            bDone = True
            print "done  %(im)d   %(n)d" % {"im" : im, "n" : len(_N.where(counts == K)[0])}

    return mins + im*dm

def resortLabelsByClusterSize(labarr):
    #  relabel cluster labels so that larger clusters have low labels
    ulabs = _N.unique(labarr)
    
    clabs  = _N.empty(len(labarr))   # copy of labs
    ulabs, cnts = _N.unique(labarr, return_counts=True)
    uc    = _N.empty((ulabs.shape[0], 2))
    uc[:, 0] = ulabs
    uc[:, 1] = cnts
    kys = _N.lexsort((uc[:, 0], uc[:, 1]))  #  sort by last column  (cnts)

    nunq = len(ulabs)
    #  longest clusters first
    ik   = ulabs[0]  #  the smallest label
    
    for nu in xrange(nunq-1, -1, -1):
        replace = _N.where(labarr == kys[nu])[0]  #   old sspots
        clabs[replace] = ik

        ik += 1
    labarr[:] = clabs[:]

def colorclusters(smkpos, labs, MS, name="", xLo=0, xHi=3):
    fig = _plt.figure(figsize=(12, 8))
    _plt.suptitle("spikes   %d" % len(labs))
    myclrs = _clrs.get_colors(MS)
    for m in xrange(MS):
        inds = _N.where(labs == m)[0]
        for k in xrange(4):
            fig.add_subplot(2, 2, k+1)
            _plt.scatter(smkpos[inds, 0], smkpos[inds, k+1], color=myclrs[m], s=9)
            _plt.xlim(xLo-(xHi-xLo)*0.1, xHi+(xHi-xLo)*0.1)
    _plt.savefig("cc%s-all" % name)
    _plt.close()

    L = 0
    for m in xrange(MS):
        fig = _plt.figure(figsize=(12, 8))
        inds = _N.where(labs == m)[0]
        _plt.suptitle("spikes   %d" % len(inds))
        L += len(inds)
        for k in xrange(4):
            fig.add_subplot(2, 2, k+1)
            _plt.scatter(smkpos[inds, 0], smkpos[inds, k+1], color=myclrs[m], s=9)
            _plt.xlim(xLo-(xHi-xLo)*0.1, xHi+(xHi-xLo)*0.1)

        _plt.savefig("cc%(n)s-%(m)d" % {"n" : name, "m" : m})
        _plt.close()


def sepHash(_x, BINS=50, blksz=20, xlo=-6, xhi=6, K=4):
    ##########################
    bins    = _N.linspace(xlo, xhi, BINS+1)
    
    cumcnts = _N.zeros(BINS)
    #####################   separate hash / nonhash indices
    nonhash = []

    totalMks = _x.shape[0]

    # only look in bins where at least minInBin marks observed.
    for ch in xrange(1, K+1):
        cnts0, bns0 = _N.histogram(_x[:, 0], bins=bins)
        mncnt       = _N.mean(cnts0)
        minBins       = _N.where(cnts0 > 0.5*mncnt)[0]   # only compare these
        mincnt      = _N.array(cnts0 * 0.02, dtype=_N.int)
        mincnt[_N.where(mincnt == 0)[0]] = 1    #  set it to at least 1
        done    = False
        inds  = _N.array([i[0] for i in sorted(enumerate(_x[:, ch]), key=lambda x:x[1], reverse=True)])

        blk = -1
        cumcnts[:] = 0

        while not done:
            """
            Start from large spikes with large mark.  cumcnts will be 0 in all 
            minBins.  Keep lowering mark until there are no more empty minBins
            """
            blk += 1
            cnts, bns = _N.histogram(_x[inds[blk*blksz:(blk+1)*blksz], 0], bins=bins)
            cumcnts += cnts
            nearEmptyBins = len(_N.where(cumcnts[minBins] < mincnt[minBins])[0])
            if nearEmptyBins < int(BINS*0.2):
                done = True
                nonhash.extend(inds[0:(blk+1)*blksz])

    unonhash = _N.unique(nonhash)  #  not hash spikes
    hashsp   = _N.setdiff1d(inds, unonhash)  #  inds is contiguous but reordered all

    return unonhash, hashsp, _N.max(_x[hashsp, 1:], axis=0)

def kmBIC(ctrs, labs, X):
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
    m = contiguous_pack2(labs, startAt=0)
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

    return(BIC), m

def emMKPOS_sep1A(nhmks, hmks, TR=5, wfNClstrs=[[1, 7], [1, 10]], spNClstrs=[[1, 2], [1, 7]], K=4, spcdim=1):
    #  wfNClstrs   [ non-hash(min, max), hash(min, max)  by waveform clustering
    TR = 2

    iNH = -1
    sNH = ["nh", "h"]

    bestLabs = []

    minSz = 5
    startClstrs = _N.empty(2, dtype=_N.int)
    iRMv_ME   = 0
    for mks in [hmks, nhmks]:
        iNH += 1
        minK, maxK = wfNClstrs[iNH]
        startCl = 0 
        bestLabMP = None
        if mks is not None:
            labs, bics, bestLab, nClstrs = _oT.EMwfBICs(mks, minK=minK, maxK=maxK, TR=TR)
            bestLab = _N.array(bestLab, dtype=_N.int)

            #    if mks == nhmks:

            ##  non-hash, do spatial clustering

            bestLabMP = _N.array(bestLab)
            #for i in _N.unique(bestLab):  # how many "neurons" did we find?
            #    print "for %(i)d   there are %(n)d" % {"i" : i, "n" : len(_N.where(bestLab == i)[0])}
            print "neurons for iNH=%(nh)d   WF nclusters %(nc)d" % {"nh" : iNH, "nc" : nClstrs}
            for nc in xrange(nClstrs):
                minK, _maxK = spNClstrs[iNH]

                inThisClstr = _N.where(bestLab == nc)[0]
                LiTC        = len(inThisClstr)

                if LiTC > 2:   #  at least 2 spikes from this neuron
                    pbestLab = _N.ones(LiTC, dtype=_N.int) * -1   #  poslabs
                    pos = mks[inThisClstr, 0:spcdim]
                    #_N.savetxt("clstr%(nh)d_%(iRM)d" % {"nh" : iNH, "iRM" : iRMv_ME}, mks[inThisClstr], fmt="%.4f %4f %.4f %4f %.4f")
                    iRMv_ME += 1
                    pos = pos.reshape(LiTC, spcdim)

                    #  1 spk / clstr maxK == LiTC   want a few more than 1 / clstr
                    #  maxK is at most _maxK unless LiTC
                    maxK = _maxK if LiTC > _maxK else LiTC-1

                    plabs, pbics, pbestLab, pClstrs = _oT.EMposBICs(pos, minK=minK, maxK=maxK, TR=2)
                    pClstrs = contiguous_pack2(pbestLab)
                else:   #  only 2 in this waveform cluster
                    pClstrs = 1
                    pbestLab= _N.zeros(LiTC, dtype=_N.int)
                pbestLab[_N.where(pbestLab >= 0)[0]] += startCl  #  only the ones used for init
                startCl += pClstrs
                bestLabMP[inThisClstr] = pbestLab
        startClstrs[iNH] = startCl

        if bestLabMP is None:
            bestLabs.append(_N.array([], dtype=_N.int))
        else:
            bestLabs.append(_N.array(bestLabMP))
    return bestLabs[0], bestLabs[1], startClstrs


def ignoreSmallMarks(marks, allFR):
    #  return maximum values on each channel.  If a given spike has mark
    #  that's smaller in every channel, ignore it.

    marks.sort(axis=0)
    print marks[:, 0]
    print marks[:, 1]
    print marks[:, 2]
    print marks[:, 3]
