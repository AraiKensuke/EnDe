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
        L += len(inds)
        for k in xrange(4):
            fig.add_subplot(2, 2, k+1)
            _plt.scatter(smkpos[inds, 0], smkpos[inds, k+1], color=myclrs[m], s=9)
            _plt.xlim(xLo-(xHi-xLo)*0.1, xHi+(xHi-xLo)*0.1)
        _plt.savefig("cc%(n)s-%(m)d" % {"n" : name, "m" : m})
        _plt.close()
    print L




def sepHash(_x, BINS=50, blksz=20, xlo=-6, xhi=6):
    ##########################
    bins    = _N.linspace(xlo, xhi, BINS+1)
    
    cumcnts = _N.zeros(BINS)
    #####################   separate hash / nonhash indices
    nonhash = []

    totalMks = _x.shape[0]

    # only look in bins where at least minInBin marks observed.
    for ch in xrange(1, 5):
    #for ch in xrange(1, 3):
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
            if len(_N.where(cumcnts[minBins] < mincnt[minBins])[0]) == 0:
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


def emMKPOS(nhmks, hmks, TR=5, minK=2, maxK=15):
    TR = 5
    minK=2
    maxK=15

    iNH = -1
    sNH = ["nh", "h"]

    bestLabs = []

    # nhid, hid = sepHash(mkpos)

    # nhmks     = mkpos[nhid]
    # hmks      = mkpos[hid]

    for mks in [nhmks, hmks]:
        iNH += 1
        labs, bics, bestLab, nClstrs = _oT.EMwfBICs(mks, minK=minK, maxK=maxK, TR=TR)
        bestLab = _N.array(labs[nClstrs-minK, 0], dtype=_N.int)

        #    if mks == nhmks:

        ##  non-hash, do spatial clustering
        startCl = 0 
        bestLabMP = _N.array(bestLab)
        maxK = 3 if (iNH == 1) else 10
        minK = 2 if (iNH == 1) else 3

        for nc in xrange(nClstrs):
            inThisClstr = _N.where(bestLab == nc)[0]
            pos = mks[inThisClstr, 0]

            pos = pos.reshape(len(inThisClstr), 1)

            if len(inThisClstr) > maxK:
                plabs, pbics, pbestLab, pClstrs = _oT.EMposBICs(pos, minK=1, maxK=maxK, TR=3)
                pClstrs = contiguous_pack2(pbestLab)
                cmp = _N.where(pbics == _N.min(pbics))[0]
            else:
                pbestLab = _N.zeros(len(inThisClstr), dtype=_N.int)
                pClstrs  = 1
            bestLabMP[inThisClstr] = pbestLab + startCl
            startCl += pClstrs
        bestLabs.append(_N.array(bestLabMP))
    return bestLabs[0], bestLabs[1]


def emMKPOS_sep(nhmks, hmks, TR=5, minK=2, maxK=15):
    TR = 2
    minK=2
    maxK=8

    iNH = -1
    sNH = ["nh", "h"]

    bestLabs = []

    minSz = 8
    startClstrs = _N.empty(2, dtype=_N.int)
    for mks in [nhmks, hmks]:
        iNH += 1
        startCl = 0 
        bestLabMP = None
        if mks is not None:
            labs, bics, bestLab, nClstrs = _oT.EMwfBICs(mks, minK=minK, maxK=maxK, TR=TR)
            bestLab = _N.array(labs[nClstrs-minK, 0], dtype=_N.int)

            #    if mks == nhmks:

            ##  non-hash, do spatial clustering

            bestLabMP = _N.array(bestLab)
            maxK = 3
            minK = 1

            splitSpace = _single_

            if splitSpace == _qrtrd_:
                maxK = 3
            if splitSpace == _halved_:
                maxK = 4
            if splitSpace == _single_:
                maxK = 7

            print "neurons for iNH=%(nh)d   nclusters %(nc)d" % {"nh" : iNH, "nc" : nClstrs}
            for nc in xrange(nClstrs):
                inThisClstr = _N.where(bestLab == nc)[0]
                pbestLab = _N.ones(len(inThisClstr), dtype=_N.int) * -1   #  poslabs
                pos = mks[inThisClstr, 0]
                pos = pos.reshape(len(inThisClstr), 1)

                if splitSpace == _qrtrd_:
                    nB_pos = _N.where((pos >= -6) & (pos < -3))[0]   #  -6 to -3
                    nF_pos = _N.where((pos >= -3) & (pos < 0))[0]   #  -3 to 0
                    pF_pos = _N.where((pos >= 0)  & (pos < 3))[0]   #  0 to 3
                    pB_pos = _N.where(pos >= 3)[0]

                    nB_pClstrs = nF_pClstrs = pF_pClstrs = pB_pClstrs = 0

                    #########
                    if len(nB_pos) >= minSz:
                        nB_plabs, nB_pbics, nB_pbestLab, nB_pClstrs = _oT.EMposBICs(pos[nB_pos], minK=minK, maxK=maxK, TR=2)
                        nB_pClstrs = contiguous_pack2(nB_pbestLab)
                        pbestLab[nB_pos] = nB_pbestLab
                    #########
                    if len(nF_pos) >= minSz:
                        nF_plabs, nF_pbics, nF_pbestLab, nF_pClstrs = _oT.EMposBICs(pos[nF_pos], minK=minK, maxK=maxK, TR=2)
                        nF_pClstrs = contiguous_pack2(nF_pbestLab)
                        pbestLab[nF_pos] = nF_pbestLab + nB_pClstrs
                    #########
                    if len(pF_pos) >= minSz:
                        pF_plabs, pF_pbics, pF_pbestLab, pF_pClstrs = _oT.EMposBICs(pos[pF_pos], minK=minK, maxK=maxK, TR=2)
                        pF_pClstrs = contiguous_pack2(pF_pbestLab)
                        pbestLab[pF_pos] = pF_pbestLab + nB_pClstrs + nF_pClstrs
                    #########
                    if len(pB_pos) >= minSz:
                        pB_plabs, pB_pbics, pB_pbestLab, pB_pClstrs = _oT.EMposBICs(pos[pB_pos], minK=minK, maxK=maxK, TR=2)
                        pB_pClstrs = contiguous_pack2(pB_pbestLab)
                        pbestLab[pB_pos] = pB_pbestLab + nB_pClstrs + nF_pClstrs + pF_pClstrs
                    pClstrs    = nB_pClstrs + nF_pClstrs + pF_pClstrs + pB_pClstrs
                elif splitSpace == _halved_:
                    n_pos = _N.where((pos >= -6) & (pos < 0))[0]   #  -6 to -3
                    p_pos = _N.where(pos >= 0)[0]   #  0 to 3

                    n_pClstrs  = p_pClstrs  = 0

                    if len(n_pos) >= minSz:
                        n_plabs, n_pbics, n_pbestLab, n_pClstrs = _oT.EMposBICs(pos[n_pos], minK=minK, maxK=maxK, TR=2)
                        n_pClstrs = contiguous_pack2(n_pbestLab)
                        pbestLab[n_pos] = n_pbestLab
                    #########
                    if len(p_pos) >= minSz:
                        p_plabs, p_pbics, p_pbestLab, p_pClstrs = _oT.EMposBICs(pos[p_pos], minK=minK, maxK=maxK, TR=10)
                        p_pClstrs = contiguous_pack2(p_pbestLab)
                        pbestLab[p_pos] = n_pbestLab + p_pClstrs
                    pClstrs    = n_pClstrs + p_pClstrs
                else:
                    plabs, pbics, pbestLab, pClstrs = _oT.EMposBICs(pos, minK=minK, maxK=maxK, TR=2)
                    pClstrs = contiguous_pack2(pbestLab)

                pbestLab[_N.where(pbestLab >= 0)[0]] += startCl  #  only the ones used for init
                startCl += pClstrs
                bestLabMP[inThisClstr] = pbestLab
        startClstrs[iNH] = startCl
        if bestLabMP is None:
            bestLabs.append(_N.array([], dtype=_N.int))
        else:
            bestLabs.append(_N.array(bestLabMP))
    return bestLabs[0], bestLabs[1], startClstrs

def mergesmallclusters(nhmks, hmks, slabs, hlabs, pmdim, clstrSzs):
    """
    slabs, hlabs are contiguous labels (both start from 0) of hash and nonhash
    """
    #  EM init sometimes creates very small clusters.  Merge these with the 
    #  larger clusters EM finds

    for hh in xrange(2):
        if hh == 0:
            labs = slabs
            mks  = nhmks
        else:
            labs = hlabs
            mks  = hmks
        smallOnes = []
        largeClstrs = []
        ulabs    = _N.unique(labs)   # labs contiguous
        lctrs      = []
        lcovs      = []

        for i in xrange(len(ulabs)):  # how many "neurons" did we find?
            sts = _N.where(labs == i)[0]
            sz  = len(sts)
            #print "for %(i)d   there are %(n)d" % {"i" : i, "n" : sz}
            if sz <= pmdim:
                smallOnes.extend(sts)
            else:
                largeClstrs.append(i)
                lctrs.append(_N.mean(mks[sts], axis=0))
                lcovs.append(_N.cov(mks[sts], rowvar=0))
        ctrs = _N.array(lctrs)
        covs = _N.array(lcovs)

        if len(smallOnes) > 0:
            smind   = _N.array(smallOnes)
            sngls   = mks[smind].reshape((smind.shape[0], 1, pmdim))
            rctrs   = ctrs.reshape((1, len(largeClstrs), pmdim))

            dmu     = (sngls - rctrs)
            iSg     = _N.linalg.inv(covs)
            qdrMKS  = _N.empty((len(largeClstrs), smind.shape[0]))
            _N.einsum("nmj,mjk,nmk->mn", dmu, iSg, dmu, out=qdrMKS)

            #  for each column, find the row with the smallest distance.  
            #  largeClstrs[row] is the cluster # that data point should be assigned to
            clst_clstr, nrn_nmb = _N.where(_N.min(qdrMKS, axis=0) == qdrMKS)

            for ii in xrange(len(nrn_nmb)):   # nrn_nmb out of order, [0...len(nrn_nmb)]
                #print "for neuron %(1)d   closest cluster is %(2)d" % {"1" : smallOnes[nrn_nmb[ii]], "2" : clst_clstr[ii]}
                labs[smallOnes[nrn_nmb[ii]]] = largeClstrs[clst_clstr[ii]]
            contiguous_pack2(labs)
            clstrSzs[hh] = len(_N.unique(labs))

def smallclustersAreNoise(nhmks, hmks, slabs, hlabs, pmdim, clstrSzs):
    """
    slabs, hlabs are contiguous labels (both start from 0) of hash and nonhash
    """
    #  EM init sometimes creates very small clusters.  Merge these with the 
    #  larger clusters EM finds

    for hh in xrange(2):
        if hh == 0:
            labs = slabs
            mks  = nhmks
        else:
            labs = hlabs
            mks  = hmks
        smallOnes = []
        largeClstrs = []
        ulabs    = _N.unique(labs)   # labs contiguous
        lctrs      = []
        lcovs      = []

        for i in xrange(len(ulabs)):  # how many "neurons" did we find?
            sts = _N.where(labs == i)[0]
            sz  = len(sts)
            #print "for %(i)d   there are %(n)d" % {"i" : i, "n" : sz}
            if sz <= pmdim:
                smallOnes.extend(sts)
            else:
                largeClstrs.append(i)
                lctrs.append(_N.mean(mks[sts], axis=0))
                lcovs.append(_N.cov(mks[sts], rowvar=0))
        ctrs = _N.array(lctrs)
        covs = _N.array(lcovs)

        if len(smallOnes) > 0:
            smind   = _N.array(smallOnes)
            sngls   = mks[smind].reshape((smind.shape[0], 1, pmdim))
            rctrs   = ctrs.reshape((1, len(largeClstrs), pmdim))

            dmu     = (sngls - rctrs)
            iSg     = _N.linalg.inv(covs)
            qdrMKS  = _N.empty((len(largeClstrs), smind.shape[0]))
            _N.einsum("nmj,mjk,nmk->mn", dmu, iSg, dmu, out=qdrMKS)

            #  for each column, find the row with the smallest distance.  
            #  largeClstrs[row] is the cluster # that data point should be assigned to
            clst_clstr, nrn_nmb = _N.where(_N.min(qdrMKS, axis=0) == qdrMKS)

            for ii in xrange(len(nrn_nmb)):   # nrn_nmb out of order, [0...len(nrn_nmb)]
                #print "for neuron %(1)d   closest cluster is %(2)d" % {"1" : smallOnes[nrn_nmb[ii]], "2" : clst_clstr[ii]}
                labs[smallOnes[nrn_nmb[ii]]] = largeClstrs[clst_clstr[ii]]
            contiguous_pack2(labs)
            clstrSzs[hh] = len(_N.unique(labs))

def emMKPOS_sep1A(nhmks, hmks, TR=5, minK=2, maxK=15):
    TR = 2
    minK=2
    maxK=8

    iNH = -1
    sNH = ["nh", "h"]

    bestLabs = []

    minSz = 8
    startClstrs = _N.empty(2, dtype=_N.int)
    iRMv_ME   = 0
    for mks in [nhmks, hmks]:
        iNH += 1
        startCl = 0 
        bestLabMP = None
        if mks is not None:
            labs, bics, bestLab, nClstrs = _oT.EMwfBICs(mks, minK=minK, maxK=maxK, TR=TR)
            bestLab = _N.array(bestLab, dtype=_N.int)

            #    if mks == nhmks:

            ##  non-hash, do spatial clustering

            bestLabMP = _N.array(bestLab)
            minK = 1
            _maxK = 7
            
            #for i in _N.unique(bestLab):  # how many "neurons" did we find?
            #    print "for %(i)d   there are %(n)d" % {"i" : i, "n" : len(_N.where(bestLab == i)[0])}
            print "neurons for iNH=%(nh)d   nclusters %(nc)d" % {"nh" : iNH, "nc" : nClstrs}
            for nc in xrange(nClstrs):
                inThisClstr = _N.where(bestLab == nc)[0]
                LiTC        = len(inThisClstr)
                pbestLab = _N.ones(LiTC, dtype=_N.int) * -1   #  poslabs
                pos = mks[inThisClstr, 0]
                _N.savetxt("clstr%(nh)d_%(iRM)d" % {"nh" : iNH, "iRM" : iRMv_ME}, mks[inThisClstr], fmt="%.4f %4f %.4f %4f %.4f")
                iRMv_ME += 1
                pos = pos.reshape(LiTC, 1)

                #  1 spk / clstr maxK == LiTC   want a few more than 1 / clstr
                #  maxK is at most _maxK unless LiTC
                maxK = _maxK if LiTC > _maxK else LiTC-1

                plabs, pbics, pbestLab, pClstrs = _oT.EMposBICs(pos, minK=minK, maxK=maxK, TR=2)
                pClstrs = contiguous_pack2(pbestLab)

                pbestLab[_N.where(pbestLab >= 0)[0]] += startCl  #  only the ones used for init
                startCl += pClstrs
                bestLabMP[inThisClstr] = pbestLab
        startClstrs[iNH] = startCl
        if bestLabMP is None:
            bestLabs.append(_N.array([], dtype=_N.int))
        else:
            bestLabs.append(_N.array(bestLabMP))

    return bestLabs[0], bestLabs[1], startClstrs


def emMKPOS_sep2(nhmks, hmks, TR=5, minK=2, maxK=15):
    """
    EM for wf cluster, heuristic density for position
    """
    iNH = -1
    sNH = ["nh", "h"]

    bestLabs = []

    minSz = 8
    startClstrs = _N.empty(2, dtype=_N.int)

    sd   = 0.01
    isd2 = 1./(sd**2)
    #xp   = _N.linspace(-6, 6, 241)
    xp   = _N.linspace(0, 3, 61)
    rxp  = xp.reshape(1, xp.shape[0])

    for mks in [nhmks, hmks]:
        if mks is not None:
            iNH += 1
            if iNH == 0:  # spatial clusters
                gk = gauKer(2)  #  0.05*2   = 0.1
                gk /= _N.sum(gk)
            if iNH == 1:
                gk = gauKer(20)  #  0.05*20   = 1
                gk /= _N.sum(gk)


            labs, bics, bestLab, nClstrs = _oT.EMwfBICs(mks, minK=minK, maxK=maxK, TR=TR)
            bestLab = _N.array(labs[nClstrs-minK, 0], dtype=_N.int)


            #    if mks == nhmks:

            ##  non-hash, do spatial clustering

            startCl = 0 
            bestLabMP = _N.array(bestLab)

            print "neurons for iNH=%(nh)d   %(nc)d" % {"nh" : iNH, "nc" : nClstrs}
            for nc in xrange(nClstrs):
                inThisClstr = _N.where(bestLab == nc)[0]
                pbestLab = _N.ones(len(inThisClstr), dtype=_N.int) * -1   #  poslabs
                pos = mks[inThisClstr, 0]
                #    if mks == nhmks:

                ##  non-hash, do spatial clustering
                rpos = pos.reshape(pos.shape[0], 1)

                dens = _N.sum(_N.exp(-0.5*isd2*(rpos-rxp)**2), axis=0)

                fdens = _N.convolve(dens, gk, mode="same")
                dfdens= _N.diff(fdens)
                brdrs = _N.where((dfdens[0:-1] < 0) & (dfdens[1:] > 0))[0]

                lbrdrs = []
                if brdrs[0] > 0:
                    lbrdrs.append(0)
                lbrdrs.extend(brdrs)
                #if brdrs[-1] < 240:
                #    lbrdrs.append(240)
                if brdrs[-1] < 60:
                    lbrdrs.append(60)

                M = len(lbrdrs) - 1
                inds = _N.empty(len(pos), dtype=_N.int)

                for m in xrange(len(lbrdrs) - 1):
                    these = _N.where((pos >= xp[lbrdrs[m]]) & (pos <= xp[lbrdrs[m+1]]))[0]
                    inds[these] = m

                #  if a cluster is very small and isolated, keep it.  If small but very wide, put it with another cluster

                for m in xrange(len(lbrdrs) - 1):
                    these = _N.where((pos >= xp[lbrdrs[m]]) & (pos <= xp[lbrdrs[m+1]]))[0]
                    inds[these] = m

                inds += startCl  #  only the ones used for init
                startCl += len(lbrdrs)-1
                bestLabMP[inThisClstr] = inds
            startClstrs[iNH] = startCl

            bestLabs.append(_N.array(bestLabMP))
        else:
            bestLabs.append(None)
    return bestLabs[0], bestLabs[1], startClstrs


def emMKPOS_sep3(nhmks, hmks, TR=5, minK=2, maxK=15):
    """
    EM for wf cluster, heuristic density for position
    """
    TR = 2
    minK=5
    maxK=15

    iNH = -1
    sNH = ["nh", "h"]

    k = 4
    bestLabs = []

    x   = _N.linspace(-6, 6, 481)
    startClstrs = _N.empty(2, dtype=_N.int)

    for mks in [nhmks, hmks]:
        startCl = 0 
        ThisNeuronI = 0

        iNH += 1
        labs, bics, bestLab, nClstrs = _oT.EMwfBICs(mks, minK=minK, maxK=maxK, TR=TR)
        bestLab = _N.array(labs[nClstrs-minK, 0], dtype=_N.int)

        allclrs = clrs.get_colors(3*nClstrs)

        for i in xrange(4):
            for j in xrange(i+1, 4):
                #fig = _plt.figure(figsize=(9, 11))

                #_plt.subplot2grid((8, 5), (0, 0), rowspan=3, colspan=3)
                for nc in xrange(nClstrs):
                    inThisClstr = _N.where(bestLab == nc)[0]
                    #_plt.scatter(mks[inThisClstr, i+1], mks[inThisClstr, j+1], s=5, color=allclrs[nc])
                    xAmp = _N.max(mks[:, i+1]) - _N.min(mks[:, i+1])
                    yAmp = _N.max(mks[:, j+1]) - _N.min(mks[:, j+1])
                    xmin = _N.min(mks[:, i+1]);     xmax = _N.max(mks[:, i+1])
                    ymin = _N.min(mks[:, j+1]);     ymax = _N.max(mks[:, j+1])
                    #_plt.xlim(xmin-xAmp*0.05, xmax+xAmp*0.05)
                    #_plt.ylim(ymin-yAmp*0.05, ymax+yAmp*0.05)
                #fig.subplots_adjust(left=0.08, bottom=0.02, top=0.92, right=0.92)
                #_plt.savefig("aaa_wf_clus_%(nh)s_%(ch1)d,%(ch2)d" % {"nh" : sNH[iNH], "ch1" : i, "ch2" : j})

        ##  non-hash, do spatial clustering

        bestLabMP = _N.array(bestLab)
        minK = 1

        #fig = _plt.figure(figsize=(12, 7))        

        sd   = 0.2 if (iNH == 0) else 1.5
        isd2 = 1./(sd**2)
        xp   = _N.linspace(-6, 6, 241)
        rxp  = xp.reshape(1, xp.shape[0])

        for nc in xrange(nClstrs):   #  each one "spike sorted"
            inThisClstr = _N.where(bestLab == nc)[0]
            #pbestLab = _N.ones(len(inThisClstr), dtype=_N.int) * -1   #  poslabs
            pos = mks[inThisClstr, 0]
            pos = pos.reshape(len(inThisClstr), 1)

            rpos = pos.reshape(pos.shape[0], 1)
            fdens    = _N.sum(_N.exp(-0.5*isd2*(rpos-rxp)**2), axis=0)
            ###  Don't consider points that don't look like clusters

            dfdens= _N.diff(fdens)
            maxs   = _N.where((dfdens[0:-1] > 0) & (dfdens[1:] < 0))[0]
            mins   = _N.where((dfdens[0:-1] < 0) & (dfdens[1:] > 0))[0]

            #  fdens[maxs+1], fdens[mins+1]   -  location of maxima

            cvFDENS = (_N.std(fdens) / _N.mean(fdens))
            IMI = (_N.std(_N.diff(maxs)) / _N.mean(_N.diff(maxs)))

            #  cvFDENS < 0.7 and IMI < 0.4  and len(maxs) > 8  throw out 

            rmList = []

            if (cvFDENS < 0.7) and (IMI < 0.4) and (len(maxs) > 7):
                #  flat, nearly uniform.  Just assign 1 cluster to this
                rmList = _N.arange(len(pos))   
            else:
                for pk in maxs:
                    if fdens[pk] < 6:
                        lftMins = _N.where(mins < pk)[0]
                        rgtMins = _N.where(mins > pk)[0]
                        lft  = lftMins[-1]   if (len(lftMins) > 0) else -1
                        rgt  = rgtMins[0]    if (len(rgtMins) > 0) else -1
                        posL = xp[mins[lft]] if (lft >= 0) else -6
                        posR = xp[mins[rgt]] if (rgt >= 0) else 6
                        lfdens = fdens[lft]  if (lft >= 0) else 1
                        rgdens = fdens[rgt]  if (rgt >= 0) else 1

                        if ((lfdens < 3) and (rgdens < 3)):  # ISOLATED PKS
                            #  ignore points from lft to rgt
                            ths = _N.where((pos > posL) & (pos < posR))[0]
                            rmList.extend(ths)
                        if ((lfdens > 4*fdens[pk]) or (rgdens > 4*fdens[pk])):
                            #  NEAR much taller peak
                            ths = _N.where((pos > posL) & (pos < posR))[0]
                            rmList.extend(ths)

                iclpos = _N.setdiff1d(_N.arange(len(pos)), _N.array(rmList))
                clpos  = pos[iclpos]
                rclpos = clpos.reshape(clpos.shape[0], 1)
                fcldens    = _N.sum(_N.exp(-0.5*isd2*(rclpos-rxp)**2), axis=0)

                #  AFTER initial pass, still have too many peaks
                dfcldens= _N.diff(fcldens)
                maxs   = _N.where((dfcldens[0:-1] > 0) & (dfcldens[1:] < 0))[0]
                mins   = _N.where((dfcldens[0:-1] < 0) & (dfcldens[1:] > 0))[0]

                if len(maxs) > 7:
                    #  just kill off the weaker peaks
                    minfcl   = _N.min(fcldens)
                    dynrange = _N.max(fcldens) - minfcl
                    for pk in maxs:
                        if fdens[pk] < minfcl + 0.3*dynrange:
                            lftMins = _N.where(mins < pk)[0]
                            rgtMins = _N.where(mins > pk)[0]
                            lft  = lftMins[-1]   if (len(lftMins) > 0) else -1
                            rgt  = rgtMins[0]    if (len(rgtMins) > 0) else -1
                            posL = xp[mins[lft]] if (lft >= 0) else -6
                            posR = xp[mins[rgt]] if (rgt >= 0) else 6
                            lfdens = fdens[lft]  if (lft >= 0) else 1
                            rgdens = fdens[rgt]  if (rgt >= 0) else 1

                              #  ignore points from lft to rgt
                            ths = _N.where((pos > posL) & (pos < posR))[0]
                            rmList.extend(ths)

                iclpos = _N.setdiff1d(_N.arange(len(pos)), _N.array(rmList))
                clpos  = pos[iclpos]
                rclpos = clpos.reshape(clpos.shape[0], 1)
                fcldens    = _N.sum(_N.exp(-0.5*isd2*(rclpos-rxp)**2), axis=0)

            if len(rmList) == len(inThisClstr):
                M = 1
                inds = _N.zeros(len(pos), dtype=_N.int)
            else:
                dfdens= _N.diff(fcldens)
                brdrs = _N.where((dfdens[0:-1] < 0) & (dfdens[1:] > 0))[0]

                if len(brdrs) == 0:
                    M = 1
                    inds = _N.zeros(len(pos), dtype=_N.int)
                else:
                    lbrdrs = []
                    if brdrs[0] > 0:
                        lbrdrs.append(0)
                    lbrdrs.extend(brdrs)
                    if brdrs[-1] < 240:
                        lbrdrs.append(240)

                    M = len(lbrdrs) - 1
                    inds = _N.empty(len(pos), dtype=_N.int)

                    for m in xrange(len(lbrdrs) - 1):
                        these = _N.where((pos >= xp[lbrdrs[m]]) & (pos <= xp[lbrdrs[m+1]]))[0]
                        inds[these]   = m

                    if len(rmList) > 0:   #  treat separately for now
                        M += 1
                        inds[rmList]  = M - 1

                #  if a cluster is very small and isolated, keep it.  If small but very wide, put it with another cluster

            pClstrs = M
            #fig.add_subplot(5, 3, nc+1)
            for pc in xrange(pClstrs):
                posClstr = _N.where(inds == pc)[0]
                #_plt.hist(mks[inThisClstr[posClstr], 0], bins=_N.linspace(-6, 6, 61), color=allclrs[pc], edgecolor=allclrs[pc])
                #_plt.xticks([-5, 0, 5])
                #_plt.yticks([])
            #_plt.title("%(1)d:%(2)d" % {"1" : ThisNeuronI, "2" : (ThisNeuronI + pClstrs)})
            ThisNeuronI += pClstrs
            #_plt.xlim(-6, 6)

            inds[_N.where(inds >= 0)[0]] += startCl  #  only the ones used for init
            startCl += pClstrs
            bestLabMP[inThisClstr] = inds
            #print bestLabMP[inThisClstr]
        #_plt.savefig("posClstrs")

        startClstrs[iNH] = startCl
        bestLabs.append(_N.array(bestLabMP))
    return bestLabs[0], bestLabs[1], startClstrs

def sepHashEM(mks):
    bics = _N.empty(2);    gmm = None
    
    #  If hash in ALL channels, consider it hash
    hashes = []

    gmms = []
    if mks.shape[0] > 500:
        for sh in xrange(1, mks.shape[1]):   # tetrode channel
            rmks = mks[:, sh].reshape(mks.shape[0], 1)
            for ncmp in [1, 2]:
                gmm = mixture.GMM(n_components=ncmp)
                gmm.fit(rmks)
                bics[ncmp-1] = gmm.bic(rmks)
                gmms.append(gmm)
            if bics[1] > bics[0]: # double
                ncmps = 1
            else:
                ncmps = 2

            if ncmps > 1:  #  is this 2 component meaningful?
                gmm = mixture.GMM(n_components=ncmp)
                gmm.fit(rmks)

                labs = gmm.predict(rmks)
                l0 = _N.where(labs == 0)[0]
                l1 = _N.where(labs == 1)[0]
                #print "len l0 %(0)d   l1 %(1)d" % {"0" : len(l0), "1" : len(l1)}
                hI = l1 if _N.mean(mks[l1, sh]) < _N.mean(mks[l0, sh]) else l0
                nhI = l0 if _N.mean(mks[l1, sh]) < _N.mean(mks[l0, sh]) else l1
                if (float(len(nhI)) / float(len(hI)) < 0.333):
                    hashes.append(hI)

        if len(hashes) == 0:#< (mks.shape[1] - 1):    
            # no hashes
            return _N.array([]), _N.arange(mks.shape[0]), None
        else:   #  
            if len(hashes) == 1:    #  1 channel w/2 cmp. The rest 1 cmp.
               h = hashes[0]
            else:  #  multiple channels w/2 cmp.  must be hash in all channels
                h = _N.intersect1d(hashes[0], hashes[1])
                for i in xrange(2, len(hashes)):
                    h = _N.intersect1d(h, hashes[i])
                
            hsh    = h
            nonh    = _N.setdiff1d(_N.arange(mks.shape[0]), hsh)
            # fig = _plt.figure(figsize=(8, 10))
            # for sh in xrange(4):
            #     fig.add_subplot(3, 2, sh+1)

            #     _plt.scatter(mks[hsh, 0], mks[hsh, sh+1], s=5, color="black")
            #     _plt.scatter(mks[nonh, 0], mks[nonh, sh+1], s=5, color="red")
            #     _plt.xlim(-6, 6)
            # fig.add_subplot(3, 2, 5)
            # _plt.hist(mks[hsh, 0], bins=50)
            # fig.add_subplot(3, 2, 6)
            # _plt.hist(mks[nonh, 0], bins=50)


            #_plt.savefig("sephash_%s" % tetlist[it])
            #_plt.close()
            return nonh, hsh, gmms
    else:
        return _N.array([]), _N.arange(mks.shape[0]), None



def splitclstrs(posmk, labS):
    #  for each cluster, see if its better as 1 or 2 cluster
    unIDs = _N.unique(labS)
    nClstrs = len(unIDs)

    startID = 0

    for uID in unIDs:
        these = _N.where(labS == uID)[0]
        if len(these) > 8:
            plabs, pbics, pbestLab, pClstrs = _oT.EMposBICs(posmk[these, 0].reshape(len(these), 1), minK=1, maxK=3, TR=1)
            if pClstrs > 1:
                print "bigger"
                lrgr = _N.where(labS > uID)[0]   #  clusters with larger IDs
                labS[lrgr] += 1
                labS[these] = pbestLab + uID
