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
            print nearEmptyBins
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

def findsmallclusters(nhmks, slabs, pmdim):
    """
    slabs, hlabs are contiguous labels (both start from 0) of hash and nonhash
    """
    #  EM init sometimes creates very small clusters.  Merge these with the 
    #  larger clusters EM finds

    labs = slabs
    mks  = nhmks
    smallClstrs = []
    s_smallClstrs = []    # spike indices
    largeClstrs = []
    ulabs    = _N.unique(labs)   # labs contiguous
    lctrs      = []
    lcovs      = []

    for i in xrange(len(ulabs)):  # how many "neurons" did we find?
        sts = _N.where(labs == i)[0]
        sz  = len(sts)
        #print "for %(i)d   there are %(n)d" % {"i" : i, "n" : sz}
        if sz <= pmdim:
            smallClstrs.append(i)
            s_smallClstrs.extend(sts)
        else:
            largeClstrs.append(i)
            lctrs.append(_N.mean(mks[sts], axis=0))
            lcovs.append(_N.cov(mks[sts], rowvar=0))
    ctrs = _N.array(lctrs)
    covs = _N.array(lcovs)

    #_plt.scatter(mks[_N.array(s_smallClstrs), 0], mks[_N.array(s_smallClstrs), 1])
    return smallClstrs, s_smallClstrs

def emMKPOS_sep1A(nhmks, hmks, TR=5, wfNClstrs=[[2, 8], [1, 4]], spNClstrs=[[1, 7], [1, 2]], K=4):
    #  wfNClstrs   [ non-hash(min, max), hash(min, max)  by waveform clustering
    TR = 2

    iNH = -1
    sNH = ["nh", "h"]

    bestLabs = []

    minSz = 5
    startClstrs = _N.empty(2, dtype=_N.int)
    iRMv_ME   = 0
    for mks in [nhmks, hmks]:
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
                    pos = mks[inThisClstr, 0]
                    #_N.savetxt("clstr%(nh)d_%(iRM)d" % {"nh" : iNH, "iRM" : iRMv_ME}, mks[inThisClstr], fmt="%.4f %4f %.4f %4f %.4f")
                    iRMv_ME += 1
                    pos = pos.reshape(LiTC, 1)

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

def emMKPOS_sep1B(nhmks, hmks, TR=5, wfNClstrs=[[2, 8], [1, 4]], spNClstrs=[[1, 7], [1, 2]]):
    #  wfNClstrs   [ non-hash(min, max), hash(min, max)  by waveform clustering
    TR = 2

    iNH = -1
    sNH = ["nh", "h"]

    bestLabs = []

    minSz = 5
    startClstrs = _N.empty(2, dtype=_N.int)
    iRMv_ME   = 0
    for mks in [nhmks, hmks]:
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

                print LiTC
                if LiTC > 2:   #  at least 2 spikes from this neuron
                    pbestLab = _N.ones(LiTC, dtype=_N.int) * -1   #  poslabs
                    pos = mks[inThisClstr, 0]
                    _N.savetxt("clstr%(nh)d_%(iRM)d" % {"nh" : iNH, "iRM" : iRMv_ME}, mks[inThisClstr], fmt="%.4f %4f %.4f %4f %.4f")
                    iRMv_ME += 1
                    posr = pos.reshape(1, LiTC)

                    #  1 spk / clstr maxK == LiTC   want a few more than 1 / clstr
                    #  at resolution of 0.02
                    bns = int(12/0.02)
                    xp = _N.linspace(-6, 6, bns + 1)
                    xpr = xp.reshape((xp.shape[0], 1))
                    sg2=0.05**2    #  resoultion x 5
                    densI = _N.exp(-0.5*(xpr-posr)*(xpr-posr)/sg2)
                    dens  = _N.sum(densI, axis=1)

                    #  we're summing things of height 1.  1, 2, 3  SDs 0.6, 0.14, 0.01, 0.000335
                    ddens = _N.diff(dens)
                    septx = _N.where(((ddens[0:-1] < 0) & (ddens[1:] >= 0)) & (dens[1:-1] < 3.35e-4))[0]

                    #  where are we at least 3 SDs from a lone spike

                    edges = _N.zeros(len(septx)+2, dtype=_N.int)
                    edges[1:-1] = septx
                    edges[-1]   = bns
                    #plabels = _N.ones(pos.shape[0], dtype=_N.int) * -1

                    for ib in xrange(len(edges)-1):
                        inrns = _N.where((pos >= xp[edges[ib]]) & (pos < xp[edges[ib+1]]))[0]
                        pbestLab[inrns] = ib
                    print pbestLab

                    pClstrs = len(edges)-1
                    print "pClstrs   %d" % pClstrs
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
                lrgr = _N.where(labS > uID)[0]   #  clusters with larger IDs
                labS[lrgr] += 1
                labS[these] = pbestLab + uID

def posMkCov0(posmk, labS, K=4):
    #  for each cluster, see if its better as 1 or 2 cluster
    unIDs = _N.unique(labS)
    nClstrs = len(unIDs)

    startID = 0

    for uID in unIDs:
        these = _N.where(labS == uID)[0]

        if len(these) > 8:
            pc1, pv1 = _ss.pearsonr(posmk[these, 0], posmk[these, 1])
            pc2, pv2 = _ss.pearsonr(posmk[these, 0], posmk[these, 2])
            if K >= 2:
                pc3, pv3 = _ss.pearsonr(posmk[these, 0], posmk[these, 3])
                pc4, pv4 = _ss.pearsonr(posmk[these, 0], posmk[these, 4])

            #if (pv1 < 0.01) or (pv2 < 0.01) or (pv3 < 0.01) or (pv4 < 0.01):
            #    print "uID %(id)d is significant:  %(pc1).3f  %(pc2).3f  %(pc3).3f  %(pc4).3f" % {"id" : uID, "pc1" : pc1, "pc2" : pc2, "pc3" : pc3, "pc4" : pc4}

