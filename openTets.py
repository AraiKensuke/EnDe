import fitutil as _fu
import cPickle as _pkl
import numpy as _N
from sklearn import mixture
import EnDedirs as _edd

_all_, _motion_, _rest_ = 0, 1, 2

def openTets(fn, tets, t0=None, t1=None, lm=None, detectHash=True, chooseSpikes=_motion_):
    """
    Open a file with marks and positions
    stets   if I give 
    """
    if lm is None:
        with open(fn, "rb") as f:
            lm = _pkl.load(f)
        f.close()

    if type(tets) is int:
        tets = [tets]
    elif type(tets[0]) is str:
        inds = []
        for i in xrange(len(tets)):
            inds.append(lm.tetlist.index(tets[i]))
        tets = inds

    t0 = 0 if (t0 is None) else t0
    t1 = lm.marks.shape[0] if (t1 is None) else t1

    if chooseSpikes == _motion_:
        minds = lm.minds
    elif chooseSpikes == _all_:
        minds = _N.array([[t0, t1]])
    else:
        lminds = []
        if lm.minds[0, 0] > 0:
            lminds.append([0, lm.minds[0, 0]])
        for i in xrange(lm.minds.shape[0] - 1):
            lminds.append([lm.minds[i, 1], lm.minds[i+1, 0]])
        if lm.minds[-1, 1] > 0:
            lminds.append([lm.minds[-1, 1], t1])
        minds = _N.array(lminds)

    inds = _N.where((minds[:, 0] >= t0) & (minds[:, 0] <=  t1))[0]

    if detectHash:
        allnhmks = []
        allhmks  = []
    else:
        allmarks = []

    for tet in tets:
        marks = []
        pos   = []
        for t in inds:
            mksl = lm.marks[minds[t, 0]:minds[t, 1], tet]
            shsl = lm.pos[minds[t, 0]:minds[t, 1]]
            nn = _N.where(_N.equal(mksl, None) == False)[0]

            for n in nn:
                for l in xrange(len(mksl[n])):
                    marks.append(mksl[n][l])
                    pos.append(shsl[n])

        posmarks = _N.empty((len(marks), 5))
        posmarks[:, 1:] = _N.array(marks)
        posmarks[:, 0]  = _N.array(pos)

        if detectHash:
            #  separate the hash
            nhid, hid, gmms = _fu.sepHashEM(posmarks)

            nhmks     = posmarks[nhid]
            hmks      = posmarks[hid]

            allnhmks.append(nhmks)
            allhmks.append(hmks)
        else:
            allmarks.append(posmarks)

    if detectHash:
        return allnhmks, allhmks, lm, gmms
    else:
        return allmarks, lm

def EMwfBICs(mks, TR=5, minK=2, maxK=15):
    """
    """
    bics = _N.empty(((maxK-minK), TR))
    labs = _N.empty((maxK-minK, TR, mks.shape[0]))

    for K in xrange(minK, maxK):
        for tr in xrange(TR):
            gmm = mixture.GMM(n_components=K, covariance_type="full")
            
            gmm.fit(mks[:, 1:])
            bics[K-minK, tr] = gmm.bic(mks[:, 1:])
            labs[K-minK, tr] = gmm.predict(mks[:, 1:])

    coords = _N.where(bics == _N.min(bics))
    bestLab = labs[coords[0][0], coords[1][0]]   #  indices in 2-D array

    nClstrs = coords[0][0] + minK   #  best # of clusters

    return labs, bics, bestLab, nClstrs

def EMBICs(mks, TR=5, minK=2, maxK=15):
    """
    """
    bics = _N.empty(((maxK-minK), TR))
    labs = _N.empty((maxK-minK, TR, mks.shape[0]))

    for K in xrange(minK, maxK):
        for tr in xrange(TR):
            gmm = mixture.GMM(n_components=K, covariance_type="full")
            
            gmm.fit(mks)
            bics[K-minK, tr] = gmm.bic(mks)
            labs[K-minK, tr] = gmm.predict(mks)

    coords = _N.where(bics == _N.min(bics))
    bestLab = labs[coords[0][0], coords[1][0]]

    nClstrs = coords[0][0] + minK

    return labs, bics, bestLab, nClstrs


def EMposBICs(pos, TR=5, minK=1, maxK=15):
    """
    """
    bics = _N.empty(((maxK-minK), TR))
    labs = _N.empty((maxK-minK, TR, pos.shape[0]), dtype=_N.int)

    for K in xrange(minK, maxK):
        for tr in xrange(TR):
            gmm = mixture.GMM(n_components=K)
            
            gmm.fit(pos)
            bics[K-minK, tr] = gmm.bic(pos)
            labs[K-minK, tr] = gmm.predict(pos)

    coords = _N.where(bics == _N.min(bics))
    bestLab = labs[coords[0][0], coords[1][0]]

    nClstrs = coords[0][0] + minK

    return labs, bics, bestLab, nClstrs


def justOneTet(fn, stet):
    with open(fn, "rb") as f:
        lm = _pkl.load(f)
    f.close()

    ind = lm.tetlist.index(stet)
    
    lm.tetlist = [stet]
    mks = _N.array(lm.marks[:, ind])
    mks = mks.reshape(mks.shape[0], 1)
    lm.marks = mks

    dmp = open(_edd.resFN("tetmarks_%s.pkl" % stet, dir="bond0402", create=True), "wb")
    _pkl.dump(lm, dmp, -1)
    dmp.close()

