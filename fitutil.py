import numpy as _N
import scipy.cluster.vq as scv
import matplotlib.pyplot as _plt

myclrs = [ "#FF0000", "#0000FF", "#00FFFF", 
           "#888888", "#FF8888", "#8888FF", "#88FF88", 
           "#990000", "#000099", "#009900", 
           "#998888", "#888899", "#889988",
           "#FF00FF", "#FFAAFF",
           "#000000", "#FF0000", "#0000FF", "#00FF00", 
           "#888888", "#FF8888", "#8888FF", "#88FF88", 
           "#990000", "#000099", "#009900", 
           "#998888", "#888899", "#889988",
           "#FF00FF", "#FFAAFF"]

def bestcluster(ITERS, smkpos, MS):
    """
    MS  # of clusters
    """
    ICR         = _N.zeros(ITERS)
    minOfMaxICR = _N.zeros(ITERS)
    maxOfMaxICR = _N.zeros(ITERS)
    #InterCR     = _N.zeros(ITERS)

    allCtrs = []
    allLabs = []

    for it in xrange(ITERS):
        ctrs, labs = scv.kmeans2(smkpos, MS)  #  big clear
        intraClusRadii = []
        maxIntraClusRadii = []
        tot  = 0
        #interClusRadii = 0

        for m in xrange(MS):
            #for m1 in xrange(m+1, MS):
            #    interClusRadii += _N.sum((ctrs[m] - ctrs[m1])**2)
                
            inClus = _N.where(labs == m)[0]

            if len(inClus) > 3:
                radii = _N.sqrt(_N.sum((smkpos[inClus] - ctrs[m])**2, axis=1))
                intraClusRadii.append(_N.sum(radii))
                maxIntraClusRadii.append(_N.max(radii))
                tot += len(inClus)
        ICR[it]    = _N.sum(intraClusRadii)/ tot
        minOfMaxICR[it] = _N.min(maxIntraClusRadii)
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
    
    z13 = z1 + z3
    bestlabs = _N.where(z13 == _N.min(z13))[0]

    return allLabs[bestlabs[0]]


def colorclusters(smkpos, labs, MS):
    fig = _plt.figure(figsize=(12, 8))

    for m in xrange(MS):
        inds = _N.where(labs == m)[0]
        for k in xrange(4):
            fig.add_subplot(2, 2, k+1)
            _plt.scatter(smkpos[inds, 0], smkpos[inds, k+1], color=myclrs[m], s=3)
