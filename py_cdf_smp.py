import numpy as _N

def smp_from_cdf(sg2s, sLLkPr, s, d_sg2s, sg2s_m1):
    """
    xt0t1    relative coordinates
    mks      absolute coordinates

    sLLkPr   spiking part
    s        silence part
    """
    sat = sLLkPr + s
    sat -= _N.max(sat, axis=0)
    pnn = _N.exp(sat)   # p not normalized

    # ###  does well when a is large
    N, M = pnn.shape
    cdf   = _N.zeros((N, M))

    for i in xrange(1, N):  #  over 
        #print cdf[i].shape
        #print pnn[:, i].shape
        cdf[i] = cdf[i-1] + pnn[i]*d_sg2s[i-1]
    cdf /= cdf[-2]     #  even if U[0,1] rand is 1, we still have some room at the end to add a bit of noise.

    rnds = _N.random.rand(M, 2)
    retRnd= _N.empty(M)

    for m in xrange(M):
        isg2 = _N.searchsorted(cdf[:, m], rnds[m, 0])
        retRnd[m] = sg2s[isg2] #+ rnds[m, 1]*d_sg2s[isg2]
    return retRnd

    #_plt.plot(sg2s[1:, 0], cdf[:, 0])


def smp_from_cdf_interp(sg2s, sLLkPr, s, d_sg2s, sg2s_m1):
    """
    xt0t1    relative coordinates
    mks      absolute coordinates

    sLLkPr   spiking part
    s        silence part
    """
    sat = sLLkPr + s
    sat -= _N.max(sat, axis=0)
    pnn = _N.exp(sat)   # p not normalized

    # ###  does well when a is large
    N, M = pnn.shape
    cdf   = _N.zeros((N, M))

    for i in xrange(1, N):  #  over 
        #print cdf[i].shape
        #print pnn[:, i].shape
        cdf[i] = cdf[i-1] + pnn[i]*d_sg2s[i-1]
    cdf /= cdf[-2]     #  even if U[0,1] rand is 1, we still have some room at the end to add a bit of noise.

    rnds = _N.random.rand(M, 2)
    retRnd= _N.empty(M)

    for m in xrange(M):
        #  btwn cdf[isg2] and cdf[isg2+1]
        #  (rnds[m,0] - cdf[isg2]) * (cdf[isg2+1] - cdf[isg2]) * d_sg2s[isg2]
        isg2 = _N.searchsorted(cdf[:, m], rnds[m, 0])
        #retRnd[m] = sg2s[isg2] #+ rnds[m, 1]*d_sg2s[isg2]
        retRnd[m] = sg2s[isg2] + ((rnds[m,0] - cdf[isg2,m]) / (cdf[isg2+1,m] - cdf[isg2,m])) * d_sg2s[isg2]
        if retRnd[m] < 0:
            """
            Why does this happen sometimes?
            """
            retRnd[m] = sg2s[isg2]
            print "retRnd[m] < 0:  %(1).3e  %(2).3e   %(3).3e  %(4).3e     rat %(5).3e" % {"1" : sg2s[isg2], "2" : rnds[m,0], "3" : cdf[isg2,m], "4" : cdf[isg2+1,m], "5" : ((rnds[m,0] - cdf[isg2,m]) / (cdf[isg2+1,m] - cdf[isg2,m]))}
    return retRnd

    #_plt.plot(sg2s[1:, 0], cdf[:, 0])


