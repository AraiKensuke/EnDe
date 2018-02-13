import numpy as _N
##  gz   = _N.zeros((ITERS, nSpks, M), dtype=_N.bool)
##  cgz  = _N.empty((ITERS, nSpks), dtype=_N.uint8)   # 8 bit int

def gz2cgz(gz):
    ITERS = gz.shape[0]
    nSpks = gz.shape[1]
    M     = gz.shape[2]

    cgz = _N.empty((ITERS, nSpks), dtype=_N.uint8)
    for it in xrange(ITERS):
        nIDs, clstrIDs = _N.where(gz[it])
        cgz[it, nIDs] = clstrIDs
    return cgz

def cgz2gz(cgz, M):
    ITERS = cgz.shape[0]
    nSpks = cgz.shape[1]

    gz = _N.zeros((ITERS, nSpks, M), dtype=_N.uint8)
    for it in xrange(ITERS):
        for n in xrange(nSpks):
            gz[it, n, cgz[it, n]] = 1
    return gz

    
