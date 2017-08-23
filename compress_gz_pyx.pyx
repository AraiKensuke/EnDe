import numpy as _N
cimport cython

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

@cython.boundscheck(False)
@cython.wraparound(False)
def cgz2gz(unsigned char[:, ::1] cgz, long M):
    cdef long ITERS = cgz.shape[0]
    cdef long nSpks = cgz.shape[1]

    gz = _N.zeros((ITERS, nSpks, M), dtype=_N.uint8)
    cdef unsigned char[:, :, ::1] gz_mv= gz
    cdef unsigned char* p_gz = &gz_mv[0, 0, 0]
    cdef unsigned char* p_cgz = &cgz[0, 0]

    cdef long it_nSpks_M, it_nSpks, it, n
    
    with nogil:
        for 0 <= it < ITERS:
            it_nSpks   = it * nSpks
            it_nSpks_M = it_nSpks * M

            for 0 <= n < nSpks:
                p_gz[it_nSpks_M + n*M + p_cgz[it_nSpks + n]] = 1

    return _N.array(gz, dtype=_N.bool)

    
