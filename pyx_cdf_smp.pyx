import numpy as _N
cimport numpy as _N
cimport cython
import time as _tm

def smp_from_cdf_interp(sg2s, sLLkPr, s, d_sg2s, sg2s_m1):
    """
    sg2s     q2ss 
    sLLkPr   q2ss x M
    pnn      q2ss x M
    xt0t1    relative coordinates
    mks      absolute coordinates

    sLLkPr   spiking part
    s        silence part
    """
    sat = sLLkPr + s
    sat -= _N.max(sat, axis=0)
    pnn = _N.exp(sat)   # p not normalized

    cdef int N, M
    # ###  does well when a is large
    __N, __M = pnn.shape
    N = __N
    M = __M
    cdf   = _N.zeros((N, M))
    cdef double[:, ::1] pnnmv = pnn
    cdef double[:, ::1] cdfmv = cdf
    cdef double[::1] d_sg2smv = d_sg2s
    cdef double[::1] sg2smv   = sg2s
    cdef double *p_cdf        = &cdfmv[0, 0]
    cdef double *p_pnn        = &pnnmv[0, 0]
    cdef double *p_d_sg2s     = &d_sg2smv[0]
    cdef double *p_sg2s       = &sg2smv[0]
    
    
    for 1 <= i < N:
        cdf[i] = cdf[i-1] + pnn[i]*d_sg2s[i-1]
    cdf /= cdf[-2]     #  even if U[0,1] rand is 1, we still have some room at the end to add a bit of noise.

    rnds = _N.random.rand(M)
    retRnd= _N.empty(M)
    cdef double[::1] retRndmv = retRnd
    cdef double[::1] rndsmv   = rnds
    cdef double *p_retRnd        = &retRndmv[0]
    cdef double *p_rnds          = &rndsmv[0]

    cdef int isg2, _isg2, m

    for 0 <= m < M:
        #  btwn cdf[isg2] and cdf[isg2+1]
        #  (rnds[m,0] - cdf[isg2]) * (cdf[isg2+1] - cdf[isg2]) * d_sg2s[isg2]
        _isg2 = _N.searchsorted(cdf[:, m], rnds[m])
        isg2  = _isg2-1
        p_retRnd[m] = p_sg2s[isg2] + ((p_rnds[m] - p_cdf[isg2*M+m]) / (p_cdf[(isg2+1)*M+m] - p_cdf[isg2*M+m])) * p_d_sg2s[isg2]

    return retRnd
