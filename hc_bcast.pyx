#  do multiple quadratics
cimport cython
from cython.parallel import parallel, prange
from libc.stdlib cimport abort, malloc, free
from libc.math cimport exp
import numpy as _N
cimport numpy as _N

@cython.boundscheck(False)
@cython.wraparound(False)
def hc_bcast1(double[:, ::1] fr, double [:, ::1] xASr, double[:, ::1] iq2r, double [:, ::1] qdrSPC, int M, int N):
    #  fxs       M x fss   
    #  fxrux     Nupx    
    #  f_intgrd  Nupx
    cdef int m, n, mN

    cdef double *p_qdrSPC   = &qdrSPC[0, 0]
    cdef double *p_fr       = &fr[0, 0]
    cdef double *p_xASr     = &xASr[0, 0]
    cdef double *p_iq2r     = &iq2r[0, 0]

    for 0 <= m < M:
        mN = m*N
        for 0 <= n < N:
            p_qdrSPC[mN+n] = (p_fr[m] - p_xASr[n])*(p_fr[m] - p_xASr[n])*p_iq2r[m]

@cython.boundscheck(False)
@cython.wraparound(False)
def hc_bcast1_par(double[:, ::1] fr, double [:, ::1] xASr, double[:, ::1] iq2r, double [:, ::1] qdrSPC, int M, int N, int nthrds=4):
    #  fxs       M x fss   
    #  fxrux     Nupx    
    #  f_intgrd  Nupx
    cdef int m, n, mN

    cdef double *p_qdrSPC   = &qdrSPC[0, 0]
    cdef double *p_fr       = &fr[0, 0]
    cdef double *p_xASr     = &xASr[0, 0]
    cdef double *p_iq2r     = &iq2r[0, 0]

    with nogil, parallel(num_threads=nthrds):
        for m in prange(M):
            mN = m*N
            for 0 <= n < N:
                p_qdrSPC[mN+n] = (p_fr[m] - p_xASr[n])*(p_fr[m] - p_xASr[n])*p_iq2r[m]
