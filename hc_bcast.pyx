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



@cython.boundscheck(False)
@cython.wraparound(False)
def hc_qdr_sum(double[:, ::1] pkFRr, double [:, ::1] mkNrms, double[:, ::1] qdrSpc, double [:, ::1] qdrMKS, double [:, ::1] cont, int M, int N):
    #  fxs       M x fss   
    #  fxrux     Nupx    
    #  f_intgrd  Nupx
    cdef int m, n, mN

    cdef double *p_qdrSpc   = &qdrSpc[0, 0]
    cdef double *p_pkFRr       = &pkFRr[0, 0]
    cdef double *p_mkNrms     = &mkNrms[0, 0]
    cdef double *p_qdrMKS     = &qdrMKS[0, 0]
    cdef double *p_cont       = &cont[0, 0]
    cdef double pkFRr_m, mkNrms_m

    for 0 <= m < M:
        mN = m*N
        pkFRr_m = p_pkFRr[m]
        mkNrms_m = p_mkNrms[m]
        for 0 <= n < N:
            p_cont[mN+n] = pkFRr_m + mkNrms_m - 0.5*(p_qdrSpc[mN+n] + p_qdrMKS[mN+n])
