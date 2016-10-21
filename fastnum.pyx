#  do multiple quadratics
cimport cython
from cython.parallel import parallel, prange
from libc.stdlib cimport abort, malloc, free
from libc.math cimport exp
import numpy as _N
cimport numpy as _N

##########
##########
@cython.boundscheck(False)
@cython.wraparound(False)
def multi_qdrtcs(double[:, :, ::1] v, double[:, :, ::1] iSg, double[:, ::1] qdr, int M, int N, int k):
    #  fxs       M x fss   
    #  fxrux     Nupx    
    #  f_intgrd  Nupx
    cdef int m, n, i, j, mNk, mkk, nk, ik

    cdef double *p_v   = &v[0, 0, 0]
    cdef double *p_iSg = &iSg[0, 0, 0]
    cdef double *p_qdr = &qdr[0, 0]
    cdef double lv          

    for m in range(M):
        mNk = m*N*k
        mkk = m*k*k
        for n in range(N):
            nk = n*k
            for i in range(k):
                ik = i*k
                for j in range(k):            
                    p_qdr[m*N + n] += p_v[mNk + nk + i] * p_iSg[mkk + ik + j] * p_v[mNk + nk + j]
                                     
@cython.boundscheck(False)
@cython.wraparound(False)
def multi_qdrtcs_par(double[:, :, ::1] v, double[:, :, ::1] iSg, double[:, ::1] qdr, int M, int N, int k, int nthrds=4):
    #  fxs       M x fss   
    #  fxrux     Nupx    
    #  f_intgrd  Nupx
    cdef int m, n, i, j, mNk, mkk, nk, ik

    cdef double *p_v   = &v[0, 0, 0]
    cdef double *p_iSg = &iSg[0, 0, 0]
    cdef double *p_qdr = &qdr[0, 0]
    cdef double lv          

    with nogil, parallel(num_threads=nthrds):
        for m in prange(M):
            mNk = m*N*k
            mkk = m*k*k
            for n in range(N):
                nk = n*k
                for i in range(k):
                    ik = i*k
                    for j in range(k):            
                        p_qdr[m*N + n] += p_v[mNk + nk + i] * p_iSg[mkk + ik + j] * p_v[mNk + nk + j]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void inner_loop(double *p_v, double *p_iSg, double *p_qdr, int N, int k, int mN, int mNk, int mkk) nogil:
    cdef int n, i, j, nk, ik, mNk_nk, mN_n, mkk_ik
    for n in range(N):
        nk = n*k
        mNk_nk = mNk + nk
        mN_n   = mN + n
        for i in range(k):
            ik = i*k
            mkk_ik = mkk + ik
            for j in range(k):            
                p_qdr[mN_n] += p_v[mNk_nk + i] * p_iSg[mkk_ik + j] * p_v[mNk_nk + j]
                                     


@cython.boundscheck(False)
@cython.wraparound(False)
def multi_qdrtcs_par_func(double[:, :, ::1] v, double[:, :, ::1] iSg, double[:, ::1] qdr, int M, int N, int k, int nthrds=4):
    #  fxs       M x fss   
    #  fxrux     Nupx    
    #  f_intgrd  Nupx
    cdef int m, mNk, mkk, nk, ik

    cdef double *p_v   = &v[0, 0, 0]
    cdef double *p_iSg = &iSg[0, 0, 0]
    cdef double *p_qdr = &qdr[0, 0]

    with nogil, parallel(num_threads=nthrds):
        for m in prange(M):
            inner_loop(p_v, p_iSg, p_qdr, N, k, m*N, m*N*k, m*k*k)

