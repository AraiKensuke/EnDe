#  do multiple quadratics
cimport cython
from cython.parallel import parallel, prange
from libc.stdlib cimport abort, malloc, free
from libc.math cimport exp
import numpy as _N
cimport numpy as _N

##########
##########
def multi_qdrtcs_simp(v, iSg, qdr, M, N, k):
    for m in xrange(M):
        for n in xrange(N):
            qdr[m, n] = 0
            for i in xrange(k):
                for j in xrange(k):            
                    qdr[m, n] += v[m, n, i] * iSg[m, i, j] * v[m, n, j]


@cython.boundscheck(False)
@cython.wraparound(False)
def multi_qdrtcs(double[:, :, ::1] v, double[:, :, ::1] iSg, double[:, ::1] qdr, int M, int N, int k):
    #  fxs       M x fss   
    #  fxrux     Nupx    
    #  f_intgrd  Nupx
    cdef int m, n, i, j, mNk, mkk, nk, ik, mNk_nk_i, mN_n

    cdef double *p_v   = &v[0, 0, 0]
    cdef double *p_iSg = &iSg[0, 0, 0]
    cdef double *p_qdr = &qdr[0, 0]
    cdef double lv          

    for m in xrange(M):
        mNk = m*N*k
        mkk = m*k*k
        for n in xrange(N):
            nk = n*k
            mN_n = m*N + n
            p_qdr[mN_n] = 0
            for i in xrange(k):
                ik = i*k
                mNk_nk_i = mNk + nk + i
                for j in xrange(k):            
                    p_qdr[mN_n] += p_v[mNk_nk_i] * p_iSg[mkk + ik + j] * p_v[mNk + nk + j]

                                     
#@cython.boundscheck(False)
#@cython.wraparound(False)
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
                p_qdr[m*N + n] = 0
                for i in range(k):
                    ik = i*k
                    for j in range(k):            
                        p_qdr[m*N + n] += p_v[mNk + nk + i] * p_iSg[mkk + ik + j] * p_v[mNk + nk + j]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void inner_loop(double *p_v, double *p_iSg, double *p_qdr, int N, int k, int mN, int mNk, int mkk) nogil:
    cdef int n, i, j, ik, mNk_nk, mN_n, mkk_ik
    cdef double tot
    for n in range(N):
        mNk_nk = mNk + n*k
        mN_n   = mN + n
        tot = 0
        p_qdr[mN_n] = 0
        for i in range(k):
            mkk_ik = mkk + i*k
            for j in range(k):            
                tot += p_v[mNk_nk + i] * p_iSg[mkk_ik + j] * p_v[mNk_nk + j]
        p_qdr[mN_n] = tot

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void inner_loop_sym(double *p_v, double *p_iSg, double *p_qdr, int N, int k, int mN, int mNk, int mkk) nogil:
    cdef int n, i, j, ik, mNk_nk, mN_n, mkk_ik
    cdef double p_v_mNk_nki
    cdef double tot
    for n in range(N):
        mNk_nk = mNk + n*k
        mN_n   = mN + n
        tot = 0

        for i in range(k):
            mkk_ik = mkk + i*k
            p_v_mNk_nki= p_v[mNk_nk + i]
            tot += p_v_mNk_nki * p_iSg[mkk_ik + i] * p_v_mNk_nki

            for j in range(i+1,k):            
                tot += 2*p_v_mNk_nki * p_iSg[mkk_ik + j] * p_v[mNk_nk + j]
        p_qdr[mN_n] = tot

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void inner_loop2(double *p_v, double *p_iSg, double *p_qdr, int N, int k, int mN, int mNk, int mkk) nogil:
    cdef int n, i, j, nk, ik, mNk_nk, mN_n, mkk_ik
    for 0 <= n < N:
        mNk_nk = mNk + n*k
        mN_n   = mN + n
        p_qdr[mN_n] = 0
        for 0 <= i < k:
            mkk_ik = mkk + i*k
            for 0 <= j < k:
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

@cython.boundscheck(False)
@cython.wraparound(False)
def multi_qdrtcs_par_func_sym(double[:, :, ::1] v, double[:, :, ::1] iSg, double[:, ::1] qdr, int M, int N, int k, int nthrds=4):
    #  fxs       M x fss   
    #  fxrux     Nupx    
    #  f_intgrd  Nupx
    cdef int m, mNk, mkk, nk, ik

    cdef double *p_v   = &v[0, 0, 0]
    cdef double *p_iSg = &iSg[0, 0, 0]
    cdef double *p_qdr = &qdr[0, 0]

    with nogil, parallel(num_threads=nthrds):
        for m in prange(M):
            inner_loop_sym(p_v, p_iSg, p_qdr, N, k, m*N, m*N*k, m*k*k)

@cython.boundscheck(False)
@cython.wraparound(False)
def multi_qdrtcs_par_func2(double[:, :, ::1] v, double[:, :, ::1] iSg, double[:, ::1] qdr, int M, int N, int k, int nthrds=4):
    #  fxs       M x fss   
    #  fxrux     Nupx    
    #  f_intgrd  Nupx
    cdef int m, mNk, mkk, nk, ik

    cdef double *p_v   = &v[0, 0, 0]
    cdef double *p_iSg = &iSg[0, 0, 0]
    cdef double *p_qdr = &qdr[0, 0]

    with nogil, parallel(num_threads=nthrds):
        for m in prange(M):
            inner_loop2(p_v, p_iSg, p_qdr, N, k, m*N, m*N*k, m*k*k)
