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
    cdef double tot, tot_j
    for n in range(N):
        mNk_nk = mNk + n*k
        mN_n   = mN + n
        tot = 0

        for i in range(k):
            mkk_ik = mkk + i*k
            p_v_mNk_nki= p_v[mNk_nk + i]
            tot += p_v_mNk_nki * p_iSg[mkk_ik + i] * p_v_mNk_nki
            totj = 0

            for j in range(i+1,k):            
                tot += 2*p_v_mNk_nki * p_iSg[mkk_ik + j] * p_v[mNk_nk + j]
                #totj += p_iSg[mkk_ik + j] * p_v[mNk_nk + j]
            #totj *= 2*p_v_mNk_nki
        p_qdr[mN_n] = tot
        #p_qdr[mN_n] = tot + totj

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void inner_loop_hc_4(double *p_v, double *p_iSg, double *p_qdr, int N, int k, int mN, int mNk, int mkk) nogil:
    #  hardcoded for k = 5.  pos + marks.  pos, marks are orthogonal
    #  In this case, should be 1 + 4 + 6 terms
    #   xSx = x[0]*x[0]*S00 + x[1]*S11*x[1] + x[2]*S22*x[2] + ... (diag trms)
    #      + 2(x[1]S12 x[2] + x[1]S13 x[3] + x[1]S14 x[4] +
    #          x[2]S23 x[3] + x[2]S24 x[4] + 
    #          x[3]S34 x[4])
    cdef int n, mNk_nk

    cdef double iS00 = p_iSg[mkk]
    cdef double iS11 = p_iSg[mkk+k+1]
    cdef double iS22 = p_iSg[mkk+2*k+2]
    cdef double iS33 = p_iSg[mkk+3*k+3]
    cdef double iS01 = 2*p_iSg[mkk+1]   #  this
    cdef double iS02 = 2*p_iSg[mkk+2]
    cdef double iS03 = 2*p_iSg[mkk+3]
    cdef double iS12 = 2*p_iSg[mkk+k+2]
    cdef double iS13 = 2*p_iSg[mkk+k+3]
    cdef double iS23 = 2*p_iSg[mkk+2*k+3]

    for n in range(N):
        mNk_nk = mNk + n*k

        #  mkk offsets pointer so we're accessing mth iSg
        #  p_iSg[mkk_ik+i*k + j] = p_iSg[i, j] of the mth

        # p_qdr[mN+n] = p_v[mNk_nk]*p_iSg[mkk]*p_v[mNk_nk] + \
        #               p_v[mNk_nk+1]*p_iSg[mkk+k+1]*p_v[mNk_nk+1] + \
        #               p_v[mNk_nk+2]*p_iSg[mkk+2*k+2]*p_v[mNk_nk+2] + \
        #               p_v[mNk_nk+3]*p_iSg[mkk+3*k+3]*p_v[mNk_nk+3] + \
        #               2*(p_v[mNk_nk]*p_iSg[mkk+1]*p_v[mNk_nk+1] + \
        #                  p_v[mNk_nk]*p_iSg[mkk+2]*p_v[mNk_nk+2] + \
        #                  p_v[mNk_nk]*p_iSg[mkk+3]*p_v[mNk_nk+3] + \
        #                  p_v[mNk_nk+1]*p_iSg[mkk+k+2]*p_v[mNk_nk+2] + \
        #                  p_v[mNk_nk+1]*p_iSg[mkk+k+3]*p_v[mNk_nk+3] + \
        #                  p_v[mNk_nk+2]*p_iSg[mkk+2*k+3]*p_v[mNk_nk+3])
        p_qdr[mN+n] = p_v[mNk_nk+3]*iS33*p_v[mNk_nk+3] + \
                      p_v[mNk_nk]*(iS00*p_v[mNk_nk] + \
                                   iS01*p_v[mNk_nk+1] + \
                                   iS02*p_v[mNk_nk+2] + \
                                   iS03*p_v[mNk_nk+3]) + \
                         p_v[mNk_nk+1]*(iS11*p_v[mNk_nk+1] + \
                                        iS12*p_v[mNk_nk+2] + \
                                        iS13*p_v[mNk_nk+3]) + \
                         p_v[mNk_nk+2]*(iS22*p_v[mNk_nk+2] + \
                                        iS23*p_v[mNk_nk+3])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void inner_loop_hc_4_diag(double *p_v, double *p_iSg, double *p_qdr, int N, int k, int mN, int mNk, int mkk) nogil:
    #  hardcoded for k = 5.  pos + marks.  pos, marks are orthogonal
    #  In this case, should be 1 + 4 + 6 terms
    #   xSx = x[0]*x[0]*S00 + x[1]*S11*x[1] + x[2]*S22*x[2] + ... (diag trms)
    #      + 2(x[1]S12 x[2] + x[1]S13 x[3] + x[1]S14 x[4] +
    #          x[2]S23 x[3] + x[2]S24 x[4] + 
    #          x[3]S34 x[4])
    cdef int n, mNk_nk

    cdef double iS00 = p_iSg[mkk]
    cdef double iS11 = p_iSg[mkk+k+1]
    cdef double iS22 = p_iSg[mkk+2*k+2]
    cdef double iS33 = p_iSg[mkk+3*k+3]

    for n in range(N):
        mNk_nk = mNk + n*k

        p_qdr[mN+n] = p_v[mNk_nk+3]*iS33*p_v[mNk_nk+3] + \
                      p_v[mNk_nk]*iS00*p_v[mNk_nk] + \
                      p_v[mNk_nk+1]*iS11*p_v[mNk_nk+1] + \
                      p_v[mNk_nk+2]*iS22*p_v[mNk_nk+2]


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
            #  write output to p_qdr[m, n].  (xn - um)iSg_m (xn - um)
            inner_loop_sym(p_v, p_iSg, p_qdr, N, k, m*N, m*N*k, m*k*k)

@cython.boundscheck(False)
@cython.wraparound(False)
def multi_qdrtcs_hard_code_4(double[:, :, ::1] v, double[:, :, ::1] iSg, double[:, ::1] qdr, int M, int N, int k):
    #  fxs       M x fss   
    #  fxrux     Nupx    
    #  f_intgrd  Nupx
    cdef int m, mNk, mkk, nk, ik

    cdef double *p_v   = &v[0, 0, 0]
    cdef double *p_iSg = &iSg[0, 0, 0]
    cdef double *p_qdr = &qdr[0, 0]

    with nogil:
        for 0 <= m < M:
            #  write output to p_qdr[m, n].  (xn - um)iSg_m (xn - um)
            inner_loop_hc_4(p_v, p_iSg, p_qdr, N, k, m*N, m*N*k, m*k*k)

@cython.boundscheck(False)
@cython.wraparound(False)
def multi_qdrtcs_hard_code_4_diag(double[:, :, ::1] v, double[:, :, ::1] iSg, double[:, ::1] qdr, int M, int N, int k):
    #  fxs       M x fss   
    #  fxrux     Nupx    
    #  f_intgrd  Nupx
    cdef int m, mNk, mkk, nk, ik

    cdef double *p_v   = &v[0, 0, 0]
    cdef double *p_iSg = &iSg[0, 0, 0]
    cdef double *p_qdr = &qdr[0, 0]

    with nogil:
        for 0 <= m < M:
            #  write output to p_qdr[m, n].  (xn - um)iSg_m (xn - um)
            inner_loop_hc_4_diag(p_v, p_iSg, p_qdr, N, k, m*N, m*N*k, m*k*k)

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

@cython.boundscheck(False)
@cython.wraparound(False)
def exp_on_arr(double[:, ::1] inp, double[:, ::1] out, long M, long N):
    cdef double *p_inp  = &inp[0, 0]
    cdef double *p_out  =  &out[0, 0]
    cdef long m, n, mN = 0

    with nogil:
        for 0 <= m < M:
            mN = m*N
            for 0 <= n < N:
                p_out[mN + n] = exp(p_inp[mN + n])


#    M1 = rat[1:] >= rnds       
#    M2 = rat[0:-1] <= rnds
#    gz[it] = (M1&M2).T



@cython.boundscheck(False)
@cython.wraparound(False)
def set_occ(double[:, ::1] crats, double[::1] rnds, char[:, ::1] gz, long M, long N):
    #  instead of doing the following:
    #M1 = crat[1:] >= rnds
    #M2 = crat[0:-1] <= rnds
    #gz = (M1&M2)
    # in python to occupation binary vector gz with 0 or 1s,
    #  call this function (with call to gz.fill(0) before calling set_occ)
    cdef long n, im, nind, m
    cdef double* p_crats = &crats[0, 0]
    cdef double* p_rnds  = &rnds[0]
    cdef char* p_gz = &gz[0, 0]
    cdef double rnd

    with nogil:
        for n in xrange(N):
            im = 0
            nind = im*N+n
            rnd  = p_rnds[n]
            while rnd >= p_crats[nind]:   #  crats
                # rnds[0] = 0.1  crats = [0, 0.2]    i expect gz[0, n] = 1
                #p_gz[im*N+n] = 0 
                p_gz[nind] = 0 
                im += 1
                nind += N
            p_gz[(im-1)*N+ n] = 1

            #  actually slower than calling gz.fill(0) before call to srch_occ
            # for im*N+n <= m < M*N+n by N:
            #     p_gz[m] = 0
                
