#  do multiple quadratics
cimport cython
from cython.parallel import parallel, prange
from libc.stdlib cimport abort, malloc, free
from libc.stdio cimport printf
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
def multi_qdrtcs_hard_code_4_diag_v2(double[:, :, ::1] v, double[:, :, ::1] iSg, double[:, ::1] qdr, int M, int N, int k):
    #  fxs       M x fss   
    #  fxrux     Nupx    
    #  f_intgrd  Nupx
    cdef int n, m, mNk, mkk, nk, ik, mNk_nk, mN

    cdef double *p_v   = &v[0, 0, 0]
    cdef double *p_iSg = &iSg[0, 0, 0]
    cdef double *p_qdr = &qdr[0, 0]

    cdef double iS00
    cdef double iS11
    cdef double iS22
    cdef double iS33


    with nogil:
        for 0 <= m < M:
            mkk  = m*k*k
            mN   = m*N
            mNk   = m*N*k
            #  write output to p_qdr[m, n].  (xn - um)iSg_m (xn - um)

            iS00 = p_iSg[mkk]
            iS11 = p_iSg[mkk+k+1]
            iS22 = p_iSg[mkk+2*k+2]
            iS33 = p_iSg[mkk+3*k+3]

            for n in range(N):
                mNk_nk = mNk + n*k

                p_qdr[mN+n] = p_v[mNk_nk+3]*iS33*p_v[mNk_nk+3] + \
                              p_v[mNk_nk]*iS00*p_v[mNk_nk] + \
                              p_v[mNk_nk+1]*iS11*p_v[mNk_nk+1] + \
                              p_v[mNk_nk+2]*iS22*p_v[mNk_nk+2]


@cython.boundscheck(False)
@cython.wraparound(False)
def full_qdrtcs_K4(double[:, ::1] pkFRr, double [:, ::1] mkNrms, double[:, ::1] exp_arg, double[:, ::1] fr, double [:, ::1] xASr, double[:, ::1] iq2r, double [:, ::1] qdrSPC, double[:, ::1] mAS, double [:, ::1] u, double[:, :, ::1] iSg, double[:, ::1] qdrMKS, int M, int N, int k):
#(double[:, ::1] pkFRr, double [:, ::1] mkNrms, double[:, ::1] exp_arg, double[:, ::1] fr, double [:, ::1] xASr, double[:, ::1] iq2r, double [:, ::1] qdrSPC, double[:, :, ::1] v, double[:, :, ::1] iSg, double[:, ::1] qdrMKS, int M, int N, int k):
                  
    #  fxs       M x fss   
    #  fxrux     Nupx    
    #  f_intgrd  Nupx
    cdef int n, m, mNk, mkk, nk, ik, mNk_nk, mN, mNn, mK, nK

    cdef double *p_mAS   = &mAS[0, 0]
    cdef double *p_u     = &u[0, 0]
    #cdef double *p_v   = &v[0, 0, 0]
    cdef double *p_iSg = &iSg[0, 0, 0]
    cdef double *p_qdrMKS = &qdrMKS[0, 0]

    cdef double pfrm
    cdef double piq2rm
    cdef double *p_qdrSPC   = &qdrSPC[0, 0]
    cdef double *p_fr       = &fr[0, 0]
    cdef double *p_xASr     = &xASr[0, 0]
    cdef double *p_iq2r     = &iq2r[0, 0]

    cdef double *p_pkFRr       = &pkFRr[0, 0]
    cdef double *p_mkNrms     = &mkNrms[0, 0]
    cdef double *p_exp_arg       = &exp_arg[0, 0]
    cdef double pkFRr_m, mkNrms_m

    cdef double iS00
    cdef double iS11
    cdef double iS22
    cdef double iS33
    cdef double iS01
    cdef double iS02
    cdef double iS03
    cdef double iS12
    cdef double iS13
    cdef double iS23
    cdef double u_m_0
    cdef double u_m_1
    cdef double u_m_2
    cdef double u_m_3


    with nogil:
        for 0 <= m < M:
            mkk  = m*k*k
            mN   = m*N
            #mNk   = m*N*k
            #mNk   = m*N*k
            mK =   m*k

            #  write output to p_qdr[m, n].  (xn - um)iSg_m (xn - um)

            iS00 = p_iSg[mkk]
            iS11 = p_iSg[mkk+k+1]
            iS22 = p_iSg[mkk+2*k+2]
            iS33 = p_iSg[mkk+3*k+3]
            iS01 = 2*p_iSg[mkk+1]   #  this
            iS02 = 2*p_iSg[mkk+2]
            iS03 = 2*p_iSg[mkk+3]
            iS12 = 2*p_iSg[mkk+k+2]
            iS13 = 2*p_iSg[mkk+k+3]
            iS23 = 2*p_iSg[mkk+2*k+3]

            u_m_0 = p_u[mK]
            u_m_1 = p_u[mK + 1]
            u_m_2 = p_u[mK + 2]
            u_m_3 = p_u[mK + 3]


            pfrm = p_fr[m]
            piq2rm= p_iq2r[m]

            pkFRr_m = p_pkFRr[m]
            mkNrms_m = p_mkNrms[m]

            for n in range(N):
                #mNk_nk = mNk + n*k
                mNn    = mN+n
                nK     = n*k
                p_qdrMKS[mNn] = (p_mAS[nK+3]-u_m_3)*iS33*(p_mAS[nK+3]-u_m_3) +\
                                (p_mAS[nK]-u_m_0)*(iS00*(p_mAS[nK]-u_m_0) + \
                                                   iS01*(p_mAS[nK+1]-u_m_1) + \
                                                   iS02*(p_mAS[nK+2]-u_m_2) + \
                                                   iS03*(p_mAS[nK+3]-u_m_3))+\
                                (p_mAS[nK+1]-u_m_1)*(iS11*(p_mAS[nK+1]-u_m_1) + \
                                                     iS12*(p_mAS[nK+2]-u_m_2) + \
                                                     iS13*(p_mAS[nK+3]-u_m_3))+\
                                (p_mAS[nK+2]-u_m_2)*(iS22*(p_mAS[nK+2]-u_m_2) + \
                                                     iS23*(p_mAS[nK+3]-u_m_3))

                p_qdrSPC[mNn] = (pfrm - p_xASr[n])*(pfrm - p_xASr[n])*piq2rm

                p_exp_arg[mNn] = pkFRr_m + mkNrms_m - 0.5*(p_qdrSPC[mNn] + p_qdrMKS[mNn])

# @cython.boundscheck(False)
# @cython.wraparound(False)
# def full_qdrtcs_K4_diag(double[:, ::1] pkFRr, double [:, ::1] mkNrms, double[:, ::1] exp_arg, double[:, ::1] fr, double [:, ::1] xASr, double[:, ::1] iq2r, double [:, ::1] qdrSPC, double[:, :, ::1] v, double[:, :, ::1] iSg, double[:, ::1] qdrMKS, int M, int N, int k):
#     #  fxs       M x fss   
#     #  fxrux     Nupx    
#     #  f_intgrd  Nupx
#     cdef int n, m, mNk, mkk, nk, ik, mNk_nk, mN, mNn

#     cdef double *p_v   = &v[0, 0, 0]
#     cdef double *p_iSg = &iSg[0, 0, 0]
#     cdef double *p_qdrMKS = &qdrMKS[0, 0]

#     cdef double pfrm
#     cdef double piq2rm
#     cdef double *p_qdrSPC   = &qdrSPC[0, 0]
#     cdef double *p_fr       = &fr[0, 0]
#     cdef double *p_xASr     = &xASr[0, 0]
#     cdef double *p_iq2r     = &iq2r[0, 0]

#     cdef double *p_pkFRr       = &pkFRr[0, 0]
#     cdef double *p_mkNrms     = &mkNrms[0, 0]
#     cdef double *p_exp_arg       = &exp_arg[0, 0]
#     cdef double pkFRr_m, mkNrms_m

#     cdef double iS00
#     cdef double iS11
#     cdef double iS22
#     cdef double iS33

#     with nogil:
#         for 0 <= m < M:
#             mkk  = m*k*k
#             mN   = m*N
#             mNk   = m*N*k
#             #  write output to p_qdr[m, n].  (xn - um)iSg_m (xn - um)

#             iS00 = p_iSg[mkk]
#             iS11 = p_iSg[mkk+k+1]
#             iS22 = p_iSg[mkk+2*k+2]
#             iS33 = p_iSg[mkk+3*k+3]

#             pfrm = p_fr[m]
#             piq2rm= p_iq2r[m]

#             pkFRr_m = p_pkFRr[m]
#             mkNrms_m = p_mkNrms[m]

#             for n in range(N):
#                 mNk_nk = mNk + n*k
#                 mNn    = mN+n
#                 p_qdrMKS[mNn] = p_v[mNk_nk+3]*iS33*p_v[mNk_nk+3] + \
#                                 p_v[mNk_nk]*iS00*p_v[mNk_nk] + \
#                                 p_v[mNk_nk+1]*iS11*p_v[mNk_nk+1] + \
#                                 p_v[mNk_nk+2]*iS22*p_v[mNk_nk+2]

#                 p_qdrSPC[mNn] = (pfrm - p_xASr[n])*(pfrm - p_xASr[n])*piq2rm

#                 p_exp_arg[mNn] = pkFRr_m + mkNrms_m - 0.5*(p_qdrSPC[mNn] + p_qdrMKS[mNn])


@cython.boundscheck(False)
@cython.wraparound(False)
def full_qdrtcs_K4_diag(double[:, ::1] pkFRr, double [:, ::1] mkNrms, double[:, ::1] exp_arg, double[:, ::1] fr, double [:, ::1] xASr, double[:, ::1] iq2r, double [:, ::1] qdrSPC, double[:, ::1] mAS, double [:, ::1] u, double[:, :, ::1] iSg, double[:, ::1] qdrMKS, int M, int N, int k):
    #  fxs       M x fss   
    #  fxrux     Nupx    
    #  f_intgrd  Nupx
    cdef int n, m, mNk, mkk, nk, ik, mNk_nk, mN, mNn, mK, nK

    cdef double *p_mAS   = &mAS[0, 0]
    cdef double *p_u     = &u[0, 0]
    #cdef double *p_v   = &v[0, 0, 0]
    cdef double *p_iSg = &iSg[0, 0, 0]
    cdef double *p_qdrMKS = &qdrMKS[0, 0]

    cdef double pfrm
    cdef double piq2rm
    cdef double *p_qdrSPC   = &qdrSPC[0, 0]
    cdef double *p_fr       = &fr[0, 0]
    cdef double *p_xASr     = &xASr[0, 0]
    cdef double *p_iq2r     = &iq2r[0, 0]

    cdef double *p_pkFRr       = &pkFRr[0, 0]
    cdef double *p_mkNrms     = &mkNrms[0, 0]
    cdef double *p_exp_arg       = &exp_arg[0, 0]
    cdef double pkFRr_m, mkNrms_m

    cdef double iS00
    cdef double iS11
    cdef double iS22
    cdef double iS33
    cdef double u_m_0
    cdef double u_m_1
    cdef double u_m_2
    cdef double u_m_3

    with nogil:#, parallel(num_threads=2):
        for 0 <= m < M:
        #for m in prange(M):
            mkk  = m*k*k
            mN   = m*N
            #mNk   = m*N*k
            mK =   m*k
            #  write output to p_qdr[m, n].  (xn - um)iSg_m (xn - um)

            iS00 = p_iSg[mkk]
            iS11 = p_iSg[mkk+k+1]
            iS22 = p_iSg[mkk+2*k+2]
            iS33 = p_iSg[mkk+3*k+3]

            u_m_0 = p_u[mK]
            u_m_1 = p_u[mK + 1]
            u_m_2 = p_u[mK + 2]
            u_m_3 = p_u[mK + 3]

            pfrm = p_fr[m]
            piq2rm= p_iq2r[m]

            pkFRr_m = p_pkFRr[m]
            mkNrms_m = p_mkNrms[m]

            for n in range(N):
                #mNk_nk = mNk + n*k
                mNn    = mN+n
                nK     = n*k
                # p_qdrMKS[mNn] = p_v[mNk_nk+3]*iS33*p_v[mNk_nk+3] + \
                #                 p_v[mNk_nk]*iS00*p_v[mNk_nk] + \
                #                 p_v[mNk_nk+1]*iS11*p_v[mNk_nk+1] + \
                #                 p_v[mNk_nk+2]*iS22*p_v[mNk_nk+2]
                p_qdrMKS[mNn] = (p_mAS[nK+3]-u_m_3)*iS33*(p_mAS[nK+3]-u_m_3) +\
                                (p_mAS[nK]-u_m_0)*iS00*(p_mAS[nK]-u_m_0) + \
                                (p_mAS[nK+1]-u_m_1)*iS11*(p_mAS[nK+1]-u_m_1) + \
                                (p_mAS[nK+2]-u_m_2)*iS22*(p_mAS[nK+2]-u_m_2)

                p_qdrSPC[mNn] = (pfrm - p_xASr[n])*(pfrm - p_xASr[n])*piq2rm

                p_exp_arg[mNn] = pkFRr_m + mkNrms_m - 0.5*(p_qdrSPC[mNn] + p_qdrMKS[mNn])


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
    #  exp(- qdrspc - qdrmk - nrm - offset)   - get occupation ratios.
    #  offset so that most likely cluster has ratio == 1
    cdef double *p_inp  = &inp[0, 0]
    cdef double *p_out  =  &out[0, 0]
    cdef long m, n, mN = 0

    with nogil:
        for 0 <= m < M:
            mN = m*N
            for 0 <= n < N:
                #p_out[mN + n] = exp(p_inp[mN + n])
                #  most of the time, we're evaluating exp(VERY NEGATIVE).
                #  instead of wasting time evaluating this, just set these
                #  ones to 0.  assignment ratio is << exp(0), practically 0.
                p_out[mN+n] = 0 if (p_inp[mN+n] < -10) else exp(p_inp[mN + n])


@cython.boundscheck(False)
@cython.wraparound(False)
def set_occ(double[:, ::1] crats, double[::1] rnds, char[:, ::1] gz, long M, long N):
    #  instead of doing the following:
    #M1 = crat[1:] >= rnds
    #M2 = crat[0:-1] <= rnds
    #gz = (M1&M2)
    # in python to occupation binary vector gz with 0 or 1s,
    #  call this function (with call to gz.fill(0) before calling set_occ)
    cdef long n, im, #nind, m
    cdef double* p_crats = &crats[0, 0]
    cdef double* p_rnds  = &rnds[0]
    cdef char* p_gz = &gz[0, 0]   #  N x M    different than c_rats
    cdef double rnd

    with nogil:
        for n in xrange(N):
            im = 0
            rnd  = p_rnds[n]
            while rnd >= p_crats[im*N+n]:   #  crats
                # rnds[0] = 0.1  crats = [0, 0.2]    i expect gz[0, n] = 1
                im += 1
            p_gz[n*M + im-1] = 1

            #  actually slower than calling gz.fill(0) before call to srch_occ
            # for im*N+n <= m < M*N+n by N:
            #     p_gz[m] = 0
                
def multiple_mat_dot_v(double[:, :, ::1] mat, double[:, ::1] vec, double[:, ::1] out, long M, long K):
    ##  mat is M x K x K
    ##  vec is M x K
    ##  out is M x K
    cdef long m, k, mKK, mK, iK, i, j
    cdef double* p_mat = &mat[0, 0, 0]
    cdef double* p_vec = &vec[0, 0]
    cdef double* p_out = &out[0, 0]

    with nogil:
        for 0 <= m < M:
            mKK = m*K*K
            mK  = m*K
            for i in xrange(K):
                iK = i*K
                p_out[mK+ i] = 0
                for k in xrange(K):
                    p_out[mK+ i] += p_mat[mKK+ iK+ k] * p_vec[mK + k]

@cython.cdivision(True)
def mean_random_indices(double[:, ::1] vs, long[::1] inds, double[::1] out, long I, long K):
    cdef long i, k, indsK
    cdef double* p_out = &out[0]
    cdef double* p_vs  = &vs[0, 0]
    cdef long* p_inds = &inds[0]
    cdef double iI     = 1./I

    with nogil:
        for 0 <= k < K:
            p_out[k] = 0

        for 0 <= i < I:
            indsK = p_inds[i]*K
            for 0 <= k < K:
                p_out[k] += p_vs[indsK + k] 

        for 0 <= k < K:
            p_out[k] *= iI

def Sg_PSI(long[::1] cls_str_ind, long[::1] clstsz, long[::1] v_sts, double[:, ::1] mks, double[:, :, ::1] _Sg_PSI, double[:, :, ::1] Sg_PSI_, double[:, ::1] u, long M, long K):
    cdef long K2, non_cnt_ind, m, k, n, nSpks, mK, i0
    cdef long* p_cls_str_ind = &cls_str_ind[0]
    cdef long* p_clstsz      = &clstsz[0]
    cdef long* p_v_sts       = &v_sts[0]
    cdef double* p_mks       = &mks[0, 0]
    cdef double* p_u         = &u[0, 0]
    cdef double* p_Sg_PSI_   = &Sg_PSI_[0, 0, 0]
    cdef double* p__Sg_PSI   = &_Sg_PSI[0, 0, 0]
    K2 = K*K
    cdef double tot

    with nogil:
        for 0 <= m < M:
            nSpks = p_cls_str_ind[m+1] - p_cls_str_ind[m]
            i0    = p_cls_str_ind[m]
            mK    = m*K
            for 0 <= k < K:
                uk = p_u[mK+k]
                tot = 0

                for 0 <= n < nSpks:
                    non_cnt_ind = p_v_sts[i0 + n]

                    tot += (p_mks[non_cnt_ind*K + k]-uk)*(p_mks[non_cnt_ind*K + k]-uk)
                p_Sg_PSI_[m*K2+k*K + k] = p__Sg_PSI[m*K2+k*K + k] + tot*0.5


