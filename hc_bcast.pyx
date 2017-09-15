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

    cdef double pfrm
    cdef double piq2rm
    cdef double *p_qdrSPC   = &qdrSPC[0, 0]
    cdef double *p_fr       = &fr[0, 0]
    cdef double *p_xASr     = &xASr[0, 0]
    cdef double *p_iq2r     = &iq2r[0, 0]

    for 0 <= m < M:
        mN = m*N
        pfrm = p_fr[m]
        piq2rm= p_iq2r[m]
        for 0 <= n < N:
            p_qdrSPC[mN+n] = (pfrm - p_xASr[n])*(pfrm - p_xASr[n])*piq2rm
            #p_qdrSPC[mN+n] = (p_fr[m] - p_xASr[n])*(p_fr[m] - p_xASr[n])*p_iq2r[m]


@cython.boundscheck(False)
@cython.wraparound(False)
def hc_sub_2_vec_K4(double[:, ::1] mAS, double [:, ::1] u, double [:, :, ::1] out, int M, int N):
    #  mAS - u.  mAS: K-dim marks from N spikes, u: M K-dim cluster centers
    #  mAS   N x K
    #  u     M x K

    cdef int K  = 4   # hardcoded
    #  output is M x N x K
    
    cdef int m, n, mK, mKN, nK, mKN_nK

    cdef double *p_mAS   = &mAS[0, 0]
    cdef double *p_u     = &u[0, 0]
    cdef double *p_out   = &out[0, 0, 0]

    cdef double u_m_0, u_m_1, u_m_2, u_m_3  # u[m, 0] to u[m, 3]
    for 0 <= m < M:
        mK = m*K
        mKN= m*K*N

        u_m_0 = p_u[mK]
        u_m_1 = p_u[mK + 1]
        u_m_2 = p_u[mK + 2]
        u_m_3 = p_u[mK + 3]
        for 0 <= n < N:
            nK = n*K
            #mKN_nK = mKN+nK

            #  mAS[n, k] = p_mAS[n*K + k]
            #  u[m, k]   = p_u[m*K + k]
            #  out[m, n, k]   = p_out[m*K*N + n*K + k]
            p_out[mKN+nK]     = p_mAS[nK]   - u_m_0
            p_out[mKN+nK + 1] = p_mAS[nK+1] - u_m_1
            p_out[mKN+nK + 2] = p_mAS[nK+2] - u_m_2
            p_out[mKN+nK + 3] = p_mAS[nK+3] - u_m_3

        #out[m, n, k] = mAS[n, k] - u[m, k]

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
    cdef double norm

    with nogil:
        for 0 <= m < M:
            mN = m*N
            pkFRr_m = p_pkFRr[m]
            mkNrms_m = p_mkNrms[m]
            #norm    = p_pkFRr[m] + p_mkNrms[m]
            for 0 <= n < N:
                #mNn = mN+n
                #p_cont[mNn] = norm - 0.5*(p_qdrSpc[mNn] + p_qdrMKS[mNn])
                p_cont[mN+n] = pkFRr_m + mkNrms_m - 0.5*(p_qdrSpc[mN+n] + p_qdrMKS[mN+n])

@cython.boundscheck(False)
@cython.wraparound(False)
def evalAtFxdMks_new(double[:, ::1] fxdMks, double[::1] l0, double[:, ::1] us, double[:, :, ::1] iSgs, double[::1] i2pidcovs, int M, int Nx, int pmdim):
    #   fxdMks:  Nx x (mdim+1)
    #   l0s   :  M
    #   us    :  M x (mdim + 1)
    #   iSgs
    
    #cdef double zs
    zs    = _N.zeros(Nx)
    cdef double[::1] zs_mv = zs
    cdef double *p_zs      = &zs_mv[0]
    cdef int mNx, jpmdim, mpmdim, mpmdim2, ixpmdim, i, j, k, ix, m
    cdef int mNxix
  
    mxval = _N.zeros((M, Nx))
    cdef double[:, ::1] mxval_mv = mxval   #  memory view
    cdef double *p_mxval         = &mxval_mv[0, 0]

    cdef double *p_fxdMks   = &fxdMks[0, 0]
    cdef double *p_l0       = &l0[0]
    cdef double *p_us       = &us[0, 0]
    cdef double *p_iSgs     = &iSgs[0, 0, 0]
    cdef double *p_i2pidcovs= &i2pidcovs[0]
    cdef double outer, tmp

    #  fxdMks is a 

    #  cmps is spatial contribution due to component m.  

    with nogil:
        for 0 <= m < M:
            mNx = m*Nx
            mpmdim = m*pmdim
            mpmdim2= mpmdim*pmdim

            for 0 <= ix < Nx:
                ixpmdim = ix*pmdim
                mNxix = mNx + ix
                tmp = 0
                for 0 <= j < pmdim:
                    jpmdim = j*pmdim
                    outer = p_fxdMks[ixpmdim+j]-p_us[mpmdim+j]
                    #for 0 <= k < pmdim:
                    k = j
                    tmp += outer*p_iSgs[mpmdim2 + jpmdim + k]*(p_fxdMks[ixpmdim+k]-p_us[mpmdim+k])
                    for j+1 <= k < pmdim:
                        #mxval[m, ix] += (fxdMks[ix, j]-us[m, j])*iSgs[m, j, k]*(fxdMks[ix, k]-us[m, k])
                        tmp += 2*outer*p_iSgs[mpmdim2 + jpmdim + k]*(p_fxdMks[ixpmdim+k]-p_us[mpmdim+k])
                    p_mxval[mNxix] = tmp

            #  cmps = i2pidcovsr*_N.exp(-0.5*_N.einsum("xmj,xmj->mx", fxdMksr-us, _N.einsum("mjk,xmk->xmj", iSgs, fxdMksr - us)))
        for 0 <= ix < Nx:
            for 0 <= m < M:
                p_zs[ix] += p_l0[m] * p_i2pidcovs[m] * exp(-0.5*p_mxval[m*Nx+ix])
                #p_zs[ix] += l0[m] * i2pidcovs[m] * _N.exp(-0.5*mxval[m, ix])

    return zs

# def rolled_out_q2(int M):
#     #  v_sts    int of size len(Asts)   all spikes in epoch
#     cdef int SPKS, strt_ind_m
#     for 0 <= m < M:
#         strt_ind_m = cls_str_ind[m]
#         SPKS = cls_str_ind[m+1]- strt_ind_m
#         fm   = f[m]

#         for 0 <= s < SPKS:
#             sxI += (xt0t1[v_sts[s+strt_ind_m]-t0] - fm)*(xt0t1[v_sts[s+strt_ind_m]-t0] - fm)
#         SL_B[m] = sxI*0.5
#     return SL_B
        
        
    # for m in xrange(M):
    #     if clstsz[m] > 0:
    #         sts = v_sts[cls_str_ind[m]:cls_str_ind[m+1]]
    #         xI = (xt0t1[sts-t0]-f[m])*(xt0t1[sts-t0]-f[m])*0.5
    #         SL_B = _N.sum(xI)  #  spiking part of likelihood
    #         #  -S/2 (likelihood)  -(a+1)
    #         sLLkPr[m] = -(0.5*clstsz[m] + _q2_a[m] + 1)*lq2x - iq2x*(_q2_B[m] + SL_B)   #  just (isig2)^{-S/2} x (isig2)^{-(_q2_a + 1)}   
    #     else:
    #         sLLkPr[m] = -(_q2_a[m] + 1)*lq2x - iq2x*_q2_B[m]
