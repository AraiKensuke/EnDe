#  do multiple quadratics
cimport cython
from cython.parallel import parallel, prange
from libc.stdlib cimport abort, malloc, free
from libc.stdio cimport printf
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

    with nogil:#, parallel(num_threads=2):
        for 0 <= m < M:
        #for m in prange(M):
            mK = m*K
            mKN= mK*N

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
def hc_sub_2_vec_K2(double[:, ::1] mAS, double [:, ::1] u, double [:, :, ::1] out, int M, int N):
    #  mAS - u.  mAS: K-dim marks from N spikes, u: M K-dim cluster centers
    #  mAS   N x K
    #  u     M x K

    cdef int K  = 2   # hardcoded
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
        for 0 <= n < N:
            nK = n*K
            #mKN_nK = mKN+nK

            #  mAS[n, k] = p_mAS[n*K + k]
            #  u[m, k]   = p_u[m*K + k]
            #  out[m, n, k]   = p_out[m*K*N + n*K + k]
            p_out[mKN+nK]     = p_mAS[nK]   - u_m_0
            p_out[mKN+nK + 1] = p_mAS[nK+1] - u_m_1

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



@cython.boundscheck(False)
@cython.wraparound(False)
def CIFatFxdMks_mv(double[::1] mv_fxdMk, double[::1] mv_l0dt_i2pidcovs, double[:, ::1] mv_us, double[:, :, ::1] mv_iSgs, double[::1] mv_f, double[::1] mv_iq2, double[::1] mv_zs, double[::1] mv_qdr_mk, double[:, ::1] mv_qdr_sp, long M, long Nx, long mdim, double dt):
    #   x:       Nx
    #   fxdMks:  mdim
    #   l0s   :  M
    #   us    :  M x mdim
    #   fs    :  M
    #   iSgs  :  M x mdim x mdim
    #   i2pidcovs  :  M x (mdim + 1) x (mdim + 1)
    #   zs:      Nx
    #   qdrsp is M x Nxdim      m*Nx + ix

    #cdef double zs
    cdef long cmdim, cmdim2, i, j, k, ix, c, cNx
    cdef double tmp
    cdef double qdr_sp
    cdef double arg

    cdef double fc, iq2c

    for 0 <= c < M:    #  calculate the mark-contribution first.
        iq2c = mv_iq2[c]
        fc = mv_f[c]
        tmp = 0
        cNx    = c*Nx

        for 0 <= j < mdim:
            k = j
            tmp += (mv_fxdMk[j]-mv_us[c,j]) * mv_iSgs[c, j, k] * (mv_fxdMk[k]-mv_us[c, k])
            for j+1 <= k < mdim:
                tmp += 2* (mv_fxdMk[j]-mv_us[c, j]) * mv_iSgs[c, j, k] * (mv_fxdMk[k]-mv_us[c, k])

        mv_qdr_mk[c] = tmp

    for 0 <= ix < Nx:  #  the mark contribution constant, modulating it by spatial contribution
        tmp = 0
        for 0 <= c < M:    #  calculate the mark-contribution first.
            #qdr_sp = (p_x[ix] - p_f[c])*(p_x[ix] - p_f[c])*p_iq2[c]

            arg = mv_qdr_sp[c, ix] +mv_qdr_mk[c]
            if arg < 16:  #  contribution large enough
                tmp += mv_l0dt_i2pidcovs[c]*exp(-0.5*arg)
        mv_zs[ix] = tmp



@cython.boundscheck(False)
@cython.wraparound(False)
#cdef void CIFatFxdMks_nogil(double *p_fxdMk, double* p_x, double* p_l0dt_i2pidcovs, double* p_us, double* p_iSgs, double* p_f, double *p_iq2, double* p_zs, double* p_qdr_mk, double* p_qdr_sp, long M, long Nx, long mdim, double dt) nogil:
cdef void CIFatFxdMks_nogil(double *p_fxdMk, double* p_l0dt_i2pidcovs, double* p_us, double* p_iSgs, double* p_f, double *p_iq2, double* p_zs, double* p_qdr_mk, double* p_qdr_sp, long M, long Nx, long mdim, double dt) nogil:
    #   x:       Nx
    #   fxdMks:  mdim
    #   l0s   :  M
    #   us    :  M x mdim
    #   fs    :  M
    #   iSgs  :  M x mdim x mdim
    #   i2pidcovs  :  M x (mdim + 1) x (mdim + 1)
    #   zs:      Nx
    #   qdrsp is M x Nxdim      m*Nx + ix

    #cdef double zs
    cdef long cmdim, cmdim2, i, j, k, ix, c, cNx
    cdef double tmp
    cdef double qdr_sp
    cdef double arg

    cdef double fc, iq2c

    for 0 <= c < M:    #  calculate the mark-contribution first.
        iq2c = p_iq2[c]
        fc = p_f[c]
        tmp = 0
        cmdim  = c*mdim
        cmdim2 = cmdim*mdim
        cNx    = c*Nx

        for 0 <= j < mdim:
            k = j
            tmp += (p_fxdMk[j]-p_us[cmdim+j]) * p_iSgs[cmdim2 + j*mdim + k] * (p_fxdMk[k]-p_us[cmdim+k])
            for j+1 <= k < mdim:
                tmp += 2* (p_fxdMk[j]-p_us[cmdim+j]) * p_iSgs[cmdim2 + j*mdim + k] * (p_fxdMk[k]-p_us[cmdim+k])

        p_qdr_mk[c] = tmp

    for 0 <= ix < Nx:  #  the mark contribution constant, modulating it by spatial contribution
        tmp = 0
        for 0 <= c < M:    #  calculate the mark-contribution first.
            #qdr_sp = (p_x[ix] - p_f[c])*(p_x[ix] - p_f[c])*p_iq2[c]

            arg = p_qdr_sp[c*Nx + ix] +p_qdr_mk[c]
            if arg < 16:  #  contribution large enough
                tmp += p_l0dt_i2pidcovs[c]*exp(-0.5*arg)
        p_zs[ix] = tmp



"""
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void CIFatFxdMks_kde_nogil(double *p_fxdMk, double* p_x, double* p_l0dt_i2pidcovs, double* p_us, double* p_iSgs, double* p_f, double *p_iq2, double* p_zs, double* p_qdr_mk, double* p_qdr_sp, long M, long Nx, long mdim, double dt) nogil:
    #   x:       Nx
    #   fxdMks:  mdim
    #   l0s   :  M
    #   us    :  M x mdim
    #   fs    :  M
    #   iSgs  :  M x mdim x mdim
    #   i2pidcovs  :  M x (mdim + 1) x (mdim + 1)
    #   zs:      Nx
    #   qdrsp is M x Nxdim      m*Nx + ix

    #cdef double zs
    cdef long cmdim, cmdim2, i, j, k, ix, c, cNx
    cdef double tmp
    cdef double qdr_sp
    cdef double arg

    cdef double fc, iq2c

    for 0 <= c < M:    #  calculate the mark-contribution first.
        iq2c = p_iq2[c]
        fc = p_f[c]
        tmp = 0
        cmdim  = c*mdim
        cmdim2 = cmdim*mdim
        cNx    = c*Nx

        for 0 <= j < mdim:
            tmp += (p_fxdMk[j]-p_us[cmdim+j]) * p_iSgs[cmdim2 + j*mdim + j] * (p_fxdMk[k]-p_us[cmdim+k])

        p_qdr_mk[c] = tmp

    for 0 <= ix < Nx:  #  the mark contribution constant, modulating it by spatial contribution
        tmp = 0
        for 0 <= c < M:    #  calculate the mark-contribution first.
            arg = p_qdr_sp[c*Nx + ix] +p_qdr_mk[c]
            if arg < 16:  #  contribution large enough
                tmp += p_l0dt_i2pidcovs[c]*exp(-0.5*arg)
        #  xt is xix 
        #  bx2 = bx*bx
        #  occ[ix] = _N.sum(_N.exp(-0.5*(xix-f[c])**2/bx2)
        p_zs[ix] = tmp
"""
