cimport cython
from cython.parallel import parallel, prange
from libc.stdlib cimport abort, malloc, free
from libc.math cimport exp
import numpy as _N
cimport numpy as _N

cdef Py_ssize_t idx, i, n = 100

@cython.boundscheck(False)
@cython.wraparound(False)
def M_times_N_f_intgrls_noOMP_raw(double[:, ::1] fxs, double[::1] ux, double[::1] iiq2, double dSilenceX, double[::1] px, double[:, ::1] f_exp_px, int M, int fss, int Nupx, int nthrds):
    #  fxs       M x fss   
    #  fxrux     Nupx    
    #  f_intgrd  Nupx
    cdef int fi, m, n
    cdef double dd, IIQ2

    cdef double *p_fxs = &fxs[0, 0]
    cdef double *p_px  = &px[0]
    cdef double *p_ux  = &ux[0]
    cdef double *p_f_exp_px = &f_exp_px[0, 0]

    for m in xrange(M):
        IIQ2 = iiq2[m]
        for fi in xrange(fss):
            p_f_exp_px[m*fss + fi] = intgrl(m, p_fxs, p_ux, p_px, M, fss, Nupx, dSilenceX, IIQ2, fi)
    #     #  f_exp_px   is M x fss

cdef double intgrl(int m, double *p_fxs, double *p_ux, double *p_px, int M, int fss, int Nupx, double dSilenceX, double IIQ2, int fi) nogil:
    cdef double dd, tot
    cdef int mM

    mM = m*fss

    tot = 0.0
    for n in xrange(Nupx):
        dd = p_fxs[mM + fi]-p_ux[n]
        tot += exp(-0.5*dd*dd*IIQ2)*p_px[n]
    tot *= dSilenceX
    return tot

@cython.boundscheck(False)
@cython.wraparound(False)
def M_times_N_f_intgrls_noOMP(double[:, ::1] fxs, double[::1] ux, double[::1] iiq2, double dSilenceX, double[::1] px, double[:, ::1] f_exp_px, int M, int fss, int Nupx, int nthrds):
    #  fxs       M x fss   
    #  ux        fss
    #  iiq2      M
    #  px        Nupx
    #  f_exp_px  M x fss

    cdef int fi, m, n
    cdef double dd, IIQ2
    cdef double[:, ::1] f_exp_px_N = f_exp_px

    for m in range(M):
        IIQ2 = iiq2[m]
        for fi in range(fss):
            f_exp_px_N[m, fi] = 0
            for n in range(Nupx):
                dd = fxs[m, fi]-ux[n]
                f_exp_px_N[m, fi] += exp(-0.5*dd*dd*IIQ2)*px[n]
            f_exp_px_N[m, fi] *= dSilenceX
        #  f_exp_px   is M x fss


@cython.boundscheck(False)
@cython.wraparound(False)
def M_times_N_q2_intgrls_noOMP(double[::1] fxs, double[::1] ux, double[::1] iiq2, double dSilenceX, double[::1] px, double[:, ::1] f_exp_px, int M, int q2ss, int Nupx, int nthrds):
    #  fxs       M
    #  ux        Nupx
    #  iiq2      M
    #  px        Nupx
    #  f_exp_px  M x q2ss

    cdef int q2i, m, n
    cdef double dd, IIQ2
    cdef double[:, ::1] f_exp_px_N = f_exp_px

    for m in range(M):
        fm = fxs[m]   #  current value of f for m-th cluster
        for q2i in range(q2ss):    #  
            f_exp_px_N[m, q2i] = 0
            IIQ2 = iiq2[q2i]
            for n in range(Nupx):
                dd = fm-ux[n]  #  ux[n] integ. variable 
                f_exp_px_N[m, q2i] += exp(-0.5*dd*dd*IIQ2)*px[n]
            f_exp_px_N[m, q2i] *= dSilenceX
        #  f_exp_px   is M x fss


@cython.boundscheck(False)
@cython.wraparound(False)
def M_times_N_f_intgrls_pure(fxs, ux, iiq2, dSilenceX, px, f_exp_px, M, fss, Nupx, nthrds):
    #  fxs       M x fss   
    #  fxrux     Nupx    
    #  f_intgrd  Nupx
    #cdef double[:, ::1] f_exp_px_N = f_exp_px   #  makes it a bit faster than pure

    for m in range(M):
        IIQ2 = iiq2[m]
        for fi in range(fss):
            f_exp_px[m, fi] = 0
            for n in range(Nupx):
                dd = fxs[m, fi]-ux[n]
                f_exp_px[m, fi] += exp(-0.5*dd*dd*IIQ2)*px[n]
            f_exp_px[m, fi] *= dSilenceX
        #  f_exp_px   is M x fss


