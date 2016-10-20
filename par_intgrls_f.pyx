cimport cython
from cython.parallel import parallel, prange
from libc.stdlib cimport abort, malloc, free
from libc.math cimport exp
import numpy as _N
cimport numpy as _N

cdef Py_ssize_t idx, i, n = 100

cdef double f_intgrl(int m, double *p_fxs, double *p_ux, double *p_px, int M, int fss, int Nupx, double dSilenceX, double IIQ2, int fi) nogil:
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
def M_times_N_f_intgrls(double[:, ::1] fxs, double[::1] ux, double[::1] iiq2, double dSilenceX, double[::1] px, double[:, ::1] f_exp_px, int M, int fss, int Nupx, int nthrds):
    #  fxs       M x fss   
    #  fx                                        rux     Nupx    
    #  f_intgrd  Nupx
    cdef int fi, m, n
    cdef double dd, IIQ2
    cdef double[:, ::1] f_exp_px_N = f_exp_px

    with nogil, parallel(num_threads=nthrds):
        for m in prange(M):
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
def M_times_N_f_intgrls_raw(double[:, ::1] fxs, double[::1] ux, double[::1] iiq2, double dSilenceX, double[::1] px, double[:, ::1] f_exp_px, int M, int fss, int Nupx, int nthrds):
    #  fxs       M x fss   
    #  fxrux     Nupx    
    #  f_intgrd  Nupx
    cdef int fi, m, n
    cdef double dd, IIQ2

    cdef double *p_fxs = &fxs[0, 0]
    cdef double *p_px  = &px[0]
    cdef double *p_ux  = &ux[0]
    cdef double *p_f_exp_px = &f_exp_px[0, 0]

    with nogil, parallel(num_threads=nthrds):
        for m in prange(M):
            IIQ2 = iiq2[m]
            for fi in xrange(fss):     #  unrolling function makes this slower
                p_f_exp_px[m*fss + fi] = f_intgrl(m, p_fxs, p_ux, p_px, M, fss, Nupx, dSilenceX, IIQ2, fi)
    #     #  f_exp_px   is M x fss
