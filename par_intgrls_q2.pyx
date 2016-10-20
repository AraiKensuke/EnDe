cimport cython
from cython.parallel import parallel, prange
from libc.stdlib cimport abort, malloc, free
from libc.math cimport exp
import numpy as _N
cimport numpy as _N

cdef Py_ssize_t idx, i, n = 100

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double q2_intgrl(int m, double *p_f, double *p_ux, double *p_px, int M, int q2ss, int Nupx, double dSilenceX, double IIQ2, int q2i) nogil:
    cdef double dd, tot
    cdef int mM

    tot = 0.0

    for n in xrange(Nupx):
        dd = p_f[m]-p_ux[n]
        tot += exp(-0.5*dd*dd*IIQ2)*p_px[n]
    tot *= dSilenceX

    return tot


@cython.boundscheck(False)
@cython.wraparound(False)
def M_times_N_q2_intgrls(double[::1] f, double[::1] ux, double[::1] iiq2xs, double dSilenceX, double[::1] px, double[:, ::1] q2_exp_px, int M, int q2ss, int Nupx, int nthrds):
    #  fxs       M x fss   
    #  fx                                        rux     Nupx    
    #  f_intgrd  Nupx
    cdef int q2i, m, n
    cdef double dd, IIQ2
    cdef double[:, ::1] q2_exp_px_N = q2_exp_px

    with nogil, parallel(num_threads=nthrds):
        for m in prange(M):
            for q2i in range(q2ss):   # various values of q2
                IIQ2 = iiq2xs[q2i]
                q2_exp_px_N[m, q2i] = 0
                for n in range(Nupx):
                    dd = f[m]-ux[n]
                    q2_exp_px_N[m, q2i] += exp(-0.5*dd*dd*IIQ2)*px[n]
                q2_exp_px_N[m, q2i] *= dSilenceX
            #  f_exp_px   is M x fss


@cython.boundscheck(False)
@cython.wraparound(False)
def M_times_N_q2_intgrls_raw(double[::1] f, double[::1] ux, double[::1] iiq2xs, double dSilenceX, double[::1] px, double[:, ::1] q2_exp_px, int M, int q2ss, int Nupx, int nthrds):
    #  fxs       M x fss   
    #  fxrux     Nupx    
    #  f_intgrd  Nupx
    cdef int q2i, m, n
    cdef double dd, IIQ2

    cdef double *p_f   = &f[0]
    cdef double *p_px  = &px[0]
    cdef double *p_ux  = &ux[0]
    cdef double *p_q2_exp_px = &q2_exp_px[0, 0]

    with nogil, parallel(num_threads=nthrds):
        for m in prange(M):
            for q2i in xrange(q2ss):     #  unrolling function makes this slower
                IIQ2 = iiq2xs[q2i]
                p_q2_exp_px[m*q2ss + q2i] = q2_intgrl(m, p_f, p_ux, p_px, M, q2ss, Nupx, dSilenceX, IIQ2, q2i)
    #     #  f_exp_px   is M x fss
