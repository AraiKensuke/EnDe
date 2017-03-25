cimport cython
from cython.parallel import parallel, prange
from libc.stdlib cimport abort, malloc, free
from libc.math cimport exp
import numpy as _N
cimport numpy as _N

cdef Py_ssize_t idx, i, n = 100

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double q2_intgrl(double p_fm, double *p_ux, double *p_px, int Nupx, double dSilenceX, double IIQ2) nogil:
    """
    integral is a Gaussian (at cluster center, width being sampled) weighted with occupation
    """
    cdef double dd, tot, hlfIIQ2

    hlfIIQ2 = -0.5*IIQ2   #  half inverse q2
    tot = 0.0

    for 0 <= n < Nupx:
        dd = p_fm-p_ux[n]
        tot += exp(dd*dd*hlfIIQ2)*p_px[n]
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
    cdef int q2i, m, n, mq2ss
    cdef double dd, IIQ2

    cdef double *p_f   = &f[0]
    cdef double *p_px  = &px[0]
    cdef double *p_ux  = &ux[0]
    cdef double *p_q2_exp_px = &q2_exp_px[0, 0]

    ##  maybe go several ierations - sample q2_exp_px roughly first.
    ##                             then 
    with nogil, parallel(num_threads=nthrds):
        for m in prange(M):
            mq2ss = m*q2ss
            for 0 <= q2i < q2ss:
                IIQ2 = iiq2xs[q2i]
                p_q2_exp_px[mq2ss + q2i] = q2_intgrl(p_f[m], p_ux, p_px, Nupx, dSilenceX, IIQ2)
    #     #  f_exp_px   is M x fss
