cimport cython
from cython.parallel import parallel, prange
from libc.stdlib cimport abort, malloc, free
from libc.math cimport exp
import numpy as _N

cdef Py_ssize_t idx, i, n = 100

@cython.boundscheck(False)
@cython.wraparound(False)
def M_times_N_intrgrls(double [:, :] fxs, double[:] ux, double [:] iiq2, double dSilenceX, double [:] px, double [:, :] f_exp_px, int M, int fss, int Nupx, int nthrds):
    #  fxs       M x fss   
    #  fxrux     Nupx    
    #  f_intgrd  Nupx
    cdef int fi, m, n
    cdef double dd, IIQ2
    with nogil, parallel(num_threads=nthrds):
        for m in prange(M, schedule="static"):
            IIQ2 = iiq2[m]
            for fi in xrange(fss):
                f_exp_px[m, fi] = 0
                for n in xrange(Nupx):
                    dd = fxs[m, fi]-ux[n]
                    f_exp_px[m, fi] += exp(-0.5*dd*dd*IIQ2)*px[n]
                f_exp_px[m, fi] *= dSilenceX
            #  f_exp_px   is M x fss

    
