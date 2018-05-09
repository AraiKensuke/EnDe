import cython
import numpy as _N
cimport numpy as _N
from libc.stdio cimport printf
from libc.math cimport sqrt


@cython.boundscheck(False)
@cython.wraparound(False)
def calc_volrat(long g_T, long[::1] g_Ms, double[::1] O, double[::1] trngs, double[::1] volrat_mk):
    #  changes in rescaled-time direction is abrupt, while over marks may not be so abrupt.  Cut box in mark direction in 4 
    cdef double tL, tH
    cdef double d1h, d1l
    cdef long ti, inside, outside, border
    cdef long m1

    cdef long   *p_g_Ms  = &g_Ms[0]
    cdef double *p_O     = &O[0]
    cdef double *p_trngs     = &trngs[0]
    cdef double *p_volrat_mk = &volrat_mk[0]

    cdef long it, inboundary, i_here, i_til_end

    cdef long    g_Tm1 = g_T-1
    cdef long g_Mm1 = g_Ms[0]-1

    inside  = 0
    outside = 0
    border  = 0
    cdef double p01, p11, p12, p13, p14, p21, p22, p23, p24, p25, p26, p31, p32, p33, p34, p41

    
    cdef long _m1, _m2, _m3
    cdef long _m1p1, _m2p1, _m3p1, m4p1

    cdef int i_start_search 

    for m1 in xrange(p_g_Ms[0]-1):
        #print "m1  %d" % m1

        inboundary = 1

        it = -1

        while (it < g_T-2) and (inboundary == 1):
            it += 1

            tL = p_trngs[it]
            tH = p_trngs[it+1]

            d1h = tH - p_O[m1]
            d2h = tH - p_O[m1+1]

            d1l = p_O[m1] - tL
            d2l = p_O[m1+1] - tL
                        

            if (((d1h > 0) or (d2h > 0)) and
                ((d1l > 0) or (d2l > 0))):
                tmp = 0.5*(d1l + d2l) / (tH - tL)
                tmp = 0 if (tmp < 0) else tmp
                tmp = 1 if (tmp > 1) else tmp
                p_volrat_mk[m1] += tmp
            else:  #  not a border
                if ((d1h < 0) and (d2h < 0)):
                    p_volrat_mk[m1] += 1

    return inside, outside, border


def find_Occ(long[::1] g_Ms, int NT, double[::1] attimes, double[::1] occ, double[::1] O):
    #  at given rescaled time, number of mark voxels < boundary (rescaled time)
    cdef double maxt, att
    cdef int inboundary, i, j, k, l, it
    cdef double *p_attimes = &attimes[0]
    cdef double *p_occ    = &occ[0]
    cdef double *p_O      = &O[0]

    for i in xrange(g_Ms[0]):
        inboundary = 1
        it = -1
        while inboundary and (it < NT-1):
            it += 1
            att = p_attimes[it]

            if p_O[i] >= att:
                p_occ[it] += 1.
            else:
                inboundary = 0
                    
