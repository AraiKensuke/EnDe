cimport cython
import numpy as _N
cimport numpy as _N
from libc.stdio cimport printf
from libc.math cimport sqrt


@cython.boundscheck(False)
@cython.wraparound(False)
def calc_volrat2(long g_T, long[::1] g_Ms, double[:, ::1] O, double[::1] trngs, double[:, ::1] volrat_mk, long g_Tf, long g_Mf):
    #  changes in rescaled-time direction is abrupt, while over marks may not be so abrupt.  Cut box in mark direction in 4 
    cdef double tL, tH
    cdef double d1h, d2h, d3h, d4h, d1l, d2l, d3l, d4l
    cdef long ti, inside, outside, border
    cdef long m1, m2, m3, m4
    cdef double dtf          = (trngs[1] - trngs[0]) / g_Tf
    cdef double fg_Mf = float(g_Mf)
    cdef double fg_Tf = float(g_Tf)

    cdef long   *p_g_Ms  = &g_Ms[0]
    cdef double *p_O     = &O[0, 0]
    cdef double *p_trngs     = &trngs[0]
    cdef double *p_volrat_mk = &volrat_mk[0, 0]

    cdef long it, inboundary, i_here, i_til_end

    cdef long g_M1 = g_Ms[1]


    cdef long    g_Tm1 = g_T-1
    cdef long g_Mm1 = g_Ms[1]-1

    inside  = 0
    outside = 0
    border  = 0
    cdef double p00, p01, p10, p11

    
    cdef long _m1, _m2, _m3
    cdef long _m1p1, _m2p1, _m3p1, m4p1

    cdef int i_start_search 

    partials = []
    for m1 in xrange(p_g_Ms[0]-1):
        #print "m1  %d" % m1
        _m1 = m1*g_M1
        _m1p1 = (m1+1)*g_M1
        for m2 in xrange(p_g_Ms[1]-1):
            _m2 = m2
            _m2p1 = m2+1
            inboundary = 1

            it = -1
            p00 = p_O[_m1 + _m2]

            p10 = p_O[_m1p1 + _m2]
            p01 = p_O[_m1 + _m2p1]

            p11 = p_O[_m1p1 + _m2p1]

            while (it < g_T-2) and (inboundary == 1):
                it += 1

                tL = p_trngs[it]
                tH = p_trngs[it+1]

                d00h = tH - p00

                #  1   
                d10h = tH - p10
                d01h = tH - p01

                #  2   
                d11h = tH - p11

                ###################################
                d00l = p00 - tL

                #  1   
                d10l = p10 - tL
                d01l = p01 - tL

                #  2   
                d11l = p11 - tL

                if (((d00h > 0) or (d10h > 0) or (d01h > 0) or (d11h > 0)) and
                    ((d00l > 0) or (d10l > 0) or (d01l > 0) or (d11l > 0))):
                    #  approximately true?
                    tmp = 0.25*(d00l + d10l + d01l + d11l) / (tH - tL)
                    tmp = 0 if (tmp < 0) else tmp
                    tmp = 1 if (tmp > 1) else tmp
                    p_volrat_mk[m1*g_Mm1 + m2] += tmp
                else:  #  not a border
                    if ((d00h < 0) and (d10h < 0) and (d01h < 0) and (d11h < 0)):
                        p_volrat_mk[m1*g_Mm1 + m2] += 1

    return inside, outside, border, partials
                    

def find_Occ2(long[::1] g_Ms, int NT, double[::1] attimes, double[::1] occ, double[:, ::1] O):
    #  at given rescaled time, number of mark voxels < boundary (rescaled time)
    cdef double maxt, att
    cdef int inboundary, i, j, k, l, it
    cdef double *p_attimes = &attimes[0]
    cdef double *p_occ    = &occ[0]
    cdef double *p_O      = &O[0, 0]

    cdef int ig_M1, g_M1

    g_M1 = g_Ms[1]

    for i in xrange(g_Ms[0]):
        ig_M1 = i*g_M1
        for l in xrange(g_Ms[1]):
            inboundary = 1
            it = -1
            while inboundary and (it < NT-1):
                it += 1
                att = p_attimes[it]

                if p_O[ig_M1 + l] >= att:
                    p_occ[it] += 1.
                else:
                    inboundary = 0
                    
