cimport cython
import numpy as _N
cimport numpy as _N

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_volrat4(long g_T, long g_M, double[:, :, :, ::1] O, double[::1] trngs, double[:, :, :, :, ::1] volrat, long g_Tf, long g_Mf, double[:, :, :, ::1] O_zoom, double[:, :, :, :, ::1] volrat_zoom):
    #  changes in rescaled-time direction is abrupt, while over marks may not be so abrupt.  Cut box in mark direction in 4 
    cdef double tL, tH
    cdef double d1h, d2h, d3h, d4h, d1l, d2l, d3l, d4l
    cdef long ti, inside, outside, border
    cdef long m1, m2, m3, m4
    cdef double dtf          = (trngs[1] - trngs[0]) / g_Tf
    cdef double fg_Mf = float(g_Mf)
    cdef double fg_Tf = float(g_Tf)

    cdef double *p_O     = &O[0, 0, 0, 0]
    cdef double *p_trngs     = &trngs[0]
    cdef double *p_volrat = &volrat[0, 0, 0, 0, 0]

    cdef long it, inboundary, i_here, i_til_end

    cdef long g_M3 = g_M*g_M*g_M
    cdef long g_M2 = g_M*g_M

    cdef long    g_Tm1 = g_T-1
    cdef long g_Mm1xTm1 = (g_M-1) * g_Tm1
    cdef long g_Mm1_2xTm1 = (g_M-1) * g_Mm1xTm1
    cdef long g_Mm1_3xTm1 = (g_M-1) * g_Mm1_2xTm1

    inside  = 0
    outside = 0
    border  = 0
    
    cdef long _m1, _m2, _m3
    cdef long _m1p1, _m2p1, _m3p1, m4p1

    print "before loop in calc_volrat4"
    tH    = 0
    print "g_M  %d" % g_M
    print "%.5f" % (tH - p_O[10*g_M*g_M*g_M + 10*g_M*g_M + 10*g_M + 10])
    print "%.5f" % (tH - O[10, 10, 10, 10])

    #for m1 in xrange(g_M-1):
    for m1 in xrange(10, 12):
        #print "m1  %d" % m1
        _m1 = m1*g_M3
        _m1p1 = (m1+1)*g_M3
        for m2 in xrange(10, 12):
        #for m2 in xrange(g_M-1):
            _m2 = m2*g_M2
            _m2p1 = (m2+1)*g_M2
            for m3 in xrange(10, 12):
            #for m3 in xrange(g_M-1):
                _m3 = m3*g_M
                _m3p1 = (m3+1)*g_M
                for m4 in xrange(10, 12):
                #for m4 in xrange(g_M-1):
                    m4p1 = m4+1
                    inboundary = 1

                    it = 0
                    tL = p_trngs[0]
                    tH = p_trngs[1]

                    d01h = tH - p_O[_m1 + _m2 + _m3+ m4] 
                    #  1   
                    d11h = tH - p_O[_m1p1 + _m2 + _m3+ m4] 
                    d12h = tH - p_O[_m1 + _m2p1 + _m3+ m4] 
                    d13h = tH - p_O[_m1 + _m2 + _m3p1+ m4] 
                    d14h = tH - p_O[_m1 + _m2 + _m3+ m4p1] 

                    #  2   
                    d21h = tH - p_O[_m1p1 + _m2p1 + _m3+ m4]     # 1 2
                    d22h = tH - p_O[_m1p1 + _m2 + _m3p1+ m4]     # 1 3
                    d23h = tH - p_O[_m1p1 + _m2 + _m3+ m4p1]     # 1 4
                    d24h = tH - p_O[_m1 + _m2p1 + _m3p1+ m4]     # 2 3
                    d25h = tH - p_O[_m1 + _m2p1 + _m3+ m4p1]     # 2 4
                    d26h = tH - p_O[_m1 + _m2 + _m3p1+ m4p1]     # 3 4

                    # 3  
                    d31h = tH - p_O[_m1p1 + _m2p1 + _m3p1+ m4]   # 1 2 3
                    d32h = tH - p_O[_m1p1 + _m2p1 + _m3+ m4p1]   # 1 2 4
                    d33h = tH - p_O[_m1p1 + _m2 + _m3p1+ m4p1]   # 1 3 4
                    d34h = tH - p_O[_m1 + _m2p1 + _m3p1+ m4p1]   # 2 3 4

                    # 4
                    d41h = tH - p_O[_m1p1 + _m2p1 + _m3p1+ m4p1] # 1 2 3 4

                    ###################################3
                    d01l = p_O[_m1 + _m2 + _m3+ m4] - tL

                    #  1   
                    d11l = p_O[_m1p1 + _m2 + _m3+ m4] - tL
                    d12l = p_O[_m1 + _m2p1 + _m3+ m4] - tL
                    d13l = p_O[_m1 + _m2 + _m3p1+ m4] - tL
                    d14l = p_O[_m1 + _m2 + _m3+ m4p1] - tL

                    #  2   
                    d21l = p_O[_m1p1 + _m2p1 + _m3+ m4]- tL     # 1 2
                    d22l = p_O[_m1p1 + _m2 + _m3p1+ m4]- tL     # 1 3
                    d23l = p_O[_m1p1 + _m2 + _m3+ m4p1]- tL     # 1 4
                    d24l = p_O[_m1 + _m2p1 + _m3p1+ m4]- tL     # 2 3
                    d25l = p_O[_m1 + _m2p1 + _m3+ m4p1]- tL     # 2 4
                    d26l = p_O[_m1 + _m2 + _m3p1+ m4p1]- tL     # 3 4

                    # 3  
                    d31l = p_O[_m1p1 + _m2p1 + _m3p1+ m4] - tL   # 1 2 3
                    d32l = p_O[_m1p1 + _m2p1 + _m3+ m4p1] - tL   # 1 2 4
                    d33l = p_O[_m1p1 + _m2 + _m3p1+ m4p1] - tL   # 1 3 4
                    d34l = p_O[_m1 + _m2p1 + _m3p1+ m4p1] - tL   # 2 3 4

                    # 4
                    d41l = p_O[_m1p1 + _m2p1 + _m3p1+ m4p1] - tL   # 1 2 3 4






                    if (((d01h > 0) or \
                         (d11h > 0) or (d12h > 0) or (d13h > 0) or (d14h > 0) or \
                         (d21h > 0) or (d22h > 0) or (d23h > 0) or (d24h > 0) or (d25h > 0) or (d26h > 0) or \
                         (d31h > 0) or (d32h > 0) or (d33h > 0) or (d34h > 0) or \
                         (d41h > 0)) and
                        ((d01l > 0) or \
                         (d11l > 0) or (d12l > 0) or (d13l > 0) or (d14l > 0) or \
                         (d21l > 0) or (d22l > 0) or (d23l > 0) or (d24l > 0) or (d25l > 0) or (d26l > 0) or \
                         (d31l > 0) or (d32l > 0) or (d33l > 0) or (d34l > 0) or \
                         (d41l > 0))):

                        print "%(th).4f     %(0).3f %(1).3f %(2).3f    %(3).3f %(4).3f %(5).3f    %(6).3f %(7).3f %(8).3f   %(9).3f %(10).3f %(11).3f   %(12).3f %(13).3f %(14).3f" % {"0" : d01h, "1" : d11h, "2" : d12h, "3" : d13h, "4" : d14h, "5" : d21h, "6" : d22h, "7" : d23h, "8" : d24h, "9" : d25h, "10" : d26h, "11" : d31h, "12" : d32h, "13" : d33h, "14" : d34h, "15" : d41h, "th" : tH}

                        print "cond 1"
                    else:  #  not a border
                        print "else"
                        if (d01h < 0) and \
                           (d11h<0) and (d12h<0) and (d13h<0) and (d14h<0) and \
                           (d21h<0) and (d22h<0) and (d23h<0) and (d24h<0) and (d25h<0) and (d26h<0) and \
                             (d31h<0) and (d32h<0) and (d33h<0) and (d34h<0) and \
                             (d41h < 0):
                            print "cond 2"
                        else:
                            print "cond 3"

                    #print d01h
