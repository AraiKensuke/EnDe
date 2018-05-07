
def calc_volrat2(int g_T, int g_M, double[:, ::1] O, double[::1] trngs, double[:, :, ::1] volrat, int g_Tf, int g_Mf, double[:, ::1] O_zoom, double[:, :, ::1] volrat_zoom):
    cdef double tL, tH
    cdef double d1h, d2h, d3h, d4h, d1l, d2l, d3l, d4l
    cdef int ti, inside, outside, border
    cdef int m1, m2
    cdef double dtf          = (trngs[1] - trngs[0]) / g_Tf
    cdef double fg_Mf = float(g_Mf)
    cdef double fg_Tf = float(g_Tf)

    cdef double *p_O     = &O[0, 0]
    cdef double *p_trngs     = &trngs[0]
    cdef double *p_volrat = &volrat[0, 0, 0]

    cdef int it, inboundary, i_here, i_til_end

    inside  = 0
    outside = 0
    border  = 0

    cdef int it_Start = 0     #  doesn't make sense to start from -1 for every new mark, because the 
    for m1 in xrange(g_M-1):
        for m2 in xrange(g_M-1):
            inboundary = 1
            #for itf in xrange(g_Tf-1):
            it = -1
            while (it < g_T-2) and (inboundary == 1):
                it += 1
                tL = p_trngs[it]
                tH = p_trngs[it+1]

                d1h = tH - p_O[m1*g_M + m2] 
                d2h = tH - p_O[(m1+1)*g_M+m2] 
                d3h = tH - p_O[m1*g_M+m2+1] 
                d4h = tH - p_O[(m1+1)*g_M + m2+1]
                d1l = p_O[m1*g_M+ m2] - tL
                d2l = p_O[(m1+1)*g_M+ m2] - tL
                d3l = p_O[m1*g_M+ m2+1] - tL
                d4l = p_O[(m1+1)*g_M+ m2+1] - tL

                if (((d1h > 0) or (d2h > 0) or \
                     (d3h > 0) or (d4h > 0)) and \
                    ((d1l > 0) or (d2l > 0) or \
                     (d3l > 0) or (d4l > 0))):
                    border += 1

                    p_volrat[m1*(g_M-1)*(g_T-1)+ m2*(g_T-1) + it] = calc_fine_volrat2(O, g_M, g_Mf, g_Tf, fg_Mf, fg_Tf, m1, m2, tL, dtf, O_zoom, volrat_zoom)
                else:  #  not a border
                    if ((d1h < 0) and (d2h < 0) and \
                        (d3h < 0) and (d4h < 0)):
                        p_volrat[m1*(g_M-1)*(g_T-1) + m2*(g_T-1) + it] = 1
                    else:
                        p_volrat[m1*(g_M-1)*(g_T-1) + m2*(g_T-1) + it] = 0
                        inboundary = 0
                        i_here = m1*(g_M-1)*(g_T-1) + m2*(g_T-1)
                        for i_til_end in xrange(it+1, g_T-1):
                            p_volrat[i_here + i_til_end] = 0

        print "%(t)d    %(in)d   %(out)d   %(bord)d" % {"t" : it, "in" : inside, "out" : outside, "bord" : border}

    return inside, outside, border





@cython.boundscheck(False)
@cython.wraparound(False)
def calc_fine_volrat2(double[:, ::1] O, int g_M, int g_Mf, int g_Tf, double fg_Mf, double fg_Tf, int m1, int m2, double t, double dtf, double[:, ::1] O_z, double[:, :, ::1] vlr_z):
    #  changes in rescaled-time direction is abrupt, while over marks may not be so abrupt.  Cut box in mark direction in 4 
    
    cdef double sm = 0.01
    #  assumption O[m1+1, m2+1] = O[m1, m2] + dO_m1
    cdef double dO_m1 = O[m1+1, m2] - O[m1, m2]
    cdef double dO_m2 = O[m1, m2+1] - O[m1, m2]

    cdef int im1f, im2f, itf, i_til_end, i_here

    cdef double *p_O     = &O[0, 0]
    cdef double *p_O_z   = &O_z[0, 0]
    cdef double *p_vlr_z = &vlr_z[0, 0, 0]
    cdef double idtf     = 1./ dtf

    #  make a finer grid for O_z

    cdef int m1m2 = g_M*m1 + m2
    cdef double tL, tH
    cdef double d1h, d2h, d3h, d4h, d1l, d2l, d3l, d4l

    cdef double ifg_Mfm1 = 1./(fg_Mf-1)
    cdef int    g_Mfm1 = g_Mf-1
    cdef int    g_Tfm1 = g_Tf-1

    cdef int inboundary 

    for im1f in xrange(g_Mf):
        for im2f in xrange(g_Mf):
            p_O_z[im1f*g_Mf + im2f] = p_O[m1m2] + im1f*ifg_Mfm1*dO_m1 + im2f*ifg_Mfm1*dO_m2

    for im1f in xrange(g_Mf-1):
        for im2f in xrange(g_Mf-1):
            inboundary = 1
            #for itf in xrange(g_Tf-1):
            itf = -1
            while (itf < g_Tfm1-1) and (inboundary == 1):
                itf += 1
                tL = t + itf * dtf
                tH = t + (itf+1) * dtf 

                d1h = tH - p_O_z[im1f*g_Mf + im2f] 
                d2h = tH - p_O_z[(im1f+1)*g_Mf+im2f] 
                d3h = tH - p_O_z[(im1f)*g_Mf+im2f+1] 
                d4h = tH - p_O_z[(im1f+1)*g_Mf + im2f+1]
                d1l = p_O_z[im1f*g_Mf+ im2f] - tL
                d2l = p_O_z[(im1f+1)*g_Mf+ im2f] - tL
                d3l = p_O_z[im1f*g_Mf+ im2f+1] - tL
                d4l = p_O_z[(im1f+1)*g_Mf+ im2f+1] - tL

                if (((d1h > 0) or (d2h > 0) or \
                     (d3h > 0) or (d4h > 0)) and \
                    ((d1l > 0) or (d2l > 0) or \
                     (d3l > 0) or (d4l > 0))):
                    #  a border
                    r1h = sm if (d1h<= 0) else (1 if (d1h>dtf) else d1h*idtf)
                    r2h = sm if (d2h<= 0) else (1 if (d2h>dtf) else d2h*idtf)
                    r3h = sm if (d3h<= 0) else (1 if (d3h>dtf) else d3h*idtf)
                    r4h = sm if (d4h<= 0) else (1 if (d4h>dtf) else d4h*idtf)

                    p_vlr_z[im1f*g_Mfm1*g_Tfm1+ im2f*g_Tfm1+ itf] = r1h*r2h*r3h*r4h
                else:  #  not a border
                    if ((d1h < 0) and (d2h < 0) and \
                        (d3h < 0) and (d4h < 0)):
                        p_vlr_z[im1f*g_Mfm1*g_Tfm1 + im2f*g_Tfm1 + itf] = 1
                    else:
                        p_vlr_z[im1f*g_Mfm1*g_Tfm1 + im2f*g_Tfm1 + itf] = 0
                        inboundary = 0
                        i_here = im1f*g_Mfm1*g_Tfm1 + im2f*g_Tfm1
                        for i_til_end in xrange(itf+1, g_Tfm1):
                            p_vlr_z[i_here + i_til_end] = 0

    return _N.mean(vlr_z)

##############################################################
########################  4d marks
##############################################################
