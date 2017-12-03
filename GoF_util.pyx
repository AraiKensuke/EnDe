cimport cython
import numpy as _N
cimport numpy as _N

#############################   2-D marks
def calc_volrat2(int g_T, int g_M, double[:, ::1] O, double[::1] trngs, double[:, :, ::1] volrat, int g_Mf, int g_Tf, double[:, ::1] O_zoom, double[:, :, ::1] volrat_zoom):
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

    for m1 in xrange(g_M-1):
        for m2 in xrange(g_M-1):
            inboundary = 1
            #for itf in xrange(g_Tf-1):
            it = -1
            while (it < g_T-1) and (inboundary == 1):
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
#def calc_fine_volrat2(double[:, ::1] O, int g_M, int g_Mf, int g_Tf, double fg_Mf, double fg_Tf, int m1, int m2, double t, double dtf, double[:, ::1] O_z, double[:, :, ::1] vlr_z):
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
            while (itf < g_Tfm1) and (inboundary == 1):
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

########################  4d marks





# @cython.boundscheck(False)
# @cython.wraparound(False)
# def calc_volrat4(double[:, :, :, ::1] O, int g_M, int g_Mf, int g_Tf, double fg_Mf, double fg_Tf, int m1, int m2, int m3, int m4, double t, double dtf, double[:, :, :, ::1] O_z, double[:, :, :, :, ::1] vlr_z):
# #def calc_volrat(O, g_M, g_Mf, g_Tf, fg_Mf, fg_Tf, m1, m2, t, dtf, O_z, vlr_z):
#     #  changes in rescaled-time direction is abrupt, while over marks may not be so abrupt.  Cut box in mark direction in 4 

#     #  assumption O[m1+1, m2+1] = O[m1, m2] + dO_m1
#     cdef double dO_m1 = O[m1+1, m2, m3, m4] - O[m1, m2, m3, m4]
#     cdef double dO_m2 = O[m1, m2+1, m3, m4] - O[m1, m2, m3, m4]
#     cdef double dO_m2 = O[m1, m2+1, m3, m4] - O[m1, m2, m3, m4]
#     cdef double dO_m2 = O[m1, m2+1, m3, m4] - O[m1, m2, m3, m4]

#     cdef int im1f, im2f, im3f, im4f, itf, i_til_end, i_here

#     cdef double *p_O     = &O[0, 0, 0, 0]
#     cdef double *p_O_z   = &O_z[0, 0, 0, 0]
#     cdef double *p_vlr_z = &vlr_z[0, 0, 0, 0, 0]
#     cdef double idtf     = 1./ dtf

#     #  make a finer grid for O_z

#     cdef int m1m2m3m4 = g_M*g_M*g_M*m1 + g_M*g_M*m2 + g_M*m3 + m4
#     cdef double tL, tH
#     cdef double d1h, d2h, d3h, d4h, d1l, d2l, d3l, d4l

#     cdef double ifg_Mfm1 = 1./(fg_Mf-1)
#     cdef int    g_Mfm1 = g_Mf-1
#     cdef int    g_Tfm1 = g_Tf-1
#     cdef int g_Mf3 = g_Mf*g_Mf*g_Mf
#     cdef int g_Mf2 = g_Mf*g_Mf

#     cdef int inboundary 

#     for im1f in xrange(g_Mf):
#         for im2f in xrange(g_Mf):
#             p_O_z[im1f*g_Mf + im2f] = p_O[m1m2] + im1f*ifg_Mfm1*dO_m1 + im2f*ifg_Mfm1*dO_m2

#     for im1f in xrange(g_Mf-1):
#         for im2f in xrange(g_Mf-1):
#             inboundary = 1
#             #for itf in xrange(g_Tf-1):
#             itf = -1
#             while (itf < g_Tfm1) and (inboundary == 1):
#                 itf += 1
#                 tL = t + itf * dtf
#                 tH = t + (itf+1) * dtf 

#                 d01h = tH - O_z[im1f,   im2f, im3f, im4f] 

#                 #  1
#                 d11h = tH - O_z[im1f+1, im2f, im3f, im4f] 
#                 d12h = tH - O_z[im1f, im2f+1, im3f, im4f] 
#                 d13h = tH - O_z[im1f, im2f, im3f+1, im4f] 
#                 d14h = tH - O_z[im1f, im2f, im3f, im4f+1] 

#                 #  2
#                 d21h = tH - O_z[im1f+1, im2f+1, im3f, im4f] 
#                 d22h = tH - O_z[im1f+1, im2f, im3f+1, im4f] 
#                 d23h = tH - O_z[im1f+1, im2f, im3f, im4f+1] 

#                 d24h = tH - O_z[im1f, im2f+1, im3f+1, im4f] 
#                 d25h = tH - O_z[im1f, im2f+1, im3f, im4f+1] 

#                 d26h = tH - O_z[im1f, im2f, im3f+1, im4f+1] 

#                 # 3
#                 d31h = tH - O_z[im1f+1,   im2f+1, im3f+1, im4f] 
#                 d32h = tH - O_z[im1f+1,   im2f+1, im3f, im4f+1] 
#                 d33h = tH - O_z[im1f+1,   im2f, im3f+1, im4f+1] 
#                 d34h = tH - O_z[im1f,   im2f+1, im3f+1, im4f+1] 

#                 # 4
#                 d41h = tH - O_z[im1f+1, im2f+1, im3f+1, im4f+1] 

                
#                 ###############################
#                 d01l = O_z[im1f,   im2f, im3f, im4f]  - tL

#                 #  1
#                 d11l = O_z[im1f+1, im2f, im3f, im4f]   - tL
#                 d12l = O_z[im1f, im2f+1, im3f, im4f]   - tL
#                 d13l = O_z[im1f, im2f, im3f+1, im4f]   - tL
#                 d14l = O_z[im1f, im2f, im3f, im4f+1]   - tL

#                 #  2
#                 d21l = O_z[im1f+1, im2f+1, im3f, im4f]   - tL
#                 d22l = O_z[im1f+1, im2f, im3f+1, im4f]   - tL
#                 d23l = O_z[im1f+1, im2f, im3f, im4f+1]   - tL
#                 d24l = O_z[im1f, im2f+1, im3f+1, im4f]   - tL
#                 d25l = O_z[im1f, im2f+1, im3f, im4f+1]   - tL
#                 d26l = O_z[im1f, im2f, im3f+1, im4f+1]   - tL

#                 # 3
#                 d31l = O_z[im1f+1,   im2f+1, im3f+1, im4f]   - tL
#                 d32l = O_z[im1f+1,   im2f+1, im3f, im4f+1]   - tL
#                 d33l = O_z[im1f+1,   im2f, im3f+1, im4f+1]   - tL
#                 d34l = O_z[im1f,   im2f+1, im3f+1, im4f+1]   - tL

#                 # 4
#                 d41l = O_z[im1f+1, im2f+1, im3f+1, im4f+1]   - tL

#                 if (((d01h > 0) or \
#                      (d11h > 0) or (d12h > 0) or (d13h > 0) or (d14h > 0) or \
#                      (d21h > 0) or (d22h > 0) or (d23h > 0) or (d24h > 0) or (d25h > 0) or (d26h > 0) or \
#                      (d31h > 0) or (d32h > 0) or (d33h > 0) or (d34h > 0) or \
#                      (d41h)) and
#                     ((d01l > 0) or \
#                      (d11l > 0) or (d12l > 0) or (d13l > 0) or (d14l > 0) or \
#                      (d21l > 0) or (d22l > 0) or (d23l > 0) or (d24l > 0) or (d25l > 0) or (d26l > 0) or \
#                      (d31l > 0) or (d32l > 0) or (d33l > 0) or (d34l > 0) or \
#                      (d41l))):
#                     #  a border
#                     r01h = sm if (d01h<=0) else (1 if (d01h>dtf) else d01h*idtf)

#                     r11h = sm if (d11h<=0) else (1 if (d11h>dtf) else d11h*idtf)
#                     r12h = sm if (d12h<=0) else (1 if (d12h>dtf) else d12h*idtf)
#                     r13h = sm if (d13h<=0) else (1 if (d13h>dtf) else d13h*idtf)
#                     r14h = sm if (d14h<=0) else (1 if (d14h>dtf) else d14h*idtf)

#                     r21h = sm if (d21h<=0) else (1 if (d21h>dtf) else d21h*idtf)
#                     r22h = sm if (d22h<=0) else (1 if (d22h>dtf) else d22h*idtf)
#                     r23h = sm if (d23h<=0) else (1 if (d23h>dtf) else d23h*idtf)
#                     r24h = sm if (d24h<=0) else (1 if (d24h>dtf) else d24h*idtf)
#                     r25h = sm if (d25h<=0) else (1 if (d25h>dtf) else d25h*idtf)
#                     r26h = sm if (d26h<=0) else (1 if (d26h>dtf) else d26h*idtf)

#                     r31h = sm if (d31h<=0) else (1 if (d31h>dtf) else d31h*idtf)
#                     r32h = sm if (d32h<=0) else (1 if (d32h>dtf) else d32h*idtf)
#                     r33h = sm if (d33h<=0) else (1 if (d33h>dtf) else d33h*idtf)
#                     r34h = sm if (d34h<=0) else (1 if (d34h>dtf) else d34h*idtf)

#                     r41h = sm if (d41h<=0) else (1 if (d41h>dtf) else d41h*idtf)

#                     p_vlr_z[im1f*g_Mfm1*g_Mfm1*g_Mfm1*g_Tfm1 + im2f*g_Mfm1*g_Mfm1*g_Tfm1 + im3f*g_Mfm1*g_Tfm1 + im4f*g_Tfm1 + itf] = r01h * r11h*r12h*r13h*r14h * r21h*r22h*r23h*r24h*r25h*r26h * r31h*r32h*r33h*r34h + r41h
#                 else:  #  not a border
#                     if (d01h < 0) and \
#                        (d11h<0) and (d12h<0) and (d13h<0) and (d14h<0) and \
#                        (d21h<0) and (d22h<0) and (d23h<0) and (d24h<0) and (d25h<0) and (d26h<0) and \
#                          (d31h<0) and (d32h<0) and (d33h<0) and (d34h<0) and \
#                          (d41h < 0):
#                     p_vlr_z[im1f*g_Mfm1*g_Mfm1*g_Mfm1*g_Tfm1 + im2f*g_Mfm1*g_Mfm1*g_Tfm1 + im3f*g_Mfm1*g_Tfm1 + im4f*g_Tfm1 + itf] = 1
#                     else:
#                     p_vlr_z[im1f*g_Mfm1*g_Mfm1*g_Mfm1*g_Tfm1 + im2f*g_Mfm1*g_Mfm1*g_Tfm1 + im3f*g_Mfm1*g_Tfm1 + im4f*g_Tfm1 + itf] = 0
#                         inboundary = 0
#                         i_here = im1f*g_Mfm1*g_Tfm1 + im2f*g_Tfm1
#                         for i_til_end in xrange(itf+1, g_Tfm1):
#                             p_vlr_z[i_here + i_til_end] = 0

#     return _N.mean(vlr_z)
