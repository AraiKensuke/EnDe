import numpy as _N
cimport numpy as _N



def calc_volrat(double[:, ::1] O, int g_M, int g_Mf, int g_Tf, double fg_Mf, double fg_Tf, int m1, int m2, double t, double dtf, double[:, ::1] O_z, double[:, :, ::1] vlr_z):
#def calc_volrat(O, g_M, g_Mf, g_Tf, fg_Mf, fg_Tf, m1, m2, t, dtf, O_z, vlr_z):
    #  changes in rescaled-time direction is abrupt, while over marks may not be so abrupt.  Cut box in mark direction in 4 

    #  assumption O[m1+1, m2+1] = O[m1, m2] + dO_m1
    dO_m1 = O[m1+1, m2] - O[m1, m2]
    dO_m2 = O[m1, m2+1] - O[m1, m2]

    #  make a finer grid for O_z
    for im1f in xrange(g_Mf):
        for im2f in xrange(g_Mf):
            O_z[im1f, im2f] = O[m1, m2] + (im1f/(fg_Mf-1))*dO_m1 + (im2f/(fg_Mf-1))*dO_m2
    #O_z[g_Mf-1, g_Mf-1] = O[m1+1, m2+1]

    for im1f in xrange(g_Mf-1):
        for im2f in xrange(g_Mf-1):
            for itf in xrange(g_Tf-1):
                tL = t + itf * dtf
                tH = t + (itf+1) * dtf 

                d1h = tH - O_z[im1f, im2f] 
                d2h = tH - O_z[im1f+1, im2f] 
                d3h = tH - O_z[im1f, im2f+1] 
                d4h = tH - O_z[im1f+1, im2f+1]
                d1l = O_z[im1f, im2f] - tL
                d2l = O_z[im1f+1, im2f] - tL
                d3l = O_z[im1f, im2f+1] - tL
                d4l = O_z[im1f+1, im2f+1] - tL

                if (((d1h > 0) or (d2h > 0) or \
                     (d3h > 0) or (d4h > 0)) and \
                    ((d1l > 0) or (d2l > 0) or \
                     (d3l > 0) or (d4l > 0))):
                    #  a border
                    if d1h > 0:
                        r1h = 1 if (d1h > dtf) else d1h / dtf
                    else:
                        r1h = 0.01  #  don't set to 0
                    if d2h > 0:
                        r2h = 1 if (d2h > dtf) else d2h / dtf
                    else:
                        r2h = 0.01 #  don't set to 0
                    if d3h > 0:
                        r3h = 1 if (d3h > dtf) else d3h / dtf
                    else:
                        r3h = 0.01  #  don't set to 0
                    if d4h > 0:
                        r4h = 1 if (d4h > dtf) else d4h / dtf
                    else:
                        r4h = 0.01  #  don't set to 0


                    vlr_z[im1f, im2f, itf] = r1h*r2h*r3h*r4h
                else:  #  not a border
                    if ((d1h < 0) and (d2h < 0) and \
                        (d3h < 0) and (d4h < 0)):
                        vlr_z[im1f, im2f, itf] = 1
                    else:
                        vlr_z[im1f, im2f, itf] = 0

                # if (((O_z[im1f, im2f] < tH) or (O_z[im1f+1, im2f] < tH) or \
                #     (O_z[im1f, im2f+1] < tH) or (O_z[im1f+1, im2f+1] < tH)) and \
                #     ((O_z[im1f, im2f] > tL) or (O_z[im1f+1, im2f] > tL) or \
                #      (O_z[im1f, im2f+1] > tL) or (O_z[im1f+1, im2f+1] > tL))):
                #     #  a border
                #     vlr_z[im1f, im2f, itf] = 
                # else:  #  not a border
                #     if ((O_z[im1f, im2f] > tH) and (O_z[im1f+1, im2f] > tH) and \
                #         (O_z[im1f, im2f+1] > tH) and (O_z[im1f+1, im2f+1] > tH)):
                #         vlr_z[im1f, im2f, itf] = 1
                #     else:
                #         vlr_z[im1f, im2f, itf] = 0

    return _N.mean(vlr_z)




        
        

"""
def calc_volrat(double[:, ::1] O, int g_M, int g_Mf, int g_Tf, double fg_Mf, double fg_Tf, int m1, int m2, double t, double dtf, double[:, ::1] O_z, double[:, :, ::1] vlr_z):
    #  changes in rescaled-time direction is abrupt, while over marks may not be so abrupt.  Cut box in mark direction in 4 

    #  assumption O[m1+1, m2+1] = O[m1, m2] + dO_m1
    cdef double dO_m1 = O[m1+1, m2] - O[m1, m2]
    cdef double dO_m2 = O[m1, m2+1] - O[m1, m2]

    cdef int im1f, im2f, itf

    cdef double *p_O     = &O[0, 0]
    cdef double *p_O_z   = &O_z[0, 0]
    cdef double *p_vlr_z = &vlr_z[0, 0, 0]
    cdef double idtf     = 1./ dtf

    #  make a finer grid for O_z

    cdef int m1m2 = g_M*m1 + m2
    cdef double tL, tH
    cdef double d1h, d2h, d3h, d4h, d1l, d2l, d3l, d4l

    cdef double ifg_Mfm1 = 1./(fg_Mf-1)

    for im1f in xrange(g_Mf):
        for im2f in xrange(g_Mf):
            p_O_z[im1f*g_Mf + im2f] = p_O[m1m2] + (im1f/(fg_Mf-1))*dO_m1 + (im2f/(fg_Mf-1))*dO_m2


    for im1f in xrange(g_Mf-1):
        for im2f in xrange(g_Mf-1):
            for itf in xrange(g_Tf-1):
                tL = t + itf * dtf
                tH = t + (itf+1) * dtf 

                d1h = tH - O_z[im1f, im2f] 
                d2h = tH - O_z[im1f+1, im2f] 
                d3h = tH - O_z[im1f, im2f+1] 
                d4h = tH - O_z[im1f+1, im2f+1]
                d1l = O_z[im1f, im2f] - tL
                d2l = O_z[im1f+1, im2f] - tL
                d3l = O_z[im1f, im2f+1] - tL
                d4l = O_z[im1f+1, im2f+1] - tL

                if (((d1h > 0) or (d2h > 0) or \
                     (d3h > 0) or (d4h > 0)) and \
                    ((d1l > 0) or (d2l > 0) or \
                     (d3l > 0) or (d4l > 0))):
                    #  a border
                    if d1h > 0:
                        r1h = 1 if (d1h > dtf) else d1h / dtf
                    else:
                        r1h = 0.01  #  don't set to 0
                    if d2h > 0:
                        r2h = 1 if (d2h > dtf) else d2h / dtf
                    else:
                        r2h = 0.01 #  don't set to 0
                    if d3h > 0:
                        r3h = 1 if (d3h > dtf) else d3h / dtf
                    else:
                        r3h = 0.01  #  don't set to 0
                    if d4h > 0:
                        r4h = 1 if (d4h > dtf) else d4h / dtf
                    else:
                        r4h = 0.01  #  don't set to 0


                    vlr_z[im1f, im2f, itf] = r1h*r2h*r3h*r4h
                else:  #  not a border
                    if ((d1h < 0) and (d2h < 0) and \
                        (d3h < 0) and (d4h < 0)):
                        vlr_z[im1f, im2f, itf] = 1
                    else:
                        vlr_z[im1f, im2f, itf] = 0

                    # if (((O_z[im1f, im2f] < tH) or (O_z[im1f+1, im2f] < tH) or \
                    #     (O_z[im1f, im2f+1] < tH) or (O_z[im1f+1, im2f+1] < tH)) and \
                    #     ((O_z[im1f, im2f] > tL) or (O_z[im1f+1, im2f] > tL) or \
                    #      (O_z[im1f, im2f+1] > tL) or (O_z[im1f+1, im2f+1] > tL))):
                    #     #  a border
                    #     vlr_z[im1f, im2f, itf] = 
                    # else:  #  not a border
                    #     if ((O_z[im1f, im2f] > tH) and (O_z[im1f+1, im2f] > tH) and \
                    #         (O_z[im1f, im2f+1] > tH) and (O_z[im1f+1, im2f+1] > tH)):
                    #         vlr_z[im1f, im2f, itf] = 1
                    #     else:
                    #         vlr_z[im1f, im2f, itf] = 0

        return _N.mean(vlr_z)

"""                        
                    
        
        
        

    # for im1f in xrange(g_Mf-1):
    #     for im2f in xrange(g_Mf-1):
    #         for itf in xrange(g_Tf-1):
    #             tL = t + itf * dtf
    #             tH = t + (itf+1) * dtf 

    #             d1h = tH - p_O_z[im1f*g_Mf + im2f] 
    #             d2h = tH - p_O_z[(im1f+1)*g_Mf+im2f] 
    #             d3h = tH - p_O_z[(im1f)*g_Mf+im2f+1] 
    #             d4h = tH - p_O_z[(im1f+1)*g_Mf + im2f+1]
    #             d1l = p_O_z[im1f*g_Mf+ im2f] - tL
    #             d2l = p_O_z[(im1f+1)*g_Mf+ im2f] - tL
    #             d3l = p_O_z[im1f*g_Mf+ im2f+1] - tL
    #             d4l = p_O_z[(im1f+1)*g_Mf+ im2f+1] - tL

    #             if (((d1h > 0) or (d2h > 0) or \
    #                  (d3h > 0) or (d4h > 0)) and \
    #                 ((d1l > 0) or (d2l > 0) or \
    #                  (d3l > 0) or (d4l > 0))):
    #                 #  a border
    #                 if d1h > 0:
    #                     r1h = 1 if (d1h > dtf) else d1h * idtf
    #                 else:
    #                     r1h = 0.01  #  don't set to 0
    #                 if d2h > 0:
    #                     r2h = 1 if (d2h > dtf) else d2h * idtf
    #                 else:
    #                     r2h = 0.01 #  don't set to 0
    #                 if d3h > 0:
    #                     r3h = 1 if (d3h > dtf) else d3h * idtf
    #                 else:
    #                     r3h = 0.01  #  don't set to 0
    #                 if d4h > 0:
    #                     r4h = 1 if (d4h > dtf) else d4h * idtf
    #                 else:
    #                     r4h = 0.01  #  don't set to 0


    #                 p_vlr_z[im1f*g_Mf*g_Mf+ im2f*g_Mf+ itf] = r1h*r2h*r3h*r4h
    #             else:  #  not a border
    #                 if ((d1h < 0) and (d2h < 0) and \
    #                     (d3h < 0) and (d4h < 0)):
    #                     p_vlr_z[im1f*g_Mf*g_Mf + im2f*g_Mf + itf] = 1
    #                 else:
    #                     p_vlr_z[im1f*g_Mf*g_Mf + im2f*g_Mf + itf] = 0

    # return _N.mean(vlr_z)






