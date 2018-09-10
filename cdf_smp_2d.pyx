import numpy as _N
cimport numpy as _N
import cython
cimport cython
import time as _tm
import utilities as _U
from libc.math cimport exp, sqrt, log
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
import matplotlib.pyplot as _plt
#uniqFN(filename, serial=False, iStart=1, returnPrevious=False)
import utilities as _U

twoPOW = _N.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072], dtype=_N.int)
cdef long[::1] v_twoPOW = twoPOW
cdef long* p_twoPOW = &v_twoPOW[0]
cdef double x_Lo, x_Hi, y_Lo, y_Hi

cdef int __NRM = 0
cdef int __IG  = 1

cdef double fourpi2 = 39.47841760435743
cdef double twpi    = 6.283185307179586

cdef double[::1] v_cpf2
cdef double[::1] v_cpq22
cdef double[::1] v_fx2
cdef double[::1] v_qx2

#####  variables for changing resolution of occupation function
#  riemann_xys:  sum over path replaced by occupation weighted sum over grid
cdef double *p_riemann_xs
cdef double *p_riemann_ys
cdef long *p_hist_all
cdef double *p_ibinszs_x, *p_ibinszs_y
cdef double *p_q2_thr
cdef int minSmps

py_Nupxs  = None
py_Nupys  = None
py_riemann_xs = None
py_riemann_ys = None
py_ibinszs_x = None
py_ibinszs_y = None
py_hist_all = None

cdef long *Nupxs
cdef long *Nupys
cdef long *p_Nupxys  #  grid # for sum over occ weighted spatial path 

py_arr_xy_0s = None
py_arr_x_0s  = None
py_arr_y_0s  = None
py_q2_th     = None

cdef long *arr_xy_0s
cdef long *arr_x_0s
cdef long *arr_y_0s
cdef long n_res   #  number of different resolutions 
#cdef double x_lo, x_hi

f_STEPS   = None
q2_STEPS  = None
f_cldz    = None
q2_cldz   = None
f_SMALL   = 10
q2_SMALL  = 10


cpf2 = None    # value of cond. post, defined only certain pts
cpq22 = None  

_NRM = 0   #  for outside visibility
_IG  = 1

fx2 = None    # grid where cond posterior function is defined and adptvly smpld
qx2 = None    # grid where cond posterior function is defined and adptvly smpld


f_lo = None
f_hi = None
q2_lo = None
q2_hi = None
dt    = None

adtv_pdf_params = _N.empty(5)

########################################################################
def init(_dt, _f_lo, _f_hi, _q2_lo, _q2_hi, _f_STEPS, _q2_STEPS, _f_SMALL, _q2_SMALL, _f_cldz, _q2_cldz, _minSmps):
    """
    init the grid used to sample the conditional posterior functions
    unless maze is highly rectangular, _f_lo, _f_hi being set same for both 
    x and y direction not a problem
    """
    global fx2, qx2, cpf2, cpq22, 
    global v_fx2, v_qx2, v_cpf2, v_cpq22, 
    global dt
    global fx_lo, fx_hi, fy_lo, fy_hi, q2_lo, q2_hi
    global f_SMALL, q2_SMALL, f_STEPS, q2_STEPS, f_cldz, q2_cldz
    global minSmps

    minSmps   = _minSmps
    f_SMALL   = _f_SMALL
    q2_SMALL  = _q2_SMALL

    f_STEPS   = _f_STEPS
    q2_STEPS  = _q2_STEPS

    f_cldz    = _f_cldz
    q2_cldz   = _q2_cldz

    f_lo      = _f_lo
    f_hi      = _f_hi
    q2_lo     = _q2_lo
    q2_hi     = _q2_hi

    fx2 = _N.linspace(f_lo, f_hi, 2**f_STEPS+1, endpoint=True)
    qx2 = _N.exp(_N.linspace(_N.log(q2_lo), _N.log(q2_hi), 2**q2_STEPS+1, endpoint=True))
    cpf2 = _N.empty(2**f_STEPS+1)
    cpq22= _N.empty(2**f_STEPS+1)

    v_fx2     = fx2
    v_qx2     = qx2
    v_cpf2    = cpf2
    v_cqf22   = cpq22

    dt        = _dt

    #q2_thrs
    #Nupxs
    
########################################################################
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def adtv_smp_cdf_interp(int itr, int forq, int cmpnt, double[::1] x, double[::1] log_p, int N, int m, double[::1] m_rnds):
    """
    each cluster has independent x over which conditional likelihood defined
    xt0t1    relative coordinates
    mks      absolute coordinates

    sLLkPr   spiking part
    s        silence part

    Caveat.  For Inverse-Gamma, if prior
    """
    global minSmps
    cdef int i
    cdef double tot = 0

    cdef double *p_x = &x[0]
    cdef double *dx   = <double*>malloc((N-1)*sizeof(double))
    cdef double *p    = <double*>malloc(N*sizeof(double))
    cdf  = _N.empty(N)
    cdef double[::1] v_cdf = cdf
    cdef double *p_cdf     = &v_cdf[0]
    cdef double *p_log_p   = &log_p[0]

    cdef double retRnd
    cdef double rnd = m_rnds[m]

    cdef int isg2, _isg2

    p_cdf[0]   = 0
    for i in xrange(N):
        p[i] = exp(p_log_p[i])
    for i in xrange(1, N):
        dx[i-1] = p_x[i]-p_x[i-1]
        #p_cdf[i] = p_cdf[i-1] + 0.5*(exp(p_log_p[i-1])+exp(p_log_p[i]))*dx[i-1]#*itot
        p_cdf[i] = p_cdf[i-1] + 0.5*(p[i-1]+p[i])*dx[i-1]#*itot

    cdf /= cdf[N-1]     #  even if U[0,1] rand is 1, we still have some room at the end to add a bit of noise.
    # if (itr > 2720) and (itr < 2745) and (forq == 1) and (cmpnt == 0):
    #     dat = _N.empty((N, 2))
    #     dat[:, 0] = x
    #     dat[:, 1] = cdf
    #     _N.savetxt("q2_cdf_%d" % itr, dat, fmt="%.4e %.4e")

    #  btwn cdf[isg2] and cdf[isg2+1]
    #  (rnds[m,0] - cdf[isg2]) * (cdf[isg2+1] - cdf[isg2]) * d_sg2s[isg2]
    #_isg2 = _N.searchsorted(cdf, rnd)
    _isg2 = cdf.searchsorted(rnd)
    isg2  = _isg2-1

    retRnd = p_x[isg2] + ((rnd - p_cdf[isg2]) / (p_cdf[isg2+1] - p_cdf[isg2])) * dx[isg2]  # unlike in above case, retRnd may be < 0
    free(dx)
    free(p)
    return retRnd

########################################################################
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def smp_f(int itr, int cmpnt, int M, long[::1] clstsz, long[::1] cls_strt_inds, long[::1] sts, 
          double[::1] xt0t1, double[::1] yt0t1, int t0, double[::1] fx, double[::1] fy, 
          double[::1] q2x, double[::1] q2y, double[::1] l0, 
          double[::1] _fx_u, double[::1] _fx_q2, 
          double[::1] _fy_u, double[::1] _fy_q2, double[::1] m_rands):
    """
    f_smv is the component of being sampled, f_fx is the fixed component
    _f_sm_u are the prior hyperparameters for f_xmv
    """
    global f_STEPS, f_SMALL, f_cldz, fx2, cpf2
    """
    f     parameter f
    _f_u  
    """
    cdef int m
    cdef double tmp
    cdef double[::1]     v_adtv_pdf_params = adtv_pdf_params
    cdef double* p_adtv_pdf_params = &v_adtv_pdf_params[0]

    cdef double* _p_q2pr       #  prior hyperparameter
    cdef double* _p_f_u        #  prior hyperparameter
    cdef double* p_f           #  component of f to be sampled
    cdef double* p_q2          #  q2 of that component
    cdef double fs, fq2        #  temporary variables

    cdef double* p_pt0t1 

    if cmpnt == 0:
        _p_q2pr = &_fx_q2[0]      #  prior hyperparameter
        _p_f_u  = &_fx_u[0]       #  prior hyperparameter
        p_pt0t1 = &xt0t1[0]
        p_f     = &fx[0]
        p_q2    = &q2x[0]
    elif cmpnt == 1:
        _p_q2pr = &_fy_q2[0]      #  prior hyperparameter
        _p_f_u  = &_fy_u[0]       #  prior hyperparameter
        p_pt0t1 = &yt0t1[0]
        p_f     = &fy[0]
        p_q2    = &q2y[0]

    cdef long* p_clstsz  = &clstsz[0]
    cdef long* p_strt_inds = &cls_strt_inds[0]
    cdef long* p_sts     = &sts[0]
    cdef double U, FQ2

    for 0 <= m < M:
        if p_clstsz[m] > 0:
            tmp = 0
            for p_strt_inds[m] <= i < p_strt_inds[m+1]:
                tmp += p_pt0t1[p_sts[i]-t0]
            fs = tmp/p_clstsz[m]      #  spiking part
            fq2= p_q2[m]/p_clstsz[m]
            U = (fs*_p_q2pr[m] + _p_f_u[m]*fq2) / (_p_q2pr[m] + fq2)
            FQ2 = (_p_q2pr[m]*fq2) / (_p_q2pr[m] + fq2)
        else:
            U   = _p_f_u[m]
            FQ2 = _p_q2pr[m]

        p_adtv_pdf_params[0] = U
        p_adtv_pdf_params[1] = FQ2
        if cmpnt == 0:
            p_adtv_pdf_params[2] = q2x[m]
            p_adtv_pdf_params[3] = fy[m]
            p_adtv_pdf_params[4] = q2y[m]
        elif cmpnt == 1:
            p_adtv_pdf_params[2] = q2y[m]  #  same component as cmptnt
            p_adtv_pdf_params[3] = fx[m]
            p_adtv_pdf_params[4] = q2x[m]


        #printf("--------------  coming from smp_f\n")
        adtvInds, N = adtv_support_pdf(cmpnt, fx2, cpf2, f_STEPS, f_cldz, f_SMALL, dt, l0[m], _NRM, adtv_pdf_params, )

        p_f[m] = adtv_smp_cdf_interp(itr, 0, cmpnt, fx2[adtvInds], cpf2[adtvInds], N, m, m_rands)
        #if ((itr > 4000) and (itr < 4500)):
        # if (itr > 1000):
        #     #printf("iter %d   m %d   cmp  %d---   clstsz %d  fs %.3f   U %.3f\n", itr, m, cmpnt, p_clstsz[m], fs, U)
        #     dat = _N.empty((len(adtvInds), 2))
        #     dat[:, 0] = fx2[adtvInds]
        #     dat[:, 1] = cpf2[adtvInds]
        #     _U.savetxtWCom("f_%(i)d_%(c)d_%(m)d.txt" % {"i" : itr, "c" : cmpnt, "m" : m}, dat, fmt="%.4e %.4e", com=("#clstsz %(cs)d  U %(U).3e   Q2 %(Q).3e\n#l0 %(l0).3e  const comp f %(f).3e  const comp q2 %(q2c).3e  var comp q2 %(q2v).3e" % {"cs" : p_clstsz[m], "U" : U, "Q" : FQ2, "f" : p_adtv_pdf_params[3], "q2c" : p_adtv_pdf_params[4], "q2v" : p_adtv_pdf_params[2], "l0" : l0[m]}))
        #     #_N.savetxt("f_%(i)d_%(c)d_%(m)d.txt" % {"i" : itr, "c" : cmpnt, "m" : m}, dat, fmt="%.4f %.4f")


########################################################################
#@cython.cdivision(True)
#@cython.boundscheck(False)
#@cython.wraparound(False)
def smp_q2(int itr, int cmpnt, int M, long[::1] clstsz, long[::1] cls_strt_inds, long[::1] sts, 
          double[::1] xt0t1, double[::1] yt0t1, int t0, double[::1] fx, double[::1] fy, 
          double[::1] q2x, double[::1] q2y, double[::1] l0, 
          double[::1] _q2_a, double[::1] _q2_B, double[::1] m_rands):
    """
    f_smv is the component of being sampled, f_fx is the fixed component
    _f_sm_u are the prior hyperparameters for f_xmv
    """
    global f_STEPS, f_SMALL, f_cldz, qx2, cpq22
    """
    f     parameter f
    _f_u  
    """
    cdef int m
    cdef double tmp
    cdef double[::1]     v_adtv_pdf_params = adtv_pdf_params
    cdef double* p_adtv_pdf_params = &v_adtv_pdf_params[0]

    cdef double* _p_q2_a = &_q2_a[0]
    cdef double* _p_q2_B  = &_q2_B[0]
    cdef double* p_f           #  component of f to be sampled
    cdef double* p_q2          #  q2 of that component
    cdef double fs, fq2        #  temporary variables

    cdef double* p_pt0t1 
    _p_q2_a = &_q2_a[0]
    _p_q2_B = &_q2_B[0]
    if cmpnt == 0:
        p_pt0t1 = &xt0t1[0]
        p_f     = &fx[0]
        p_q2    = &q2x[0]
    elif cmpnt == 1:
        p_pt0t1 = &yt0t1[0]
        p_f     = &fy[0]
        p_q2    = &q2y[0]

    cdef long* p_clstsz  = &clstsz[0]
    cdef long* p_strt_inds = &cls_strt_inds[0]
    cdef long* p_sts     = &sts[0]
    cdef double SL_a, SL_B

    #  v_sts   spike times, (5 10 11 16) (3 7 9)
    for m in xrange(M):
        if p_clstsz[m] > 0:
            SL_B= 0
            for p_strt_inds[m] <= i < p_strt_inds[m+1]:
                SL_B += (p_pt0t1[p_sts[i]-t0]-p_f[m])*(p_pt0t1[p_sts[i]-t0]-p_f[m])
            SL_B *= 0.5
            #  (2 SL/B - _p_q2_B) / (2SL_a - 2 _p_q2_a - 2)   -->  estimator
            #   [sum( )^2+ B-B] / [M + 2a + -1 2 - 2a - 2 ]
            SL_B += _p_q2_B[m]

            #  -S/2 (likelihood)  -(a+1)
            SL_a = 0.5*p_clstsz[m] + _p_q2_a[m] + 1
        else:
            SL_a = _p_q2_a[m]
            SL_B = _p_q2_B[m]

        p_adtv_pdf_params[0] = SL_a
        p_adtv_pdf_params[1] = SL_B

        if cmpnt == 0:
            p_adtv_pdf_params[2] = fx[m]
            p_adtv_pdf_params[3] = q2y[m]
            p_adtv_pdf_params[4] = fy[m]
        elif cmpnt == 1:
            p_adtv_pdf_params[2] = fy[m]  #  same component as cmptnt
            p_adtv_pdf_params[3] = q2x[m]
            p_adtv_pdf_params[4] = fx[m]

        #printf("adtv_pdf_params  %.3f  %.3f  %.3f\n", p_adtv_pdf_params[2], p_adtv_pdf_params[3], p_adtv_pdf_params[4])
        adtvInds, N = adtv_support_pdf(cmpnt, qx2, cpq22, q2_STEPS, f_cldz, f_SMALL, dt, l0[m], _IG, adtv_pdf_params, )

        p_q2[m] = adtv_smp_cdf_interp(itr, 1, cmpnt, qx2[adtvInds], cpq22[adtvInds], N, m, m_rands)
        if p_clstsz[m] > 0:
            if p_q2[m] > 10:
                s = _N.empty((len(adtvInds), 2))
                s[:, 0] = qx2[adtvInds]
                s[:, 1] = cpq22[adtvInds]
                _N.savetxt("bad.dat", s, fmt="%.4e %.4e")
        
        # if (itr > 1000):
        #     #printf("iter %d   m %d   cmp  %d---   clstsz %d  fs %.3f   U %.3f\n", itr, m, cmpnt, p_clstsz[m], fs, U)
        #     dat = _N.empty((len(adtvInds), 2))
        #     dat[:, 0] = qx2[adtvInds]
        #     dat[:, 1] = cpq22[adtvInds]
        #     _U.savetxtWCom("q2_%(i)d_%(c)d_%(m)d.txt" % {"i" : itr, "c" : cmpnt, "m" : m}, dat, fmt="%.4e %.4e", com=("#clstsz %(cs)d  a %(a).3f   B %(B).3f\n#l0 %(l0).3e  const comp q2 %(q2).3e  const comp f %(fc).3e  var comp f %(fv).3e" % {"cs" : p_clstsz[m], "a" : SL_a, "B" : SL_B, "q2" : p_adtv_pdf_params[3], "fc" : p_adtv_pdf_params[4], "fv" : p_adtv_pdf_params[2], "l0" : l0[m]}))


@cython.cdivision(True)
cdef double pdfIG(int cmpnt, double fc_x, double fc_y, double q2_x, double q2_y, double a, double B, long iStart_x, long iStart_y, long iStart_xy, long Nupx, long Nupy, double ibnsz_x, double ibnsz_y, double dt, double l0, double xL, double xH, double yL, double yH) nogil:
    #  Value of pdf @ fc.  
    #  fxd_IIQ2    1./q2_c
    #  Mc          - spiking + prior  mean
    #  Sigma2c     - spiking + prior  variance
    #  p_riemann_x - points at which integral discretized
    #  p_px        - occupancy
    global py_riemann_xs
    global py_riemann_ys
    global py_hist_all

    cdef double hlfIIQ2_x = -0.5/q2_x
    cdef double hlfIIQ2_y = -0.5/q2_y
    cdef double sptlIntgrl = 0.0
    cdef double ddx2 = 0
    cdef int nxY, nx, ny, m, iLx, iRx, iLy, iRy
    cdef double sdx = sqrt(q2_x)
    cdef double sdy = sqrt(q2_y)

    #iLx = int((fc_x-7*sdx-xL)*ibnsz_x) - 2
    iLx = <int>((fc_x-7*sdx-xL)*ibnsz_x) - 2
    iRx = <int>((fc_x+7*sdx-xL)*ibnsz_x) + 2
    iLx = iLx if iLx >= 0 else 0
    iRx = iRx if iRx <= Nupx else Nupx
    iLy = <int>((fc_y-7*sdy-yL)*ibnsz_y) - 2
    iRy = <int>((fc_y+7*sdy-yL)*ibnsz_y) + 2
    iLy = iLy if iLy >= 0 else 0
    iRy = iRy if iRy <= Nupy else Nupy

    for iLx <= nx < iRx:    #  integrate
        nxY = nx * Nupy + iStart_xy
        #ddx2 = (fc_x-p_riemann_xs[nx]) * (fc_x-p_riemann_xs[nx])*hlfIIQ2_x
        ddx2 = (fc_x-p_riemann_xs[iStart_x + nx]) * (fc_x-p_riemann_xs[iStart_x + nx])*hlfIIQ2_x
        for iLy <= ny < iRy:    #  integrate
            if p_hist_all[nxY + ny] > 0:
                ddy = fc_y-p_riemann_ys[iStart_y + ny]
                sptlIntgrl += exp(ddx2 + ddy*ddy*hlfIIQ2_y)*p_hist_all[nxY + ny]
    sptlIntgrl *= ((dt*l0)/sqrt(fourpi2*q2_x*q2_y))

    #printf("***  a  %.3e  q2_x  %.3e    q2_y  %.3e   B  %.3e   %.3e\n", a, q2_x, q2_y, B, sptlIntgrl)
    if cmpnt == 0:
        return -(a + 1)*log(q2_x) - B/q2_x-sptlIntgrl
    else:
        return -(a + 1)*log(q2_y) - B/q2_y-sptlIntgrl

"""
def pdfIG(int cmpnt, double fc_x, double fc_y, double q2_x, double q2_y, double a, double B, long iStart_x, long iStart_y, long iStart_xy, long Nupx, long Nupy, double ibnsz_x, double ibnsz_y, double dt, double l0, double xL, double xH, double yL, double yH):

    #  Value of pdf @ fc.  
    #  fxd_IIQ2    1./q2_c
    #  Mc          - spiking + prior  mean
    #  Sigma2c     - spiking + prior  variance
    #  p_riemann_x - points at which integral discretized
    #  p_px        - occupancy
    global py_riemann_xs
    global py_riemann_ys
    global py_hist_all
    global fourpi2
    cdef double hlfIIQ2_x = -0.5/q2_x
    cdef double hlfIIQ2_y = -0.5/q2_y
    cdef double sptlIntgrl = 0.0
    cdef double ddx2 = 0
    cdef int nxY, nx, ny, m, iLx, iRx, iLy, iRy
    cdef double sdx = sqrt(q2_x)
    cdef double sdy = sqrt(q2_y)

    iLx = int((fc_x-7*sdx-xL)*ibnsz_x)
    iRx = int((fc_x+7*sdx-xL)*ibnsz_x)
    iLx = iLx if iLx >= 0 else 0
    iRx = iRx if iRx <= Nupx else Nupx
    iLy = int((fc_y-7*sdy-yL)*ibnsz_y)
    iRy = int((fc_y+7*sdy-yL)*ibnsz_y)
    iLy = iLy if iLy >= 0 else 0
    iRy = iRy if iRy <= Nupy else Nupy

    for iLx <= nx < iRx:    #  integrate
        nxY = nx * Nupy
        #ddx2 = (fc_x-p_riemann_xs[nx]) * (fc_x-p_riemann_xs[nx])*hlfIIQ2_x
        #ddx2 = (fc_x-p_riemann_x[nx]) * (fc_x-p_riemann_x[nx])*hlfIIQ2_x
        ddx2 = (fc_x-py_riemann_xs[iStart_x + nx]) * (fc_x-py_riemann_xs[iStart_x + nx])*hlfIIQ2_x
        for iLy <= ny < iRy:    #  integrate
            if py_hist_all[iStart_xy + nxY + ny] > 0:
                #ddy = fc_y-p_riemann_y[ny]
                ddy = fc_y-py_riemann_ys[iStart_y + ny]
                #sptlIntgrl += exp(ddx2 + ddy*ddy*hlfIIQ2_y)*p_hist[nxY + ny]
                sptlIntgrl += exp(ddx2 + ddy*ddy*hlfIIQ2_y)*py_hist_all[iStart_xy + nxY + ny]
    sptlIntgrl *= ((dt*l0)/sqrt(fourpi2*q2_x*q2_y))

    #printf("***  a  %.3e  q2_x  %.3e    q2_y  %.3e   B  %.3e   %.3e\n", a, q2_x, q2_y, B, sptlIntgrl)
    if cmpnt == 0:
        return -(a + 1)*log(q2_x) - B/q2_x-sptlIntgrl
    else:
        return -(a + 1)*log(q2_y) - B/q2_y-sptlIntgrl
"""



########################################################################
@cython.cdivision(True)
cdef double pdfNRM(int cmpnt, double fc_x, double fc_y, double q2_x, double q2_y, double Mc, double Sigma2c, double *p_riemann_x, double *p_riemann_y, long *p_hist, long Nupx, long Nupy, double ibnsz_x, double ibnsz_y, double dt, double l0, double xL, double xH, double yL, double yH) nogil:
    #  Value of pdf @ fc.  
    #  Mc          - spiking + prior  mean
    #  Sigma2c     - spiking + prior  variance
    #  p_riemann_x - points at which integral discretized
    #  p_px        - occupancy
    global fourpi2
    cdef double hlfIIQ2_x = -0.5/q2_x
    cdef double hlfIIQ2_y = -0.5/q2_y
    cdef double sptlIntgrl = 0.0
    cdef double ddx2 = 0
    cdef int nxY, nx, ny, m, iLx, iRx, iLy, iRy
    cdef double sdx = sqrt(q2_x)
    cdef double sdy = sqrt(q2_y)

    iLx = <int>((fc_x-7*sdx-xL)*ibnsz_x) - 2
    iRx = <int>((fc_x+7*sdx-xL)*ibnsz_x) + 2
    iLx = iLx if iLx >= 0 else 0
    iRx = iRx if iRx <= Nupx else Nupx
    iLy = <int>((fc_y-7*sdy-yL)*ibnsz_y) - 2
    iRy = <int>((fc_y+7*sdy-yL)*ibnsz_y) + 2
    iLy = iLy if iLy >= 0 else 0
    iRy = iRy if iRy <= Nupy else Nupy

    for iLx <= nx < iRx:    #  integrate
        nxY = nx * Nupy
        #ddx2 = (fc_x-p_riemann_xs[nx]) * (fc_x-p_riemann_xs[nx])*hlfIIQ2_x
        ddx2 = (fc_x-p_riemann_x[nx]) * (fc_x-p_riemann_x[nx])*hlfIIQ2_x
        for iLy <= ny < iRy:    #  integrate
            if p_hist[nxY + ny] > 0:
                ddy = fc_y-p_riemann_y[ny]
                sptlIntgrl += exp(ddx2 + ddy*ddy*hlfIIQ2_y)*p_hist[nxY + ny]
    sptlIntgrl *= ((dt*l0)/sqrt(fourpi2*q2_x*q2_y))

    if cmpnt == 0:
        return -0.5*(fc_x-Mc)*(fc_x-Mc)/Sigma2c-sptlIntgrl
    else:
        return -0.5*(fc_y-Mc)*(fc_y-Mc)/Sigma2c-sptlIntgrl


########################################################################
@cython.cdivision(True)
def l0_spatial(long M, double dt, double[::1] v_fxd_fc_x, double[::1] v_fxd_fc_y, double[::1] v_fxd_q2_x, double[::1] v_fxd_q2_y, double[::1] v_l0_exp_hist):
    #  Value of pdf @ fc.  
    #  Mc          - spiking + prior  mean
    #  Sigma2c     - spiking + prior  variance
    #  p_riemann_x - points at which integral discretized
    #  p_px        - occupancy
    global p_riemann_xs;      p_riemann_ys;  global p_hist_all
    global x_Lo, x_Hi, y_Lo, y_Hi
    global fourpi2

    cdef double hlfIIQ2_x, hlfIIQ2_y
    cdef double sptlIntgrl = 0.0
    cdef long Nupx, Nupy, iStart_xy, iStart_x, iStart_y
    cdef int nxY, nx, ny, m, iLx, iRx, iLy, iRy
    cdef double ibnsz_x, ibnsz_y, sd_x, sd_y
    cdef double *p_l0_exp_hist = &v_l0_exp_hist[0]
    cdef double *p_fxd_fc_x    = &v_fxd_fc_x[0]
    cdef double *p_fxd_fc_y    = &v_fxd_fc_y[0]
    cdef double *p_fxd_q2_x    = &v_fxd_q2_x[0]
    cdef double *p_fxd_q2_y    = &v_fxd_q2_y[0]
    cdef double fc_x, fc_y, q2c_x, q2c_y

    ##  calculate 
    for m in xrange(M):
        fc_x = p_fxd_fc_x[m]
        q2c_x= p_fxd_q2_x[m]
        fc_y = p_fxd_fc_y[m]
        q2c_y= p_fxd_q2_y[m]

        getOccHist(q2c_x, q2c_y, &Nupx, &Nupy, &iStart_xy, &iStart_x, &iStart_y, &ibnsz_x, &ibnsz_y)
        sptlIntgrl = 0.0
        hlfIIQ2_x = -0.5/q2c_x
        hlfIIQ2_y = -0.5/q2c_y
        
        sd_x = sqrt(q2c_x)
        sd_y = sqrt(q2c_y)

        #################################
        iLx = int((fc_x-7*sd_x-x_Lo)*ibnsz_x) -2
        iRx = int((fc_x+7*sd_x-x_Lo)*ibnsz_x) + 2
        iLx = iLx if iLx >= 0 else 0
        iRx = iRx if iRx <= Nupx else Nupx
        #################################
        iLy = int((fc_y-7*sd_y-y_Lo)*ibnsz_y) - 2
        iRy = int((fc_y+7*sd_y-y_Lo)*ibnsz_y) + 2
        iLy = iLy if iLy >= 0 else 0
        iRy = iRy if iRy <= Nupy else Nupy

        # printf("%f  %f\n", ibnsz_x, ibnsz_y)

        # if m == 0:
        #     printf("iStart_xy  %ld   iStart_x  %ld   iStart_y  %ld\n", iStart_xy, iStart_x, iStart_y)
        #     printf("iLx %d   iLy %d\n", iLx, iLy)
        #     printf("iRx %d   iRy %d\n", iRx, iRy)
        #     printf("fc  %.3f  %.3f\n", fc_x, fc_y)
        #     printf("sd  %.3f  %.3f\n", sd_x, sd_y)

        for iLx <= nx < iRx:    #  integrate
            nxY = nx * Nupy
            ddx2 = (fc_x-p_riemann_xs[iStart_x + nx]) * (fc_x-p_riemann_xs[iStart_x + nx])*hlfIIQ2_x
            for iLy <= ny < iRy:    #  integrate
                #printf("---%d\n", iStart + nxY + ny)
                if p_hist_all[iStart_xy + nxY + ny] != 0:
                    #printf("%.3f\n", ddx2 + ddy*ddy*hlfIIQ2_y)
                    ddy = fc_y-p_riemann_ys[iStart_y + ny]
                    #sptlIntgrl += exp(ddx2 + ddy*ddy*hlfIIQ2_y)*p_hist_all[iStart+nxY + ny]
                    sptlIntgrl += exp(ddx2 + ddy*ddy*hlfIIQ2_y)*p_hist_all[iStart_xy+nxY + ny]

        #sptlIntgrl *= (dt/sqrt(twpi*q2c_x*q2c_y))

        sptlIntgrl *= (dt/sqrt(fourpi2*q2c_x*q2c_y))

        p_l0_exp_hist[m] = sptlIntgrl


########################################################################
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def adtv_support_pdf(int cmpnt, double[::1] gx, double[::1] cond_pstr,
                     int STEPS, int cldz, int small,
                     double dt, double l0, 
                     int dist, double[::1] params):
    """
    This gets called M times per Gibbs sample
    gx      grid @ which to sample pdf.  
    cond_pstr      storage of sampled values
    STEPS   gx has size 2**STEPS+1
    cldz    current level of discretization
    small   small probability
    dist    __NRM or __IT
    params  
    rieman_x  for spatial integral, grid for integration
    px      occupation
    """

    #  STEPS= 5,   IS = 3
    #  cldz=3, skp=2**cldz, start=0    stop=2**STEPS+1
    #  0 8 16 24 32           32/8 + 1  = 5 elements
    #  skp/2  = 4
    #
    #  cldz=2, skp=2**(cldz+1), start=2**cl 
    #   4 12 20 28              # 4 elements    2**2  
    #  skp/4 = 2    next lvl
    #
    #  cldz=1, skp=2**(cldz+1)
    #  2 6 10 14 18 22 26 30    # 8 elements     skp/4 = 1
    #
    #  cldz=0, skp=2**(cldz+1)
    #  1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31   # 16 elements

    #cldz = 5   # current level of discretization.  initl_skp  = 2**cldz
    #  Skip sizes go like 8 8 4 2.   2**
    global p_riemann_xs;      global p_riemann_ys;  global p_hist_all;      global p_ibinszs_x, p_ibinszs_y
    global x_Lo, x_Hi
    global minSmps

    cdef int bDone= 0

    cdef double pmax = -1.e300
    cdef double gpmax= -1.e300   #  global pmax

    cdef int lvl  = 0  #  how deep in discretization level.  maximum val of cldz+1
    cdef int L

    #intvs_each_lvl = _N.empty((cldz+1)*3, dtype=_N.int)*-1
    cdef int *intvs_each_lvl = <int *>malloc(sizeof(int)*(cldz+1)*3)#, dtype=_N.int)*-1
    cdef int iBdL, iBdR
    cdef int iBdL_glbl = p_twoPOW[STEPS]+2#2**STEPS+2
    cdef int iBdR_glbl = -1

    cdef int nSmpd      = 0
    cdef int ix, skp, strt, stop, nxt_lvl_skp, initl_skp
    cdef double pthresh

    cdef double *p_cond_pstr = &cond_pstr[0]
    cdef double *p_gx        = &gx[0]

    cdef double *p_params    = &params[0]

    cdef int imax   = -1
    cdef int gimax  = -1
    cdef double pdfAtSpkML
    cdef double posSpkML

    ####  
    ####  params conditioned on
    ####  priors    
    cdef double _q2_a, _q2_B
    cdef double _f_u, _f_q2
    #_q2_a, _q2_B
    #_f_u, _f_q2
    ####  likelihood terms :  f_c
    cdef double fc = 0
    cdef long Nupx, Nupy, iStart_xy, iStart_x, iStart_y
    cdef double ibnsz_x, ibnsz_y
    cdef int istepbck = 0

    cdef double Mc, Sigma2c, q2_cnd_on, a, B, f_cnd_on, iq2_cnd_on, f_smp_cmp, q2_smp_cmp

    if dist == __NRM:
        Mc        = p_params[0]   #  observed spikes part
        Sigma2c   = p_params[1]   #  observed spikes part
        q2_smp_cmp= p_params[2]   #  if gx is range of fx, this is q2x
        f_cnd_on  = p_params[3]   #  component of f conditioned on (set fixed)
        q2_cnd_on = p_params[4]   #  other f comp and q2 (both comps) conditioned on
    else:
        a         = p_params[0]   #  observed spikes part
        B         = p_params[1]   #  observed spikes part
        f_smp_cmp = p_params[2]   #  if gx is range of q2x, this is fx
        q2_cnd_on = p_params[3]   #  other f comp and q2 (both comps) conditione
        f_cnd_on  = p_params[4]   #  component of f conditioned on (set fixed)
        #printf("q2_cnd_on  %.4f\n", q2_cnd_on)


    #initial point, set pmax, gmax
    if dist == __NRM:   ################## __NRM
        getOccHist(q2_smp_cmp, q2_cnd_on, &Nupx, &Nupy, &iStart_xy, &iStart_x, &iStart_y, &ibnsz_x, &ibnsz_y)

        #   pdf @ f_{xy} = Mc

        posSpkML = Mc
        if cmpnt == 0:
            #pdfAtSpkML = pdfNRM(cmpnt,          Mc, f_cnd_on, q2_smp_cmp, q2_cnd_on, Mc, Sigma2c, &p_riemann_xs[iStart_x], &p_riemann_ys[iStart_y], &p_hist_all[iStart_xy], Nupx, Nupy, ibnsz_x, ibnsz_y, dt, l0, x_Lo, x_Hi, y_Lo, y_Hi)
            p_cond_pstr[0] = pdfNRM(cmpnt, p_gx[0], f_cnd_on, q2_smp_cmp, q2_cnd_on, Mc, Sigma2c, &p_riemann_xs[iStart_x], &p_riemann_ys[iStart_y], &p_hist_all[iStart_xy], Nupx, Nupy, ibnsz_x, ibnsz_y, dt, l0, x_Lo, x_Hi, y_Lo, y_Hi)
        else:  #  cmpnt == 1
            pdfAtSpkML     = pdfNRM(cmpnt, f_cnd_on, Mc,      q2_cnd_on, q2_smp_cmp, Mc, Sigma2c, &p_riemann_xs[iStart_x], &p_riemann_ys[iStart_y], &p_hist_all[iStart_xy], Nupx, Nupy, ibnsz_x, ibnsz_y, dt, l0, x_Lo, x_Hi, y_Lo, y_Hi)
            p_cond_pstr[0] = pdfNRM(cmpnt, f_cnd_on, p_gx[0], q2_cnd_on, q2_smp_cmp, Mc, Sigma2c, &p_riemann_xs[iStart_x], &p_riemann_ys[iStart_y], &p_hist_all[iStart_xy], Nupx, Nupy, ibnsz_x, ibnsz_y, dt, l0, x_Lo, x_Hi, y_Lo, y_Hi)
    if dist == __IG:   ################## __IG
        posSpkML = B/(a+1)
        getOccHist(p_gx[0], q2_cnd_on, &Nupx, &Nupy, &iStart_xy, &iStart_x, &iStart_y, &ibnsz_x, &ibnsz_y)
        #printf("p_gx %.3e   cn %.3f   %ld  %ld\n", p_gx[0], q2_cnd_on, Nupx, Nupy)
        #printf("p_gx %.3e   cn %.3f   %ld  %ld\n", p_gx[1], q2_cnd_on, Nupx, Nupy)
        #printf("xy %ld   x %ld   y %ld\n", iStart_xy, iStart_x, iStart_y)

        if cmpnt == 0:
            #p_cond_pstr[0] = pdfIG(cmpnt, f_smp_cmp, f_cnd_on, p_gx[0], q2_cnd_on, a, B, &p_riemann_xs[iStart_x], &p_riemann_ys[iStart_y], &p_hist_all[iStart_xy], Nupx, Nupy, ibnsz_x, ibnsz_y, dt, l0, x_Lo, x_Hi, y_Lo, y_Hi)
            pdfAtSpkML     = pdfIG(cmpnt, f_smp_cmp, f_cnd_on, B/(a+1), q2_cnd_on, a, B, iStart_x, iStart_y, iStart_xy, Nupx, Nupy, ibnsz_x, ibnsz_y, dt, l0, x_Lo, x_Hi, y_Lo, y_Hi)
            p_cond_pstr[0] = pdfIG(cmpnt, f_smp_cmp, f_cnd_on, p_gx[0], q2_cnd_on, a, B, iStart_x, iStart_y, iStart_xy, Nupx, Nupy, ibnsz_x, ibnsz_y, dt, l0, x_Lo, x_Hi, y_Lo, y_Hi)
        else:  #  cmpnt == 1
            #p_cond_pstr[0] = pdfIG(cmpnt, f_cnd_on, f_smp_cmp, q2_cnd_on, p_gx[0], a, B, &p_riemann_xs[iStart_x], &p_riemann_ys[iStart_y], &p_hist_all[iStart_xy], Nupx, Nupy, ibnsz_x, ibnsz_y, dt, l0, x_Lo, x_Hi, y_Lo, y_Hi)
            pdfAtSpkML     = pdfIG(cmpnt, f_cnd_on, f_smp_cmp, q2_cnd_on, B/(a+1), a, B, iStart_x, iStart_y, iStart_xy, Nupx, Nupy, ibnsz_x, ibnsz_y, dt, l0, x_Lo, x_Hi, y_Lo, y_Hi)
            p_cond_pstr[0] = pdfIG(cmpnt, f_cnd_on, f_smp_cmp, q2_cnd_on, p_gx[0], a, B, iStart_x, iStart_y, iStart_xy, Nupx, Nupy, ibnsz_x, ibnsz_y, dt, l0, x_Lo, x_Hi, y_Lo, y_Hi)

    # elif dist == __IG:
    #     getOccHist(p_gx[0], &Nupx, &iStart, &ibnsz)
    #     p_cond_pstr[0] = pdfIG(p_gx[0], f_cnd_on, a, B, &p_riemann_xs[iStart], &p_hist_all[iStart], Nupx, ibnsz, dt, l0, x_Lo, x_Hi)
    pmax = p_cond_pstr[0]
    gmax = pmax
    imax = 0

    while (cldz > -1) and (bDone == 0):
        if lvl == 0:  # 1st 2 passes skip sizes are the same
            strt = 0   #  first point will have been done already
            skp  = p_twoPOW[cldz]          # 2**cldz
            #print "init skip %d skp" % skp
            initl_skp  = skp
            stop  = p_twoPOW[STEPS]-strt   # 2**STEPS-strt
            nxt_lvl_skp = skp/2
        else:
            strt = iBdL - nxt_lvl_skp
            if strt < 0:
                strt = iBdL + nxt_lvl_skp
            stop = iBdR + nxt_lvl_skp
            if stop > p_twoPOW[STEPS]:      # 2**STEPS:
                stop = iBdR - nxt_lvl_skp
            skp  = p_twoPOW[cldz+1]        # 2**(cldz+1)
            nxt_lvl_skp = skp/4

        ################  these 2 must be done together, and after strt, skp set
        cldz  -= 1
        lvl   += 1
        ################  these 2 must be done together, and after strt, skp set
        if dist == __NRM:
            ##  for each f cond posterior calculated at:
            #   - q2 conditioned on
            #   - Mc (likelihood + prior) center of observation
            #   - Sigma2c (likelihood + prior) width of observation
            #   prior

            getOccHist(q2_smp_cmp, q2_cnd_on, &Nupx, &Nupy, &iStart_xy, &iStart_x, &iStart_y, &ibnsz_x, &ibnsz_y)
            for strt <= ix < stop+1 by skp:
                if cmpnt == 0:
                    p_cond_pstr[ix] = pdfNRM(cmpnt, p_gx[ix], f_cnd_on, q2_smp_cmp, q2_cnd_on, Mc, Sigma2c, &p_riemann_xs[iStart_x], &p_riemann_ys[iStart_y], &p_hist_all[iStart_xy], Nupx, Nupy, ibnsz_x, ibnsz_y, dt, l0, x_Lo, x_Hi, y_Lo, y_Hi)
                else:
                    p_cond_pstr[ix] = pdfNRM(cmpnt, f_cnd_on, p_gx[ix], q2_cnd_on, q2_smp_cmp, Mc, Sigma2c, &p_riemann_xs[iStart_x], &p_riemann_ys[iStart_y], &p_hist_all[iStart_xy], Nupx, Nupy, ibnsz_x, ibnsz_y, dt, l0, x_Lo, x_Hi, y_Lo, y_Hi)

                if p_cond_pstr[ix] > pmax:
                    pmax = p_cond_pstr[ix]   # pmax updated each time grid made finer
                    imax = ix
        if dist == __IG:
            ##  for each f cond posterior calculated at:
            #   - q2 conditioned on
            #   - Mc (likelihood + prior) center of observation
            #   - Sigma2c (likelihood + prior) width of observation
            #   prior

            for strt <= ix < stop+1 by skp:
                #printf("p_gx %.3e   cn %.3f   %ld  %ld\n", p_gx[ix], q2_cnd_on, Nupx, Nupy)

                if cmpnt == 0:
                    getOccHist(p_gx[ix], q2_cnd_on, &Nupx, &Nupy, &iStart_xy, &iStart_x, &iStart_y, &ibnsz_x, &ibnsz_y)   
                    #p_cond_pstr[ix] = pdfIG(cmpnt, f_smp_cmp, f_cnd_on, p_gx[ix], q2_cnd_on, a, B, &p_riemann_xs[iStart_x], &p_riemann_ys[iStart_y], &p_hist_all[iStart_xy], Nupx, Nupy, ibnsz_x, ibnsz_y, dt, l0, x_Lo, x_Hi, y_Lo, y_Hi)
                    p_cond_pstr[ix] = pdfIG(cmpnt, f_smp_cmp, f_cnd_on, p_gx[ix], q2_cnd_on, a, B, iStart_x, iStart_y, iStart_xy, Nupx, Nupy, ibnsz_x, ibnsz_y, dt, l0, x_Lo, x_Hi, y_Lo, y_Hi)
                else:  #  cmpnt == 1
                    getOccHist(q2_cnd_on, p_gx[ix], &Nupx, &Nupy, &iStart_xy, &iStart_x, &iStart_y, &ibnsz_x, &ibnsz_y)
                    #p_cond_pstr[ix] = pdfIG(cmpnt, f_cnd_on, f_smp_cmp, q2_cnd_on, p_gx[ix], a, B, &p_riemann_xs[iStart_x], &p_riemann_ys[iStart_y], &p_hist_all[iStart_xy], Nupx, Nupy, ibnsz_x, ibnsz_y, dt, l0, x_Lo, x_Hi, y_Lo, y_Hi)
                    p_cond_pstr[ix] = pdfIG(cmpnt, f_cnd_on, f_smp_cmp, q2_cnd_on, p_gx[ix], a, B, iStart_x, iStart_y, iStart_xy, Nupx, Nupy, ibnsz_x, ibnsz_y, dt, l0, x_Lo, x_Hi, y_Lo, y_Hi)
                if p_cond_pstr[ix] > pmax:
                    pmax = p_cond_pstr[ix]   # pmax updated each time grid made finer
                    imax = ix

        # else:
        #     ##  for each q2 cond posterior calculated at:
        #     #   we need mean of x @ spikes, q2, # of spikes
        #     #   prior
        #     #   p_gx[ix]   values of q2
        #     for strt <= ix < stop+1 by skp:
        #         getOccHist(p_gx[ix], &Nupx, &iStart, &ibnsz)
        #         p_cond_pstr[ix] = pdfIG(p_gx[ix], f_cnd_on, a, B, &p_riemann_xs[iStart], &p_hist_all[iStart], Nupx, ibnsz, dt, l0, x_Lo, x_Hi)

        #         if p_cond_pstr[ix] > pmax:
        #             pmax = p_cond_pstr[ix]   # pmax updated each time grid made finer
        #             imax = ix

        if pmax > gpmax:   #  stop when 
            gpmax = pmax
            pthresh= gpmax-small   #  _N.exp(-12) == 6.144e-6
            gimax = imax

        #  start half the current skip size before and after iBdL and iBdR
        # now find left and right bounds

        ix = strt  #  sampling has started from istrt.  

        #printf("strt %d     stop %d     skip %d\n", strt, stop, skp)
        istepbck = 0
        while (ix < gimax) and (p_cond_pstr[ix] < pthresh):
            # if we don't do this while, we still end up stepping back. 
            istepbck += 1
            ix += skp  
        #  at this point, p_cond_str[ix] >= pthresh or ix >= gimax
        if istepbck > 0:
            ix -= skp  # step back, so p_cond_str[ix] is now < pthresh.
        if ix >= 0:
            iBdL = ix
        else:
            while ix < 0:
                ix += skp
            iBdL = ix
        ##########
        istepbck = 0
        ix = stop
        while (ix > gimax) and (p_cond_pstr[ix] < pthresh):
            istepbck += 1
            ix -= skp
        if istepbck > 0:
            ix += skp

        if ix <= p_twoPOW[STEPS]:
            iBdR = ix
        else:     #  rewind if we go past right limit.  Happens when distribution too wide, right hand is still > pthresh
            while ix > p_twoPOW[STEPS]:
                ix -= skp
            iBdR = ix

        iBdL_glbl = iBdL if (iBdL < iBdL_glbl) else iBdL_glbl
        iBdR_glbl = iBdR if (iBdR > iBdR_glbl) else iBdR_glbl

        intvs_each_lvl[3*(lvl-1)]   = iBdL
        intvs_each_lvl[3*(lvl-1)+1] = iBdR
        intvs_each_lvl[3*(lvl-1)+2] = skp
        nSmpd += (iBdR - iBdL) / skp + 1

        bDone = 1 if (nSmpd > minSmps) else 0


    # if (gpmax < 0) or (pdfAtSpkML < 0):
    #    printf("gpmax %.3e    pdfAtSpkML %.3e\n", gpmax, pdfAtSpkML)
        
    # if pdfAtSpkML > gpmax:
    #     printf("      %d  A) %.3e   B) %.3e     C) %.3e   D) %.3e\n", dist, p_gx[gimax], posSpkML, gpmax, pdfAtSpkML)
    # else:
    #     printf("regul %d A) %.3e   B) %.3e     C) %.3e   D) %.3e\n", dist, p_gx[gimax], posSpkML, gpmax, pdfAtSpkML)

    #  reconstruct lstFM
    narr_FM     = _N.empty(nSmpd, dtype=_N.int)
    cdef int ii0 = 0
    cdef int ii1 = 0
    cdef int ii = 0
    cdef int l, ll

    #print intvs_each_lvl
    for l in xrange(lvl):
        ii0 = intvs_each_lvl[3*l]
        ii1 = intvs_each_lvl[3*l+1]
        skp = intvs_each_lvl[3*l+2]
        L   = (ii1-ii0)/skp + 1
        narr_FM[ii:ii+L] = _N.arange(ii0, ii1+1, skp)
        #for il in xrange(L):
        #    narr_FM[il] = ii0 + il*skp   #  list of finely sampled points
        ii += L

    #free(narr_FM)
    #  The first iBdL is iBdL_glbl.  We always work inwards
    
    lft = _N.arange(0, iBdL_glbl - iBdL_glbl % initl_skp, initl_skp)
    rgt = _N.arange(iBdR_glbl - iBdR_glbl % initl_skp + initl_skp, 2**STEPS+2, initl_skp)
    narr_FM.sort()
    #  midInds should be same as _N.arange(iBdL_glbl, iBdR_glbl+1, fine_skip)
    L   = (lft.shape[0] + rgt.shape[0] + narr_FM.shape[0])
    adtvInds = _N.empty(L, dtype=_N.int)
    cdef long[::1] v_adtvInds = adtvInds
    cdef long* p_adtvInds     = &v_adtvInds[0]
    adtvInds[0:lft.shape[0]] = lft
    adtvInds[lft.shape[0]:lft.shape[0] + narr_FM.shape[0]] = narr_FM
    adtvInds[lft.shape[0] + narr_FM.shape[0]:lft.shape[0] + narr_FM.shape[0]+rgt.shape[0]] = rgt

    #  nSmpd == narr_FM.shape[0]
    # print "----------"
    # print nSmpd
    # print adtvInds

    for 0 <= l < L:   #  largest value should be 0, so exp doesn't -> 0
        p_cond_pstr[p_adtvInds[l]] -= gpmax

    # print adtvInds.shape[0]
    # print narr_FM.shape[0]
    # print "----------"
    # for l in adtvInds:   #  largest value should be 0, so exp doesn't -> 0
    #     if p_cond_pstr[l] > gpmax:   ####  TEMPORARY  for debug purposes
    #         print "------------   woa, %d" % l
        #     for ll in xrange(lvl):
        #         ii0 = intvs_each_lvl[3*ll]
        #         ii1 = intvs_each_lvl[3*ll+1]
        #         printf("       %d  %d     %d\n", ii0, ii1, intvs_each_lvl[3*ll+2])

        #     print adtvInds
        #     print narr_FM
        #     print "offending val %(pcp).3e @ %(x).3e " % {"pcp" : p_cond_pstr[l], "x" : p_gx[l]}

        #     getOccHist(p_gx[l], &Nupx, &iStart, &dSilenceX, &ibnsz)            
        #     print iStart
        #     print dSilenceX
        #     print ibnsz
        #     print Nupx
        #     tryagain = pdfIG(p_gx[l], f_cnd_on, a, B, &p_riemann_xs[iStart], &p_hist_all[iStart], Nupx, ibnsz, dt, l0, dSilenceX, x_Lo, x_Hi)
        #     print "try again val %.3e" % tryagain
        #     p_cond_pstr[l] = tryagain
        #     gpmax = tryagain
        # p_cond_pstr[l] -= gpmax
    free(intvs_each_lvl)
    return adtvInds, L



###########################################################################
###########################################################################
###########################################################################
#####################  OCCUPATION FUNCTIONS



def init_occ_resolutions(_x_Lo, _x_Hi, _y_Lo, _y_Hi, v_q2_thr, v_Nupxs, v_Nupys):
    #  BIN a bit smaller than width
    # q2_thr, Nupxs
    # 0.02^2         0.05^2          0.1^2,          0.2^2,    
    # 1000(0.012)    600(0.02)       200 (0.06)      100 (0.12)
    # 0.5^2,     1^2,    6^2,   100000^2  
    #       40         20      6      2

    # init(-6, 6, _N.array([600, 600, 200, 100, 40, 20, 6, 2])
    # x_Lo lower limit for integration
    # x_Hi upper limit for integration

    #  p_pxs[i], bins = _N.histogram(Nupx[i]+1)
    #  gets called once per epoch
    #  

    global py_Nupxs;      global py_Nupys;    
    global py_ibinszs_x;  global py_ibinszs_y
    global py_hist_all;   global py_riemann_xs;   global py_riemann_ys
    global p_hist_all;    global Nupxs;    global Nupys;  global n_res
    global p_q2_thr;  global arr_xy_0s; global arr_x_0s; global arr_y_0s
    global p_riemann_xs;  global p_riemann_ys
    global p_ibinszs_x, p_ibinszs_y
    global x_Lo, x_Hi, y_Lo, y_Hi
    global py_q2_thr;     global py_arr_xy_0s;  
    global py_arr_x_0s;   global py_arr_y_0s

    cdef int i, j, totxy = 0, totx = 0, toty = 0
    cdef double dx, x

    x_Lo = _x_Lo;    x_Hi = _x_Hi
    y_Lo = _y_Lo;    y_Hi = _y_Hi
    
    n_res         = len(v_Nupxs)  # n_res resolutions for occupational histgrm

    print "init_occ_res 0"
    py_Nupxs     = _N.empty(n_res, dtype=_N.int)
    cdef long[::1] mv_Nupxs     = py_Nupxs
    Nupxs = &mv_Nupxs[0]
    py_Nupys     = _N.empty(n_res, dtype=_N.int)
    cdef long[::1] mv_Nupys     = py_Nupys
    Nupys = &mv_Nupys[0]
    py_ibinszs_x     = _N.empty(n_res)
    cdef double[::1] mv_ibinszs_x     = py_ibinszs_x
    p_ibinszs_x = &mv_ibinszs_x[0]
    py_ibinszs_y     = _N.empty(n_res)
    cdef double[::1] mv_ibinszs_y     = py_ibinszs_y
    p_ibinszs_y = &mv_ibinszs_y[0]
    py_hist_all   = _N.empty(_N.sum(v_Nupxs*v_Nupys), dtype=_N.int)
    cdef long[::1] mv_hist_all = py_hist_all
    p_hist_all     = &mv_hist_all[0]

    py_riemann_xs   = _N.empty(_N.sum(v_Nupxs))
    cdef double[::1] mv_riemann_xs = py_riemann_xs
    p_riemann_xs     = &mv_riemann_xs[0]
    py_riemann_ys   = _N.empty(_N.sum(v_Nupys))
    cdef double[::1] mv_riemann_ys = py_riemann_ys
    p_riemann_ys     = &mv_riemann_ys[0]

    #p_hist_all     = <long*>malloc(_N.sum(v_Nupxs*v_Nupys)*sizeof(long))
    #p_riemann_xs = <double*>malloc(_N.sum(v_Nupxs)*sizeof(double))
    #p_riemann_ys = <double*>malloc(_N.sum(v_Nupys)*sizeof(double))


    #Nupxs        = <long*>malloc(n_res*sizeof(long))
    #Nupys        = <long*>malloc(n_res*sizeof(long))
    #p_ibinszs_x  = <double*>malloc(n_res*sizeof(double))
    #p_ibinszs_y  = <double*>malloc(n_res*sizeof(double))

    py_arr_xy_0s  = _N.empty(n_res, dtype=_N.int)
    cdef long[::1] mv_arr_xy_0s = py_arr_xy_0s
    arr_xy_0s       = &mv_arr_xy_0s[0]#<long*>malloc(n_res*sizeof(long)) 
    py_arr_x_0s   = _N.empty(n_res, dtype=_N.int)
    cdef long[::1] mv_arr_x_0s  = py_arr_x_0s
    arr_x_0s        = &mv_arr_x_0s[0]#<long*>malloc(n_res*sizeof(long)) 
    py_arr_y_0s   = _N.empty(n_res, dtype=_N.int)
    cdef long[::1] mv_arr_y_0s = py_arr_y_0s
    arr_y_0s       = &mv_arr_y_0s[0]#<long*>malloc(n_res*sizeof(long)) 
    py_q2_thr   = _N.empty(n_res)
    cdef double[::1] mv_q2_thr = py_q2_thr
    p_q2_thr       = &mv_q2_thr[0]#<long*>malloc(n_res*sizeof(long)) 

    # arr_x_0s       = <long*>malloc(n_res*sizeof(long)) 
    # arr_y_0s       = <long*>malloc(n_res*sizeof(long)) 
    # p_q2_thr     = <double*>malloc(n_res*sizeof(double))
    #  all n_res histograms to be stored in a flat array
    #print("size  %d\n" % _N.sum(v_Nupxs*v_Nupys))
    #p_hist_all     = <long*>malloc(_N.sum(v_Nupxs*v_Nupys)*sizeof(long))
    #p_riemann_xs = <double*>malloc(_N.sum(v_Nupxs)*sizeof(double))
    #p_riemann_ys = <double*>malloc(_N.sum(v_Nupys)*sizeof(double))

    print "init_occ_res 3"
    for 0 <= i < n_res:
        p_q2_thr[i] = v_q2_thr[i]
        Nupxs[i] = v_Nupxs[i]
        Nupys[i] = v_Nupys[i]

        arr_xy_0s[i] = totxy
        arr_x_0s[i] = totx
        arr_y_0s[i] = toty

        printf("**  %d  %d  %d\n", totx, toty, totxy)

        dx = float(x_Hi - x_Lo) / Nupxs[i]
        dy = float(y_Hi - y_Lo) / Nupys[i]
        py_ibinszs_x[i] = 1./dx
        py_ibinszs_y[i] = 1./dy
        x  = x_Lo + 0.5*dx
        y  = y_Lo + 0.5*dy

        for 0 <= j < Nupxs[i]:
            py_riemann_xs[totx+j] = x
            x += dx

        for 0 <= j < Nupys[i]:
            py_riemann_ys[toty+j] = y
            y += dy


        totxy += py_Nupxs[i] * py_Nupys[i]
        totx += py_Nupxs[i]
        toty += py_Nupys[i]
    print "init_occ_res 4"


def clean_occ_resolutions():
    global p_hist_all;   global p_riemann_xs;   global p_q2_thr
    global Nupxs;      global arr_xy_0s
    free(p_hist_all);    free(p_riemann_xs);    free(p_q2_thr)
    free(Nupxs);       free(arr_xy_0s)

def change_occ_hist(pthx, pthy, _x_Lo, _x_Hi, _y_Lo, _y_Hi):
    global p_hist_all;    global Nupxs;         global Nupys
    global n_res
    global p_q2_thr;    global arr_xy_0s;        global p_riemann_xs
    #N_pth = len(pthx)

    ##  
    cdef int i, j, iNy
    for 0 <= n < n_res:
        hist, bns, v3 = _N.histogram2d(pthx, pthy, bins=[_N.linspace(_x_Lo, _x_Hi, Nupxs[n]+1), _N.linspace(_y_Lo, _y_Hi, Nupys[n]+1)], normed=False)

        for 0 <= i < Nupxs[n]:
            iNy = i * Nupys[n]
            for 0 <= j < Nupys[n]:
                p_hist_all[arr_xy_0s[n] + iNy+j] = hist[i, j]

    #  for each
    
    
# set BOTH the correct occupation density AND riemann_x
cdef void getOccHist(double gau_q2_x, double gau_q2_y, long *Nupx, long *Nupy, long *iStart_xy, long *iStart_x, long *iStart_y, double *ibnsz_x, double *ibnsz_y):
    global Nupxs;      global Nupys;  global arr_xy_0s
    global n_res;      global p_q2_thr
    global p_ibinszs_x; p_ibinszs_y
    cdef int i

    #  for higher q2 (wider), lower resolution OK
    #  each Nupx[i] set so spatial resolution for sum is 1/3 (i-1)th level
    for 0 <= i < n_res:  
        if (gau_q2_x < p_q2_thr[i]) and (gau_q2_y < p_q2_thr[i]):
            Nupx[0]   = Nupxs[i]
            Nupy[0]   = Nupys[i]
            iStart_xy[0] = arr_xy_0s[i]
            iStart_x[0] = arr_x_0s[i]
            iStart_y[0] = arr_y_0s[i]
            ibnsz_x[0]= p_ibinszs_x[i]
            ibnsz_y[0]= p_ibinszs_y[i]
            #printf("returning   i %d    n_res %d\n", i, n_res)
            return 
    #  
    Nupx[0]   = Nupxs[n_res-1]
    Nupy[0]   = Nupxs[n_res-1]
    iStart_xy[0] = arr_xy_0s[n_res-1]
    iStart_x[0] = arr_x_0s[n_res-1]
    iStart_y[0] = arr_y_0s[n_res-1]
    ibnsz_x[0]= p_ibinszs_x[n_res-1]
    ibnsz_y[0]= p_ibinszs_y[n_res-1]
