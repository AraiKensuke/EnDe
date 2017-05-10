import numpy as _N
cimport numpy as _N
cimport cython
import time as _tm
from libc.math cimport exp, sqrt, log
from libc.stdlib cimport malloc, free
import matplotlib.pyplot as _plt

twoPOW = _N.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072], dtype=_N.int)
cdef long[::1] v_twoPOW = twoPOW
cdef long* p_twoPOW = &v_twoPOW[0]

cdef int __NRM = 0
cdef int __IG  = 1

cdef double twpi = 6.283185307

cdef double[::1] v_cpf2
cdef double[::1] v_cpq22
cdef double[::1] v_fx2
cdef double[::1] v_qx2

f_STEPS   = None
q2_STEPS  = None
f_cldz    = None
q2_cldz   = None
f_SMALL   = 10
q2_SMALL  = 10


cpf2 = None
cpq22 = None    # value of cond. post, defined only certain pts

_NRM = 0   #  for outside visibility
_IG  = 1

fx2 = None
qx2 = None    # adaptive points where cond. post f, q2 defined

f_lo = None
f_hi = None
q2_lo = None
q2_hi = None
dt    = None
dSilenceX= None

adtv_pdf_params = _N.empty(3)

########################################################################
def init(_dt, _f_lo, _f_hi, _q2_lo, _q2_hi, _f_STEPS, _q2_STEPS, _f_SMALL, _q2_SMALL, _f_cldz, _q2_cldz):
    global fx2, qx2, cpf2, cpq22, 
    global v_fx2, v_qx2, v_cpf2, v_cpq22, 
    global dt, dSilenceX
    global f_lo, f_hi, q2_lo, q2_hi
    global f_SMALL, q2_SMALL, f_STEPS, q2_STEPS, f_cldz, q2_cldz

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
    

########################################################################
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def adtv_smp_cdf_interp(double[::1] x, double[::1] log_p, int N, int m, double[::1] m_rnds):
    """
    each cluster has independent x over which conditional likelihood defined
    xt0t1    relative coordinates
    mks      absolute coordinates

    sLLkPr   spiking part
    s        silence part
    """
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

    #  btwn cdf[isg2] and cdf[isg2+1]
    #  (rnds[m,0] - cdf[isg2]) * (cdf[isg2+1] - cdf[isg2]) * d_sg2s[isg2]
    _isg2 = _N.searchsorted(cdf, rnd)
    isg2  = _isg2-1

    retRnd = p_x[isg2] + ((rnd - p_cdf[isg2]) / (p_cdf[isg2+1] - p_cdf[isg2])) * dx[isg2]  # unlike in above case, retRnd may be < 0
    free(dx)
    free(p)
    return retRnd

########################################################################
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def smp_f(int M, long[::1] clstsz, long[::1] cls_strt_inds, long[::1] sts, 
          double[::1] fx, double[::1] pxO_full, int Nupx,
          double[::1] xt0t1, int t0, 
          double[::1] f, double[::1] q2, double[::1] l0, 
          double[::1] _f_u, double[::1] _f_q2, double[::1] m_rands):
    global f_STEPS, f_SMALL, f_cldz, dSilenceX
    """
    f     parameter f
    _f_u  
    """
    cdef int m
    cdef double tmp
    cdef double* _p_q2pr = &_f_q2[0]
    cdef double* _p_f_u  = &_f_u[0]
    cdef double* p_f     = &f[0]
    cdef double* p_q2    = &q2[0]
    cdef double fs, fq2

    cdef long* p_clstsz  = &clstsz[0]
    cdef double* p_xt0t1 = &xt0t1[0]
    cdef long* p_strt_inds = &cls_strt_inds[0]
    cdef long* p_sts     = &sts[0]
    cdef double[::1]     v_adtv_pdf_params = adtv_pdf_params
    cdef double* p_adtv_pdf_params = &v_adtv_pdf_params[0]
    cdef double U, FQ2

    for 0 <= m < M:
        if p_clstsz[m] > 0:
            tmp = 0
            for p_strt_inds[m] <= i < p_strt_inds[m+1]:
                tmp += p_xt0t1[p_sts[i]-t0]
            fs = tmp/p_clstsz[m]
            fq2= p_q2[m]/p_clstsz[m]
            U = (fs*_p_q2pr[m] + _p_f_u[m]*fq2) / (_p_q2pr[m] + fq2)
            FQ2 = (_p_q2pr[m]*fq2) / (_p_q2pr[m] + fq2)
        else:
            U   = _p_f_u[m]
            FQ2 = _p_q2pr[m]

        p_adtv_pdf_params[0] = U
        p_adtv_pdf_params[1] = FQ2
        p_adtv_pdf_params[2] = q2[m]

        adtvInds, N = adtv_support_pdf(fx2, cpf2, f_STEPS, f_cldz, f_SMALL, fx, pxO_full, Nupx, dt, l0[m], dSilenceX, _NRM, adtv_pdf_params, )

        p_f[m] = adtv_smp_cdf_interp(fx2[adtvInds], cpf2[adtvInds], N, m, m_rands)

########################################################################
#@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def smp_q2(int M, long[::1] clstsz, long[::1] cls_strt_inds, long[::1] sts, 
          double[::1] fx, double[::1] pxO_full, int Nupx,
          double[::1] xt0t1, int t0, 
          double[::1] f, double[::1] q2, double[::1] l0, 
          double[::1] _q2_a, double[::1] _q2_B, double[::1] m_rands):
    cdef int m
    cdef double tmp, fm
    cdef double* _p_q2_a = &_q2_a[0]
    cdef double* _p_q2_B  = &_q2_B[0]
    cdef double* p_f     = &f[0]
    cdef double* p_q2    = &q2[0]
    cdef double fs, fq2

    cdef long* p_clstsz  = &clstsz[0]
    cdef double* p_xt0t1 = &xt0t1[0]
    cdef long* p_strt_inds = &cls_strt_inds[0]
    cdef long* p_sts     = &sts[0]
    cdef double[::1]     v_adtv_pdf_params = adtv_pdf_params
    cdef double* p_adtv_pdf_params = &v_adtv_pdf_params[0]
    cdef double SL_a, SL_B

    #  v_sts   spike times, (5 10 11 16) (3 7 9)
    #  
    for m in xrange(M):
        if p_clstsz[m] > 0:
            fm = f[m]

            SL_B= 0
            for p_strt_inds[m] <= i < p_strt_inds[m+1]:
                SL_B += (p_xt0t1[p_sts[i]-t0]-fm)*(p_xt0t1[p_sts[i]-t0]-fm)
            SL_B *= 0.5
            SL_B += _p_q2_B[m]

            #  -S/2 (likelihood)  -(a+1)
            SL_a = 0.5*p_clstsz[m] + _p_q2_a[m] + 1
        else:
            SL_a = _p_q2_a[m]
            SL_B = _p_q2_B[m]
            fm = f[m]

        p_adtv_pdf_params[0] = SL_a
        p_adtv_pdf_params[1] = SL_B
        p_adtv_pdf_params[2] = fm

        adtvInds, N = adtv_support_pdf(qx2, cpq22, q2_STEPS, f_cldz, f_SMALL, fx, pxO_full, Nupx, dt, l0[m], dSilenceX, _IG, adtv_pdf_params, )
        p_q2[m] = adtv_smp_cdf_interp(qx2[adtvInds], cpq22[adtvInds], N, m, m_rands)


########################################################################
@cython.cdivision(True)
cdef double pdfIG(double q2c, double fxd_f, double a, double B, double* p_riemann_x, double *p_px, int Nupx, double dt, double l0, double dSilenceX) nogil:
    #  
    cdef double hlfIIQ2 = -0.5/q2c
    cdef double sptlIntgrl = 0.0
    cdef double dd
    cdef int n


    for n in xrange(Nupx):   #  spatial integral
        dd = fxd_f-p_riemann_x[n]
        sptlIntgrl += exp(dd*dd*hlfIIQ2)*p_px[n]
    sptlIntgrl *= ((dt*l0)/sqrt(twpi*q2c))*dSilenceX
    
    return -(a + 1)*log(q2c) - B/q2c-sptlIntgrl

########################################################################
@cython.cdivision(True)
cdef double pdfNRM(double fc, double fxd_q2, double fxd_IIQ2, double Mc, double Sigma2c, double *p_riemann_x, double *p_px, int Nupx, double dt, double l0, double dSilenceX) nogil:
    #  Value of pdf @ fc.  
    #  fxd_IIQ2    1./q2_c
    #  Mc          - spiking + prior  mean
    #  Sigma2c     - spiking + prior  variance
    #  p_riemann_x - points at which integral discretized
    #  p_px        - occupancy
    cdef double hlfIIQ2 = -0.5*fxd_IIQ2
    cdef double sptlIntgrl = 0.0
    cdef double dd = 0
    cdef int n

    ##  calculate 
    for n in xrange(Nupx):
        dd = fc-p_riemann_x[n]
        sptlIntgrl += exp(dd*dd*hlfIIQ2)*p_px[n]
    sptlIntgrl *= ((dt*l0)/sqrt(twpi*fxd_q2))*dSilenceX

    return -0.5*(fc-Mc)*(fc-Mc)/Sigma2c-sptlIntgrl

########################################################################
@cython.cdivision(True)
def adtv_support_pdf(double[::1] gx, double[::1] cond_pstr,
                     int STEPS, int cldz, int small,
                     double[::1] riemann_x, double[::1] px, int Nupx,
                     double dt, double l0, double dSilenceX,
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
    cdef int bDone= 0

    cdef double pmax = -1.e200
    cdef double gpmax= -1.e200   #  global pmax

    cdef int lvl  = 0  #  how deep in discretization level.  maximum val of cldz+1
    cdef int L

    #intvs_each_lvl = _N.empty((cldz+1)*3, dtype=_N.int)*-1
    cdef int *intvs_each_lvl = <int *>malloc(sizeof(int)*(cldz+1)*3)#, dtype=_N.int)*-1
    cdef int iBdL, iBdR
    cdef int iBdL_glbl = p_twoPOW[STEPS]+2#2**STEPS+2
    cdef int iBdR_glbl = -1

    cdef int nSmpd      = 0
    cdef int minSmps    = 20
    cdef int ix, skp, strt, stop, nxt_lvl_skp, initl_skp
    cdef double pthresh

    cdef double *p_cond_pstr = &cond_pstr[0]
    cdef double *p_riemann_x = &riemann_x[0]
    cdef double *p_px        = &px[0]
    cdef double *p_gx        = &gx[0]

    cdef double *p_params    = &params[0]

    cdef int imax   = -1
    cdef int gimax  = -1

    ####  
    ####  params conditioned on
    cdef fxd_f, fxd_q2
    #fxd_f, fxd_q2
    ####  priors    
    cdef double _q2_a, _q2_B
    cdef double _f_u, _f_q2
    #_q2_a, _q2_B
    #_f_u, _f_q2
    ####  likelihood terms :  f_c
    cdef double fc = 0

    cdef double Mc, Sigma2c, q2_cnd_on, a, B, f_cnd_on

    if dist == __NRM:
        Mc        = p_params[0]
        Sigma2c   = p_params[1]
        q2_cnd_on = p_params[2]
    else:
        a         = p_params[0]
        B         = p_params[1]
        f_cnd_on  = p_params[2]

    cdef int iFine = 0    #  elements in the fine part
    while (cldz > -1) and (bDone == 0):
        if lvl == 0:  # 1st 2 passes skip sizes are the same
            strt = 0
            skp  = p_twoPOW[cldz]          # 2**cldz
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

            #for ix in xrange(strt, stop+1, skp):  
            for strt <= ix < stop+1 by skp:
                #pdfNRM(double fc, double fxd_IIQ2, double Mc, double Sigma2c, double* p_riemann_x, double* p_px, int Nupx)
                p_cond_pstr[ix] = pdfNRM(p_gx[ix], q2_cnd_on, 1./q2_cnd_on, Mc, Sigma2c, p_riemann_x, p_px, Nupx, dt, l0, dSilenceX)

                if p_cond_pstr[ix] > pmax:
                    pmax = p_cond_pstr[ix]   # pmax updated each time grid made finer
                    imax = ix
        else:
            ##  for each q2 cond posterior calculated at:
            #   we need mean of x @ spikes, q2, # of spikes
            #   prior
            for strt <= ix < stop+1 by skp:
                p_cond_pstr[ix] = pdfIG(p_gx[ix], f_cnd_on, a, B, p_riemann_x, p_px, Nupx, dt, l0, dSilenceX)

                if p_cond_pstr[ix] > pmax:
                    pmax = p_cond_pstr[ix]   # pmax updated each time grid made finer
                    imax = ix

        if pmax > gpmax:   #  stop when 
            gpmax = pmax
            pthresh= gpmax-small   #  _N.exp(-12) == 6.144e-6
            gimax = imax

        #  start half the current skip size before and after iBdL and iBdR
        # now find left and right bounds

        ix = strt

        while (ix < gimax) and (p_cond_pstr[ix] < pthresh):
            ix += skp
        ix -= skp
        if ix >= 0:
            iBdL = ix
        else:
            while ix < 0:
                ix += skp
            iBdL = ix
        ix = stop
        while (ix > gimax) and (p_cond_pstr[ix] < pthresh):
            ix -= skp
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

    #  reconstruct lstFM
    narr_FM     = _N.empty(nSmpd, dtype=_N.int)
    cdef int ii0 = 0
    cdef int ii1 = 0
    cdef int ii = 0
    cdef int l

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
    free(intvs_each_lvl)
    #  The first iBdL is iBdL_glbl.  We always work inwards
    
    lft = _N.arange(0, iBdL_glbl - iBdL_glbl % initl_skp, initl_skp)
    rgt = _N.arange(iBdR_glbl - iBdR_glbl % initl_skp + initl_skp, 2**STEPS+2, initl_skp)
    narr_FM.sort()
    #  midInds should be same as _N.arange(iBdL_glbl, iBdR_glbl+1, fine_skip)
    L   = (lft.shape[0] + rgt.shape[0] + narr_FM.shape[0])
    adtvInds = _N.empty(L, dtype=_N.int)
    adtvInds[0:lft.shape[0]] = lft
    adtvInds[lft.shape[0]:lft.shape[0] + narr_FM.shape[0]] = narr_FM
    adtvInds[lft.shape[0] + narr_FM.shape[0]:lft.shape[0] + narr_FM.shape[0]+rgt.shape[0]] = rgt

    for l in adtvInds:   #  largest value should be 0, so exp doesn't -> 0
        p_cond_pstr[l] -= gpmax
    
    return adtvInds, L



