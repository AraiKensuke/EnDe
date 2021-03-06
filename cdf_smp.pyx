import numpy as _N
cimport numpy as _N
cimport cython
import time as _tm
from libc.math cimport exp, sqrt, log
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
import matplotlib.pyplot as _plt
#uniqFN(filename, serial=False, iStart=1, returnPrevious=False)
import utilities as _U

twoPOW = _N.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072], dtype=_N.int)
cdef long[::1] v_twoPOW = twoPOW
cdef long* p_twoPOW = &v_twoPOW[0]
cdef double x_Lo
cdef double x_Hi

cdef int __NRM = 0
cdef int __IG  = 1

cdef double twpi = 6.283185307

cdef double[::1] v_cpf2
cdef double[::1] v_cpq22
cdef double[::1] v_fx2
cdef double[::1] v_qx2

#####  variables for changing resolution of occupation function
cdef double *p_riemann_xs, 
cdef double *p_px_all
cdef double *p_ibinszs
cdef double *p_q2_thr
cdef double *dSilenceXs
cdef int minSmps

cdef long *Nupxs
cdef long *arr_0s
cdef long n_res   #  number of different resolutions 
cdef double x_lo, x_hi

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
#dSilenceX= None

adtv_pdf_params = _N.empty(3)

########################################################################
def init(_dt, _f_lo, _f_hi, _q2_lo, _q2_hi, _f_STEPS, _q2_STEPS, _f_SMALL, _q2_SMALL, _f_cldz, _q2_cldz, _minSmps):
    global fx2, qx2, cpf2, cpq22, 
    global v_fx2, v_qx2, v_cpf2, v_cpq22, 
    global dt
    global f_lo, f_hi, q2_lo, q2_hi
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
def adtv_smp_cdf_interp(double[::1] x, double[::1] log_p, int N, int m, double[::1] m_rnds):
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
def smp_f(int M, long[::1] clstsz, long[::1] cls_strt_inds, long[::1] sts, 
#          double[::1] fx, double[::1] pxO_full, 
          double[::1] xt0t1, int t0, 
          double[::1] f, double[::1] q2, double[::1] l0, 
          double[::1] _f_u, double[::1] _f_q2, double[::1] m_rands):
    global f_STEPS, f_SMALL, f_cldz
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

        adtvInds, N = adtv_support_pdf(fx2, cpf2, f_STEPS, f_cldz, f_SMALL, dt, l0[m], _NRM, adtv_pdf_params, )

        p_f[m] = adtv_smp_cdf_interp(fx2[adtvInds], cpf2[adtvInds], N, m, m_rands)

########################################################################
#@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def smp_q2(int M, long[::1] clstsz, long[::1] cls_strt_inds, long[::1] sts, 
           double[::1] xt0t1, int t0, 
           double[::1] f, double[::1] q2, double[::1] l0, 
           double[::1] _q2_a, double[::1] _q2_B, double[::1] m_rands, int ep):
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

        adtvInds, N = adtv_support_pdf(qx2, cpq22, q2_STEPS, f_cldz, f_SMALL, dt, l0[m], _IG, adtv_pdf_params, )
        p_q2[m] = adtv_smp_cdf_interp(qx2[adtvInds], cpq22[adtvInds], N, m, m_rands)
        dat = _N.empty((len(adtvInds), 2))
        dat[:, 0] = qx2[adtvInds]
        dat[:, 1] = cpq22[adtvInds]
        # fn = _U.uniqFN("adtv%d" % m, serial=True, iStart=0)
        # _N.savetxt(fn, dat, fmt="%.4e %.4e")


########################################################################
@cython.cdivision(True)
cdef double pdfIG(double q2c, double fxd_f, double a, double B, double* p_riemann_x, double *p_px, long Nupx, double ibnsz, double dt, double l0, double dSilenceX, double xL, double xH) nogil:
    #  
    cdef double hlfIIQ2 = -0.5/q2c
    cdef double sptlIntgrl = 0.0
    cdef double dd
    cdef int n, iL, iR
    cdef double sd = sqrt(q2c)

    iL = int((fxd_f-6*sd-xL)*ibnsz)
    iR = int((fxd_f+6*sd-xL)*ibnsz)
    iL = iL if iL >= 0 else 0
    iR = iR if iR <= Nupx else Nupx

    #for n in xrange(Nupx):   #  spatial integral
    for iL <= n < iR:    #  integrate
        dd = fxd_f-p_riemann_x[n]
        sptlIntgrl += exp(dd*dd*hlfIIQ2)*p_px[n]
    sptlIntgrl *= ((dt*l0)/sqrt(twpi*q2c))*dSilenceX
    
    return -(a + 1)*log(q2c) - B/q2c-sptlIntgrl

########################################################################
@cython.cdivision(True)
cdef double pdfNRM(double fc, double fxd_q2, double fxd_IIQ2, double Mc, double Sigma2c, double *p_riemann_x, double *p_px, long Nupx, double ibnsz, double dt, double l0, double dSilenceX, double xL, double xH) nogil:
    #  Value of pdf @ fc.  
    #  fxd_IIQ2    1./q2_c
    #  Mc          - spiking + prior  mean
    #  Sigma2c     - spiking + prior  variance
    #  p_riemann_x - points at which integral discretized
    #  p_px        - occupancy
    cdef double hlfIIQ2 = -0.5*fxd_IIQ2
    cdef double sptlIntgrl = 0.0
    cdef double dd = 0
    cdef int n, iL, iR, iL_, iR_
    cdef double sd = sqrt(fxd_q2)
    
    iL = int((fc-6*sd-xL)*ibnsz)
    iR = int((fc+6*sd-xL)*ibnsz)
    iL = iL if iL >= 0 else 0
    iR = iR if iR <= Nupx else Nupx

    ##  calculate 
    #for n in xrange(Nupx):    #  integrate
    for iL <= n < iR:    #  integrate
        dd = fc-p_riemann_x[n]
        sptlIntgrl += exp(dd*dd*hlfIIQ2)*p_px[n]
    sptlIntgrl *= ((dt*l0)/sqrt(twpi*fxd_q2))*dSilenceX

    # if iDBG == 1:
    #     iL_ = int((fc-6*sd-xL)*ibnsz)
    #     iR_ = int((fc+6*sd-xL)*ibnsz)
    #     printf("Nupx---  %d", Nupx)
    #     printf("sptlIntgrl----  %d  %d    %.4f   %d %d\n", iL, iR, sptlIntgrl, iL_, iR_)
    #     printf("fc %.4e   sd %.4e    Mc %.4e  Sigma2c %.4e\n", fc, sd, Mc, Sigma2c)

    return -0.5*(fc-Mc)*(fc-Mc)/Sigma2c-sptlIntgrl


########################################################################
@cython.cdivision(True)
def l0_spatial(long M, double dt, double[::1] v_fxd_fc, double[::1] v_fxd_q2, double[::1] v_l0_exp_px):
    #  Value of pdf @ fc.  
    #  fxd_IIQ2    1./q2_c
    #  Mc          - spiking + prior  mean
    #  Sigma2c     - spiking + prior  variance
    #  p_riemann_x - points at which integral discretized
    #  p_px        - occupancy
    global p_riemann_xs;      global p_px_all
    global x_Lo, x_Hi

    cdef double hlfIIQ2
    cdef double sptlIntgrl = 0.0
    cdef long Nupx, iStart
    cdef int n, m, iL, iR
    cdef double dSilenceX, ibnsz, sd
    cdef double *p_l0_exp_px = &v_l0_exp_px[0]
    cdef double *p_fxd_fc    = &v_fxd_fc[0]
    cdef double *p_fxd_q2    = &v_fxd_q2[0]

    ##  calculate 
    for m in xrange(M):
        fc = p_fxd_fc[m]
        q2c= p_fxd_q2[m]
        getOccDens(q2c, &Nupx, &iStart, &dSilenceX, &ibnsz)
        sptlIntgrl = 0.0
        hlfIIQ2 = -0.5/q2c
        sd = sqrt(q2c)

        iL = int((fc-6*sd-x_Lo)*ibnsz)

        iR = int((fc+6*sd-x_Lo)*ibnsz)
        iL = iL if iL >= 0 else 0
        iR = iR if iR <= Nupx else Nupx

        #for n in xrange(Nupx):
        for iL <= n < iR:
            sptlIntgrl += exp(((p_riemann_xs[iStart+n]-fc)*(p_riemann_xs[iStart+n]-fc))*hlfIIQ2)*p_px_all[iStart+n]
        sptlIntgrl *= (dt/sqrt(twpi*q2c))*dSilenceX
        p_l0_exp_px[m] = sptlIntgrl

########################################################################
@cython.cdivision(True)
def adtv_support_pdf(double[::1] gx, double[::1] cond_pstr,
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
    global p_riemann_xs;      global p_px_all;      global p_ibinszs
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
    cdef long Nupx, iStart
    cdef double dSilenceX, ibnsz
    cdef int istepbck = 0

    cdef double Mc, Sigma2c, q2_cnd_on, a, B, f_cnd_on, iq2_cnd_on

    if dist == __NRM:
        Mc        = p_params[0]
        Sigma2c   = p_params[1]
        q2_cnd_on = p_params[2]
        iq2_cnd_on= 1./q2_cnd_on
    else:
        a         = p_params[0]
        B         = p_params[1]
        f_cnd_on  = p_params[2]

    #initial point, set pmax, gmax
    if dist == __NRM:
        getOccDens(q2_cnd_on, &Nupx, &iStart, &dSilenceX, &ibnsz)
        p_cond_pstr[0] = pdfNRM(p_gx[0], q2_cnd_on, iq2_cnd_on, Mc, Sigma2c, &p_riemann_xs[iStart], &p_px_all[iStart], Nupx, ibnsz, dt, l0, dSilenceX, x_Lo, x_Hi)
    elif dist == __IG:
        getOccDens(p_gx[0], &Nupx, &iStart, &dSilenceX, &ibnsz)
        p_cond_pstr[0] = pdfIG(p_gx[0], f_cnd_on, a, B, &p_riemann_xs[iStart], &p_px_all[iStart], Nupx, ibnsz, dt, l0, dSilenceX, x_Lo, x_Hi)
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

            getOccDens(q2_cnd_on, &Nupx, &iStart, &dSilenceX, &ibnsz)
            for strt <= ix < stop+1 by skp:
                p_cond_pstr[ix] = pdfNRM(p_gx[ix], q2_cnd_on, iq2_cnd_on, Mc, Sigma2c, &p_riemann_xs[iStart], &p_px_all[iStart], Nupx, ibnsz, dt, l0, dSilenceX, x_Lo, x_Hi)

                if p_cond_pstr[ix] > pmax:
                    pmax = p_cond_pstr[ix]   # pmax updated each time grid made finer
                    imax = ix
        else:
            ##  for each q2 cond posterior calculated at:
            #   we need mean of x @ spikes, q2, # of spikes
            #   prior
            #   p_gx[ix]   values of q2
            for strt <= ix < stop+1 by skp:
                getOccDens(p_gx[ix], &Nupx, &iStart, &dSilenceX, &ibnsz)
                p_cond_pstr[ix] = pdfIG(p_gx[ix], f_cnd_on, a, B, &p_riemann_xs[iStart], &p_px_all[iStart], Nupx, ibnsz, dt, l0, dSilenceX, x_Lo, x_Hi)

                if p_cond_pstr[ix] > pmax:
                    pmax = p_cond_pstr[ix]   # pmax updated each time grid made finer
                    imax = ix

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

        #     getOccDens(p_gx[l], &Nupx, &iStart, &dSilenceX, &ibnsz)            
        #     print iStart
        #     print dSilenceX
        #     print ibnsz
        #     print Nupx
        #     tryagain = pdfIG(p_gx[l], f_cnd_on, a, B, &p_riemann_xs[iStart], &p_px_all[iStart], Nupx, ibnsz, dt, l0, dSilenceX, x_Lo, x_Hi)
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
def init_occ_resolutions(_x_Lo, _x_Hi, v_q2_thr, v_Nupxs, ):    
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
    #  
    global p_px_all;    global Nupxs;    global n_res
    global p_q2_thr;    global arr_0s;   global p_riemann_xs
    global p_ibinszs
    global dSilenceXs
    global x_Lo, x_Hi

    cdef int i, j, tot = 0
    cdef double dx, x
    x_Lo = _x_Lo
    x_Hi = _x_Hi
    n_res         = len(v_Nupxs)

    Nupxs        = <long*>malloc(n_res*sizeof(long))
    p_ibinszs    = <double*>malloc(n_res*sizeof(double))
    arr_0s       = <long*>malloc(n_res*sizeof(long))
    dSilenceXs   = <double*>malloc(n_res*sizeof(double))
    p_q2_thr     = <double*>malloc(n_res*sizeof(double))
    p_px_all     = <double*>malloc(_N.sum(v_Nupxs)*sizeof(double))
    p_riemann_xs = <double*>malloc(_N.sum(v_Nupxs)*sizeof(double))


    for 0 <= i < n_res:
        p_q2_thr[i] = v_q2_thr[i]
        Nupxs[i] = v_Nupxs[i]
        arr_0s[i] = tot

        dx = float(x_Hi - x_Lo) / Nupxs[i]
        p_ibinszs[i] = 1./dx
        x  = x_Lo + 0.5*dx
        for 0 <= j < Nupxs[i]:
            p_riemann_xs[tot+j] = x
            #print "%.4f" % p_riemann_xs[tot+j]
            x += dx

        tot += Nupxs[i]

    #print "ibinszs"
    #for 0 <= i < n_res:
    #    print("%.4f\n" %  p_ibinszs[i])

def clean_occ_resolutions():
    global p_px_all;   global p_riemann_xs;   global p_q2_thr
    global Nupxs;      global arr_0s
    free(p_px_all);    free(p_riemann_xs);    free(p_q2_thr)
    free(Nupxs);       free(arr_0s)

def change_occ_px(pth, _x_Lo, _x_Hi):
    global p_px_all;    global Nupxs;         global n_res
    global p_q2_thr;    global arr_0s;        global p_riemann_xs
    N_pth = len(pth)
    global dSilenceXs

    cdef int i, j
    for 0 <= i < n_res:
        dSilenceXs[i] = (N_pth / float(Nupxs[i]))*(_x_Hi - _x_Lo)
        px, bns = _N.histogram(pth, _N.linspace(_x_Lo, _x_Hi, Nupxs[i]+1), normed=True)

        for 0 <= j < Nupxs[i]:
            p_px_all[arr_0s[i] + j] = px[j]

    #  for each
    
    
# set BOTH the correct occupation density AND riemann_x
cdef void getOccDens(double gau_q2, long *Nupx, long *iStart, double *dSilenceX, double *ibnsz):
    global Nupxs;      global arr_0s
    global n_res;      global p_q2_thr
    global p_ibinszs
    cdef int i

    for 0 <= i < n_res:
        if gau_q2 < p_q2_thr[i]:
            Nupx[0]   = Nupxs[i]
            iStart[0] = arr_0s[i]
            dSilenceX[0]= dSilenceXs[i]
            ibnsz[0]= p_ibinszs[i]
            return 
