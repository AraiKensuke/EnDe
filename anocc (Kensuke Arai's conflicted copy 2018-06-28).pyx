import numpy as _N

cdef double twpi = 6.283185307

gau_x = None
cdef double[::1] mv_gau_x
cdef double *p_gau_x

def init():
    global gau_x, mv_gau_x, p_gau_x

    _x          = _N.cumsum(0.125*_N.linspace(0, 18, 19))
    gau_x       = _N.empty(19 + 18)
    gau_x[0:18] = -x[18:0:-1]
    gau_x[18:]  = x
    
    


def approx_path_w_gau(xp, pcs=100):
    N  = xp.shape[0]
    Dt = N/pcs

    mns  = _N.empty(pcs)
    sd2s = _N.empty(pcs)
    isd2s = _N.empty(pcs)

    it = -1
    for t in xrange(pcs):
        it += 1
        xs = xp[Dt*t:Dt*t+Dt]

        sd = _N.std(xs)
        mns[it] = _N.mean(xs)
        sd2s[it] = sd*sd
        isd2s[it] = 1./(sd*sd)

    return Dt, mns, sd2s, isd2s

def smp_f(int M, double[::1] mp, double[::1] sigp2, double long[::1] clstsz, long[::1] cls_strt_inds, 
          long[::1] sts, double[::1] xt0t1, int t0, 
          double[::1] f, double[::1] q2, double[::1] l0, 
          double[::1] _f_u, double[::1] _f_q2, double[::1] m_rands):
    global fx

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

        #  from U - FQ2*pts_at[fi]   #  -20, 20
        ifx = -1
        for fx in gau_x:
            ifx += 1
            pdf[ifx] = _N.exp(-0.5*(fx - U)**2/FQ2 - _N.exp(-l0/_N.sqrt(2*_N.pi*p_q2[m]) * \
                                                 _N.sum((1./_N.sqrt(2*_N.pi*(p_q2[m]+sigp2))) * \
                                                        _N.exp(-0.5*(fx - mp)*(fx-mp) / (p_q2[m] + sig2p)))))
            
        
        

