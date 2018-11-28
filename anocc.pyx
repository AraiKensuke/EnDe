import numpy as _N


def approx_occhist_w_gau(xp, fN):
    """
    approximate occupational histogram with gaussians.  
    fN bins of xp.  then smooth histogram, and find its peaks.
    fit peaks with Gaussian, and return
    """
    N   = xp.shape[0]

    x0 = _N.min(xp)
    x1 = _N.max(xp)

    fN  = 401   #  # of bin boundary points. # of bins is fN-1

    #  spatial histogram
    fs  = _N.linspace(-6, 6, fN, endpoint=True)  # bin edges
    cnts, bins = _N.histogram(xp, bins=fs)   #  give bins= the bin boundaries
    dx = _N.diff(bins)[0]             # bin widths
    x = 0.5*(bins[1:] + bins[0:-1])   # bin centers

    #  smooth the spatial histogram
    smth_krnl = 2
    gk        = gauKer(smth_krnl) 
    gk        /= _N.sum(gk)
    fcnts = _N.convolve(cnts, gk, mode="same")
    dfcnts= _N.diff(fcnts)

    xp_inn = _N.where((dfcnts[0:-1] <= 0) & (dfcnts[1:] > 0))[0]

    pcs    = xp_inn.shape[0] + 1
    xp_bds = _N.zeros(pcs+1, dtype=_N.int)
    xp_bds[1:pcs] = xp_inn
    xp_bds[pcs]   = fN-1

    mns = _N.empty(pcs)
    sds = _N.empty(pcs)

    for t in xrange(pcs):
        mns[t] = _N.dot(cnts[xp_bds[t]:xp_bds[t+1]], x[xp_bds[t]:xp_bds[t+1]]) / _N.sum(cnts[xp_bds[t]:xp_bds[t+1]])

        p      = cnts[xp_bds[t]:xp_bds[t+1]] / float(_N.sum(cnts[xp_bds[t]:xp_bds[t+1]]))
        sds[t] = _N.dot(p, (x[xp_bds[t]:xp_bds[t+1]]-mns[t])*(x[xp_bds[t]:xp_bds[t+1]]-mns[t]))

    return pcs, mns, sds



@cython.cdivision(True)
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
    

def smp_f(int itr, int M, int Np, double areaUnder, double[::1] mp, double[::1] sig2p, 
          long[::1] clstsz, long[::1] cls_strt_inds, 
          long[::1] sts, double[::1] xt0t1, int t0, 
          double[::1] f, double[::1] q2, double[::1] l0, 
          double[::1] _f_u, double[::1] _f_q2, double[::1] m_rands):
    global p_f_x, mv_f_x, f_x, std_gau_x, p_std_gau_x, cpf_f, mv_cpf_f, p_cpf_f, dt, N_gau_x
    cdef double fx, zx
    cdef long ifx

    cdef int m, ip
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
    cdef double U, FQ2, clstrW, nonsp

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
        clstrW = sqrt(FQ2)

        zx = clstrW*p_std_gau_x[0]  #  since we do 
        fx = U + zx
        p_f_x[0] = fx
        nonsp = 0
        for ip in xrange(Np):
            nonsp += (1./sqrt(twpi*(p_q2[m]+sig2p[ip]))) * \
                     exp(-0.5*(fx - mp[ip])*(fx-mp[ip]) / (p_q2[m] + sig2p[ip]))

        p_cpf_f[0] = -0.5*zx*zx/FQ2 - areaUnder*dt*l0[m] * nonsp
        maxp = p_cpf_f[0]
        imaxp= 0

        for ifx in xrange(1, N_gau_x):
            zx = clstrW*p_std_gau_x[ifx]  #  since we do 
            fx = U + zx
            p_f_x[ifx] = fx
            nonsp = 0
            for ip in xrange(Np):
                nonsp += (1./sqrt(twpi*(p_q2[m]+sig2p[ip]))) * \
                           exp(-0.5*(fx - mp[ip])*(fx-mp[ip]) / (p_q2[m] + sig2p[ip]))

            p_cpf_f[ifx] = -0.5*zx*zx/FQ2 - areaUnder*dt*l0[m] * nonsp
            if p_cpf_f[ifx] > maxp:
                maxp = p_cpf_f[ifx]
                imaxp= ifx
        for ifx in xrange(N_gau_x):
            p_cpf_f[ifx] -= maxp

        p_f[m] = adtv_smp_cdf_interp(itr, mv_f_x, mv_cpf_f, N_gau_x, m, m_rands)


        if (itr % 50 == 0):
            printf("iter %d   m %d  ---   clstsz %d  fs %.3f   U %.3f  clstrW %.3f\n", itr, m, p_clstsz[m], fs, U, clstrW)
            dat = _N.empty((N_gau_x, 2))
            dat[:, 0] = f_x
            dat[:, 1] = cpf_f
            _N.savetxt("ga_f_%(i)d_%(m)d.txt" % {"i" : itr, "m" : m}, dat, fmt="%.4f %.4e")

                                                            
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def adtv_smp_cdf_interp(int itr, double[::1] x, double[::1] log_p, int N, int m, double[::1] m_rnds):
    """
    each cluster has independent x over which conditional likelihood defined
    xt0t1    relative coordinates
    mks      absolute coordinates

    sLLkPr   spiking part
    s        silence part

    Caveat.  For Inverse-Gamma, if prior
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

    if (itr % 50 == 0):
        dat = _N.empty((N, 2))
        dat[:, 0] = x
        dat[:, 1] = cdf
        _N.savetxt("ga_cdf_%(i)d_%(m)d.txt" % {"i" : itr, "m" : m}, dat, fmt="%.4f %.4e")

    #  btwn cdf[isg2] and cdf[isg2+1]
    #  (rnds[m,0] - cdf[isg2]) * (cdf[isg2+1] - cdf[isg2]) * d_sg2s[isg2]
    #_isg2 = _N.searchsorted(cdf, rnd)
    _isg2 = cdf.searchsorted(rnd)
    isg2  = _isg2-1

    retRnd = p_x[isg2] + ((rnd - p_cdf[isg2]) / (p_cdf[isg2+1] - p_cdf[isg2])) * dx[isg2]  # unlike in above case, retRnd may be < 0
    free(dx)
    free(p)
    return retRnd
