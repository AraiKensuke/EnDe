def f_spiking_portion(double[::1] xt0t1, long t0, long[::1] v_sts, long[::1] strt_inds, long[::1] clstsz, double[::1] q2, double[::1] _f_u, double[::1] q2pr, long M, double[::1] U, double[::1] FQ2):
    """
    Contribution of Spiking to conditional posterior of f
    xt0t1     trajectory from t0:t1.  time t0 @ index 0 
    t0       
    v_sts     flattened spike times for all clusters
    strt_inds start index in v_sts for each cluster
    clstsz    size of each of M clusters
    M         # of clusters
    q2        current value of spatial variance
    _f_u      prior mean for f
    q2pr      (maybe discounted), variance of prior for f
    U         (output) mean of spiking portion
    FQ2       (output) variance of spiking portion
    """
    cdef int m, L, i
    cdef double*  p_xt0t1     = &xt0t1[0]
    cdef long*    p_v_sts     = &v_sts[0]
    cdef long*    p_strt_inds = &strt_inds[0]
    cdef long*    p_clstsz    = &clstsz[0]
    cdef double*  p_q2        = &q2[0]
    cdef double*  _p_f_u      = &_f_u[0]
    cdef double*  _p_q2pr     = &q2pr[0]

    cdef double*  p_U         = &U[0]
    cdef double*  p_FQ2       = &FQ2[0]

    cdef double   tmp, fq2, fs

    with nogil:
        for 0 <= m < M:
            if p_clstsz[m] > 0:
                tmp = 0
                for p_strt_inds[m] <= i < p_strt_inds[m+1]:
                    tmp += p_xt0t1[p_v_sts[i]-t0]
                fs = tmp/p_clstsz[m]
                fq2= p_q2[m]/p_clstsz[m]
                p_U[m] = (fs*_p_q2pr[m] + _p_f_u[m]*fq2) / (_p_q2pr[m] + fq2)
                p_FQ2[m] = (_p_q2pr[m]*fq2) / (_p_q2pr[m] + fq2)
            else:
                p_U[m]   = _p_f_u[m]
                p_FQ2[m] = _p_q2pr[m]

