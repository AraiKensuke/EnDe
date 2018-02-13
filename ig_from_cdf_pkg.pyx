import scipy.stats as _ss
import time as _tm
import pickle
import numpy as _N
cimport numpy as _N   #  this is very important
from libc.math cimport log
from libc.stdlib cimport atol

cdef double Bl, Bh, log_al, log_ah, i_Bh_m_Bl, i_lah_m_lal
cdef long cdf_N, la_N, N2, N3, N2N3
nmpy_CDF    = None
cdef double *CDF

def init():
    global nmpy_CDF, Bl, Bh, log_al, log_ah, i_Bh_m_Bl, i_lah_m_lal, la_N, cdf_N, N2, N3, N2N3, CDF
    with open("ig_table.dmp", "rb") as f:
        lm = pickle.load(f)
    f.close()
    Bl = lm["Bl"]
    Bh = lm["Bh"]
    log_al = lm["log_al"]
    log_ah = lm["log_ah"]
    la_N   = lm["la_N"]
    cdf_N   = lm["cdf_N"]
    N2     = cdf_N+1
    N3     = la_N+1
    N2N3   = N2*N3
    i_Bh_m_Bl   = 1./ (Bh - Bl)
    i_lah_m_lal = 1./ (log_ah - log_al)
    
    nmpy_CDF    = _N.array(lm["sg2_at_cdfs"])
    cdef double[:, :, ::1] v_CDF     = nmpy_CDF
    CDF = &v_CDF[0, 0, 0]

#  passing in a rnds[i] doesn't speed this up
def sampIG_single(double a, double B):
    global nmpy_CDF, Bl, Bh, log_al, log_ah, i_Bh_m_Bl, i_lah_m_lal, cdf_N, la_N, N2, N3, N2N3
    cdef double pB = (B - Bl) * i_Bh_m_Bl

    # #  find appropriate a
    cdef double log_a    = log(a)

    cdef double da       = la_N * (log_a - log_al) * i_lah_m_lal
    cdef long ia           = <long>(da)
    cdef double pa        = da - ia


    cdef double rnd = _N.random.rand()
    cdef long ic  = <long>(rnd*cdf_N)
    cdef double pc  = rnd*cdf_N - ic
    cdef double m1pc  = 1-pc
    cdef double m1pa  = 1-pa
    cdef double m1pB  = 1-pB
    cdef long i0      = ia*N2 + ic

    rval = m1pc*((m1pa*CDF[ia*N2+ic]+pa*CDF[(ia+1)*N2+ic])*m1pB + \
                 (m1pa*CDF[N2N3 + ia*N2 + ic]+pa*CDF[N2N3 + (ia+1)*N2 + ic])*pB) + \
        pc*((m1pa*CDF[ia*N2+ic+1]+pa*CDF[(ia+1)*N2+ic+1])*m1pB + \
            (m1pa*CDF[N2N3 + ia*N2 + ic+1]+pa*CDF[N2N3 + (ia+1)*N2 + ic+1])*pB)

    return rval

