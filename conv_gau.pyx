import numpy as _N
cimport numpy as _N
from libc.math cimport sqrt, log
import pickle
cimport cython

#  convolution    p(x) x N(x - u) dx

cdef double _k
cdef long _i0
cdef long _Nf, _Nq2
cdef double _f_L
cdef double _f_H
_tab = None
cdef double *p_tab


def build_gau_pxdx_tbl(px, x_grid, x_path, fs, iq2s, i0, k, f_L, f_H, usepx=False):
    global _k, _i0, _Nf, _Nq2, _f_L, _f_H, _tab, p_tab
    # fs from [100 x 100]
    #cdef double *gau_occ_table
    _Nf  = fs.shape[0]
    _Nq2 = iq2s.shape[0]

    #_tab = _N.empty((_Nf+1, _Nq2+1))
    _tab = _N.empty((_Nf, _Nq2))
    cdef double[:, ::1] v_tab = _tab
    p_tab        = &v_tab[0, 0]
    

    cdef long fi, qi
    cdef double f, iq2
    fi = -1

    cdef double isqrt2pi = 1./sqrt(2*_N.pi)

    for f in fs:
        fi += 1

        if usepx:
            dd2 = (f-x_grid)*(f-x_grid)
        else:
            dd2 = (f-x_path)*(f-x_path)
        qi = -1
        for iq2 in iq2s:
            qi += 1
            if usepx:
                p_tab[fi*_Nq2 + qi] = -_N.sum(sqrt(iq2)*isqrt2pi*_N.exp(-0.5*dd2*iq2)*px)
            else:
                p_tab[fi*_Nq2 + qi] = -_N.sum(sqrt(iq2)*isqrt2pi*_N.exp(-0.5*dd2*iq2))

    # pcklme = {}
    # pcklme["info"] = "build table of convolutions of Gaussians with a range of means and variances with the spatial occupation density p(x)."
    # pcklme["q2s"] = 1/iq2s
    # pcklme["fs"]  = fs
    # pcklme["Nf"] = Nf
    # pcklme["k"] = k
    # pcklme["f_L"] = f_L
    # pcklme["f_H"] = f_H
    # pcklme["xpath"] = x_path
    # pcklme["tbl"] = nmpy_gau_occ_table
    _k  = k
    _i0 = i0
    _f_L= f_L
    _f_H= f_H


    # dmp = open("conv_tbl.dmp", "wb")
    # pickle.dump(pcklme, dmp, -1)
    # dmp.close()






def init():
    global _k, _i0, _Nf, _f_L, _f_H, _tab
    with open("conv_tbl.dmp", "rb") as f:
        ld = pickle.load(f)
    f.close()

    _k  = ld["k"]
    _i0 = ld["i0"]
    _Nf = ld["Nf"]
    _f_L= ld["f_L"]
    _f_H= ld["f_H"]
    _tab= ld["tbl"]

@cython.cdivision(True)
cdef double conv_px(double f, double q) nogil:
    global _k, _i0, _Nf, _f_L, _f_H, _tab, p_tab
    #  read from table of pre-calculated convolutions 
    #  of p(x) [the spatial occupancy] with N(x | f, q**2)

    #  q = exp(k x (i - i0) x 0.5)     --->   log q = k x (i - i0) x 0.5
    #  i = (2/k) log q + i0
    #  so given a q' I want, i' = (2/k) log q' + i0


    #qi_r = (2./k)*_N.log(q) - i0    

    cdef double qi_r = (2/_k)*log(q) + _i0    

    cdef long qi   = <long>qi_r
    cdef double p_qi_r  = qi_r - qi

    cdef double fi_r    = (_Nf-1)*((f - _f_L) / (_f_H - _f_L))
    cdef long fi   = <long>fi_r
    #print "----  %(q).3f  %(f).3f" % {"q" : qi_r, "f" : fi_r}
    cdef double p_fi_r= fi_r - fi
    #print p_fi_r

    cdef double ans

    """
    if f >= _f_H:
        ans = (1-p_qi_r)*_tab[fi,qi] + p_qi_r*_tab[fi, qi+1]
    else:
        ans = (1-p_qi_r)*((1-p_fi_r)*_tab[fi,qi] + p_fi_r*_tab[fi+1, qi]) + \
              p_qi_r*((1-p_fi_r)*_tab[fi, qi+1] + p_fi_r*_tab[fi+1, qi+1])
    """
    if f >= _f_H:
        ans = (1-p_qi_r)*p_tab[_Nq2*fi+qi] + p_qi_r*p_tab[_Nq2*fi+qi+1]
    else:
        ans = (1-p_qi_r)*((1-p_fi_r)*p_tab[_Nq2*fi+qi] + p_fi_r*p_tab[_Nq2*(fi+1)+ qi]) + \
              p_qi_r*((1-p_fi_r)*p_tab[_Nq2*fi+qi+1] + p_fi_r*p_tab[_Nq2*(fi+1)+ qi+1])

    return ans
    # return (1-p_qi_r)*((1-p_fi_r)*_tab[fi, qi] + p_fi_r*_tab[fi+1, qi]) + \
    #     p_qi_r*((1-p_fi_r)*_tab[fi, qi+1] + p_fi_r*_tab[fi+1, qi+1])
