import numpy as _N

class priors:
    #  in array, value for [non_hash, hash]
    #  for hash
    #  cluster center uncertainty
    _fx_u    = _N.array([0.2, 0.2])
    _fy_u    = _N.array([0., 0.])
    _fx_q2   = _N.array([4., 4.])
    _fy_q2   = _N.array([4., 4.])
    #_f_q2  =  [100., 100.] #  cluster center uncertainty

    #  _ss.invgamma.rvs(q2_a_ + 1, scale=q2_B_)
    #  mean is B / (a-1) for a > 1
    #  mode is B / (a+1) for a > 1
    _q2x_a   = _N.array([1.1, 1.1])
    _q2x_B   = _N.array([2e-2, 2e-2])
    _q2y_a   = _N.array([1.1, 1.1])
    _q2y_B   = _N.array([2e-2, 2e-2])

    #  _ss.gamma.rvs(l0_a_, scale=(1/l0_B_))
    #  mean is l0_a_*l0_B_.  Small l0_a_ gives higher variability
    _l0_a   = [1.1, 1.1]   # (a-1) / B    
    _l0_B   = [1./200, 1/4.]   #0.1/(1./2000)
    ##

    _u_Sg   = None
    _u_u    = None

    _Sg_nu   = None
    _Sg_PSI    = None

