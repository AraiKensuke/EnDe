class priors:
    #  in array, value for [non_hash, hash]
    #  for hash
    _f_u    = [0., 0.]    
    _f_q2  =  [16., 16.] #  cluster center uncertainty

    #  _ss.invgamma.rvs(q2_a_ + 1, scale=q2_B_)
    #  mean is B / (a-1) for a > 1
    #  mode is B / (a+1) for a > 1
    _q2_a   = [-1, -1]
    _q2_B   = [0, 0]

    #  _ss.gamma.rvs(l0_a_, scale=(1/l0_B_))
    #  mean is l0_a_*l0_B_.  Small l0_a_ gives higher variability
    _l0_a   = [1., 1.]       #  x^{a-1} x 
    _l0_B   = [0., 0.]   #0.1/(1./2000)

