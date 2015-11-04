import numpy as _N

isqrt2pi = 1/_N.sqrt(2*_N.pi)

def kerFr(atMark, sptl, tr_mks, mdim, Bx, cBm, bx):
    """
    Return me firing rate if I were to only to count spikes w/ mark "atMark"

    fld_x is    (1, dimNx)      return firing rate at these x values
    #tr_pos is dim (nSpks, 1)    training positions
    tr_mks        (nSpks x k)   training marks
    atMark        (1 x k)       look at only spikes w/ this mark
    all_pos
    """
    iB2    = 1/(cBm*cBm)

    nSpks  = tr_mks.shape[0]
    q4mk   = -0.5*_N.sum((tr_mks - atMark) * (tr_mks - atMark), axis=1)*iB2
    #  fld_x - tr_pos  gives array (# positions to evaluate x # of trainingated for every new spike
    #occ    = _N.sum(_N.exp(-0.5*iBx2*(fld_x - all_pos)**2), axis=1)  #  this piece doesn't need to be evaluated for every new spike
    #occ    = _N.sum(_N.exp(-0.5*ibx2*(oo.xpr - oo.all_pos)**2), axis=1)  #  this piece doesn't need to be evaluated for every new spike

    fr1= (isqrt2pi * isqrt2pi**mdim)*(1./nSpks)*(1./Bx)*(1./cBm**mdim)*_N.sum(_N.exp(sptl + q4mk), axis=1)  #  return me # of positions to evaluate
    #fr1 /= (occ + Tot_occ*0.01)

    """   EQUIVALENT TO
    fr2     = _N.zeros(fld_x.shape[0])
    for n in xrange(nSpks):
        fr2 += _N.exp(-0.5*iB2*_N.sum((tr_mks[n]-atMark)**2)) * _N.exp(-0.5*iBx2*(fld_x[:, 0] - tr_pos[n])**2)
    fr2 *= (isqrt2pi * isqrt2pi**mdim)*(1./nSpks)*(1./Bx)*(1./cBm**mdim)
    """

    return fr1

def Lambda(fld_x, tr_pos, all_pos, Bx, bx):
    """
    return me a function of position.  Call for each new received mark.

    fld_x is    (1, dimNx)   likelihood at these x values
    tr_pos is dim (nSpks, 1) 
    tr_mks        (nSpks x k)
    atMark        (1 x k)
    """
    iBx2   = 1/(bx*bx)

    nSpks  = tr_pos.shape[0]
    sptl   = -0.5*iBx2*(fld_x - tr_pos)**2  #  this piece doesn't need to be evaluated for every new spike
    occ    = _N.sum(_N.exp(-0.5*iBx2*(fld_x - all_pos)**2), axis=1)  #  this piece doesn't need to be evaluated for every new spike
    #occ    = _N.sum(_N.exp(-0.5*ibx2*(oo.xpr - oo.all_pos)**2), axis=1)  #  this piece doesn't need to be evaluated for every new spike
    Tot_occ  = _N.sum(occ)
    print Tot_occ


    Lam    = isqrt2pi*(1./nSpks)*(1./Bx)*_N.sum(_N.exp(sptl), axis=1)  #  return me # of positions to evaluate
    Lam    /= (occ + Tot_occ*0.01)

    return Lam
