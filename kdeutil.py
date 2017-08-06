import numpy as _N
import matplotlib.pyplot as _plt

isqrt2pi = 1/_N.sqrt(2*_N.pi)

def kerFr(atMark, sptl, tr_mks, mdim, Bx, cBm, bx, dxp, occ):
    """
    Return me firing rate if I were to only to count spikes w/ mark "atMark"

    fld_x is    (1, dimNx)      return firing rate at these x values
    #tr_pos is dim (nSpks, 1)    training positions
    sptl          (training positions - field positions)**2
    tr_mks        (nSpks x k)   training marks
    atMark        (1 x k)       mark of spike to be decoded
    all_pos
    """
    iB2    = 1/(cBm*cBm)

    nTrSpks  = tr_mks.shape[0]

    #  q4mk  shape (nSpks, )   treat as (1, nSpks) when q4mk + sptl
    #  sptl  shape (Nx, nSpks)

    #  sum over mark dims
    q4mk   = -0.5*_N.sum((tr_mks - atMark) * (tr_mks - atMark), axis=1)*iB2

    #  fld_x - tr_pos  gives array (# positions to evaluate x # of trainingated for every new spike
    #occ    = _N.sum(_N.exp(-0.5*iBx2*(fld_x - all_pos)**2), axis=1)  #  this piece doesn't need to be evaluated for every new spike
    #occ    = _N.sum(_N.exp(-0.5*ibx2*(oo.xpr - oo.all_pos)**2), axis=1)  #  this piece doesn't need to be evaluated for every new spike

    #  if H diagonal with diagonal cB**2
    #  det(H) == (cB^2)^mdim), so |det(H)|^(1/2) == cB^(2*mdim*0.5) == cB^mdim

    fr1= (isqrt2pi * isqrt2pi**mdim)*(1./Bx)*(1./nTrSpks)*(1./cBm**mdim)*_N.sum(_N.exp(sptl + q4mk), axis=1) / (occ * 0.001)  #  return me # of positions to evaluate

    # fr1 = _N.zeros(sptl.shape[0])

    # for n in xrange(nTrSpks):
    #     fr1 += _N.exp(q4mk[n])*_N.exp(sptl[:, n])
    # fr1 *= (isqrt2pi * isqrt2pi**mdim)*(1./Bx)*(1./nTrSpks)*(1./cBm**mdim)*dxp / (occ * 0.001)*2000
    #  sptl


    """   EQUIVALENT TO
    fr2     = _N.zeros(fld_x.shape[0])
    for n in xrange(nSpks):
        fr2 += _N.exp(-0.5*iB2*_N.sum((tr_mks[n]-atMark)**2)) * _N.exp(-0.5*iBx2*(fld_x[:, 0] - tr_pos[n])**2)
    fr2 *= (isqrt2pi * isqrt2pi**mdim)*(1./nSpks)*(1./Bx)*(1./cBm**mdim)
    """

    return fr1

def Lambda(fld_x, tr_pos, all_pos, Bx, bx, dxp, occ):
    """
    return me a function of position.  Call for each new received mark.

    fld_x is    (dimNx x 1)   likelihood at these x values
    tr_pos is dim (nSpks)      #  these get treated as if (1 x nSpks)
    tr_mks        (nSpks x k)
    atMark        (1 x k)
    """
    iBx2   = 1/(bx*bx)

    #      sptl    is Nx x nSpks (= tr_pos.shape[0])
    sptl   = -0.5*iBx2*(fld_x - tr_pos)**2  #  this piece doesn't need to be evaluated for every new spike
    #occ    = (1/_N.sqrt(2*_N.pi*bx*bx))*_N.sum(_N.exp(-0.5*iBx2*(fld_x - all_pos)**2), axis=1)*dxp #  this piece doesn't need to be evaluated for every new spike
    #  occ[i]: amount time in ms spent in bin i. sum(occ) = total time in ms

    CtsPrBin    = isqrt2pi*(1./bx)*_N.sum(_N.exp(sptl), axis=1)*dxp  #  sum(Lam)*dxp == number of spikes observed all bins
    Lam    = CtsPrBin / (occ*0.001)  #  divide by time (s) in each bin
    #  sum(occupancy) == time spent in each bin

    return Lam

def evalAtFxdMks_new(fxdMks, l0, us, Sgs, iSgs, i2pidcovsr):
    Nx, pmdim     = fxdMks.shape
    #  fxdMks is a 
    fxdMksr= fxdMks.reshape(Nx, 1, pmdim)

    #  
    #  cmps is spatial contribution 
    cmps = i2pidcovsr*_N.exp(-0.5*_N.einsum("xmj,xmj->mx", fxdMksr-us, _N.einsum("mjk,xmk->xmj", iSgs, fxdMksr - us)))

    zs = _N.sum(l0*cmps, axis=0)

    return zs
    """

    fld = fxdMks[:, 0].reshape(Nx, 1)    #  fld is (Nx, 1)

    ll  = _N.zeros(Nx)
    mdim= pmdim - 1

    for n in xrange(us.shape[0]):
        atMk= fxdMks[n, 1:]
        x = us[n, 0]
        m = us[n, 1:]
        #ll += (l0[n, 0]*(1/_N.sqrt(2*_N.pi*Sgs[n,0, 0])) * _N.exp(-0.5*(fld - us[n, 0])*(fld - us[n, 0])*iSgs[n,0, 0]))[:, 0]
        ll += (l0[n, 0]*(1/_N.sqrt(2*_N.pi*Sgs[n,0, 0])) * _N.exp(-0.5*(fld - x)*(fld - x)*iSgs[n,0, 0]) * (1/_N.sqrt(2*_N.pi)**mdim)*(1./_N.sqrt(_N.linalg.det(Sgs[n,1:, 1:]))) * _N.exp(-0.5*_N.dot((atMk - m), _N.dot(iSgs[n, 1:, 1:], (atMk - m)))))[:, 0] 
    """

    return ll
