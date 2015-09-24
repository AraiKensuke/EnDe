import numpy as _N
import matplotlib.pyplot as _plt

def generateMvt(N, vAmp=1):
    """
    vAmp  is speed amplitude
    """
    x = 0
    pos = _N.empty(N)

    dirc = -1 if _N.random.rand() < 0.5 else 1

    done = False
    v    = 0
    for n in xrange(N):
        v = 0.9995*v + 0.00007*_N.random.randn()
        x += vAmp*_N.abs(v)*dirc

        if (x > 6) and (dirc > 0):
            done = True
        elif (x < -6) and (dirc < 0):
            done = True
        if done:
            dirc = -1 if _N.random.rand() < 0.5 else 1
            if dirc < 0:
                if x <= -6:  #  -6.1 -> -0.1
                    x = x + 6
                else:  #  6.1 -> -0.1
                    x = -1*(x - 6)
            if dirc > 0:
                if x <= -6:  #  -6.1 -> 0.1
                    x = -1*(x + 6)
                else:  #  6.1 -> 0.1
                    x = x - 6
        pos[n] = x
        done = False

    return pos

def compareLklhds(dec, t0, t1, tet=0, scale=1.):
    it0 = int(t0*scale)
    it1 = int(t1*scale)

    pg   = 0
    onPg = 0
    
    for t in xrange(it0, it1):
        if dec.marks[t, tet] is not None:
            if onPg == 0:
                fig = _plt.figure(figsize=(11, 8))        
            fig.add_subplot(4, 5, onPg + 1)
            _plt.plot(dec.xp, dec.Lklhd[0, t])
            _plt.axvline(x=dec.pos[t])
            _plt.yticks([])
            _plt.xticks([-6, -3, 0, 3, 6])
            _plt.title("t = %.3f" % (float(t) / scale))
            fig.add_subplot(4, 5, onPg + 2)
            _plt.plot(dec.xp, dec.pX_Nm[t])        
            _plt.axvline(x=dec.pos[t])
            _plt.yticks([])
            _plt.xticks([-6, -3, 0, 3, 6])
            _plt.title("t = %.3f" % (float(t) / scale))
            onPg += 2

        if onPg >= 20:
            fig.subplots_adjust(wspace=0.35, hspace=0.35, left=0.08, right=0.92, top=0.92, bottom=0.08)
            _plt.savefig("compareLklhds_pg=%d" % pg)
            _plt.close()
            pg += 1
            onPg = 0

    if onPg > 0:
        fig.subplots_adjust(wspace=0.15, hspace=0.15, left=0.08, right=0.92, top=0.92, bottom=0.08)
        _plt.savefig("compareLklhds_pg=%d" % pg)
        _plt.close()



# def kerLklhd(atMark, lklhd_x, tr_pos, tr_mks, atMark, mdim, Bx, cBm, bx, t0=None, t1=None):
#     """
#     return me a function of position.  Call for each new received mark.

#     lklhd_x is    (1, dimNx)   likelihood at these x values
#     tr_pos is dim (nSpks, 1) 
#     tr_mks        (nSpks x k)
#     atMark        (1 x k)
#     """
#     iB2    = 1/(cBm*cBm)

#     nSpks  = 0
#     Lklhd  = 0

#     q4mk = -0.5*_N.sum((trng_mks - atMark) * (trng_mks - atMark), axis=1)*iB2

#     _N.exp(-0.5*((lklhd_x - tr_pos)*iBx)**2)  #  this piece doesn't need to be evaluated for every new spike

#     _N.sum(_N.exp(q4mk))
#     (isqrt2pi * isqrt2pi**mdim)*(1./nSpks)*(1./Bx)*(1./cBm**mdim)

