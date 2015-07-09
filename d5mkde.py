import numpy as _N
import pickle
import EnDedirs as _edd

class kde:
    setname = None
    xA = None;    mA = None     #  grid for kde histogram
    Nx = None;    Nm = None     #  how many bins in histogram
    Bx = None;    cBm = None    #  bandwidth
    bx = None;                  #  bandwidth

    Ntr= None;                  #  length of data used to construct kde

    kde= None

    def __init__(self, setname):
        self.setname = setname

    #  xtr is training stimuli
    def est(self, xtr, mdim, markstr, xA, mA, Ntr, Nx, Nm, Bx, cBm, bx, t0=None, t1=None, filename="kde.dump"):
        oo  = self

        oo.xA = xA;    oo.mA = mA
        oo.Nx = Nx;    oo.Nm = Nm
        oo.Bx = Bx;    oo.cBm = cBm
        oo.bx = bx
        oo.Ntr= Ntr
        print "t0 %(0)d  t1 %(1)d" % {"0" : t0, "1" : t1}
        if (t0 is not None) and (t1 is not None):
            oo.Ntr = t1-t0
        else:
            t0 = 0
            t1 = oo.Ntr
        
        isqrt2pi = 1/_N.sqrt(2*_N.pi)

        nard = [oo.Nx]
        for d in xrange(mdim):
            nard.append(oo.Nm)

        oo.kde = _N.zeros(nard)
        oo.kde = oo.kde.reshape(oo.Nx, Nm**mdim)
        print "kde.shape"
        print oo.kde.shape

        iBx = 1./Bx
        xR  = _N.linspace(-xA, xA, Nx).reshape(Nx, 1)
        mR  = _N.linspace(0, mA, Nm)
        mkpts = Nm**mdim   #  # grid pts on which to compute KDE

        args = []
        for d in xrange(mdim):
            args.append(mR)

        mk_grid = _N.array(_N.meshgrid(*args, indexing="ij"))

        mk_grid = mk_grid.reshape(mdim, mkpts).T
        mk_grid = mk_grid.reshape(1, mdim*mkpts)

        isnone = _N.equal(markstr, None)
        ispks  = _N.where(isnone == False)[0]
        it0t1  = _N.where((ispks >= t0) & (ispks < t1))[0]
        print ispks
        print it0t1

        iB2    = 1/(cBm*cBm)

        print len(ispks[it0t1])
        for ui in ispks[it0t1]:   # iterate over time, training
            mklst = markstr[ui]

            for mk in mklst[1:]:   #  sum over spikes
                #  output this onto a Nx x Nm for each spike.  
                #  Here oo.kde is a vector
                #  Explicit sum over marks

                #  Either I need a 
                #print mk
                mk_tile = _N.tile(mk, mkpts)
                mk_tile = mk_tile.reshape(1, mkpts*mdim)
                #print mk_grid.shape
                #print mk_tile.shape
                #print (xR - xtr[ui]).shape
                mprt = _N.multiply(mk_grid - mk_tile, mk_grid - mk_tile)
                mprt = mprt.reshape(mkpts, mdim)
                mprt = _N.sum(mprt, axis=1)

                mprt = mprt.reshape(1, mkpts)
                #print mprt.shape  # components of mark consecutive.  we want to sum every 3
                #                xprt = (xR - xtr[ui])**2
                #print xprt.shape
                #print (mprt + xprt).shape
                oo.kde += _N.exp(-0.5*((xR - xtr[ui])*iBx)**2 - 0.5 * mprt*iB2)

        # oo.kde *= (isqrt2pi * isqrt2pi)*(1./oo.N)*(1./Bx)*(1./Bm)
        #  x(t) - x_j   
        #  lambda(x, m) 

        #  K is a function of x(t)
        #  denom is also a function of x(t)

        """
        denom = _N.zeros(oo.Nx)
        for n in xrange(oo.N):
            denom += _N.exp(-0.5*((xR - oo.xtr[n])*(xR - oo.xtr[n]))/bx)
        """
        ###  func of x times func of m.  lives on a grid 
        #xR = xR.reshape(Nx, 1)
        xtr = xtr.reshape(1, Ntr)
        denom = _N.sum(_N.exp(-0.5*((xR - xtr)/bx)**2), axis=1)
        denom *= isqrt2pi*(1./bx)

        denom = denom.reshape(Nx, 1)
        print oo.kde.shape
        oo.kde /= denom

        dmp = open(_edd.resFN(filename, dir=oo.setname, create=True), "wb")
        pickle.dump(oo, dmp)
        dmp.close()
