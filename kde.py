import numpy as _N
import pickle
import EnDedirs as _edd

class kde:
    setname = None
    xA = None;    mA = None     #  grid for kde histogram
    Nx = None;    Nm = None     #  how many bins in histogram
    Bx = None;    Bm = None     #  bandwidth
    bx = None;                  #  bandwidth

    Ntr= None;                  #  length of data used to construct kde

    kde= None

    def __init__(self, setname):
        self.setname = setname
        
    def est(self, xtr, markstr, xA, mA, Ntr, Nx, Nm, Bx, Bm, bx):
        oo  = self
        oo.xA = xA;    oo.mA = mA
        oo.Nx = Nx;    oo.Nm = Nm
        oo.Bx = Bx;    oo.Bm = Bm
        oo.bx = bx
        oo.Ntr= Ntr
        
        isqrt2pi = 1/_N.sqrt(2*_N.pi)

        oo.kde = _N.zeros((Nx, Nm))

        xR  = _N.linspace(-xA, xA, Nx).reshape(Nx, 1)
        mR  = _N.linspace(0, mA, Nm).reshape(1, Nm)

        isnone = _N.equal(markstr, None)
        ispks  = _N.where(isnone == False)[0]

        for ui in ispks:   # iterate over time, training
            mklst = markstr[ui]

            for mk in mklst:
                #  output this onto a Nx x Nm for each spike.  Then Sum
                oo.kde += _N.exp(-0.5*((xR - xtr[ui])/Bx)**2)*_N.exp(-0.5*((mR - mk)/Bm)**2)

        #oo.kde *= (isqrt2pi * isqrt2pi)*(1./oo.N)*(1./Bx)*(1./Bm)
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
        xR = xR.reshape(Nx, 1)
        xtr = xtr.reshape(1, Ntr)
        denom = _N.sum(_N.exp(-0.5*((xR - xtr)/bx)**2), axis=1)
        denom *= isqrt2pi*(1./bx)

        denom = denom.reshape(Nx, 1)
        oo.kde /= denom

        dmp = open(_edd.resFN("kde.dump", dir=oo.setname, create=True), "wb")
        pickle.dump(oo, dmp)
        dmp.close()
