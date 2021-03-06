import utils as _U
import numpy as _N
import pickle
import EnDedirs as _edd
import d5mkde as _kde
from filter import gauKer
mvn = _N.random.multivariate_normal
import matplotlib.pyplot as _plt

###  For this one, we allow for 

_non_hash_ = 0
_hash_     = 1
_both_     = 2

class mixNormDatPoi:
    k      = 2      #  data dimensionality
    N      = 500    #  no of datas
    M      = 3      #  no components
    Mh_strt      = 0      #  start component for hash clusters

    um      = None   #  mean of marks for each neuron
    Cov     = None   #  cov  of marks for each neuron

    uP      = None  #  mean of place fields for each neuron
    stdP    = None  #  std  of place fields for each neuron
    stdPMags= None  #

    marksH  = None
    marksNH = None

    lmd     = None  #  

    pos      = None   #  position
    neurIDs  = None #  neural identity?
    neurIDsf = None #  neural identity?

    alp    = None   #  weight of each place field of each particular neuron
    md     = 3.4      #  How far are clusters separated, compared to acov

    min_var   = 0.2
    max_var   = 1.

    bWtrack = False  #  movement is from Wtrack

    Fu     = 0.97
    Fstd   = 0.97
    Falp   = 0.975

    Fm     = 0.999

    sivu   = 0.1        #  position mean 
    sivs   = 0.03
    siva   = 0.5
    sivm   = 0.02       #  mark mean wander
    xA     = 4
    mA     = 4

    Nx     = 50
    Nm     = 50
    Bx     = 0.1
    Bm     = 0.1
    bx     = 0.1

    vAmp   = 1.
    constV = False
    nLf    = None
    nRh    = None
    pLR    = 0.5

    tetlist= ["01"]

    def create(self, setname, pos=None, thresh=-1000000):
        #  time series to model ratio of the states
        oo  = self
        k   = oo.k
        N   = oo.N   #  length of observation
        M   = oo.M   #  # of cells

        oo.uP   = _N.empty((M, N))     #  
        oo.stdP = _N.empty((M, N))     #  
        
        oo.um   = _N.empty((M, N, k))     #  
        oo.Cov = _N.empty((M, k, k))
        oo.neurIDs= _N.zeros((N, M), dtype=_N.int)
        #oo.neurIDsf= _N.empty((N, M))
        oo.dt  = 0.001
        if oo.stdPMags is None:
            oo.stdPMags = _N.ones(M)*0.1

        oo.alp = _N.empty((M, N))

        mr  = _N.random.rand(N)

        if oo.bWtrack:
            oo.pos = _U.generateMvt(N, vAmp=oo.vAmp, constV=oo.constV, pLR=oo.pLR, nLf=oo.nLf, nRh=oo.nRh)

        else:
            oo._pos   = _N.empty(N)
            oo.pos   = _N.empty(N)
            oo._pos[0] = 0.3*_N.random.randn()
            for n in xrange(N-1):
                oo._pos[n+1] = 0.9995*oo._pos[n] + 0.04*_N.random.randn()
            oo.pos[:] = oo._pos[:]#_N.convolve(oo._pos, gk, mode="same")
        oo.mvpos = oo.pos
            
        Ns = N/50
        _ts   = _N.linspace(0, N, Ns, endpoint=False)
        ts    = _N.linspace(0, N, N, endpoint=False)

        #  First generate 
        #  step update is 50ms.  

        #gk = gauKer(2)
        #gk /= _N.sum(gk)
        
        #oo.pos[:] = _N.interp(ts, _ts, oo._pos)

        Ns = N/1000
        _ts   = _N.linspace(0, N, Ns, endpoint=False)

        oo._uP   = _N.empty((M, Ns))     #  
        oo._um   = _N.empty((M, Ns, oo.k))     #  
        oo._stdP = _N.empty((M, Ns))     #  
        oo._alp  = _N.empty((M, Ns))     #  

        #########  Initial conditions
        if oo.uP0 is not None:   #  M x Npf  matrix
            oo._uP[:, 0]  = 0
        else:
            oo._uP[:, 0]  = _N.random.randn(M)

        if oo.alp0 is not None:  #  M x Npf  matrix
            oo._alp[:, 0] =   oo.alp0
        else:
            oo._alp[:, 0] = _N.random.randn(M)

        if oo.um0 is not None:  #  M x k  matrix
            oo._um[:, 0, :] =   0
        else:
            oo._um[:, 0]   = _N.random.randn(oo.k)*oo.md   

        ######  BUILD latent place field centers and mark centers
        for m in xrange(M):
            for ik in xrange(k):
                oo.Cov[m, ik, ik] = oo.min_var + (oo.max_var - oo.min_var)*_N.random.rand()
            for ik1 in xrange(k):
                for ik2 in xrange(ik1 + 1, k):        #  set up cov. matrices
                    oo.Cov[m, ik1, ik2] = 0.4*(0.6+0.4*_N.abs(_N.random.rand()))*_N.sqrt(oo.Cov[m, ik1, ik1]*oo.Cov[m, ik2, ik2])
                    oo.Cov[m, ik2, ik1] = oo.Cov[m, ik1, ik2]

        oo._stdP[:, 0] = 0.1*_N.random.randn()
        for n in xrange(Ns-1):   # slow changes
            for m in xrange(M):
                for ik in xrange(k):
                    oo._um[m, n+1, ik] = oo.Fm*oo._um[m, n, ik] + oo.sivm*_N.random.randn()
                    oo._uP[m, n+1] = oo.Fu*oo._uP[m, n] + oo.sivu*_N.random.randn()
                    oo._stdP[m, n+1] = oo.Fstd*oo._stdP[m, n] + oo.sivs*_N.random.randn()
                    oo._alp[m, n+1] = oo.Falp*oo._alp[m, n] + oo.siva*_N.random.randn()

        oo._uP[:, :]  += oo.uP0.reshape((M, 1))
        oo._um[:, :, :]  += oo.um0.reshape((M, 1, k))
        for m in xrange(M):
            for ik in xrange(k):
                oo.um[m, :, ik]   = _N.interp(ts, _ts, oo._um[m, :, ik])
                oo.uP[m]   = _N.interp(ts, _ts, oo._uP[m])
                oo.stdP[m] = _N.interp(ts, _ts, _N.abs(oo._stdP[m])) + oo.stdPMags[m]
                oo.alp[m] = _N.log(_N.interp(ts, _ts, _N.abs(oo._alp[m])))

        ######  now create spikes
        oo.marksH     = _N.empty((oo.N, 1), dtype=list)
        oo.marksNH    = _N.empty((oo.N, 1), dtype=list)

        nspks = 0
        cut   = 0
        for m in xrange(M):#  For each cluster
            if m < oo.Mh_strt:
                marks = oo.marksNH
            else:
                marks = oo.marksH
            mkid = oo.markIDs[m]     #  for the mark
            fr          = _N.zeros(oo.N)
            wdP     = 2*oo.stdP[m]**2

            fr[:]   += _N.exp(oo.alp[m] - (oo.pos - oo.uP[m])**2 / wdP)

            rands = _N.random.rand(oo.N)
            thrX  = _N.where(rands < fr*oo.dt)[0]

            #oo.neurIDsf[:, m] = fr
            oo.neurIDs[thrX, m] = 1
            for n in xrange(len(thrX)):   # iterate over time
                tm = thrX[n]
                mk    = mvn(oo.um[mkid, tm], oo.Cov[mkid], size=1)[0]
                if (_N.sum(mk < thresh) == oo.k):
                    oo.neurIDs[thrX[n], m] = 0
                    cut += 1
                else:
                    if marks[tm, 0] is None:      #  separate hash, non-hash
                        marks[tm, 0] = [mk]
                    else:
                        marks[tm, 0].append(mk)

                    nspks += 1

        #kde = _kde.kde(setname)
        #oo.marks = None
        #kde.est(oo.pos, oo.marks, oo.k, oo.xA, oo.mA, oo.Nx, oo.Nm, oo.Bx, oo.Bm, oo.bx, t0=t0, t1=t1, filename="kde.dump")

        print "tot spikes created %(n)d   cut %(c)d" % {"n" : nspks, "c" : cut}
        dmp = open(_edd.resFN("marks.pkl", dir=setname, create=True), "wb")
        pickle.dump(oo, dmp, -1)
        dmp.close()

    def makeMarks(self, marktype=_both_):
        """
        combine hash, non_hash marks
        """
        oo = self
        oo.marks      = _N.empty((oo.N, 1), dtype=list)

        if (marktype == _both_) or (marktype == _non_hash_):
            nonhashMks       = _N.equal(oo.marksNH[:, 0], None)
            inds             = _N.where(nonhashMks == False)[0]
            oo.marks[inds, 0] = oo.marksNH[inds, 0]
        if (marktype == _both_) or (marktype == _hash_):
            hashMks       = _N.equal(oo.marksH[:, 0], None)
            inds             = _N.where(hashMks == False)[0]
            for t in inds:
                if oo.marks[t, 0] is not None:
                    oo.marks[t, 0].extend(oo.marksH[t, 0])
                else:
                    oo.marks[t, 0] = oo.marksH[t, 0]

    def evalAtFxdMks(self, fxdMks, t):
        oo     = self
        iSgs= _N.linalg.inv(oo.covs)

        Nx     = fxdMks.shape[0]

        cmps= _N.empty((oo.M, Nx))
        for m in xrange(oo.M):
            cmps[m] = weight*_N.exp(-0.5*_N.sum(_N.multiply(fxdMks-oo.us[m], _N.dot(iSgs[m], (fxdMks - oo.us[m]).T).T), axis=1))
            bnans = _N.isnan(cmps[m])
            if len(_N.where(bnans is True)[0]) > 0:
                print _N.linalg.det(oo.covs[m])
                print 1/_N.sqrt(_N.linalg.det(oo.covs[m]))
                print -0.5*_N.sum(_N.multiply(fxdMks-oo.us[m], _N.dot(iSgs[m], (fxdMks - oo.us[m]).T).T), axis=1)
                print oo.us[m]
                print iSgs[m]
                print oo.covs[m]

    #def dump(self):
        
