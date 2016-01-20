import utils as _U
import numpy as _N
import pickle
import EnDedirs as _edd
import d5mkde as _kde
from filter import gauKer
mvn = _N.random.multivariate_normal
import matplotlib.pyplot as _plt


class mixNormDatPoi:
    k      = 2      #  data dimensionality
    N      = 500    #  no of datas
    M      = 3      #  no components

    um      = None   #  mean of marks for each neuron
    Cov     = None   #  cov  of marks for each neuron

    uP      = None  #  mean of place fields for each neuron
    stdP    = None  #  std  of place fields for each neuron
    stdPMags= None  #

    lmd     = None  #  
    Npf     = None  #  number of place fields per neuron

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

    tetlist= ["01"]

    def create(self, setname, pos=None):
        #  time series to model ratio of the states
        oo  = self
        k   = oo.k
        N   = oo.N   #  length of observation
        M   = oo.M
        Npf = oo.Npf #  number of place fields per neuron

        oo.uP   = _N.empty((M, Npf, N))     #  
        oo.stdP = _N.empty((M, Npf, N))     #  
        
        oo.um   = _N.empty((M, N, k))     #  
        oo.Cov = _N.empty((M, k, k))
        oo.neurIDs= _N.zeros((N, M), dtype=_N.int)
        #oo.neurIDsf= _N.empty((N, M))
        oo.dt  = 0.001
        if oo.stdPMags is None:
            oo.stdPMags = _N.ones(M*Npf)*0.1

        oo.alp = _N.empty((M, Npf, N))

        mr  = _N.random.rand(N)

        if oo.bWtrack:
            oo.pos = _U.generateMvt(N, vAmp=oo.vAmp, constV=oo.constV)

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

        oo._uP   = _N.empty((M, Npf, Ns))     #  
        oo._um   = _N.empty((M, Ns, oo.k))     #  
        oo._stdP = _N.empty((M, Npf, Ns))     #  
        oo._alp  = _N.empty((M, Npf, Ns))     #  
        oo._pfdst= _N.empty((M, Npf-1, Ns))     #  distance between place flds

        #########  Initial conditions
        if oo.uP0 is not None:   #  M x Npf  matrix
            oo._uP[:, :, 0]  = 0
        else:
            oo._uP[:, :, 0]  = _N.random.randn(M, Npf)

        if oo.alp0 is not None:  #  M x Npf  matrix
            oo._alp[:, :, 0] =   oo.alp0
        else:
            oo._alp[:, :, 0] = _N.random.randn(M, Npf)

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
                    oo.Cov[m, ik1, ik2] = 0.04*(0.1+_N.abs(_N.random.randn()))
                    oo.Cov[m, ik2, ik1] = oo.Cov[m, ik1, ik2]

            ##  place field set up
            for np in xrange(1, Npf):
                mlt = 1 if _N.random.rand() < 0.5 else -1
                oo._uP[m, np] = oo._uP[m, np-1] - mlt*(1 + 0.3*_N.random.rand())
        oo._stdP[:, :, 0] = 0.1*_N.random.randn()
        for n in xrange(Ns-1):   # slow changes
            for m in xrange(M):
                for ik in xrange(k):
                    oo._um[m, n+1, ik] = oo.Fm*oo._um[m, n, ik] + oo.sivm*_N.random.randn()
                for mpf in xrange(Npf):
                    oo._uP[m, mpf, n+1] = oo.Fu*oo._uP[m, mpf, n] + oo.sivu*_N.random.randn()
                    oo._stdP[m, mpf, n+1] = oo.Fstd*oo._stdP[m, mpf, n] + oo.sivs*_N.random.randn()
                    oo._alp[m, mpf, n+1] = oo.Falp*oo._alp[m, mpf, n] + oo.siva*_N.random.randn()

        oo._uP[:, :, :]  += oo.uP0.reshape((M, Npf, 1))
        #oo._um[:, :, :]  += oo.um0.reshape((M, 1, 1))
        oo._um[:, :, :]  += oo.um0.reshape((M, 1, k))
        for m in xrange(M):
            for ik in xrange(k):
                oo.um[m, :, ik]   = _N.interp(ts, _ts, oo._um[m, :, ik])
            for mpf in xrange(Npf):
                oo.uP[m, mpf]   = _N.interp(ts, _ts, oo._uP[m, mpf])
                oo.stdP[m, mpf] = _N.interp(ts, _ts, _N.abs(oo._stdP[m, mpf])) + oo.stdPMags[m, mpf]
                oo.alp[m, mpf] = _N.log(_N.interp(ts, _ts, _N.abs(oo._alp[m, mpf])))

        ######  now create spikes
        oo.marks    = _N.empty((oo.N, 1), dtype=list)

        nspks = 0
        for m in xrange(M):#  For each cluster
            fr          = _N.zeros(oo.N)
            for mpf in xrange(Npf):
                wdP     = 2*oo.stdP[m, mpf]**2

                fr[:]   += _N.exp(oo.alp[m, mpf] - (oo.pos - oo.uP[m, mpf])**2 / wdP)

            rands = _N.random.rand(oo.N)
            thrX  = _N.where(rands < fr*oo.dt)[0]

            #oo.neurIDsf[:, m] = fr
            oo.neurIDs[thrX, m] = 1
            for n in xrange(len(thrX)):   # iterate over time
                tm = thrX[n]
                mk    = mvn(oo.um[m, tm], oo.Cov[m], size=1)[0]
                if oo.marks[tm, 0] is None:
                    oo.marks[tm, 0] = [mk]
                else:
                    oo.marks[tm, 0].append(mk)
                nspks += 1

        #kde = _kde.kde(setname)
        #oo.marks = None
        #kde.est(oo.pos, oo.marks, oo.k, oo.xA, oo.mA, oo.Nx, oo.Nm, oo.Bx, oo.Bm, oo.bx, t0=t0, t1=t1, filename="kde.dump")

        print "total spikes created %d" % nspks
        dmp = open(_edd.resFN("marks.pkl", dir=setname, create=True), "wb")
        pickle.dump(oo, dmp, -1)
        dmp.close()

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
        
