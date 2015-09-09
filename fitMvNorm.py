import stats_util as s_u
import numpy as _N
import pickle
import matplotlib.pyplot as _plt
import scipy.cluster.vq as scv
import scipy.stats as _ss

mvn    = _N.random.multivariate_normal
class fitMvNorm:
    myclrs = [ "#FF0000", "#0000FF", "#00FFFF", 
               "#888888", "#FF8888", "#8888FF", "#88FF88", 
               "#990000", "#000099", "#009900", 
               "#998888", "#888899", "#889988",
               "#FF00FF", "#FFAAFF",
               "#000000", "#FF0000", "#0000FF", "#00FF00", 
               "#888888", "#FF8888", "#8888FF", "#88FF88", 
               "#990000", "#000099", "#009900", 
               "#998888", "#888899", "#889988",
               "#FF00FF", "#FFAAFF"]

    ##################  hyper parameters - do not change during Gibbs
    # HYPER PARAMS for prior covariance: nu, PSI
    PR_cov_a  = 3
    PR_cov_B  = None

    # HYPER PARAMS mean: nu, PSI
    PR_mu_mu = None
    PR_mu_sg = None
    iPR_mu_sg= None

    #  HYPER PARAMS mixture coeff
    PR_m_alp   = None
    #####

    #  how many clusters do I think there are
    M        = 10

    ITERS = 1

    #  samples of mu, cov.  Storage for Gibbs samples
    scov = None
    smu  = None
    sm      = None   #  cluster weight

    mnd  = None
    #  augmented variables
    gz   = None

    pmdim = None

    bPosInd = False

    def __init__(self, ITERS, M, k):
        oo = self

        oo.ITERS = ITERS

        #  sampled variables
        oo.scov = _N.zeros((oo.ITERS, M, k, k))
        oo.smu  = _N.empty((oo.ITERS, M, k))
        oo.sm   = _N.ones((oo.ITERS, M, 1))/M

        ###  
        oo.PR_cov_B = _N.tile(_N.eye(k)*0.5, M).T.reshape(M, k, k)  #  we only look at the diagonal component
        oo.PR_cov_a = _N.ones(M, dtype=_N.int)
        oo.PR_mu_mu = _N.zeros((M, k))

        oo.PR_mu_sg = _N.tile(_N.eye(k)*0.5, M).T.reshape(M, k, k)
        oo.iPR_mu_sg= _N.linalg.inv(oo.PR_mu_sg)

        oo.PR_m_alp = _N.ones(M) * (1./M)

        ###  posterior parameters
        oo.us   = _N.zeros((M, k))
        oo.covs = _N.zeros((M, k, k))
        oo.ms   = _N.zeros((M, 1))

        oo.M    = M
        
    #  M initial guess # of clusters
    #  k
    #  pos, mk    position and mark at spike time
    def init0(self, M, k, pos, mk, n1, n2, sepHash=False, pctH=0.7, MS=None):
        print "init0"
        oo = self
        _x   = _N.empty((n2-n1, k))
        _x[:, 0]    = pos
        _x[:, 1:]   = mk
        N   = n2-n1

        #  Gibbs sampling 
        ################  init cluster centers
        if sepHash:  #  treat hash spikes seperately
            bgCh  = _N.max(_x[:, 1:], axis=1)
            inds  = _N.array([i[0] for i in sorted(enumerate(bgCh), key=lambda x:x[1])])
            #  hash are lowest 70%
            #  signal are highest 30%

            pH        = int(bgCh.shape[0]*pctH)
            if MS is None:
                MH        = int(M*pctH)
                MS        = M - MH
            else:
                MH        = M - MS

            sigInds       = inds[pH:]
            smkpos        = _x[sigInds]
            scrH, labH = scv.kmeans2(_x[inds[0:pH]], MH)

            ITERS  = 30
            mnDsts = _N.zeros(ITERS)

            allCtrs = []
            allLabs = []

            for it in xrange(ITERS):
                ctrs, labs = scv.kmeans2(smkpos, MS)  #  big clear
                dist = []
                tot  = 0
                for m in xrange(MS):
                    inClus = _N.where(labs == m)[0]

                    if len(inClus) > 3:
                        dist.append(_N.mean(_N.sqrt(_N.sum((smkpos[inClus] - ctrs[m])**2, axis=1)))*len(inClus))
                        tot += len(inClus)
                mnDsts[it] = _N.sum(dist)/ tot

                allCtrs.append(ctrs)
                allLabs.append(labs)

            bestIn = _N.where(mnDsts == _N.min(mnDsts))[0]

            lab        = _N.array(labH.tolist() + (allLabs[bestIn] + MH).tolist())
            x          = _N.empty((n2-n1, k))
            x[:, 0]    = _x[inds, 0]
            x[:, 1:]   = _x[inds, 1:]

            """
            print "seperate hash"
            bgCh  = _N.max(_x[:, 1:], axis=1)
            inds  = _N.array([i[0] for i in sorted(enumerate(bgCh), key=lambda x:x[1])])
            #  hash are lowest 70%
            #  signal are highest 30%

            pH        = int(bgCh.shape[0]*pctH)
            if MS is None:
                MH        = int(M*pctH)
                MS        = M - MH
            else:
                MH        = M - MS

            scrH, labH = scv.kmeans2(_x[inds[0:pH]], MH)
            scrB, labB = scv.kmeans2(_x[inds[pH:]], MS)  #  big clear

            _plt.scatter(_x[inds[0:pH], 0], _x[inds[0:pH], 1], color="black", s=2)

            cc = 0
            clsz = _N.empty(MS, dtype=_N.int)

            for c in xrange(MS):
                thisCl = _N.where(labB == c)[0] + pH
                clsz[c] = len(thisCl)
                scrB2, labB2 = scv.kmeans2(_x[inds[thisCl]], 2)
                #  thisCl 
                
                #  look at 
                for cb in xrange(2):
                    dU = _N.mean(_x[inds[thisCl][thisClS]], axis=)
                    print _N.cov()
                    thisClS = _N.where(labB2 == cb)[0]
                    #print thisClS
                    #_plt.scatter(_x[inds[thisCl][thisClS], 0], _x[inds[thisCl][thisClS], 1], color=oo.myclrs[cc], s=2)
                    cc += 1



            lab        = _N.array(labH.tolist() + (labB + MH).tolist())
            x          = _N.empty((n2-n1, k))
            x[:, 0]    = _x[inds, 0]
            x[:, 1:]   = _x[inds, 1:]
            """
        else:
            x = _x
            scr, lab = scv.kmeans2(x, M)

        SI = N / M
        covAll = _N.cov(x.T)
        dcovMag= _N.diagonal(covAll)*0.005

        for im in xrange(M):
            kinds = _N.where(lab == im)[0]  #  inds

            if len(kinds) > 0:
                oo.scov[0, im] = _N.cov(x[kinds], rowvar=0)
                oo.smu[0, im]  = _N.mean(x[kinds], axis=0)
            else:
                oo.smu[0, im]  = _N.mean(x[SI*im:SI*(im+1)], axis=0)
                oo.scov[0, im] = covAll*0.125

    #  I need to know initial values smu, scov, sm.  Generate gz 
    def fit(self, M, k, pos, mk, n1, n2):
        oo = self
        mnd    = oo.mnd
        x   = _N.empty((n2-n1, k))
        x[:, 0]    = pos
        x[:, 1:]   = mk
        N   = n2-n1
        oo.gz   = _N.zeros((oo.ITERS, N, M), dtype=_N.int)
        M   = oo.M
        oo.pmdim = k

        covAll = _N.cov(x.T)
        dcovMag= _N.diagonal(covAll)*0.125

        #  termporary containers
        expTrm = _N.empty((M, N))
        expArg = _N.empty((M, N))
        crats = _N.zeros((M+1, N))
        rands = _N.random.rand(N, 1)
        dirArgs = _N.empty(M, dtype=_N.int)

        rsum = _N.empty((1, N))
        skpM   = _N.arange(0, N)*M

        for it in xrange(oo.ITERS-1):
            iscov = _N.linalg.inv(oo.scov[it])
            norms = 1/_N.sqrt(2*_N.pi*_N.linalg.det(oo.scov[it]))
            norms = norms.reshape(M, 1)

            for im in xrange(M):
                expArg[im] = -0.5*_N.sum(_N.multiply((x-oo.smu[it, im]), _N.dot(x-oo.smu[it, im], iscov[im])), axis=1)   #  expArg[im] is size N

            rexpArg = expArg.T.reshape(M*N)
            lrgInM = expArg.argmax(axis=0)
            lrgstArgs = rexpArg[skpM+lrgInM]
            expArg0 = expArg - lrgstArgs

            expTrm = _N.exp(expArg0)
            rats = oo.sm[it]*expTrm*norms  #  shape is M x oo.N
            _N.sum(rats, axis=0, out=rsum[0, :])

            rats /= rsum   #  each column of "rats" sums to 1

            for im in xrange(M):
                crats[im+1] = rats[im] + crats[im]

            rands = _N.random.rand(N)
            rrands = _N.tile(rands, M).reshape(M, N)
            ###  THIS once broke because we had an empty cluster
            irw, icl = _N.where((rrands >= crats[:-1]) & (rrands <= crats[1:]))

            oo.gz[it+1, icl, irw] = 1   #  we must clean out gz

            #  _N.sum(oo.gz...) sz M   its vec of num. of obs of each state 'm'
            _N.add(oo.PR_m_alp, _N.sum(oo.gz[it+1], axis=0), out=dirArgs)

            da = _N.random.dirichlet(dirArgs)
            oo.sm[it+1, :, 0] = _N.random.dirichlet(dirArgs)
            
            for im in xrange(M):
                minds = _N.where(oo.gz[it+1, :, im] == 1)[0]

                if len(minds) > 0:
                    clstx    = x[minds]
                    mc       = _N.sum(clstx, axis=0)
                    Nm       = clstx.shape[0]

                    # hyp
                    ########  POSITION
                    ##  mean of posterior distribution of cluster means
                    #  sigma^2 and mu are the current Gibbs-sampled values

                    for ik in xrange(oo.pmdim):
                        po_mu_sg = 1. / (1 / oo.PR_mu_sg[im, ik, ik] + Nm / oo.scov[it, im, ik, ik])
                        po_mu_mu  = (oo.PR_mu_mu[im, ik] / oo.PR_mu_sg[im, ik, ik] + mc[ik] / oo.scov[it, im, ik, ik]) * po_mu_sg

                        oo.smu[it+1, im, ik] = po_mu_mu + _N.sqrt(po_mu_sg)*_N.random.randn()

                        ##  dof of posterior distribution of cluster covariance
                        po_sg_a = oo.PR_cov_a[im] + 0.5*Nm
                        po_sg_B = oo.PR_cov_B[im, ik, ik] + 0.5*_N.sum((clstx[:, ik] - oo.smu[it+1, im, ik])**2)
                        oo.scov[it+1, im, ik, ik] = _ss.invgamma.rvs(po_sg_a, scale=po_sg_B)
                else:  #  no marks assigned to this cluster 
                    oo.scov[it+1, im] = oo.scov[it, im]
                    oo.smu[it+1, im]  = oo.smu[it, im]

        #  When I say prior for mu, I mean I have hyper parameters mu_mu and mu_sg.
        #  hyperparameters are not sampled
        hITERS = oo.ITERS/2
        oo.us[:]  = _N.mean(oo.smu[hITERS:], axis=0)
        oo.covs[:] = _N.mean(oo.scov[hITERS:], axis=0)
        oo.ms[:]  = _N.mean(oo.sm[hITERS:], axis=0).reshape(oo.M, 1)

        oo.dat = x

    def set_priors_and_initial_values(self):
        """
        after a first run, 
        """
        oo = self
        mid = oo.ITERS/2
        #  hyperparameters describe the priors, and are estimated from the 
        #  posterior of the parameter
        #  the posteriors are now priors
        oo.PR_m_alp[:] = _N.sum(_N.mean(oo.gz[mid:oo.ITERS-1], axis=0), axis=0)
        print "=========   set_priors"
        print oo.PR_m_alp
        #  prior of cluster center is current
        #  posterior distribution of cluster center
        oo.PR_mu_mu[:] = _N.mean(oo.smu[mid:oo.ITERS-1], axis=0)
        oo.PR_mu_sg[:] = _N.mean(oo.scov[mid:oo.ITERS-1], axis=0)
        #  prior of cluster center is current
        #  posterior distribution of cluster center
        oo.PR_cov_B[:] = _N.mean(oo.scov[mid:oo.ITERS-1], axis=0)
        oo.PR_cov_a[:] = _N.sum(oo.gz[oo.ITERS-1], axis=0)

        # last sampled values will be starting values

        oo.sm[0]   = oo.sm[oo.ITERS-1]
        oo.smu[0]  = oo.smu[oo.ITERS-1]
        oo.scov[0] = oo.scov[oo.ITERS-1]

        #oo.gz   = _N.zeros((oo.ITERS, oo.mnd.N, oo.M), dtype=_N.int)
        oo.gz[:,:,:] = 0

    def pNkmk_x(self, varXfxdMks, Nx):
        """
        varXfxdMks    grid of [variable X, fixed marks]     [Nx  x  mdim+1]
        """
        oo = self

        #  given the mark, give me the firing rate as a function of x
        
        return zs


    def evalAll(self, Ngrd):
        oo     = self
        x0      = min(oo.dat[:, 0])
        x1      = max(oo.dat[:, 0])
        y0      = min(oo.dat[:, 1])
        y1      = max(oo.dat[:, 1])

        x      = _N.linspace(x0, x1, Ngrd)#.reshape(Ngrd, 1)
        y      = _N.linspace(y0, y1, Ngrd)#.reshape(1, Ngrd)

        xg, yg = _N.meshgrid(x, y)

        xy = _N.array([xg, yg])
        xy = xy.reshape(oo.pmdim, Ngrd*Ngrd)
        #   xy.T    goes from lower left, scans right, to upper right

        us  = _N.mean(oo.smu[100:], axis=0)
        Sgs = _N.mean(oo.scov[100:], axis=0)
        iSgs= _N.linalg.inv(Sgs)

        cmps= _N.empty((oo.M, Ngrd, Ngrd))
        for m in xrange(oo.M):
            cmps[m] = 1/_N.sqrt(2*_N.pi*_N.linalg.det(Sgs[m]))*_N.exp(-0.5*_N.sum(_N.multiply(xy.T-us[m], _N.dot(iSgs[m], (xy.T - us[m]).T).T), axis=1)).reshape(Ngrd, Ngrd)

        ms  = _N.mean(oo.sm[100:], axis=0).reshape(oo.M, 1, 1)
        zs = ms*cmps

        smpMn   = _N.mean(oo.dat, axis=0)

        _plt.scatter(oo.dat[:, 0], oo.dat[:, 1], s=10)
        _plt.imshow(_N.sum(zs, axis=0), origin="lower", extent=(x0, x1, y0, y1))

        return zs

    def evalAtFxdMks(self, fxdMks):
        oo     = self
        iSgs= _N.linalg.inv(oo.covs)

        Nx     = fxdMks.shape[0]

        cmps= _N.empty((oo.M, Nx))
        for m in xrange(oo.M):
            cmps[m] = 1/_N.sqrt(2*_N.pi*_N.linalg.det(oo.covs[m]))*_N.exp(-0.5*_N.sum(_N.multiply(fxdMks-oo.us[m], _N.dot(iSgs[m], (fxdMks - oo.us[m]).T).T), axis=1))
            bnans = _N.isnan(cmps[m])
            if len(_N.where(bnans is True)[0]) > 0:
                print _N.linalg.det(oo.covs[m])
                print 1/_N.sqrt(_N.linalg.det(oo.covs[m]))
                print -0.5*_N.sum(_N.multiply(fxdMks-oo.us[m], _N.dot(iSgs[m], (fxdMks - oo.us[m]).T).T), axis=1)
                print oo.us[m]
                print iSgs[m]
                print oo.covs[m]
                

        zs = _N.sum(oo.ms*cmps, axis=0)

        return zs
