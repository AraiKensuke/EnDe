import stats_util as s_u
import numpy as _N
import pickle
import matplotlib.pyplot as _plt
import scipy.cluster.vq as scv
import scipy.stats as _ss
import fitutil as _fu
import time as _tm

mvn    = _N.random.multivariate_normal
class fitMvNorm:
    ##################  hyper parameters - do not change during Gibbs
    # HYPER PARAMS for prior covariance: nu, PSI
    PR_cov_nu    = 3
    PR_cov_PSI   = None
    po_cov_nu    = None
    po_cov_PSI   = None

    # HYPER PARAMS mean: nu, PSI
    po_mu_mu = None
    po_mu_sg = None
    PR_mu_mu = None
    PR_mu_sg = None
    iPR_mu_sg= None

    #  HYPER PARAMS mixture coeff
    PR_m_alp   = None
    po_alpha   = None
    #####

    #  how many clusters do I think there are
    M        = 10
    M      = 15

    ITERS = 1
    
    AR    = 0.5   #  % increase per 1 minute
    AR0   = 1     #  baseline AR - for messier fit, AR > 1 can still stop adaptation?

    #  samples of mu, cov.  Storage for Gibbs samples
    scov = None
    smu  = None
    sm      = None   #  cluster weight

    mnd  = None
    #  augmented variables
    gz   = None

    pmdim = None

    bPosInd = False

    iSgs  = None
    i2pidcovs = None
    i2pidcovsr = None

    def __init__(self, ITERS, M, k):
        """
        
        """
        oo = self

        oo.M    = M
        oo.k    = k
        oo.ITERS = ITERS
        
        #  sampled variables
        oo.scov = _N.zeros((oo.ITERS, M, k, k))
        oo.smu  = _N.empty((oo.ITERS, M, k))
        oo.sm   = _N.ones((oo.ITERS, M, 1))/M

        ###  
        #  
        #  parameters of cluster covariance.  Becomes prior
        oo.po_cov_PSI = _N.empty((ITERS, M, k, k))
        oo.po_cov_nu  = _N.empty((ITERS, M), dtype=_N.int)
        #  priors
        oo.PR_cov_PSI = _N.tile(_N.eye(k)*0.5, M).T.reshape(M, k, k)
        oo.PR_cov_nu  = _N.ones(M, dtype=_N.int)
        #  parameters of cluster mean.  Becomes prior
        oo.po_mu_mu = _N.zeros((ITERS, M, k))
        oo.po_mu_sg = _N.empty((ITERS, M, k, k))
        oo.PR_mu_mu = _N.zeros((M, k))
        oo.PR_mu_sg = _N.tile(_N.eye(k)*30, M).T.reshape(M, k, k)

        oo.iPR_mu_sg= _N.linalg.inv(oo.PR_mu_sg)

        #  parameters of cluster weights
        oo.PR_m_alp = _N.ones(M) * (1./M)
        oo.po_alpha = _N.empty((ITERS, M))

        ###  posterior parameters
        oo.us   = _N.zeros((M, k))
        oo.covs = _N.zeros((M, k, k))
        oo.ms   = _N.zeros((M, 1))

    #  M initial guess # of clusters
    #  k
    #  pos, mk    position and mark at spike time
    def init0(self, pos, mk, n1, n2, sepHash=False, pctH=0.7, MS=None, sepHashMthd=0, doTouchUp=False, MF=None, kmeansinit=True):
        """
        M       total number of clusters

        MS      number of clusters assigned to signal
        MF      number of clusters used for initial fit
        M - MF  If doing touchup, number of clusters to assign to this
        """
        print "init0"
        oo = self

        k  = oo.k
        MF = oo.M if MF is None else MF

        print "MF  %d" % MF

        _x   = _N.empty((n2-n1, k))
        _x[:, 0]    = pos
        _x[:, 1:]   = mk
        N   = n2-n1

        #  Gibbs sampling 
        ################  init cluster centers
        if sepHash:  #  treat hash spikes seperately
            print "sepHashMthd  %d" % sepHashMthd
            if sepHashMthd == 0:
                ##########################
                bgCh  = _N.max(_x[:, 1:], axis=1)
                inds  = _N.array([i[0] for i in sorted(enumerate(bgCh), key=lambda x:x[1])])
                #  hash are lowest 70%
                #  signal are highest 30%

                pH        = int(bgCh.shape[0]*pctH)
                if MS is None:
                    MH        = int(M*pctH)
                    MS        = MF - MH
                else:
                    MH        = MF - MS

                sigInds       = inds[pH:]
                smkpos        = _x[sigInds]
                labH = _fu.bestcluster(50, _x[inds[0:pH]], MH)

                labS = _fu.bestcluster(50, smkpos, MS)
            else:  #  sepHashMthd == 1
                ##########################
                BINS    = 20
                bins    = _N.linspace(-6, 6, BINS+1)

                cumcnts = _N.zeros(BINS)

                blksz   = 20

                #####################   separate hash / nonhash indices
                nonhash = []
                for ch in xrange(1, 5):
                    done    = False
                    inds  = _N.array([i[0] for i in sorted(enumerate(_x[:, ch]), key=lambda x:x[1], reverse=True)])

                    blk = -1
                    cumcnts[:] = 0

                    while not done:
                        blk += 1
                        cnts, bns = _N.histogram(_x[inds[blk*blksz:(blk+1)*blksz], 0], bins=bins)
                        cumcnts += cnts
                        if len(_N.where(cumcnts < 2)[0]) <= 3:
                            done = True
                            nonhash.extend(inds[0:(blk+1)*blksz])

                unonhash = _N.unique(nonhash)  #  not hash spikes
                hashsp   = _N.setdiff1d(inds, unonhash)  #  inds is contiguous but reordered all

                ##  place-specific firing of 
                #_x[:, 0] *= 5

                MH        = MF - MS

                sigInds       = unonhash
                smkpos        = _x[sigInds]

                #labS = _fu.bestcluster(50, smkpos, MS)
                labS, MSA = _fu.findBestClusterBySplit(smkpos, MS, oo.k, 15)
                if MSA > MS:
                    MH        = MF - MS
                    MS        = MSA

                _fu.colorclusters(smkpos, labS, MSA)

                labH = _fu.bestcluster(50, _x[hashsp], MH)
                #bins = _N.linspace(-30, 30, 101)
                #labH = _fu.positionalClusters(_x[hashsp, 0], bins, MH)
                #histdat = _plt.hist(_x[hashsp, 0], bins=bins)
                #_N.savetxt("hist", histdat[0], fmt="%.4f")
                #_N.savetxt("hash", _x[hashsp], fmt="%.4f %.4f %.4f %.4f %.4f")
                #_x[:, 0] /= 5
                #scrH, labH = scv.kmeans2(_x[hashsp], MH)


            ##################
            lab        = _N.array(labH.tolist() + (labS + MH).tolist())
            x          = _N.empty((n2-n1, k))
            if sepHashMthd == 0:
                x[:, 0]    = _x[inds, 0]
                x[:, 1:]   = _x[inds, 1:]
            else:
                x[0:len(hashsp)] = _x[hashsp]
                x[len(hashsp):]  = _x[sigInds]
        else:  #  don't separate hash from signal marks. simple kmeans2
            x = _x
            if not kmeansinit:  #  just random initial conditions
                print "random initial conditions"
                lab = _N.array(_N.random.rand(N)*MF, dtype=_N.int)
            else:
                scr, lab = scv.kmeans2(x, MF)

        #  now assign the cluster we've found to Gaussian mixtures
        SI = N / MF
        covAll = _N.cov(x.T)
        dcovMag= _N.diagonal(covAll)*0.005

        for im in xrange(MF):
            kinds = _N.where(lab == im)[0]  #  inds

            if len(kinds) > 6:   # problem when cov is not positive def.
                oo.smu[0, im]  = _N.mean(x[kinds], axis=0)
                oo.scov[0, im] = _N.cov(x[kinds], rowvar=0)
                oo.sm[0, im]   = float(len(kinds)+1) / (N+MF)
            else:
                #oo.smu[0, im]  = _N.mean(x[sigInds], axis=0)
                oo.smu[0, im]  = _N.mean(x, axis=0)
                oo.scov[0, im] = covAll*0.125
                oo.sm[0, im]   = float(len(kinds)+1) / (N+MF)



    def fit(self, M, pos, mk, n1, n2, init=False):
        """
        Fit, with the inverting done in blocks
        """
        oo = self
        k      = oo.k
        mnd    = oo.mnd
        x   = _N.empty((n2-n1, k))
        x[:, 0]    = pos
        x[:, 1:]   = mk
        N   = n2-n1
        oo.pmdim = k
        oo.gz   = _N.zeros((oo.ITERS, N, M), dtype=_N.int)
        if init:
            oo.PR_m_alp[:] = 1. / M   #  initial

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

        Nms   = _N.empty((M, 1, 1), dtype=_N.int)
        mcs   = _N.empty((M, k))
        clstxs= []

        ####  stuff used repeatedly
        k_zeros    = _N.zeros(k)
        pr_iSg_Mu = _N.einsum("mjk,mk->mj", oo.iPR_mu_sg, oo.PR_mu_mu)

        #  
        for it in xrange(oo.ITERS-1):
            t1 = _tm.time()
            if it % 50 == 0:
                print it
            iscov = _N.linalg.inv(oo.scov[it, 0:M])
            #print iscov

            norms = 1/_N.sqrt(2*_N.pi*_N.linalg.det(oo.scov[it, 0:M]))
            norms = norms.reshape(M, 1)
            t2 = _tm.time()

            ####  THIS IS THE BOTTLE NECK
            for im in xrange(M):
                expArg[im] = -0.5*_N.einsum("nj,nj->n", x - oo.smu[it, im], _N.dot(x - oo.smu[it, im], iscov[im]))
                #expArg[im] = -0.5*_N.sum(_N.multiply((x-oo.smu[it, im]), _N.dot(x-oo.smu[it, im], iscov[im])), axis=1)   #  expArg[im] is size N

            t3 = _tm.time()
            rexpArg = expArg.T.reshape(M*N)
            lrgInM = expArg.argmax(axis=0)
            lrgstArgs = rexpArg[skpM+lrgInM]
            expArg0 = expArg - lrgstArgs

            expTrm = _N.exp(expArg0)

            rats = oo.sm[it, 0:M]*expTrm*norms  #  shape is M x oo.N

            _N.sum(rats, axis=0, out=rsum[0, :])   
            rats /= rsum   #  each column of "rats" sums to 1

            for im in xrange(M):
                crats[im+1] = rats[im] + crats[im]
            t4 = _tm.time()
            rands = _N.random.rand(N)
            rrands = _N.tile(rands, M).reshape(M, N)
            ###  THIS once broke because we had an empty cluster
            irw, icl = _N.where((rrands >= crats[:-1]) & (rrands <= crats[1:]))

            ##############  GENERATE cluster membership
            oo.gz[it+1, icl, irw] = 1   #  we must clean out gz

            #  _N.sum(oo.gz...) sz M   its vec of num. of obs of each state 'm'
            _N.add(oo.PR_m_alp[0:M], _N.sum(oo.gz[it+1], axis=0), out=oo.po_alpha[it+1])

            ##############  SAMPLE WEIGHTS
            oo.sm[it+1, 0:M, 0] = _N.random.dirichlet(oo.po_alpha[it+1])

            clstxs = []
            mindss = []
            mcs = _N.empty((M, k))   # cluster sample means

            t5 = _tm.time()            
            for im in xrange(M):  # 111111111111111
                minds    = _N.where(oo.gz[it+1, :, im] == 1)[0]
                Nms[im,0,0]      = minds.shape[0]
                mindss.append(minds)
            oo.po_mu_sg[it+1] = _N.linalg.inv(oo.iPR_mu_sg + Nms*iscov)

            for im in xrange(M):  # 222222222222222
                if Nms[im,0,0] > 0:
                    clstx    = x[mindss[im]]
                    mcs[im]       = _N.mean(clstx, axis=0)
                else:
                    clstx    = k_zeros
                    mcs[im]       = clstx
                clstxs.append(clstx)

                # hyp
                ########  POSITION
                ##  mean of posterior distribution of cluster means
                #  sigma^2 and mu are the current Gibbs-sampled values

                ##  mean of posterior distribution of cluster means

            oo.po_mu_mu[it+1] = _N.einsum("mjk,mk->mj", oo.po_mu_sg[it+1], pr_iSg_Mu + Nms[:,:,0]*_N.einsum("mjk,mk->mj", iscov, mcs))
            # dot(MATRIX, vector)   
            
                ##############  SAMPLE MEANS
                #  this can be done without

            rn3    = _N.random.randn(M, k)
            C      = _N.linalg.cholesky(oo.po_mu_sg[it+1])
            oo.smu[it+1] = oo.po_mu_mu[it+1] + _N.einsum("njk,nk->nj", C, rn3)

            for im in xrange(M):  # 3333333333333333
                Nm = Nms[im,0,0]

                if Nm >= oo.pmdim:
                    clstx = clstxs[im]
                    ##  dof of posterior distribution of cluster covariance
                    oo.po_cov_nu[it+1, im] = oo.PR_cov_nu[im] + Nm
                    ##  dof of posterior distribution of cluster covariance
                    oo.po_cov_PSI[it+1, im] = oo.PR_cov_PSI[im] + _N.dot((clstx - oo.smu[it+1, im]).T, (clstx-oo.smu[it+1, im]))

                    ##############  SAMPLE COVARIANCES
                    oo.scov[it+1, im] = s_u.sample_invwishart(oo.po_cov_PSI[it+1, im], oo.po_cov_nu[it+1, im])
                else:  #  no marks assigned to this cluster 
                    oo.scov[it+1, im] = oo.scov[it, im]
                    oo.smu[it+1, im]  = oo.smu[it, im]
                    oo.po_mu_sg[it+1, im] = oo.PR_mu_sg[im]
                    oo.po_mu_mu[it+1, im] = oo.PR_mu_mu[im]
                    oo.po_cov_nu[it+1, im] = oo.PR_cov_nu[im]
                    ##  dof of posterior distribution of cluster covariance
                    oo.po_cov_PSI[it+1, im] = oo.PR_cov_PSI[im]

            t6 = _tm.time()
            # print "-----"
            # print (t2-t1)
            # print (t3-t2)
            # print (t4-t3)
            # print (t5-t4)
            # print (t6-t5)

        #  When I say prior for mu, I mean I have hyper parameters mu_mu and mu_sg.
        #  hyperparameters are not sampled

        print oo.po_alpha[oo.ITERS-1]
        hITERS = int(oo.ITERS*0.75)
        oo.us[:]  = _N.mean(oo.smu[hITERS:oo.ITERS], axis=0)
        oo.covs[:] = _N.mean(oo.scov[hITERS:oo.ITERS], axis=0)
        oo.ms[:]  = _N.mean(oo.sm[hITERS:oo.ITERS], axis=0).reshape(oo.M, 1)

        oo.dat = x

    ###
    ###   length of previous (encoding episode + decoding episode) * AR
    ###   AR = adaptation rate - per minute
    ###   
    ###
    def set_priors_and_initial_values(self, telapse=0):
        """
        after a first run, 
    
        telapse in ms.  AR
        """
        oo = self
        mid = oo.ITERS/2
        #  hyperparameters describe the priors, and are estimated from the 
        #  posterior of the parameter
        #  the posteriors are now priors
        oo.PR_m_alp[:] = _N.mean(oo.po_alpha[mid:], axis=0)
        print "=========   set_priors"
        print oo.PR_m_alp
        #  prior of cluster center is current
        #  posterior distribution of cluster center
        tS = _N.sqrt(telapse / 60000.)
        oo.PR_mu_mu[:] = _N.mean(oo.po_mu_mu[mid:], axis=0)
        oo.PR_mu_sg[:] = _N.mean(oo.po_mu_sg[mid:], axis=0)*(oo.AR0+oo.AR*tS)
        print "adaptation factor  %.3f" % (oo.AR0+oo.AR*tS)
        #oo.PR_mu_sg[:] = _N.mean(oo.po_mu_sg[mid:], axis=0)*5
        #oo.PR_mu_sg[:] = _N.mean(oo.po_mu_sg[mid:], axis=0)*1.08
        oo.iPR_mu_sg   = _N.linalg.inv(oo.PR_mu_sg)
        #  prior of cluster center is current
        #  posterior distribution of cluster center
        oo.PR_cov_nu[:] = _N.mean(oo.po_cov_nu[mid:], axis=0)
        oo.PR_cov_PSI[:] = _N.mean(oo.po_cov_PSI[mid:], axis=0)

        # last sampled values will be starting values

        oo.sm[0]   = oo.sm[oo.ITERS-1]
        oo.smu[0]  = oo.smu[oo.ITERS-1]
        oo.scov[0] = oo.scov[oo.ITERS-1]

        #oo.gz   = _N.zeros((oo.ITERS, oo.mnd.N, oo.M), dtype=_N.int)
        oo.gz[:,:,:] = 0

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

    def evalAtFxdMks_new(self, fxdMks):
        oo     = self

        Nx     = fxdMks.shape[0]
        fxdMksr= fxdMks.reshape(Nx, 1, oo.pmdim)

        cmps = oo.i2pidcovsr*_N.exp(-0.5*_N.einsum("xmj,xmj->mx", fxdMksr-oo.us, _N.einsum("mjk,xmk->xmj", oo.iSgs, fxdMksr - oo.us)))
                
        zs = _N.sum(oo.ms*cmps, axis=0)

        return zs
        
        
