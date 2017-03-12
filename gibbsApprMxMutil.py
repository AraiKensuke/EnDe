import numpy as _N
from fitutil import  emMKPOS_sep1A, emMKPOS_sep1B, sepHash, colorclusters, findsmallclusters, splitclstrs, posMkCov0, contiguous_pack2
from posteriorUtil import MAPvalues2
import clrs 
from filter import gauKer
import time as _tm
from EnDedirs import resFN, datFN
import matplotlib.pyplot as _plt
import fastnum as _fm
import hc_bcast as _hcb
import scipy.stats as _ss
import openTets as _oT
import utilities as _U

twpi = 2*_N.pi
wdSpc = 1

def initClusters(oo, K, x, mks, t0, t1, Asts, doSepHash=True, xLo=0, xHi=3, oneCluster=False, nzclstr=False):
    n0 = 0
    n1 = len(Asts)

    _x   = _N.empty((n1-n0, K+1))
    _x[:, 0]    = x[Asts+t0]
    _x[:, 1:]   = mks[Asts+t0]

    if oneCluster:
        unonhash = _N.arange(len(Asts))
        hashsp   = _N.array([])
        hashthresh = _N.min(_x[:, 1:], axis=0)   #  no hash spikes

        labS     = _N.zeros(len(Asts), dtype=_N.int)
        labH     = _N.array([], dtype=_N.int)
        clstrs   = _N.array([0, 1])
        lab      = _N.array(labS.tolist() + (labH + clstrs[0]).tolist())
        M        = 1
        MF       = 1
        flatlabels = _N.zeros(len(Asts), dtype=_N.int)
    else:
        if not doSepHash:
            unonhash = _N.arange(len(Asts))
            hashsp   = _N.array([])
            hashthresh = _N.min(_x[:, 1:], axis=0)   #  no hash spikes

            ###   1 cluster
            # labS = _N.zeros(len(Asts), dtype=_N.int)
            # labH = _N.array([], dtype=_N.int)
            # clstrs = _N.array([0, 1])
        else:
            unonhash, hashsp, hashthresh = sepHash(_x, BINS=20, blksz=5, xlo=oo.xLo, xhi=oo.xHi)
            #  hashthresh is dim 2

            # print len(unonhash)
            # print "--------"
            # print len(hashsp)
            # fig = _plt.figure(figsize=(5, 10))
            # fig.add_subplot(3, 1, 1)
            # _plt.scatter(_x[hashsp, 1], _x[hashsp, 2], color="red")
            # _plt.scatter(_x[unonhash, 1], _x[unonhash, 2], color="black")
            # fig.add_subplot(3, 1, 2)
            # _plt.scatter(_x[hashsp, 0], _x[hashsp, 1], color="red")
            # _plt.scatter(_x[unonhash, 0], _x[unonhash, 1], color="black")
            # fig.add_subplot(3, 1, 3)
            # _plt.scatter(_x[hashsp, 0], _x[hashsp, 2], color="red")
            # _plt.scatter(_x[unonhash, 0], _x[unonhash, 2], color="black")


        # len(hashsp)==len(labH)
        # len(unonhash)==len(labS)
        if (len(unonhash) > 0) and (len(hashsp) > 0): 
            labS, labH, clstrs = emMKPOS_sep1B(_x[unonhash], _x[hashsp])
        elif len(unonhash) == 0:
            labS, labH, clstrs = emMKPOS_sep1B(None, _x[hashsp], TR=5)
        else:
            labS, labH, clstrs = emMKPOS_sep1B(_x[unonhash], None, TR=5)
        if doSepHash:
            splitclstrs(_x[unonhash], labS)
            posMkCov0(_x[unonhash], labS)

            #mergesmallclusters(_x[unonhash], _x[hashsp], labS, labH, K+1, clstrs)
            smallClstrID, spksInSmallClstrs = findsmallclusters(_x[unonhash], labS, K+1)

            print smallClstrID
            _N.savetxt("labSb4", labS, fmt="%d")
            for nid in smallClstrID:
                ths = _N.where(labS == nid)[0]
                labS[ths] = -1#clstrs[0]+clstrs[1]-1  # -1 first for easy cpack2
            _N.savetxt("labS", labS, fmt="%d")

            # 0...clstrs[0]-1     clstrs[0]...clstrs[0]+clstrs[1]-1  (no nz)
            # 0...clstrs[0]-2     clstrs[0]-1...clstrs[0]+clstrs[1]-2  (no nz)
            contiguous_pack2(labS, startAt=-1)

            clstrs[0] = len(_N.unique(labS)) 
            clstrs[1] = len(_N.unique(labH))

            print "----------"
            print clstrs
            print "----------"
            # labS [0...#S]   labH [#S...#S+#H]
            
            nzspks = _N.where(labS == -1)[0]
            labS[nzspks] = clstrs[0]+clstrs[1]-1   #  highest ID

            contiguous_pack2(labH, startAt=(clstrs[0]-1))
            _N.savetxt("labH", labH, fmt="%d")
            _N.savetxt("labS", labS, fmt="%d")

            #contiguous_pack2(labH, startAt=(_N.max(labS)+1))

            nonnz = _N.where(labS < clstrs[0]-1)[0]
            nz    = _N.where(labS == clstrs[0]+clstrs[1]-1)[0]
            _plt.scatter(_x[hashsp, 0], _x[hashsp, 1], color="black")
            _plt.scatter(_x[unonhash[nonnz], 0], _x[unonhash[nonnz], 1], color="blue")
            _plt.scatter(_x[unonhash[nz], 0], _x[unonhash[nz], 1], color="red")

            #colorclusters(_x[hashsp], labH, clstrs[1], name="hash", xLo=xLo, xHi=xHi)
            #colorclusters(_x[unonhash], labS, clstrs[0], name="nhash", xLo=xLo, xHi=xHi)


    #     #fig = _plt.figure(figsize=(7, 10))
    #     #fig.add_subplot(2, 1, 1)

        flatlabels = _N.ones(n1-n0, dtype=_N.int)*-1   # 
        #cls = clrs.get_colors(clstrs[0] + clstrs[1])

        for i in labS:
            these = _N.where(labS == i)[0]

            if len(these) > 0:
                flatlabels[unonhash[these]] = i
            #_plt.scatter(_x[unonhash[these], 0], _x[unonhash[these], 1], color=cls[i])
        #for i in xrange(clstrs[1]):
        for i in labH:
            these = _N.where(labH == i)[0]

            if len(these) > 0:
                flatlabels[hashsp[these]] = i 
            #_plt.scatter(_x[hashsp[these], 0], _x[hashsp[these], 1], color=cls[i+clstrs[0]])

        MF     = clstrs[0] + clstrs[1]   #  includes noise
        if nzclstr:
            ths = _N.where(flatlabels == -1)[0]
            flatlabels[ths] = MF - 1
            M = int((clstrs[0]-1) * 1.3 + clstrs[1]) + 2   #  20% more clusters
        else:
            M = int(clstrs[0] * 1.3 + clstrs[1]) + 2   #  20% more clusters
        print "cluters:  %d" % M

    Mwonz     = M if (nzclstr is False) else M-1
    #####  MODES  - find from the sampling
    oo.sp_prmPstMd = _N.zeros((oo.epochs, 3*Mwonz))   # mode of params
    oo.sp_hypPstMd  = _N.zeros((oo.epochs, (2+2+2)*Mwonz)) # hyperparam
    oo.mk_prmPstMd = [_N.zeros((oo.epochs, Mwonz, K)),
                      _N.zeros((oo.epochs, Mwonz, K, K))]
                      # mode of params
    oo.mk_hypPstMd  = [_N.zeros((oo.epochs, Mwonz, K)),
                       _N.zeros((oo.epochs, Mwonz, K, K)), # hyperparam
                       _N.zeros((oo.epochs, Mwonz, 1)), # hyperparam
                       _N.zeros((oo.epochs, Mwonz, K, K))]

    print labS
    print labH
    _N.savetxt("flatlabels", flatlabels, fmt="%d")
    ##################

    # flatlabels + lab = same content, but flatlabels are temporally correct
    return labS, labH, flatlabels, Mwonz, MF, hashthresh, clstrs

def declare_params(_M, K, nzclstr=False, uAll=None, SgAll=None):
    ######################################  INITIAL VALUE OF PARAMS
    M        = _M if not nzclstr else _M + 1   #  l0, q2, f, u, Sg include nzcl
    l0       = _N.array([11.,]*M)
    q2       = _N.array([0.04]*M)
    f        = _N.empty(M)
    u       = _N.zeros((M, K))   #  center
    Sg      = _N.tile(_N.identity(K), M).T.reshape((M, K, K))*0.1
    return l0, f, q2, u, Sg

def declare_prior_hyp_params(M, MF, K, x, mks, Asts, t0, priors, labS, labH):
    #  PRIORS.  These get updated after each EPOCH
    #  priors  prefixed w/ _
    _f_u    = _N.zeros(M);    _f_q2  = _N.ones(M)*16 #  wide
    _q2_a   = _N.ones(M)*0.01;    _q2_B  = _N.ones(M)*1e-3
    _l0_a   = _N.ones(M)*0.5;     _l0_B  = _N.ones(M)

    iclstr  = -1
    for clstr_id in xrange(M):
        _f_u[clstr_id]    = priors._f_u[0]
        _f_q2[clstr_id]   = priors._f_q2[0]
        #  inverse gamma
        _q2_a[clstr_id]   = priors._q2_a[0]
        _q2_B[clstr_id]   = priors._q2_B[0]
        _l0_a[clstr_id]   = priors._l0_a[0]
        _l0_B[clstr_id]   = priors._l0_B[0]
        
    for lab in [labS, labH]:
        iclstr  += 1
        uniq_ids = _N.unique(lab)

        print "--------------------------------"
        print uniq_ids
        print "--------------------------------"
        for clstr_id in lab:
            _f_u[clstr_id]    = priors._f_u[iclstr]
            _f_q2[clstr_id]   = priors._f_q2[iclstr]
            #  inverse gamma
            _q2_a[clstr_id]   = priors._q2_a[iclstr]
            _q2_B[clstr_id]   = priors._q2_B[iclstr]
            _l0_a[clstr_id]   = priors._l0_a[iclstr]
            _l0_B[clstr_id]   = priors._l0_B[iclstr]

    #mkmn    = _N.mean(mks[Asts+t0], axis=0)   #  let's use
    #mkcv    = _N.cov(mks[Asts+t0], rowvar=0)
    ############
    #_u_u    = _N.tile(mkmn, M).T.reshape((M, K))
    #_u_Sg   = _N.tile(_N.identity(K), M).T.reshape((M, K, K))*20  #  this 

    allSg   = _N.zeros((K, K))
    sd  = _N.sort(mks[Asts], axis=0)    #  first index is tetrode

    mins= _N.min(sd, axis=0);     maxs= _N.max(sd, axis=0)

    Wdth= sd[-1] - sd[0]
    ctr = sd[0] + 0.5*(sd[-1] - sd[0])
    _u_u    = _N.tile(ctr, M).T.reshape((M, K))

    _N.fill_diagonal(allSg, (5*Wdth)**2)

    #  xcorr(1, 2)**2 / var1 var2
    for ix in xrange(K):
        for iy in xrange(ix + 1, K):
            pc, pv = _ss.pearsonr(mks[Asts, ix], mks[Asts, iy])
            allSg[ix, iy]  = pc*pc * _N.sqrt(allSg[ix, ix] * allSg[iy, iy])
            allSg[iy, ix]  = allSg[ix, iy]

    #_u_Sg   = _N.tile(_N.identity(K), M).T.reshape((M, K, K))  #  this 
    _u_Sg   = _N.tile(allSg, M).T.reshape((M, K, K))  #  this 
    _u_iSg  = _N.linalg.inv(_u_Sg)
    ############
    _Sg_nu  = _N.ones((M, 1))*(K*1.01)
    _Sg_PSI = _N.tile(_N.identity(K), M).T.reshape((M, K, K))*0.05

    return _l0_a, _l0_B, _f_u, _f_q2, _q2_a, _q2_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI

def init_params_hyps(oo, M, MF, K, l0, f, q2, u, Sg, _l0_a, _l0_B, _f_u, _f_q2, _q2_a, _q2_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI, Asts, t0, x, mks, flatlabels, nHSclusters, nzclstr=False, signalClusters=None):
    """
    M is # of clusters excluding noize
    """
    nSpks = len(flatlabels)   # 0..nHclusters[0] are non-hash, nHclusters[0]..nHcluster[0]+nHcluster[1] are hash

    nSpks = len(flatlabels)
    for im in xrange(MF):  #if lab < 0, these marks not used for init
        kinds = _N.where(flatlabels == im)[0]  #  inds
        #print "len   %d" % len(kinds)
        f[im]  = _N.mean(x[Asts[kinds]+t0], axis=0)
        u[im]  = _N.mean(mks[Asts[kinds]+t0], axis=0)
        if len(kinds) > 1:
            q2[im] = _N.std(x[Asts[kinds]+t0], axis=0)**2
        else:
            q2[im] = 0.1  #   just don't know about this one
        if len(kinds) > K:
            Sg[im] = _N.cov(mks[Asts[kinds]+t0], rowvar=0)
        else:
            Sg[im] = _N.cov(mks[Asts+t0], rowvar=0)
        l0[im] = (len(kinds) / float(nSpks))*100

    for im in xrange(MF, M):  #if lab < 0, these marks not used for init
        f[im]  = _N.random.randn()*3
        u[im]  = _N.random.rand(K)
        q2[im] = 100
        Sg[im] = _N.identity(K)*20
        l0[im] = 100

    if nzclstr:   # the nzclstr has no hyperparams
        print "using the noise cluster"
        #  l0 / sqrt(2*pi*50**2)
        print "M   ''''''''''''    %d" % M
        l0[M] = 1000000.   #  ~ 0.1Hz    #  1000/sqrt(2*pi*300**2)  #  seems better to start high here
        q2[M] = 500**2
        if signalClusters is not None:
            Sg[M] = _N.cov(mks[Asts[signalClusters]], rowvar=0) * 20
            u[M]  = _N.mean(mks[Asts[signalClusters]], axis=0)
        else:
            sd  = _N.sort(mks[Asts], axis=0)    #  first index is tetrode
            mins= _N.min(sd, axis=0);     maxs= _N.max(sd, axis=0)
            Wdth= sd[-1] - sd[0]
            ctr = sd[0] + 0.5*(sd[-1] - sd[0])
            u[M]    = ctr

            _N.fill_diagonal(Sg[M], (5*Wdth)**2)
            #Sg[M] = _N.cov(mks[Asts], rowvar=0)*10000
            #u[M]  = _N.mean(mks[Asts], axis=0)

            print Sg[M]
        f[M]  = 0

    oo.sp_prmPstMd[0, oo.ky_p_l0::3] = l0[0:M]
    oo.sp_prmPstMd[0, oo.ky_p_f::3] = f[0:M]
    oo.sp_prmPstMd[0, oo.ky_p_q2::3] = q2[0:M]
    oo.mk_prmPstMd[oo.ky_p_u][0] = u[0:M]
    oo.mk_prmPstMd[oo.ky_p_Sg][0] = Sg[0:M]

def stochasticAssignment(oo, epc, it, Msc, M, K, l0, f, q2, u, Sg, _f_u, _u_u, _f_q2, _u_Sg, Asts, t0, mASr, xASr, rat, econt, gz, qdrMKS, freeClstr, hashthresh, cmp2Existing, nthrds=1):
    #  Msc   Msc signal clusters
    #  M     all clusters, including nz clstr.  M == Msc when not using nzclstr
    #  Gibbs sampling
    #  parameters l0, f, q2
    #  mASr, xASr   just the mark, position of spikes btwn t0 and t1
    #qdrMKS2 = _N.empty(qdrMKS.shape)
    t1 = _tm.time()
    nSpks = len(Asts)
    twpi = 2*_N.pi

    Kp1      = K+1
    #rat      = _N.zeros(M+1)
    pc       = _N.zeros(M)

    ur         = u.reshape((M, 1, K))
    fr         = f.reshape((M, 1))    # centers
    #print q2
    iq2        = 1./q2
    iSg        = _N.linalg.inv(Sg)
    iq2r       = iq2.reshape((M, 1))  
    try:
        ##  warnings because l0 is 0
        isN = _N.where(q2 <= 0)[0]
        if len(isN) > 0:
            q2[isN] = 0.3

        is0 = _N.where(l0 <= 0)[0]
        if len(is0) > 0:
            l0[is0] = 0.001

        pkFR       = _N.log(l0) - 0.5*_N.log(twpi*q2)   #  M
    except RuntimeWarning:
        print "WARNING"
        print l0
        print q2

    mkNrms = _N.log(1/_N.sqrt(twpi*_N.linalg.det(Sg)))
    mkNrms = mkNrms.reshape((M, 1))   #  M x 1

    rnds       = _N.random.rand(nSpks)

    pkFRr      = pkFR.reshape((M, 1))
    dmu        = (mASr - ur)     # mASr 1 x N x K,     ur  is M x 1 x K
    N          = mASr.shape[1]
    #t2 = _tm.time()
    #_N.einsum("mnj,mjk,mnk->mn", dmu, iSg, dmu, out=qdrMKS)
    #t3 = _tm.time()
    _fm.multi_qdrtcs_par_func(dmu, iSg, qdrMKS, M, N, K, nthrds=nthrds)

    #  fr is    M x 1, xASr is 1 x N, iq2r is M x 1
    #qdrSPC     = (fr - xASr)*(fr - xASr)*iq2r  #  M x nSpks   # 0.01s
    qdrSPC     = _N.empty((M, N))
    _hcb.hc_bcast1(fr, xASr, iq2r, qdrSPC, M, N)

    ###  how far is closest cluster to each newly observed mark

    #  mAS = mks[Asts+t0] 
    #  xAS = x[Asts + t0]   #  position @ spikes

    if cmp2Existing:   #  compare only non-hash spikes and non-hash clusters
        # realCl = _N.where(freeClstr == False)[0]
        # print freeClstr.shape
        # print realCl.shape

        abvthrEachCh = mASr[0] > hashthresh    #  should be NxK of
        abvthrAtLeast1Ch = _N.sum(abvthrEachCh, axis=1) > 0   # N x K
        newNonHashSpks   = _N.where(abvthrAtLeast1Ch)[0]

        newNonHashSpksMemClstr = _N.ones(len(newNonHashSpks), dtype=_N.int) * (M-1)   #  initially, assign all of them to noise cluster

        #print "spikes not hash"
         #print abvthrInds
        abvthrEachCh = u[0:Msc] > hashthresh  #  M x K  (M includes noise)
        abvthrAtLeast1Ch = _N.sum(abvthrEachCh, axis=1) > 0
        
        knownNonHclstrs  = _N.where(abvthrAtLeast1Ch & (freeClstr == False) & (q2[0:Msc] < wdSpc))[0]
        

        #print "clusters not hash"

        #  Place prior for freeClstr near new non-hash spikes that are far 
        #  from known clusters that are not hash clusters 


        nNrstMKS_d = _N.sqrt(_N.min(qdrMKS[knownNonHclstrs], axis=0)/K)  #  dim len(sts)
        nNrstSPC_d = _N.sqrt(_N.min(qdrSPC[knownNonHclstrs], axis=0))
        #  for each spike, distance to nearest non-hash cluster
        # print nNrstMKS_d
        # print nNrstSPC_d
        # print "=============="
        s = _N.empty((len(newNonHashSpks), 3))
        #  for each spike, distance to nearest cluster
        s[:, 0] = newNonHashSpks
        s[:, 1] = nNrstMKS_d[newNonHashSpks]
        s[:, 2] = nNrstSPC_d[newNonHashSpks]
        _N.savetxt(resFN("qdrMKSSPC%d" % epc, dir=oo.outdir), s, fmt="%d %.3e %.3e")

        dMK     = nNrstMKS_d[newNonHashSpks]
        dSP     = nNrstSPC_d[newNonHashSpks]

        ###  assignment into 

        farMKinds = _N.where(dMK > 4)[0]    # 
        #  mean of prior for center - mean of farMKinds
        #  cov  of prior for center - how certain am I of mean?  
        farSPinds = _N.where(dSP > 4)[0]  #  4 std. deviations away

        farMKSPinds = _N.union1d(farMKinds, farSPinds)
        print farMKinds
        print newNonHashSpks
        
        ##  points in newNonHashSpks but not in farMKinds
        notFarMKSPinds = _N.setdiff1d(_N.arange(newNonHashSpks.shape[0]), farMKSPinds)

        farMKSP = _N.empty((len(farMKSPinds), K+1))
        farMKSP[:, 0]  = xASr[0, newNonHashSpks[farMKSPinds]]
        farMKSP[:, 1:] = mASr[0, newNonHashSpks[farMKSPinds]]
        notFarMKSP = _N.empty((len(notFarMKSPinds), K+1))
        notFarMKSP[:, 0]  = xASr[0, newNonHashSpks[notFarMKSPinds]]
        notFarMKSP[:, 1:] = mASr[0, newNonHashSpks[notFarMKSPinds]]

        # farSP = _N.empty((len(farSPinds), K+1))
        # farMK = _N.empty((len(farMKinds), K+1))
        # farSP[:, 0]  = xASr[0, farSPinds]
        # farSP[:, 1:] = mASr[0, farSPinds]
        # farMK[:, 0]  = xASr[0, farMKinds]
        # farMK[:, 1:] = mASr[0, farMKinds]

        minK = 1
        maxK = farMKSPinds.shape[0] / K
        maxK = maxK if (maxK < 6) else 6

        freeClstrs = _N.where(freeClstr == True)[0]
        if maxK >= 2:
            print "coming in here"
            #labs, bics, bestLab, nClstrs = _oT.EMBICs(farMKSP, minK=minK, maxK=maxK, TR=1)
            labs, labsH, clstrs = emMKPOS_sep1B(farMKSP, None, TR=1, wfNClstrs=[[1, 4], [1, 4]], spNClstrs=[[1, 4], [1, 3]])
            nClstrs = clstrs[0]
            bestLab    = labs

            cls = clrs.get_colors(nClstrs)

            _U.savetxtWCom(resFN("newSpksMKSP%d" % epc, dir=oo.outdir), farMKSP, fmt="%.3e %.3e %.3e %.3e %.3e", com=("# number of clusters %d" % nClstrs))
            _U.savetxtWCom(resFN("newSpksMKSP_nf%d" % epc, dir=oo.outdir), notFarMKSP, fmt="%.3e %.3e %.3e %.3e %.3e", com=("# number of clusters %d" % nClstrs))

            L = len(freeClstrs)
            
            unqLabs = _N.unique(bestLab)

            upto    = nClstrs if nClstrs < L else L  #  this should just count large clusters
            ii  = -1
            fig = _plt.figure()
            
            for fid in unqLabs[0:upto]:
                iths = farMKSPinds[_N.where(bestLab == fid)[0]]
                ths = newNonHashSpks[iths]

                for w in xrange(K):
                    fig.add_subplot(2, 2, w+1)
                    _plt.scatter(xASr[0, ths], mASr[0, ths, w], color=cls[ii])

                if len(ths) > K:
                    ii += 1
                    im = freeClstrs[ii]   # Asts + t0 gives absolute time
                    newNonHashSpksMemClstr[iths] = im

                    _u_u[im]  = _N.mean(mASr[0, ths], axis=0)
                    u[im]     = _u_u[im]
                    _f_u[im]  = _N.mean(xASr[0, ths], axis=0)
                    f[im]     = _f_u[im]
                    q2[im]    = _N.std(xASr[0, ths], axis=0)**2 * 9
                    #  l0 = Hz * sqrt(2*_N.pi*q2)
                    l0[im]    =   10*_N.sqrt(q2[im])
                    _f_q2[im] = 1
                    _u_Sg[im] = _N.cov(mASr[0, ths], rowvar=0)*25
                    print "ep %(ep)d  new   cluster #  %(m)d" % {"ep" : epc, "m" : im}
                    print _u_u[im]
                    print _f_u[im]
                    print _f_q2[im]
                else:
                    print "too small    this prob. doesn't represent a cluster"

            _plt.savefig("newspks%d" % epc)


            # #######  known clusters
            # for fid in unqLabs[0:upto]:
            #     iths = farMKSPinds[_N.where(bestLab == fid)[0]]
            #     ths = newNonHashSpks[iths]

            #     for w in xrange(K):
            #         fig.add_subplot(2, 2, w+1)
            #         _plt.scatter(xASr[0, ths], mASr[0, ths, w], color=cls[ii])

            #     if len(ths) > K:
            #         ii += 1
            #         im = freeClstrs[ii]   # Asts + t0 gives absolute time
            #         newNonHashSpksMemClstr[iths] = im

            #         _u_u[im]  = _N.mean(mASr[0, ths], axis=0)
            #         u[im]     = _u_u[im]
            #         _f_u[im]  = _N.mean(xASr[0, ths], axis=0)
            #         f[im]     = _f_u[im]
            #         q2[im]    = _N.std(xASr[0, ths], axis=0)**2 * 9
            #         #  l0 = Hz * sqrt(2*_N.pi*q2)
            #         l0[im]    =   10*_N.sqrt(q2[im])
            #         _f_q2[im] = 1
            #         _u_Sg[im] = _N.cov(mASr[0, ths], rowvar=0)*25
            #         print "ep %(ep)d  new   cluster #  %(m)d" % {"ep" : epc, "m" : im}
            #         print _u_u[im]
            #         print _f_u[im]
            #         print _f_q2[im]
            #     else:
            #         print "too small    this prob. doesn't represent a cluster"

            # _plt.savefig("newspks%d" % epc)


        else:  #  just one cluster
            im = freeClstrs[0]   # Asts + t0 gives absolute time

            _u_u[im]  = _N.mean(mASr[0, newNonHashSpks[farMKSPinds]], axis=0)
            _f_u[im]  = _N.mean(xASr[0, newNonHashSpks[farMKSPinds]], axis=0)
            _u_Sg[im] = _N.cov(mASr[0, newNonHashSpks[farMKSPinds]], rowvar=0)*16
            _f_q2[im] = _N.std(xASr[0, newNonHashSpks[farMKSPinds]], axis=0)**2 * 16

        # ##  kernel density estimate
        # xs  = _N.linspace(-6, 6, 101)
        # xsr = xs.reshape(101, 1)
        # isg2= 1/(0.1**2)   #  spatial kernel bandwidth

        # # fig = _plt.figure(figsize=(6, 9))
        # # fig.add_subplot(1, 2, 1)
        # # _plt.scatter(xASr[0, newNonHashSpks[farMKinds]], mASr[0, newNonHashSpks[farMKinds], 0])
        # # fig.add_subplot(1, 2, 2)
        # # _plt.scatter(xASr[0, newNonHashSpks[farSPinds]], mASr[0, newNonHashSpks[farSPinds], 0])

        # freeClstrs = _N.where(freeClstr == True)[0]
        # L = len(freeClstrs)

        # jjj = 0
        # if (len(farSPinds) >= Kp1) and (len(farMKinds) >= Kp1):
        #     jjj = 1
        #     l1 = L/2

        #     for l in xrange(l1):  # mASr  is 1 x N x K
        #         im = freeClstrs[l]   # Asts + t0 gives absolute time
        #         _u_u[im]  = _N.mean(mASr[0, newNonHashSpks[farMKinds]], axis=0)
        #         y   = _N.exp(-0.5*(xsr - xASr[0, newNonHashSpks[farMKinds]])**2 * isg2)
        #         yc  = _N.sum(y, axis=1)
        #         ix  = _N.where(yc == _N.max(yc))[0][0]
        #         _f_u[im]  = xs[ix]
        #         _u_Sg[im] = _N.cov(mASr[0, newNonHashSpks[farMKinds]], rowvar=0)*30
        #         _f_q2[im] = _N.std(xASr[0, newNonHashSpks[farMKinds]], axis=0)**2 * 30
        #     # _plt.figure()
        #     # _plt.plot(xs, yc)

        #     for l in xrange(l1, L):
        #         im = freeClstrs[l]   # Asts + t0 gives absolute time
        #         _u_u[im]  = _N.mean(mASr[0, newNonHashSpks[farSPinds]], axis=0)
        #         y   = _N.exp(-0.5*(xsr - xASr[0, newNonHashSpks[farSPinds]])**2 * isg2)
        #         yc  = _N.sum(y, axis=1)
        #         ix  = _N.where(yc == _N.max(yc))[0][0]
        #         _f_u[im]  = xs[ix]
        #         _u_Sg[im] = _N.cov(mASr[0, newNonHashSpks[farSPinds]], rowvar=0)*30
        #         _f_q2[im] = _N.std(xASr[0, newNonHashSpks[farSPinds]], axis=0)**2 * 30
        #     # _plt.figure()
        #     # _plt.plot(xs, yc)

        # elif (len(farSPinds) >= Kp1) and (len(farMKinds) < Kp1):
        #     jjj = 2
        #     for l in xrange(L):
        #         im = freeClstrs[l]   # Asts + t0 gives absolute time
        #         _u_u[im]  = _N.mean(mASr[0, newNonHashSpks[farSPinds]], axis=0)
        #         y   = _N.exp(-0.5*(xsr - xASr[0, newNonHashSpks[farSPinds]])**2 * isg2)
        #         yc  = _N.sum(y, axis=1)
        #         ix  = _N.where(yc == _N.max(yc))[0][0]
        #         _f_u[im]  = xs[ix]
        #         _u_Sg[im] = _N.cov(mASr[0, newNonHashSpks[farSPinds]], rowvar=0)*30
        #         _f_q2[im] = _N.std(xASr[0, newNonHashSpks[farSPinds]], axis=0)**2 * 30
        #     # _plt.figure()
        #     # _plt.plot(xs, yc)

        # elif (len(farSPinds) < Kp1) and (len(farMKinds) >= Kp1):
        #     jjj = 3
        #     for l in xrange(L):
        #         im = freeClstrs[l]   # Asts + t0 gives absolute time
        #         _u_u[im]  = _N.mean(mASr[0, newNonHashSpks[farMKinds]], axis=0)
        #         y   = _N.exp(-0.5*(xsr - xASr[0, newNonHashSpks[farMKinds]])**2 * isg2)
        #         yc  = _N.sum(y, axis=1)
        #         ix  = _N.where(yc == _N.max(yc))[0][0]
        #         _f_u[im]  = xs[ix]
        #         _u_Sg[im] = _N.cov(mASr[0, newNonHashSpks[farMKinds]], rowvar=0)*30
        #         _f_q2[im] = _N.std(xASr[0, newNonHashSpks[farMKinds]], axis=0)**2 * 30
        #     # _plt.figure()
        #     # _plt.plot(xs, yc)

        """
        print "^^^^^^^^"
        print freeClstrs
        print "set priors for freeClstrs   %d" % jjj
        #print _u_u[freeClstrs]
        #print _u_Sg[freeClstrs]
        print _f_u[freeClstrs]
        print _f_q2[freeClstrs]
        """

        #if len(farSPinds) > 10:


        #  set the priors of the freeClusters to be near the far spikes


    ####  outside cmp2Existing here
    #   (Mx1) + (Mx1) - (MxN + MxN)
    #cont       = pkFRr + mkNrms - 0.5*(qdrSPC + qdrMKS)
    cont = _N.empty((M, N))
    _hcb.hc_qdr_sum(pkFRr, mkNrms, qdrSPC, qdrMKS, cont, M, N)

    mcontr     = _N.max(cont, axis=0).reshape((1, nSpks))  
    cont       -= mcontr
    _N.exp(cont, out=econt)

    for m in xrange(M):
        rat[m+1] = rat[m] + econt[m]

    rat /= rat[M]
    """
    # print f
    # print u
    # print q2
    # print Sg
    # print l0
    """

    # print rat

    M1 = rat[1:] >= rnds
    M2 = rat[0:-1] <= rnds

    gz[it] = (M1&M2).T

    if cmp2Existing:
        #  gz   is ITERS x N x Mwowonz   (N # of spikes in epoch)
        gz[it, newNonHashSpks] = False   #  not a member of any of them
        gz[it, newNonHashSpks, newNonHashSpksMemClstr] = True


def finish_epoch(oo, nSpks, epc, ITERS, gz, l0, f, q2, u, Sg, _f_u, _f_q2, _q2_a, _q2_B, _l0_a, _l0_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI, smp_sp_hyps, smp_sp_prms, smp_mk_hyps, smp_mk_prms, freeClstr, M, K):
    #  finish epoch doesn't deal with noise cluster
    tt2 = _tm.time()

    gkMAP    = gauKer(2)
    frm   = int(0.7*ITERS)  #  have to test for stationarity

    if nSpks > 0:
        #  ITERS x nSpks x M   
        occ   = _N.mean(_N.mean(gz[frm:ITERS-1], axis=0), axis=0)

    oo.smp_sp_hyps = smp_sp_hyps
    oo.smp_sp_prms = smp_sp_prms
    oo.smp_mk_hyps = smp_mk_hyps
    oo.smp_mk_prms = smp_mk_prms

    l_trlsNearMAP = []
    MAPvalues2(epc, smp_sp_prms, oo.sp_prmPstMd, frm, ITERS, M, 3, occ, gkMAP, l_trlsNearMAP)
    l0[0:M]         = oo.sp_prmPstMd[epc, oo.ky_p_l0::3]
    f[0:M]          = oo.sp_prmPstMd[epc, oo.ky_p_f::3]
    q2[0:M]         = oo.sp_prmPstMd[epc, oo.ky_p_q2::3]
    MAPvalues2(epc, smp_sp_hyps, oo.sp_hypPstMd, frm, ITERS, M, 6, occ, gkMAP, None)
    _f_u[:]       = oo.sp_hypPstMd[epc, oo.ky_h_f_u::6]
    _f_q2[:]      = oo.sp_hypPstMd[epc, oo.ky_h_f_q2::6]
    _q2_a[:]      = oo.sp_hypPstMd[epc, oo.ky_h_q2_a::6]
    _q2_B[:]      = oo.sp_hypPstMd[epc, oo.ky_h_q2_B::6]
    _l0_a[:]      = oo.sp_hypPstMd[epc, oo.ky_h_l0_a::6]
    _l0_B[:]      = oo.sp_hypPstMd[epc, oo.ky_h_l0_B::6]

    #pcklme["cp%d" % epc] = _N.array(smp_sp_prms)
    #trlsNearMAP = _N.array(list(set(trlsNearMAP_D)))+frm   #  use these trials to pick out posterior params for MARK part

    #oo.mk_prmPstMd = [ epochs, M, K
    #                      epochs, M, K, K ]

    #oo.mk_hypPstMd  = [ epochs, M, K
    #                    epochs, M, K, K
    #                    epochs, M, 1
    #                    epochs, M, K, K

    #smp_mk_prms = [   K, ITERS, M
    #                  K, K, ITERS, M
    #smp_mk_hyps = [   K, ITERS, M
    #                  K, K, ITERS, M
    #                  1, ITERS, M
    #                  K, K, ITERS, M

    ##  params and hyper parms for mark

    for m in xrange(M):
        MAPtrls = l_trlsNearMAP[m]
        if len(MAPtrls) == 0:  #  none of them.  causes nan in mean
            MAPtrls = _N.arange(frm, ITERS, 10)
        #print MAPtrls
        u[m] = _N.median(smp_mk_prms[0][:, frm:, m], axis=1)

        Sg[m] = _N.mean(smp_mk_prms[1][:, :, frm:, m], axis=2)
        oo.mk_prmPstMd[oo.ky_p_u][epc, m] = u[m]
        oo.mk_prmPstMd[oo.ky_p_Sg][epc, m]= Sg[m]
        _u_u[m]    = _N.mean(smp_mk_hyps[oo.ky_h_u_u][:, frm:, m], axis=1)
        _u_Sg[m]   = _N.mean(smp_mk_hyps[oo.ky_h_u_Sg][:, :, frm:, m], axis=2)

        _Sg_nu[m]  = _N.mean(smp_mk_hyps[oo.ky_h_Sg_nu][0, frm:, m], axis=0)
        _Sg_PSI[m] = _N.mean(smp_mk_hyps[oo.ky_h_Sg_PSI][:, :, frm:, m], axis=2)
        oo.mk_hypPstMd[oo.ky_h_u_u][epc, m]   = _u_u[m]
        oo.mk_hypPstMd[oo.ky_h_u_Sg][epc, m]  = _u_Sg[m]
        oo.mk_hypPstMd[oo.ky_h_Sg_nu][epc, m] = _Sg_nu[m]
        oo.mk_hypPstMd[oo.ky_h_Sg_PSI][epc, m]= _Sg_PSI[m]
        #print _u_Sg[m]

    u[0:M]         = oo.mk_prmPstMd[oo.ky_p_u][epc]
    Sg[0:M]        = oo.mk_prmPstMd[oo.ky_p_Sg][epc]

    ###  hack here.  If we don't reset the prior for 
    ###  what happens when a cluster is unused?
    ###  l0 -> 0, and at the same time, the variance increases.
    ###  the prior then gets pushed to large values, but
    ###  then it becomes difficult to bring it back to small
    ###  values once that cluster becomes used again.  So
    ###  we would like unused clusters to have l0->0, but keep the
    ###  variance small.  That's why we will reset a cluster

    sq25  = 5*_N.sqrt(q2)

    if M > 1:
        occ = _N.mean(_N.sum(gz[frm:], axis=1), axis=0)  # avg. # of marks assigned to this cluster
        socc = _N.sort(occ)
        minAss = (0.5*(socc[-2]+socc[-1])*0.01)  #  if we're 100 times smaller than the average of the top 2, let's consider it empty

    if oo.resetClus and (M > 1):
        for m in xrange(M):
            #  Sg and q2 are treated differently.  Even if no spikes are
            #  observed, q2 is updated, while Sg is not.  
            #  This is because NO spikes in physical space AND trajectory
            #  information contains information about the place field.
            #  However, in mark space, not observing any marks tells you
            #  nothing about the mark distribution.  That is why f, q2
            #  are updated when there are no spikes, but u and Sg are not.

            if q2[m] < 0:
                print "????????????????"
                print q2
                print "q2[%(m)d] = %(q2).3f" % {"m" : m, "q2" : q2[m]}
                print smp_sp_prms[0, :, m]
                print smp_sp_prms[1, :, m]
                print smp_sp_prms[2, :, m]
                print smp_sp_hyps[4, :, m]
                print smp_sp_hyps[5, :, m]
            if ((occ[m] < minAss) and (l0[m] / _N.sqrt(twpi*q2[m]) < 1)) or \
                                  (f[m] < oo.xLo-sq25[m]) or \
                                  (f[m] > oo.xHi+sq25[m]):
                print "resetting  cluster %(m)d   %(l0).3f  %(f).3f" % {"m" : m, "l0" : (l0[m] / _N.sqrt(twpi*q2[m])), "f" : f[m]}

                _q2_a[m] = 1e-4
                _q2_B[m] = 1e-3
                _f_q2[m] = 4
                _u_Sg[m] = _N.identity(K)*9
                _l0_a[m] = 1e-4
                freeClstr[m] = True
            else:
                freeClstr[m] = False


    rsmp_sp_prms = smp_sp_prms.swapaxes(1, 0).reshape(ITERS, 3*M, order="F")

    _N.savetxt(resFN("posParams_%d.dat" % epc, dir=oo.outdir), rsmp_sp_prms, fmt=("%.4f %.4f %.4f " * M))   #  the params for the non-noise
    #_N.savetxt(resFN("posHypParams.dat", dir=oo.outdir), smp_sp_hyps[:, :, 0].T, fmt="%.4f %.4f %.4f %.4f %.4f %.4f")


