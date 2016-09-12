import numpy as _N
from fitutil import  emMKPOS_sep1A, sepHashEM, sepHash, colorclusters, mergesmallclusters, splitclstrs
from posteriorUtil import MAPvalues2
import clrs 
from filter import gauKer
import time as _tm
from EnDedirs import resFN, datFN
import matplotlib.pyplot as _plt

twpi = 2*_N.pi

def initClusters(oo, K, x, mks, t0, t1, Asts, doSepHash=True, xLo=0, xHi=3):
    n0 = 0
    n1 = len(Asts)

    _x   = _N.empty((n1-n0, K+1))
    _x[:, 0]    = x[Asts+t0]
    _x[:, 1:]   = mks[Asts+t0]

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

        fig = _plt.figure(figsize=(5, 10))
        fig.add_subplot(3, 1, 1)
        _plt.scatter(_x[hashsp, 1], _x[hashsp, 2], color="red")
        _plt.scatter(_x[unonhash, 1], _x[unonhash, 2], color="black")
        fig.add_subplot(3, 1, 2)
        _plt.scatter(_x[hashsp, 0], _x[hashsp, 1], color="red")
        _plt.scatter(_x[unonhash, 0], _x[unonhash, 1], color="black")
        fig.add_subplot(3, 1, 3)
        _plt.scatter(_x[hashsp, 0], _x[hashsp, 2], color="red")
        _plt.scatter(_x[unonhash, 0], _x[unonhash, 2], color="black")

    if (len(unonhash) > 0) and (len(hashsp) > 0):
        labS, labH, clstrs = emMKPOS_sep1A(_x[unonhash], _x[hashsp])
    elif len(unonhash) == 0:
        labS, labH, clstrs = emMKPOS_sep1A(None, _x[hashsp], TR=5)
    else:
        labS, labH, clstrs = emMKPOS_sep1A(_x[unonhash], None, TR=5)
    if doSepHash:
        splitclstrs(_x[unonhash], labS)
        mergesmallclusters(_x[unonhash], _x[hashsp], labS, labH, K+1, clstrs)

        #_N.savetxt("hash", hashsp)
        #_N.savetxt("nhash", unonhash)
        #colorclusters(_x[hashsp], labH, clstrs[1], name="hash", xLo=xLo, xHi=xHi)
        #colorclusters(_x[unonhash], labS, clstrs[0], name="nhash", xLo=xLo, xHi=xHi)

    #fig = _plt.figure(figsize=(7, 10))
    #fig.add_subplot(2, 1, 1)

    flatlabels = _N.ones(n1-n0, dtype=_N.int)*-1   # 
    #cls = clrs.get_colors(clstrs[0] + clstrs[1])
    for i in xrange(clstrs[0]):
        these = _N.where(labS == i)[0]

        if len(these) > 0:
            flatlabels[unonhash[these]] = i
        #_plt.scatter(_x[unonhash[these], 0], _x[unonhash[these], 1], color=cls[i])
    for i in xrange(clstrs[1]):
        these = _N.where(labH == i)[0]

        if len(these) > 0:
            flatlabels[hashsp[these]] = i + clstrs[0]
        #_plt.scatter(_x[hashsp[these], 0], _x[hashsp[these], 1], color=cls[i+clstrs[0]])

    MF     = clstrs[0] + clstrs[1]
    M = MF#int(MF * 1.1) + 2   #  20% more clusters
    print "cluters:  %d" % M


    #####  MODES  - find from the sampling
    oo.sp_prmPstMd = _N.zeros((oo.epochs, 3*M))   # mode of params
    oo.sp_hypPstMd  = _N.zeros((oo.epochs, (2+2+2)*M)) # hyperparam
    oo.mk_prmPstMd = [_N.zeros((oo.epochs, M, K)),
                      _N.zeros((oo.epochs, M, K, K))]
                      # mode of params
    oo.mk_hypPstMd  = [_N.zeros((oo.epochs, M, K)),
                       _N.zeros((oo.epochs, M, K, K)), # hyperparam
                       _N.zeros((oo.epochs, M, 1)), # hyperparam
                       _N.zeros((oo.epochs, M, K, K))]

    ##################
    lab      = _N.array(labS.tolist() + (labH + clstrs[0]).tolist())

    return labS, labH, lab, flatlabels, M, MF, hashthresh

def declare_params(_M, K, nzclstr=False, uAll=None, SgAll=None):
    ######################################  INITIAL VALUE OF PARAMS
    M        = _M if not nzclstr else _M + 1
    l0       = _N.array([11.,]*M)
    q2       = _N.array([0.04]*M)
    f        = _N.empty(M)
    u       = _N.zeros((M, K))   #  center
    Sg      = _N.tile(_N.identity(K), M).T.reshape((M, K, K))*0.1
    return l0, f, q2, u, Sg

def declare_prior_hyp_params(M, MF, K, x, mks, Asts, t0):
    #  PRIORS.  These get updated after each EPOCH
    #  priors  prefixed w/ _
    _f_u    = _N.zeros(M);    _f_q2  = _N.ones(M)*16 #  wide
    #  inverse gamma
    #_q2_a   = _N.ones(M)*4;    _q2_B  = _N.ones(M)*1e-3
    _q2_a   = _N.ones(M)*0.5;    _q2_B  = _N.ones(M)*1e-3
    _l0_a   = _N.ones(M)*1.1;     _l0_B  = _N.ones(M)*(1/30.)
    mkmn    = _N.mean(mks[Asts+t0], axis=0)
    mkcv    = _N.cov(mks[Asts+t0], rowvar=0)
    _u_u    = _N.tile(mkmn, M).T.reshape((M, K))
    _u_Sg   = _N.tile(_N.identity(K), M).T.reshape((M, K, K))*1
    _u_iSg  = _N.linalg.inv(_u_Sg)
    _Sg_nu  = _N.ones((M, 1))*(K*1.01)
    _Sg_PSI = _N.tile(_N.identity(K), M).T.reshape((M, K, K))*0.05

    return _l0_a, _l0_B, _f_u, _f_q2, _q2_a, _q2_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI

def init_params_hyps(oo, M, MF, K, l0, f, q2, u, Sg, Asts, t0, x, mks, flatlabels, nzclstr=False):
    """
    M is # of clusters excluding noize
    """
    nSpks = len(flatlabels)
    for im in xrange(MF):  #if lab < 0, these marks not used for init
        kinds = _N.where(flatlabels == im)[0]  #  inds
        #print "len   %d" % len(kinds)
        f[im]  = _N.mean(x[Asts[kinds]+t0], axis=0)
        #_f_u[im]= f[im]
        #_f_q2[im]= 0.2
        u[im]  = _N.mean(mks[Asts[kinds]+t0], axis=0)
        #_u_u[im]= u[im]
        #_u_Sg[im] = _N.identity(K)
        q2[im] = _N.std(x[Asts[kinds]+t0], axis=0)**2
        Sg[im] = _N.cov(mks[Asts[kinds]+t0], rowvar=0)
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
        l0[M] = 200.   #  ~ 0.1Hz
        q2[M] = 50**2
        Sg[M] = _N.cov(mks[Asts], rowvar=0)#*100
        f[M]  = 0
        u[M]  = _N.mean(mks[Asts], axis=0)

    oo.sp_prmPstMd[0, oo.ky_p_l0::3] = l0[0:M]
    oo.sp_prmPstMd[0, oo.ky_p_f::3] = f[0:M]
    oo.sp_prmPstMd[0, oo.ky_p_q2::3] = q2[0:M]
    oo.mk_prmPstMd[oo.ky_p_u][0] = u[0:M]
    oo.mk_prmPstMd[oo.ky_p_Sg][0] = Sg[0:M]

def stochasticAssignment(oo, it, Msc, M, K, l0, f, q2, u, Sg, _f_u, _u_u, Asts, t0, mASr, xASr, rat, econt, gz, qdrMKS, freeClstr, hashthresh, cmp2Existing):
    #  Msc   Msc signal clusters
    #  M     all clusters, including nz clstr.  M == Msc when not using nzclstr
    #  Gibbs sampling
    #  parameters l0, f, q2
    nSpks = len(Asts)
    twpi = 2*_N.pi

    #rat      = _N.zeros(M+1)
    pc       = _N.zeros(M)

    ur         = u.reshape((1, M, K))
    fr         = f.reshape((M, 1))    # centers
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

        pkFR       = _N.log(l0) - 0.5*_N.log(twpi*q2)
    except RuntimeWarning:
        print "WARNING"
        print l0
        print q2

    mkNrms = _N.log(1/_N.sqrt(twpi*_N.linalg.det(Sg)))
    mkNrms = mkNrms.reshape((M, 1))

    rnds       = _N.random.rand(nSpks)

    pkFRr      = pkFR.reshape((M, 1))
    dmu        = (mASr - ur)

    _N.einsum("nmj,mjk,nmk->mn", dmu, iSg, dmu, out=qdrMKS)
    qdrSPC     = (fr - xASr)*(fr - xASr)*iq2r  #  M x nSpks

    ###  how far is closest cluster to each newly observed mark

    realCl = _N.where(freeClstr == False)[0]
    #print qdrMKS
    #print qdrMKS[realCl]
    nNrstMKS_d = _N.sqrt(_N.min(qdrMKS[realCl], axis=0)/K)  #  dim len(sts)
    nNrstSPC_d = _N.sqrt(_N.min(qdrSPC[realCl], axis=0))

    #  mAS = mks[Asts+t0] 
    #  xAS = x[Asts + t0]   #  position @ spikes

    #  CheckFarFromExistingClusters()
    if cmp2Existing:
        # print "---------------qdrSPC"
        # print qdrSPC
        # print "---------------qdrSPC"
        # fig = _plt.figure()
        # fig.add_subplot(2, 1, 1)
        # _plt.hist(nNrstMKS_d, bins=30)
        # fig.add_subplot(2, 1, 2)
        # _plt.hist(nNrstSPC_d, bins=30)

        # fig = _plt.figure()
        # fig.add_subplot(1, 1, 1)
        # _plt.hist(x[Asts+t0], bins=30)

        ####  mks
        #abvthrEachCh = mks[Asts+t0] > hashthresh
        abvthrEachCh = mASr[:, 0] > hashthresh
        abvthrAtLeast1Ch = _N.sum(abvthrEachCh, axis=1) > 0
        abvthrInds   = _N.where(abvthrAtLeast1Ch)[0]

        print "MKS"
        farMKS = _N.where((nNrstMKS_d > 1) & abvthrAtLeast1Ch)[0]
        print "SPACE"
        farSPC  = _N.where((nNrstSPC_d > 2))[0]
        print "len(farSPC) is %d" % len(farSPC)
        print "len(farMK) is %d" % len(farMKS)


        iused = 0  #  use up to 3
        bDone = False
        # fig = _plt.figure(figsize=(8, 5))
        # fig.add_subplot(2, 1, 1)
        # _plt.scatter(x[Asts + t0], mks[Asts+t0, 0], color="black", s=2)
        bDoMKS = (len(farMKS) > 0)
        bDoSPC = (len(farSPC) > 0)

        for m in xrange(Msc):
            #if freeClstr[m] and (not bDone) and bDoMKS:
            if freeClstr[m]:
                #these     = (Asts+t0)[abvthrInds]  #  in absolute coords
                _f_u[m] = _N.mean(xASr[0, abvthrInds], axis=0)
                _u_u[m]   = _N.mean(mASr[abvthrInds, 0], axis=0)
                l0[m]     = _N.random.rand()*10
                if (iused < 2) and bDoMKS:
                    #these     = (Asts+t0)[farMKS] #Asts + t0 is t-index from 0
                    f[m]      = _N.mean(xASr[0, farMKS], axis=0)
                    u[m]      = _N.mean(mASr[farMKS, 0], axis=0)
                else:
                    f[m]      = _N.mean(xASr[0, abvthrInds], axis=0)
                    u[m]      = _N.mean(mASr[abvthrInds, 0], axis=0)

                    # print "BEG far markise"
                    # _plt.scatter(x[these], mks[these, 0], color="red", s=4)
                    # print "f[m]=%(1)s   u[m]=%(2)s" % {"1" : str(f[m]), "2" : str(u[m])}
                    freeClstr[m] = False
                    iused += 1
            # else:
            #     print "NOTTTTTTT far markise"
            #     print "f[m]=%(1)s   u[m]=%(2)s" % {"1" : str(f[m]), "2" : str(u[m])}

        bDone = False
        # fig.add_subplot(2, 1, 2)
        # _plt.scatter(x[Asts + t0], mks[Asts+t0, 0], color="black", s=3)
        iused = 0
        for m in xrange(Msc):
            #if freeClstr[m] and (not bDone) and bDoSPC:
            if freeClstr[m]:
                #these     = (Asts+t0)[abvthrInds]
                _f_u[m] = _N.mean(xASr[0, abvthrInds], axis=0)
                _u_u[m]   = _N.mean(mASr[abvthrInds, 0], axis=0)
                l0[m]     = _N.random.rand()*10

                # _f_u[m] = _N.mean(x[these], axis=0)
                # _u_u[m]   = _N.mean(mks[these], axis=0)
                # l0[m]     = _N.random.rand()*10

                if (iused < 2) and bDoSPC:
                    #these     = (Asts+t0)[farMKS] #Asts + t0 is t-index from 0
                    f[m]      = _N.mean(xASr[0, farSPC], axis=0)
                    u[m]      = _N.mean(mASr[farSPC, 0], axis=0)
                else:
                    f[m]      = _N.mean(xASr[0, abvthrInds], axis=0)
                    u[m]      = _N.mean(mASr[abvthrInds, 0], axis=0)
                    """
                    if (iused < 2) and bDoSPC:
                    these     = (Asts+t0)[farSPC]
                    f[m]      = _N.mean(x[these], axis=0)
                    u[m]      = _N.mean(mks[these], axis=0)
                    else:
                    f[m]      = _N.mean(x[these], axis=0)
                    u[m]      = _N.mean(mks[these], axis=0)
                    """
                    # _plt.scatter(x[these], mks[these, 0], color="red", s=4)
                    # print "BEG far spatial"
                    # print "f[m]=%(1)s   u[m]=%(2)s" % {"1" : str(f[m]), "2" : str(u[m])}
                    freeClstr[m] = False
                    iused += 1

            # else:
            #     print "NOTTTTTTT far spatial"
            #     print "f[m]=%(1)s   u[m]=%(2)s" % {"1" : str(f[m]), "2" : str(u[m])}

    ####  outside cmp2Existing here
    cont       = pkFRr + mkNrms - 0.5*(qdrSPC + qdrMKS)

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
        u[m] = _N.mean(smp_mk_prms[0][:, MAPtrls, m], axis=1)
        Sg[m] = _N.mean(smp_mk_prms[1][:, :, MAPtrls, m], axis=2)
        oo.mk_prmPstMd[oo.ky_p_u][epc, m] = u[m]
        oo.mk_prmPstMd[oo.ky_p_Sg][epc, m]= Sg[m]
        _u_u[m]    = _N.mean(smp_mk_hyps[oo.ky_h_u_u][:, MAPtrls, m], axis=1)
        _u_Sg[m]   = _N.mean(smp_mk_hyps[oo.ky_h_u_Sg][:, :, MAPtrls, m], axis=2)
        _Sg_nu[m]  = _N.mean(smp_mk_hyps[oo.ky_h_Sg_nu][0, MAPtrls, m], axis=0)
        _Sg_PSI[m] = _N.mean(smp_mk_hyps[oo.ky_h_Sg_PSI][:, :, MAPtrls, m], axis=2)
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
    occ = _N.mean(_N.sum(gz[frm:], axis=1), axis=0)  # avg. # of marks assigned to this cluster
    socc = _N.sort(occ)
    minAss = (0.5*(socc[-2]+socc[-1])*0.01)  #  if we're 100 times smaller than the average of the top 2, let's consider it empty

    print occ

    if oo.resetClus:
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
                freeClstr[m] = True
            else:
                freeClstr[m] = False


    rsmp_sp_prms = smp_sp_prms.swapaxes(1, 0).reshape(ITERS, 3*M, order="F")

    _N.savetxt(resFN("posParams_%d.dat" % epc, dir=oo.outdir), rsmp_sp_prms, fmt=("%.4f %.4f %.4f " * M))
    #_N.savetxt(resFN("posHypParams.dat", dir=oo.outdir), smp_sp_hyps[:, :, 0].T, fmt="%.4f %.4f %.4f %.4f %.4f %.4f")


