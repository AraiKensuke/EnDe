import numpy as _N
from fitutil import  emMKPOS_sep1A, sepHash, colorclusters, contiguous_pack2
from posteriorUtil import MAPvalues2, gam_inv_gam_dist_ML
import clrs 
from filter import gauKer
import time as _tm
from EnDedirs import resFN, datFN
import matplotlib.pyplot as _plt
import scipy.stats as _ss
import openTets as _oT
import utilities as _U
import posteriorUtil as _pU

twpi = 2*_N.pi
wdSpc = 1

def initClusters(oo, K, x, mks, t0, t1, Asts, doSepHash=True, xLo=0, xHi=3, oneCluster=False):
    n0 = 0
    n1 = len(Asts)
    print "gibbsApprMxMutil.initClusters   %d spikes" % (n1-n0)

    _x   = _N.empty((n1-n0, K+1))
    _x[:, 0]    = x[Asts+t0]
    _x[:, 1:]   = mks[Asts+t0]

    if oneCluster:
        unonhash = _N.arange(len(Asts))
        hashsp   = _N.array([])
        if len(Asts > 0):
            hashthresh = _N.min(_x[:, 1:], axis=0)   #  no hash spikes
        else:
            hashthresh = -100.

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
        if (len(unonhash) > 0) and (len(hashsp) > 0):  # REAL DATA
            labH, labS, clstrs = emMKPOS_sep1A(_x[unonhash], _x[hashsp], K=K, TR=3)
        elif len(unonhash) == 0:
            labS, labH, clstrs = emMKPOS_sep1A(None, _x[hashsp], TR=5, K=K)
        else:
            labS, labH, clstrs = emMKPOS_sep1A(_x[unonhash], None, TR=5, K=K)
            #  labs, labh are at this point both starting from 0
        if doSepHash:
            contiguous_pack2(labH, startAt=0)
            clstrs[0] = len(_N.unique(labH))
            clstrs[0] = 2 if clstrs[0] == 1 else clstrs[0]  # at least 2 hash clstrs
            contiguous_pack2(labS, startAt=clstrs[0])
            clstrs[1] = len(_N.unique(labS)) 

            _N.savetxt("labH", labH, fmt="%d")
            _N.savetxt("labS", labS, fmt="%d")

            #colorclusters(_x[hashsp], labH, clstrs[0], name="hash", xLo=xLo, xHi=xHi)
            #colorclusters(_x[unonhash], labS, clstrs[1], name="nhash", xLo=xLo, xHi=xHi)


    #     #fig = _plt.figure(figsize=(7, 10))
    #     #fig.add_subplot(2, 1, 1)

        flatlabels = _N.ones(n1-n0, dtype=_N.int)*-1   # 

        for i in labS:
            these = _N.where(labS == i)[0]

            if len(these) > 0:
                flatlabels[unonhash[these]] = i
        for i in labH:
            these = _N.where(labH == i)[0]

            if len(these) > 0:
                flatlabels[hashsp[these]] = i 
        print flatlabels

        MS     = int(clstrs[1]) 
        #MS = MS + 2 if (MS < 3) else int(_N.ceil(MS*1.1)+1)
        MS = MS + 1
        M      = clstrs[0] + MS
        print "------------"
        print "hash clusters %d" % clstrs[0]
        print "signal clusters %d" % MS
        print "------------"

        #M = int(clstrs[0] + clstrs[1]) + 1   #  20% more clusters
        print "clusters:  %d" % M

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

    _N.savetxt("flatlabels", flatlabels, fmt="%d")
    ##################

    # flatlabels + lab = same content, but flatlabels are temporally correct
    return labS, labH, flatlabels, M, hashthresh, clstrs


def declare_params(M, K, uAll=None, SgAll=None):
    ######################################  INITIAL VALUE OF PARAMS
    l0       = _N.array([11.,]*M)
    q2       = _N.array([0.04]*M)
    f        = _N.empty(M)
    u       = _N.zeros((M, K))   #  center
    Sg      = _N.tile(_N.identity(K), M).T.reshape((M, K, K))*0.1
    return l0, f, q2, u, Sg

def declare_prior_hyp_params(M, clstrs, K, x, mks, Asts, t0, priors, labS, labH):
    #  PRIORS.  These get updated after each EPOCH
    #  priors  prefixed w/ _
    _f_u    = _N.zeros(M);    _f_q2  = _N.ones(M)*16 #  wide
    _q2_a   = _N.ones(M)*0.01;    _q2_B  = _N.ones(M)*1e-3
    _l0_a   = _N.ones(M)*0.5;     _l0_B  = _N.ones(M)

    iclstr  = -1
    iprior  = 0
    for lab in [labH, labS]:

        uniq_ids = _N.unique(lab)
        #  [0,...3]  (at most 3 clstrs hashes)  [1, 2, ...] signal
        for clstr_id in lab:
            iprior = 0 if clstr_id < clstrs[0] else 1

            _f_u[clstr_id]    = priors._f_u[iprior]
            _f_q2[clstr_id]   = priors._f_q2[iprior]
            #  inverse gamma
            _q2_a[clstr_id]   = priors._q2_a[iprior]
            _q2_B[clstr_id]   = priors._q2_B[iprior]
            _l0_a[clstr_id]   = priors._l0_a[iprior]
            _l0_B[clstr_id]   = priors._l0_B[iprior]

    #mkmn    = _N.mean(mks[Asts+t0], axis=0)   #  let's use
    #mkcv    = _N.cov(mks[Asts+t0], rowvar=0)
    ############
    #_u_u    = _N.tile(mkmn, M).T.reshape((M, K))
    #_u_Sg   = _N.tile(_N.identity(K), M).T.reshape((M, K, K))*20  #  this 

    if len(Asts) > 0:
        allSg   = _N.zeros((K, K))
        sd  = _N.sort(mks[Asts], axis=0)    #  first index is tetrode

        mins= _N.min(sd, axis=0);     maxs= _N.max(sd, axis=0)

        Wdth= sd[int(0.95*len(sd))] - sd[int(0.05*len(sd))]
        ctr = sd[int(0.05*len(sd))] + 0.5*(Wdth)
        _u_u    = _N.tile(ctr, M).T.reshape((M, K))

        _N.fill_diagonal(allSg, (0.4*Wdth)**2)

        #  xcorr(1, 2)**2 / var1 var2
        for ix in xrange(K):
            for iy in xrange(ix + 1, K):
                pc, pv = _ss.pearsonr(mks[Asts, ix], mks[Asts, iy])
                allSg[ix, iy]  = pc*pc * _N.sqrt(allSg[ix, ix] * allSg[iy, iy])
                allSg[iy, ix]  = allSg[ix, iy]
    else:
        allSg   = _N.eye(K, K)   # no spikes
        _u_u    = _N.tile(_N.zeros(K), M).T.reshape((M, K))

    #_u_Sg   = _N.tile(_N.identity(K), M).T.reshape((M, K, K))  #  this 
    priors._u_u = _N.array(_u_u[0])
    _u_Sg = _N.array(_N.tile(allSg*2*2, M).T.reshape((M, K, K)))  #  I want to visit most of the possible space
    priors._u_Sg = _N.array(_u_Sg)
    _u_iSg  = _N.linalg.inv(_u_Sg)

    ##  prior of _Sg_PSI
    ############
    _Sg_nu  = _N.ones((M, 1))*(K*2.01)   #  we're reasonably sure about width of clusters in make space
    print "prior ---------  Sg"
    print allSg
    _Sg_PSI = _N.tile(allSg/(K*2.01), M).T.reshape((M, K, K))
    priors._Sg_nu = _N.array(_Sg_nu[0])
    priors._Sg_PSI = _N.array(_Sg_PSI[0])

    return _l0_a, _l0_B, _f_u, _f_q2, _q2_a, _q2_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI

def init_params_hyps(oo, M, K, l0, f, q2, u, Sg, _l0_a, _l0_B, _f_u, _f_q2, _q2_a, _q2_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI, Asts, t0, x, mks, flatlabels, nHSclusters, signalClusters=None):
    """
    M is # of clusters excluding noize
    """
    for im in xrange(M):  #if lab < 0, these marks not used for init
        kinds = _N.where(flatlabels == im)[0]  #  inds
        nSpks = len(kinds)
        print "im  %(im)d  len   %(n)d" % {"im" : im, "n" : len(kinds)}
        if nSpks > 0:
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
        else:
            f[im]  = 0
            u[im]  = _N.zeros(K)
            q2[im] = 1.
            Sg[im] = _N.eye(K)
            l0[im] = 1

    oo.sp_prmPstMd[0, oo.ky_p_l0::3] = l0
    oo.sp_prmPstMd[0, oo.ky_p_f::3] = f
    oo.sp_prmPstMd[0, oo.ky_p_q2::3] = q2
    oo.mk_prmPstMd[oo.ky_p_u][0] = u
    oo.mk_prmPstMd[oo.ky_p_Sg][0] = Sg


def finish_epoch2(oo, nSpks, epc, ITERS, gz, l0, f, q2, u, Sg, _f_u, _f_q2, _q2_a, _q2_B, _l0_a, _l0_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI, smp_sp_hyps, smp_sp_prms, smp_mk_hyps, smp_mk_prms, freeClstr, M, K, priors, m1stSignalClstr, ):
    #  finish epoch doesn't deal with noise cluster
    tt2 = _tm.time()

    blksz= 20
    initBlk = 10
    #frms = _pU.findstat(smp_sp_prms, blksz, initBlk)

    #frms *= blksz
    frms = _N.ones(M, dtype=_N.int)*5000

    occ = None
    if nSpks > 0:
        #  gz is ITERS x nSpks x M
        occ   = _N.empty(M)
        for m in xrange(M):
           occ[m]   = _N.mean(_N.mean(gz[frms[m]:ITERS-1, :, m], axis=0), axis=0)

    ##  

    oo.smp_sp_hyps = smp_sp_hyps
    oo.smp_sp_prms = smp_sp_prms
    oo.smp_mk_hyps = smp_mk_hyps
    oo.smp_mk_prms = smp_mk_prms

    l_trlsNearMAP = []
    
    skp = 2
    print "-------------"
    #  marginal posteriors of the spatial and cluster params
    for m in xrange(M):
        frm       = frms[m]
        f_smps    = smp_sp_prms[1, frm::skp, m]
        _f_u[m]      = _N.mean(f_smps)
        _f_q2[m]     = _N.std(f_smps)**2
        #  hyper params to be copied to _f_u, _f_s
        
        ##########################
        _l0_a[m], _l0_B[m] = _pU.gam_inv_gam_dist_ML(smp_sp_prms[0, frm::skp, m], dist=_pU._GAMMA, clstr=m)
        #print "ML fit of smps  _l0_a[%(m)d] %(a).3f  _l0_B[%(m)d] %(B).3f" % {"m" : m, "a" : _l0_a[m], "B" : _l0_B[m]}

        ##########################
        _q2_a[m], _q2_B[m] = _pU.gam_inv_gam_dist_ML(smp_sp_prms[2, frm::skp, m], dist=_pU._INV_GAMMA, clstr=m)
        #print "ML fit of smps  _q2_a[%(m)d] %(a).3f  _q2_B[%(m)d] %(B).3f" % {"m" : m, "a" : _q2_a[m], "B" : _q2_B[m]}

    oo.sp_prmPstMd[epc, oo.ky_p_f::3] = _f_u
    oo.sp_prmPstMd[epc, oo.ky_p_l0::3] = (_l0_a - 1) / _l0_B
    oo.sp_prmPstMd[epc, oo.ky_p_q2::3] = _q2_B / (_q2_a + 1) 
    #  go through each cluster, find the iters that are 

    #for
    MAPvalues2(epc, smp_sp_hyps, oo.sp_hypPstMd, frms, ITERS, M, 6, occ, None)

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
        frm  = frms[m]
        u[m] = _N.mean(smp_mk_prms[0][:, frm:, m], axis=1)

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

    u         = oo.mk_prmPstMd[oo.ky_p_u][epc]
    Sg        = oo.mk_prmPstMd[oo.ky_p_Sg][epc]

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
        #occ = _N.mean(_N.sum(gz[frm:], axis=1), axis=0)  # avg. # of marks assigned to this cluster
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

            bBad = (_l0_a[m] is None) or (_l0_B[m] is None) or _N.isnan(_l0_a[m]) or _N.isnan(_l0_B[m]) or (_q2_a[m] is None) or (_q2_B[m] is None) or _N.isnan(_q2_a[m]) or _N.isnan(_q2_B[m])
            if bBad:
                print "cluster who's posterior difficult to estimate found.   occupancy %d" % occ[m]
            if bBad or ((occ[m] < minAss) and (l0[m] / _N.sqrt(twpi*q2[m]) < 0.1)) or \
               (f[m] < oo.xLo-sq25[m]) or (f[m] > oo.xHi+sq25[m]):
                print "resetting  cluster %(m)d   %(l0).3f  %(f).3f" % {"m" : m, "l0" : (l0[m] / _N.sqrt(twpi*q2[m])), "f" : f[m]}
                iclstr = 1 if m >= m1stSignalClstr else 0
                _q2_a[m] = priors._q2_a[iclstr]
                _q2_B[m] = priors._q2_B[iclstr]
                _f_u[m]  = priors._f_u[iclstr]
                _f_q2[m] = priors._f_q2[iclstr]
                _l0_a[m] = priors._l0_a[iclstr]
                _l0_B[m] = priors._l0_B[iclstr]
                _u_u[m]  = priors._u_u
                _u_Sg[m]  = priors._u_Sg[0]
                _Sg_nu[m]  = priors._Sg_nu
                _Sg_PSI[m]  = priors._Sg_PSI
                print _Sg_nu[m]
                print _Sg_PSI[m]
                
                oo.sp_prmPstMd[epc, oo.ky_p_l0+3*m] = 0   # no effect on decode
                oo.sp_prmPstMd[epc, oo.ky_p_q2+3*m] = 10000.

                freeClstr[m] = True
            else:
                freeClstr[m] = False

    
    rsmp_sp_prms = smp_sp_prms.swapaxes(1, 0).reshape(ITERS, 3*M, order="F")

    _N.savetxt(resFN("posParams_%d.dat" % epc, dir=oo.outdir), rsmp_sp_prms, fmt=("%.4f %.4f %.4f " * M))   #  the params for the non-noise
    #_N.savetxt(resFN("posHypParams.dat", dir=oo.outdir), smp_sp_hyps[:, :, 0].T, fmt="%.4f %.4f %.4f %.4f %.4f %.4f")



