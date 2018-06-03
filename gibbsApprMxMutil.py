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

def initClusters(oo, M_max, K, x, mks, t0, t1, Asts, doSepHash=True, xLo=0, xHi=3, oneCluster=False, spcdim=1):
    n0 = 0
    n1 = len(Asts)
    print "gibbsApprMxMutil.initClusters   %d spikes" % (n1-n0)

    
    print (n1-n0)
    _x   = _N.empty((n1-n0, K+spcdim))
    
    _x[:, 0:spcdim]    = x[Asts+t0]
    _x[:, spcdim:]   = mks[Asts+t0]

    if oneCluster:
        unonhash = _N.arange(len(Asts))
        hashsp   = _N.array([])
        if len(Asts > 0):
            hashthresh = _N.min(_x[:, spcdim:], axis=0)   #  no hash spikes
        else:
            hashthresh = -100.

        labS     = _N.zeros(len(Asts), dtype=_N.int)
        labH     = _N.array([], dtype=_N.int)
        clstrs   = _N.array([0, 1])
        lab      = _N.array(labS.tolist() + (labH + clstrs[0]).tolist())
        M        = 1
        MF       = 1
        M_use    = 1
        flatlabels = _N.zeros(len(Asts), dtype=_N.int)
    else:
        if not doSepHash:
            unonhash = _N.arange(len(Asts))
            hashsp   = _N.array([])
            print _x.shape
            hashthresh = _N.min(_x[:, spcdim:], axis=0)   #  no hash spikes


        else:
            unonhash, hashsp, hashthresh = sepHash(_x, BINS=20, blksz=5, xlo=oo.xLo, xhi=oo.xHi, K=K)


        if (len(unonhash) > 0) and (len(hashsp) > 0):  # REAL DATA
            labH, labS, clstrs = emMKPOS_sep1A(_x[unonhash], _x[hashsp], K=K, TR=3)
        elif len(unonhash) == 0:
            labH, labS, clstrs = emMKPOS_sep1A(None, _x[hashsp], TR=1, K=K)
        else:
            print "---------  doSepH==False"   #  doSepHash == False
            labH, labS, clstrs = emMKPOS_sep1A(_x[unonhash], None, TR=1, K=K, spcdim=spcdim)
            #  labs, labh are at this point both starting from 0
        if doSepHash:
            contiguous_pack2(labH, startAt=0)
            clstrs[0] = len(_N.unique(labH))
            clstrs[0] = 2 if clstrs[0] == 1 else clstrs[0]  # at least 2 hash clstrs

            print "clstrs[0]:  %d" % clstrs[0]
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
        MS = MS + 5
        M_use      = clstrs[0] + MS
        print "------------"
        print "hash clusters %d" % clstrs[0]
        print "signal clusters %d" % MS
        print "------------"

        #M = int(clstrs[0] + clstrs[1]) + 1   #  20% more clusters
        print "clusters:  %d" % M_use


    _N.savetxt("flatlabels", flatlabels, fmt="%d")
    ##################

    # flatlabels + lab = same content, but flatlabels are temporally correct
    return labS, labH, flatlabels, M_use, hashthresh, clstrs


def declare_params(M, K, uAll=None, SgAll=None, spcdim=1):
    ######################################  INITIAL VALUE OF PARAMS
    l0       = _N.array([11.,]*M)
    if spcdim == 1:
        q2       = _N.array([0.04]*M)
        f        = _N.empty(M)
    else:
        q2       = _N.array(([0.04]*M, spcdim))
        f        = _N.empty((M, spcdim))
    u       = _N.zeros((M, K))   #  center
    Sg      = _N.tile(_N.identity(K), M).T.reshape((M, K, K))*0.1
    return l0, f, q2, u, Sg

def declare_prior_hyp_params(M, clstrs, K, x, mks, Asts, t0, priors, labS, labH, spcdim=1):
    #  PRIORS.  These get updated after each EPOCH
    #  priors  prefixed w/ _
    if spcdim == 1:
        _f_u    = _N.zeros(M);    _f_q2  = _N.ones(M)*16 #  wide
        _q2_a   = _N.ones(M)*0.01;    _q2_B  = _N.ones(M)*1e-3
    else:
        _f_u    = _N.zeros((M, spcdim));    _f_q2  = _N.ones((M, spcdim))*16 #  wide
        _q2_a   = _N.ones((M, spcdim))*0.01;    _q2_B  = _N.ones((M, spcdim))*1e-3
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

        #_N.fill_diagonal(allSg, (0.4*Wdth)**2)
        _N.fill_diagonal(allSg, (0.6*Wdth)**2)

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

    _u_Sg = _N.empty((M, K, K))
    for m in xrange(M):
        _u_Sg[m] = allSg*2*2

    #_u_Sg = _N.array(_N.tile(allSg*2*2, M).T.reshape((M, K, K)))  #  I want to visit most of the possible space    ---   creates non-contiguous array
    priors._u_Sg = _N.array(_u_Sg[0])
    _u_iSg  = _N.linalg.inv(_u_Sg)

    ##  prior of _Sg_PSI
    ############
    #_Sg_nu  = _N.ones((M, 1))*(K*2.01)   #  we're reasonably sure about width of clusters in make space
    _Sg_nu  = _N.ones(M)*(K*2.01)   #  we're reasonably sure about width of clusters in make space
    print "prior ---------  Sg"
    print allSg
    #_Sg_PSI = _N.tile(allSg/(K*2.01), M).T.reshape((M, K, K))
    _Sg_PSI = _N.empty((M, K, K))

    for m in xrange(M):
        _Sg_PSI[m] = allSg/(K*2.01)
    
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

    oo.sp_prmPstMd[oo.ky_p_l0:oo.ky_p_l0+3*M:3] = l0
    oo.sp_prmPstMd[oo.ky_p_f:oo.ky_p_f+3*M:3] = f
    oo.sp_prmPstMd[oo.ky_p_q2:oo.ky_p_q2+3*M:3] = q2
    oo.mk_prmPstMd[oo.ky_p_u][0:M] = u
    oo.mk_prmPstMd[oo.ky_p_Sg][0:M] = Sg


def finish_epoch2(oo, nSpks, epc, ITERS, gz, l0, f, q2, u, Sg, _f_u, _f_q2, _q2_a, _q2_B, _l0_a, _l0_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI, smp_sp_prms, smp_mk_prms, smp_mk_hyps, freeClstr, M_use, K, priors, m1stSignalClstr, ):
    #  finish epoch doesn't deal with noise cluster
    tt2 = _tm.time()

    #frms = _pU.stationary_from_Z(smp_sp_prms)
    frms = _pU.find_good_clstrs_and_stationary_from(M_use, smp_sp_prms[:, 0:ITERS])

    print "frms------------------"
    print frms
    occ = None
    if nSpks > 0:
        #  gz is ITERS x nSpks x M
        occ   = _N.empty(M_use)
        for m in xrange(M_use):
           #occ[m]   = _N.mean(_N.sum(gz[frms[m]:ITERS-1:10, :, m], axis=1), axis=0)
            if ITERS-frms[m] > 3000:
                occ[m]   = _N.mean(_N.sum(gz[frms[m]:ITERS:30, :, m], axis=1), axis=0)
            elif ITERS-frms[m] > 2000:
                occ[m]   = _N.mean(_N.sum(gz[frms[m]:ITERS:20, :, m], axis=1), axis=0)
            else:
                occ[m]   = _N.mean(_N.sum(gz[frms[m]:ITERS, :, m], axis=1), axis=0)

            print "occupation for m=%(m)d   %(o)f" % {"m" : m, "o" : occ[m]}

    ##  
    l_trlsNearMAP = []
    
    skp = 2
    print "-------------"
    #  marginal posteriors of the spatial and cluster params
    for m in xrange(M_use):
        frm       = frms[m]
        f_smps    = smp_sp_prms[1, frm:ITERS:skp, m]
        _f_u[m]      = _N.mean(f_smps)
        _f_q2[m]     = _N.std(f_smps)**2
        #  hyper params to be copied to _f_u, _f_s
        
        ##########################
        _l0_a[m], _l0_B[m] = _pU.gam_inv_gam_dist_ML(smp_sp_prms[0, frm:ITERS:skp, m], dist=_pU._GAMMA, clstr=m)
        #print "ML fit of smps  _l0_a[%(m)d] %(a).3f  _l0_B[%(m)d] %(B).3f" % {"m" : m, "a" : _l0_a[m], "B" : _l0_B[m]}

        ##########################
        _q2_a[m], _q2_B[m] = _pU.gam_inv_gam_dist_ML(smp_sp_prms[2, frm:ITERS:skp, m], dist=_pU._INV_GAMMA, clstr=m)
        #print "ML fit of smps  _q2_a[%(m)d] %(a).3f  _q2_B[%(m)d] %(B).3f" % {"m" : m, "a" : _q2_a[m], "B" : _q2_B[m]}

    #  modes
    oo.sp_prmPstMd[oo.ky_p_f::3] = _f_u
    oo.sp_prmPstMd[oo.ky_p_l0::3] = (_l0_a - 1) / _l0_B
    oo.sp_prmPstMd[oo.ky_p_q2::3] = _q2_B / (_q2_a + 1) 
    #  go through each cluster, find the iters that are 

    #for
    #MAPvalues2(epc, smp_sp_hyps, oo.sp_hypPstMd, frms, ITERS, M, 6, occ, None)

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

    for m in xrange(M_use):
        frm  = frms[m]
        u[m] = _N.median(smp_mk_prms[0][:, frm:ITERS, m], axis=1)

        Sg[m] = _N.median(smp_mk_prms[1][:, :, frm:ITERS, m], axis=2)
        oo.mk_prmPstMd[oo.ky_p_u][m] = u[m]
        oo.mk_prmPstMd[oo.ky_p_Sg][m]= Sg[m]
        _u_u[m]    = _N.median(smp_mk_hyps[oo.ky_h_u_u][:, frm:ITERS, m], axis=1)
        _u_Sg[m]   = _N.median(smp_mk_hyps[oo.ky_h_u_Sg][:, :, frm:ITERS, m], axis=2)

        _Sg_nu[m]  = _N.median(smp_mk_hyps[oo.ky_h_Sg_nu][frm:ITERS, m], axis=0)
        _Sg_PSI[m] = _N.median(smp_mk_hyps[oo.ky_h_Sg_PSI][:, :, frm:ITERS, m], axis=2)
        # oo.mk_hypPstMd[oo.ky_h_u_u][epc, m]   = _u_u[m]
        # oo.mk_hypPstMd[oo.ky_h_u_Sg][epc, m]  = _u_Sg[m]
        # oo.mk_hypPstMd[oo.ky_h_Sg_nu][epc, m] = _Sg_nu[m]
        # oo.mk_hypPstMd[oo.ky_h_Sg_PSI][epc, m]= _Sg_PSI[m]

    u         = oo.mk_prmPstMd[oo.ky_p_u]
    Sg        = oo.mk_prmPstMd[oo.ky_p_Sg]

    ###  hack here.  If we don't reset the prior for 
    ###  what happens when a cluster is unused?
    ###  l0 -> 0, and at the same time, the variance increases.
    ###  the prior then gets pushed to large values, but
    ###  then it becomes difficult to bring it back to small
    ###  values once that cluster becomes used again.  So
    ###  we would like unused clusters to have l0->0, but keep the
    ###  variance small.  That's why we will reset a cluster

    sq25  = 5*_N.sqrt(q2)

    if M_use > 1:
        #occ = _N.mean(_N.sum(gz[frm:], axis=1), axis=0)  # avg. # of marks assigned to this cluster
        #socc = _N.sort(occ)
        minAss = K    #(0.5*(socc[-2]+socc[-1])*0.01)  #  if we're 100 times smaller than the average of the top 2, let's consider it empty

    if oo.resetClus and (M_use > 1):
        for m in xrange(M_use):
            #  Sg and q2 are treated differently.  Even if no spikes are
            #  observed, q2 is updated, while Sg is not.  
            #  This is because NO spikes in physical space AND trajectory
            #  information contains information about the place field.
            #  However, in mark space, not observing any marks tells you
            #  nothing about the mark distribution.  That is why f, q2
            #  are updated when there are no spikes, but u and Sg are not.

            breset = False
            bBad = (_l0_a[m] is None) or (_l0_B[m] is None) or _N.isnan(_l0_a[m]) or _N.isnan(_l0_B[m]) or (_q2_a[m] is None) or (_q2_B[m] is None) or _N.isnan(_q2_a[m]) or _N.isnan(_q2_B[m])
            if bBad:
                breset = True
                if not _N.isnan(occ[m]):
                    print "cluster %(m)d posterior difficult to estimate found.   occupancy %(o).3f" % {"m" : m, "o" : occ[m]}
                else:
                    print "cluster %d posterior difficult to estimate found.   occupancy was nan" % m
            v1 = _f_q2[m] / oo.sp_prmPstMd[oo.ky_p_q2+3*m]
            v2 = ITERS - frms[m]
            v3 = oo.sp_prmPstMd[oo.ky_p_q2+3*m]
            if (oo.sp_prmPstMd[oo.ky_p_f+3*m] < oo.xLo-sq25[m]) or (oo.sp_prmPstMd[oo.ky_p_f+3*m] > oo.xHi+sq25[m]):
                breset = True
                print "spatial cener too far away from edges  %d" % m
            elif (v1 > 5.):
                breset = True
                print "high uncertainty relative to width  %d" % m
                
            elif (v2 < 4000):
                if (v1 > 1.) and (occ[m] < K):
                    breset = True
                    print "not long enough stationarity %d" % m
                #  else very clean 
            elif (v3 < 1e-4) and (occ[m] < K):
                breset = True
                print "too narrow %d" % m
            elif (oo.sp_prmPstMd[oo.ky_p_q2+3*m] > 2) and (oo.sp_prmPstMd[oo.ky_p_l0+3*m]/_N.sqrt(oo.sp_prmPstMd[oo.ky_p_q2+3*m]) < 1.25) and occ[m] < K:
                breset = True
                print "too low firing rate %d" % m
            elif (freeClstr[m] and occ[m] < K):
                breset = True
                print "free cluster and not enough occupancy %d" % m
            if breset:
                print "****RESET cluster %(m)d  with o %(o)f" % {"m" : m, "o" : occ[m]}
                reset_cluster(epc, m, l0, f, q2, freeClstr, _q2_a, _q2_B, _f_u, _f_q2, _l0_a, _l0_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI, oo, priors, m1stSignalClstr)
            else:
                print "****NO RESET cl %(m)d.   occ %(o).3f   %(1).3e  %(2).3e  %(3).3e" % {"o" : occ[m], "1" : v1, "2" : v2, "3" : v3, "m" : m}

                freeClstr[m] = False

    
    #rsmp_sp_prms = smp_sp_prms.swapaxes(1, 0).reshape(ITERS, 3*M_use, order="F")

    print "freeClstr---------------"
    print freeClstr

    #_N.savetxt(resFN("posParams_%d.dat" % epc, dir=oo.outdir), rsmp_sp_prms, fmt=("%.4f %.4f %.4f " * M_use))   #  the params for the non-noise
    #_N.savetxt(resFN("posHypParams.dat", dir=oo.outdir), smp_sp_hyps[:, :, 0].T, fmt="%.4f %.4f %.4f %.4f %.4f %.4f")


def copy_slice_params(M_use, l0_M, f_M, q2_M, u_M, Sg_M):
    l0 = _N.array(l0_M[0:M_use], copy=True)
    f  = _N.array(f_M[0:M_use], copy=True)
    q2 = _N.array(q2_M[0:M_use], copy=True)
    u  = _N.array(u_M[0:M_use], copy=True)
    Sg = _N.array(Sg_M[0:M_use], copy=True)

    return l0, f, q2, u, Sg


def copy_slice_hyp_params(M_use, _l0_a_M, _l0_B_M, _f_u_M, _f_q2_M, _q2_a_M, _q2_B_M, _u_u_M, _u_Sg_M, _Sg_nu_M, _Sg_PSI_M):
    _l0_a  = _N.array(_l0_a_M[0:M_use], copy=True)
    _l0_B  = _N.array(_l0_B_M[0:M_use], copy=True)
    _f_u   = _N.array(_f_u_M[0:M_use], copy=True)
    _f_q2  = _N.array(_f_q2_M[0:M_use], copy=True)
    _q2_a  = _N.array(_q2_a_M[0:M_use], copy=True)
    _q2_B  = _N.array(_q2_B_M[0:M_use], copy=True)
    _u_u   = _N.array(_u_u_M[0:M_use], copy=True)
    _u_Sg  = _N.array(_u_Sg_M[0:M_use], copy=True)
    _Sg_nu = _N.array(_Sg_nu_M[0:M_use], copy=True)
    _Sg_PSI= _N.array(_Sg_PSI_M[0:M_use], copy=True)

    return _l0_a, _l0_B, _f_u, _f_q2, _q2_a, _q2_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI

def reset_cluster(epc, m, l0, f, q2, freeClstr, _q2_a, _q2_B, _f_u, _f_q2, _l0_a, _l0_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI, oo, priors, m1stSignalClstr):
    #print "resetting  cluster %(m)d   %(l0).3f  %(f).3f" % {"m" : m, "l0" : (l0[m] / _N.sqrt(twpi*q2[m])), "f" : f[m]}
    iclstr = 1 if m >= m1stSignalClstr else 0
    _q2_a[m] = priors._q2_a[iclstr]
    _q2_B[m] = priors._q2_B[iclstr]
    _f_u[m]  = priors._f_u[iclstr]
    _f_q2[m] = priors._f_q2[iclstr]
    _l0_a[m] = priors._l0_a[iclstr]
    _l0_B[m] = priors._l0_B[iclstr]
    _u_u[m]  = priors._u_u
    _u_Sg[m]  = priors._u_Sg
    _Sg_nu[m]  = priors._Sg_nu
    _Sg_PSI[m]  = priors._Sg_PSI
                
    oo.sp_prmPstMd[oo.ky_p_l0+3*m] = 0   # no effect on decode
    oo.sp_prmPstMd[oo.ky_p_q2+3*m] = 10000.

    freeClstr[m] = True
    
def copy_back_params(M_use, l0, f, q2, u, Sg, M_max, l0_M, f_M, q2_M, u_M, Sg_M):
    #  re-copy working (small) parameters to the master (big) parameters 
    l0_M[0:M_use] = l0
    f_M[0:M_use]  = f
    q2_M[0:M_use] = q2
    u_M[0:M_use]  = u
    Sg_M[0:M_use] = Sg

def copy_back_hyp_params(M_use, _l0_a, _l0_B, _f_u, _f_q2, _q2_a, _q2_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI, M_max, _l0_a_M, _l0_B_M, _f_u_M, _f_q2_M, _q2_a_M, _q2_B_M, _u_u_M, _u_Sg_M, _Sg_nu_M, _Sg_PSI_M):
    #  re-copy working (small) hyperparameters to master (big) hyperparameters 
    _l0_a_M[0:M_use] = _l0_a
    _l0_B_M[0:M_use] = _l0_B
    _f_u_M[0:M_use]  = _f_u
    _f_q2_M[0:M_use] = _f_q2
    _q2_a_M[0:M_use] = _q2_a
    _q2_B_M[0:M_use] = _q2_B

    _u_u_M[0:M_use]  = _u_u
    _u_Sg_M[0:M_use] = _u_Sg

    _Sg_nu_M[0:M_use]  = _Sg_nu
    _Sg_PSI_M[0:M_use] = _Sg_PSI

def contiguous_inuse(M_use, M_max, K, freeClstr, l0, f, q2, u, Sg, _l0_a, _l0_B, _f_u, _f_q2, _q2_a, _q2_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI, smp_sp_prms, smp_mk_prms, sp_prmPstMd, mk_prmPstMd, gz, priors):
    #  method called after Gibbs iters are completed
    #  work only with small parameter array, not Master.
    #  _l0_a  (for example) are posterior (to become prior) hyp params calculated from l0s

    #  We do this after finishEpoch2.  smp_sp_prms are filled during Gibbs
    #  iter, so they must also be made contiguous.  
    freeIDs = _N.where(freeClstr[0:M_use] == True)[0]
    if len(freeIDs > 0):
        mf = freeIDs[0]   #  1st free cluster.  Only do stuff after mf

        temp3   = _N.empty((3, smp_sp_prms.shape[1]))
        tempK   = _N.empty((K, smp_sp_prms.shape[1]))
        tempKxK = _N.empty((K, K, smp_sp_prms.shape[1]))
        #  IDs in use past the 
        inuseIDs = _N.where(freeClstr[mf+1:M_use] == False)[0] + mf + 1
    else:
        mf = M_use
        inuseIDs = _N.array([], dtype=_N.int)

    if len(inuseIDs > 0):  #  free cluster between clusters in use.
        freeIDsM = _N.where(freeIDs < inuseIDs[-1])[0]
        Lu     =  len(inuseIDs)
        Lf     = len(freeIDsM)

        imf    = -1

        #iclstr = 1 if m >= m1stSignalClstr else 0

        for imu in xrange(Lu-1, -1, -1):
            imf += 1

            if imf < Lf:
                #  inuseIDs[imu] -> freeClstr[imf]
                print "%(1)d  --> %(2)d" % {"1" : inuseIDs[imu], "2" : freeIDs[imf]}
                #freeClstr[imf] = inuseIDs[imu]
                if freeIDs[imf] < inuseIDs[imu]:
                    l0[freeIDs[imf]]        = l0[inuseIDs[imu]]
                    l0[inuseIDs[imu]]        = 0.1
                    f[freeIDs[imf]]        = f[inuseIDs[imu]]
                    f[inuseIDs[imu]]        = 0
                    q2[freeIDs[imf]]        = q2[inuseIDs[imu]]
                    q2[inuseIDs[imu]]        = 1
                    u[freeIDs[imf]]        = u[inuseIDs[imu]]
                    u[inuseIDs[imu]]        = 0
                    Sg[freeIDs[imf]]        = Sg[inuseIDs[imu]]
                    Sg[inuseIDs[imu]]        = _N.eye(K)

                    #  hyper params
                    _l0_a[freeIDs[imf]]        = _l0_a[inuseIDs[imu]]
                    _l0_a[inuseIDs[imu]]        = priors._l0_a[1]
                    _l0_B[freeIDs[imf]]        = _l0_B[inuseIDs[imu]]
                    _l0_B[inuseIDs[imu]]        = priors._l0_B[1]

                    _f_u[freeIDs[imf]]        = _f_u[inuseIDs[imu]]
                    _f_u[inuseIDs[imu]]        = priors._f_u[1]
                    _f_q2[freeIDs[imf]]        = _f_q2[inuseIDs[imu]]
                    _f_q2[inuseIDs[imu]]        = priors._f_q2[1]

                    _q2_a[freeIDs[imf]]        = _q2_a[inuseIDs[imu]]
                    _q2_a[inuseIDs[imu]]        = priors._q2_a[1]
                    _q2_B[freeIDs[imf]]        = _q2_B[inuseIDs[imu]]
                    _q2_B[inuseIDs[imu]]        = priors._q2_B[1]

                    _u_u[freeIDs[imf]]        = _u_u[inuseIDs[imu]]
                    _u_u[inuseIDs[imu]]        = priors._u_u#[0]
                    _u_Sg[freeIDs[imf]]        = _u_Sg[inuseIDs[imu]]
                    _u_Sg[inuseIDs[imu]]        = priors._u_Sg#[0]

                    _Sg_nu[freeIDs[imf]]        = _Sg_nu[inuseIDs[imu]]
                    _Sg_nu[inuseIDs[imu]]        = priors._Sg_nu
                    _Sg_PSI[freeIDs[imf]]        = _Sg_PSI[inuseIDs[imu]]
                    _Sg_PSI[inuseIDs[imu]]        = priors._Sg_PSI


                    #  smp_sp_prms  is 3 x ITERS x M
                    temp3[:, :]    = smp_sp_prms[:, :, freeIDs[imf]]
                    smp_sp_prms[:, :, freeIDs[imf]] = smp_sp_prms[:, :, inuseIDs[imu]]
                    smp_sp_prms[:, :, inuseIDs[imu]] = temp3
                    #  smp_mk_prms  is K x ITERS x M
                    tempK[:, :]    = smp_mk_prms[0][:, :, freeIDs[imf]]
                    smp_mk_prms[0][:, :, freeIDs[imf]] = smp_mk_prms[0][:, :, inuseIDs[imu]]
                    smp_mk_prms[0][:, :, inuseIDs[imu]] = tempK
                    tempKxK[:, :, :]    = smp_mk_prms[1][:, :, :, freeIDs[imf]]
                    smp_mk_prms[1][:, :, :, freeIDs[imf]] = smp_mk_prms[1][:, :, :, inuseIDs[imu]]
                    smp_mk_prms[1][:, :, :, inuseIDs[imu]] = tempKxK

                    #oo.sp_prmPstMd = _N.zeros(3*M_use)   # mode params
                    sp_prmPstMd[3*freeIDs[imf]:3*(freeIDs[imf]+1)]        = sp_prmPstMd[3*inuseIDs[imu]:3*(inuseIDs[imu]+1)]
                    sp_prmPstMd[0 + 3*inuseIDs[imu]]        = 1e-15   #  neutralize this cluster
                    sp_prmPstMd[2 + 3*inuseIDs[imu]]        = 10000   #  neutralize this cluster

                    mk_prmPstMd[0][freeIDs[imf]]        = mk_prmPstMd[0][inuseIDs[imu]]
                    mk_prmPstMd[1][freeIDs[imf]]        = mk_prmPstMd[1][inuseIDs[imu]]

                    freeClstr[inuseIDs[imu]] = True
                    freeClstr[freeIDs[imf]] = False

                    #  in each gibbs Iter, which spks are assigned to cluster 0
                    gibbsIter_old, spkIDs_old   = _N.where(gz[:, :, inuseIDs[imu]] == True)   
                    #  in each gibbs Iter, which spks are assigned to cluster 1
                    gibbsIter_new, spkIDs_new   = _N.where(gz[:, :, freeIDs[imf]] == True)

                    gz[gibbsIter_old, spkIDs_old, inuseIDs[imu]]  = False
                    gz[gibbsIter_old, spkIDs_old, freeIDs[imf]]  = True

                    gz[gibbsIter_new, spkIDs_new, freeIDs[imf]]  = False
                    gz[gibbsIter_new, spkIDs_new, inuseIDs[imu]]  = True

            
    else:
        freeIDs = _N.where(freeClstr[0:M_use] == True)[0]
        inuseIDs = _N.where(freeClstr[0:M_use] == False)[0]
        print "didn't need to do anything, inuse are all contiguous"
        print "M_use is %(Mu)d   len(freeIDs) %(fI)d    len(inuseIDs) %(iI)d" % {"Mu" : M_use, "fI" : len(freeIDs), "iI" : len(inuseIDs)}
        


    print "after cont"
    print freeClstr[0:M_use]
