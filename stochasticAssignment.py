import fastnum as _fm
import hc_bcast as _hcb
import numpy as _N
from fitutil import  emMKPOS_sep1A, emMKPOS_sep1B, sepHash, colorclusters, findsmallclusters, splitclstrs, posMkCov0, contiguous_pack2
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
import pickle

twpi = 2*_N.pi
wdSpc = 1

def stochasticAssignment(oo, epc, it, M, K, l0, f, q2, u, Sg, iSg, _f_u, _u_u, _f_q2, _u_Sg, Asts, t0, mASr, xASr, rat, econt, gz, qdrMKS, freeClstr, hashthresh, m1stHashClstr, cmp2Existing, nthrds=1):
    #  Msc   Msc signal clusters
    #  M     all clusters, including nz clstr.  M == Msc when not using nzclstr
    #  Gibbs sampling
    #  parameters l0, f, q2
    #  mASr, xASr   just the mark, position of spikes btwn t0 and t1
    #  qdrMKS   quadratic distance from all marks to the M cluster centers
    t1 = _tm.time()
    nSpks = len(Asts)
    twpi = 2*_N.pi

    Kp1      = K+1
    pc       = _N.zeros(M)

    ur         = u.reshape((M, 1, K))
    fr         = f.reshape((M, 1))    # centers
    #print q2
    iq2        = 1./q2
    iq2r       = iq2.reshape((M, 1))  
    pkFR       = _N.log(l0) - 0.5*_N.log(twpi*q2)   #  M

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

    if cmp2Existing and (M > 1):   #  compare only non-hash spikes and non-hash clusters
        # realCl = _N.where(freeClstr == False)[0]
        # print freeClstr.shape
        # print realCl.shape

        #print "largest mark obs."
        maxMKS = _N.max(mASr[0], axis=0)

        abvthrEachCh = mASr[0] > hashthresh    #  should be NxK of
        abvthrAtLeast1Ch = _N.sum(abvthrEachCh, axis=1) > 0   # N x K
        belowthrAllCh    = _N.sum(abvthrEachCh, axis=1) == 0   # N x K
        newNonHashSpks   = _N.where(abvthrAtLeast1Ch)[0]
        #  newNonHashSpks inevitably will contain hash spks
        #  let's try to clean up some more.  If 
        #mASr[0, newNonHashSpks]
        
        #nonhash spks are far from H clstrs.  The closest one should be > 2 away

        #print "spikes not hash"
         #print abvthrInds
        abvthrEachCh = u > hashthresh  #  M x K  (M includes noise)
        abvthrAtLeast1Ch = _N.sum(abvthrEachCh, axis=1) > 0
        belowthrAllCh    = _N.sum(abvthrEachCh, axis=1) == 0   # N x K
        
        #knownNonHclstrs  = _N.where(abvthrAtLeast1Ch & (freeClstr == False) & (q2 < wdSpc))[0]   
        allClstrs = _N.arange(M)
        knownNonHclstrs  = _N.where((freeClstr == False) & (allClstrs < m1stHashClstr))[0]
        knownHclstrs     = _N.where((freeClstr == False) & (allClstrs >= m1stHashClstr))[0]

        freeClstrs      = _N.where(freeClstr == True)[0]        
        knownClstrs       = _N.where(freeClstr == False)[0]        

        newNonHashSpksMemClstr = _N.ones(len(newNonHashSpks), dtype=_N.int) * (M-1)   #  initially, assign all of them to noise cluster


        #print "clusters not hash"

        #  Place prior for freeClstr near new non-hash spikes that are far 
        #  from known clusters that are not hash clusters 

        #  _N.min(qdrMKS[knownNonHclstrs], axis=0) is size N
        #  _N.min(qdrMKS[knownHclstrs], axis=0) is also size N
        fp = open(resFN("stochAss_cmp2Existing%d" % epc, dir=oo.outdir), "w")
        if len(knownNonHclstrs) > 0:   #  there are signal clusters available
            nNrstMKS_d = _N.sqrt(_N.min(qdrMKS[knownNonHclstrs], axis=0))  #  dim len(sts)
            #nNrstMKS_2H_d = _N.sqrt(_N.min(qdrMKS[knownHclstrs], axis=0))  #  dim len(sts)
            nNrstSPC_d = _N.sqrt(_N.min(qdrSPC[knownNonHclstrs], axis=0))
            closest_clust  = _N.sqrt(_N.min(qdrMKS[knownClstrs], axis=0))
            closest_clust_r= closest_clust.reshape((N, 1))
            #print _N.sqrt(qdrMKS[knownClstrs]).shape
            #print closest_clust_r.shape
            nNrstMKS_i     = _N.where(_N.sqrt(qdrMKS[knownClstrs].T) == closest_clust_r)[1]    #  index of closest cluster, regardless of hash or signal
        else:
            print "len(knownNonHclstrs)  is == 0   %d" % len(knownNonHclstrs)

        #newNonHashSpks_refnd = newNonHashSpks
        #  for each spike, distance to nearest non-hash cluster
        # print nNrstMKS_d
        # print nNrstSPC_d
        # print "=============="

        closer2nonhash = _N.where(nNrstMKS_i < m1stHashClstr)[0]
        newNonHashSpks_r = _N.intersect1d(closer2nonhash, newNonHashSpks)
        print ":;;;;;;;;;;;;;;;"
        print N
        print nNrstMKS_i.shape
        print newNonHashSpks.shape
        print newNonHashSpks_r.shape
        print ":;;;;;;;;;;;;;;;"

        dMK     = nNrstMKS_d[newNonHashSpks_r]
        dSP     = nNrstSPC_d[newNonHashSpks_r]

        s = _N.empty((len(newNonHashSpks_r), 4+K))
        #  for each non-hash spike, distance to nearest cluster
        s[:, 0] = 0
        s[:, 1] = nNrstMKS_d[newNonHashSpks_r]
        s[:, 2] = nNrstSPC_d[newNonHashSpks_r]
        s[:, 3] = xASr[0, newNonHashSpks_r]
        s[:, 4:4+K] = mASr[0, newNonHashSpks_r]
        xmkPrt = "%.3e " * (1+K)
        _N.savetxt(resFN("qdrMKSSPC%d" % epc, dir=oo.outdir), s, fmt=("%d %.3e %.3e " + xmkPrt))

        #  from newnonHashSpks, select only those whose nearest is non-hash


        #  
        ###  assignment into 

        farMKinds = _N.where(dMK > 3)[0]    # 
        #  mean of prior for center - mean of farMKinds
        #  cov  of prior for center - how certain am I of mean?  
        farSPinds = _N.where(dSP > 2)[0]  #  4 std. deviations away

        fp.write("far in space only\n")
        farInSpcOnly = _N.setdiff1d(farSPinds, farMKinds)
        fp.write("%s\n" % str(farInSpcOnly))
        farInMkOnly  = _N.setdiff1d(farMKinds, farSPinds)
        fp.write("far in mark only\n")
        fp.write("%s\n" % str(farInMkOnly))
        fp.write("far in both\n")
        farInBoth    = _N.intersect1d(farSPinds, farMKinds)
        fp.write("%s\n" % str(farInBoth))

        #  only far in SP           _N.setdiff1d(SP, MK)   #  only far in space
        #  only far in MK           _N.setdiff1d(MK, SP)   #  only far in mk
        #  far in both MK and SP    _N.intersect1d(MK, SP)

        #  If I find a cluster far from any known cluster, I should
        #  prior u        u_u   mean of new.  u_Sg  covariance of new x 4
        #  prior f        f_u   mean of new.  f_q2  variance of new x 4
        #  prior sig2     no set
        #  prior Sg       no set
        #  prior l0       no set


        ii = 0
        dcov = _N.zeros((K, K))
        print "----------------------------------------------"
        reassign_l8r = []  
        farClusts = []
        while ii < (len(freeClstrs)-1):  #  keep one fat
            farClusts.append([])
            im = freeClstrs[ii]   # Asts + t0 gives absolute time
            ii += 1

            ths = newNonHashSpks_r
            if (ii % 3 == 0):
                if len(farInBoth) > 2:
                    ths = newNonHashSpks_r[farInBoth]
                elif len(farInSpcOnly) > 2:
                    ths = newNonHashSpks_r[farInSpcOnly]
                elif len(farInMkOnly) > 2:
                    ths = newNonHashSpks_r[farInMkOnly]
            elif (ii % 3 == 1):
                if len(farInSpcOnly) > 2:
                    ths = newNonHashSpks_r[farInSpcOnly]
                elif len(farInMkOnly) > 2:
                    ths = newNonHashSpks_r[farInMkOnly]
                elif len(farInBoth) > 2:
                    ths = newNonHashSpks_r[farInBoth]
            if (ii % 3 == 2):
                if len(farInMkOnly) > 2:
                    ths = newNonHashSpks_r[farInMkOnly]
                elif len(farInBoth) > 2:
                    ths = newNonHashSpks_r[farInBoth]
                elif len(farInSpcOnly) > 2:
                    ths = newNonHashSpks_r[farInSpcOnly]

            
            if len(ths) > 2:
                fp.write("setting priors for cluster %d\n" % im)
                #_u_u[im]  = _N.mean(mASr[0, ths], axis=0)
                u[im]     = _N.mean(mASr[0, ths], axis=0)
                Sg[im]     = _N.cov(mASr[0, ths], rowvar=0)
                fp.write("setting u to %s\n" % str(u[im]))
                xmk = _N.empty((len(ths), K+1))
                xmk[:, 0] = xASr[0, ths]
                xmk[:, 1:] = mASr[0, ths]
                farClusts[ii-1].append(xmk)

                _f_u[im]  = _N.mean(xASr[0, ths])
                f[im]     = _N.mean(xASr[0, ths])
                fp.write("setting f to %s\n" % str(f[im]))
                #q2[im]     = _N.std(xASr[0, ths])**2/10  #  bound to contain noise
                q2[im]     = _N.std(xASr[0, ths])**2/50.  #  bound to contain noise
                l0[im]     = 200*_N.sqrt(q2[im])
                _f_q2[im] = 2**2
                imap = _N.empty((len(ths), 2), dtype=_N.int)
                imap[:, 0] = ths
                imap[:, 1] = im
                reassign_l8r.append(imap)

                # cov       = _N.cov(mASr[0, ths], rowvar=0)
                # if len(ths) <= K:
                #     _N.fill_diagonal(dcov, _N.diagonal(cov)*9)
                #     _u_Sg[im] = dcov
                # else:
                #     _u_Sg[im] = cov*9

            #  l0 = Hz * sqrt(2*_N.pi*q2)


            #_u_Sg[im] = _N.cov(mASr[0, ths], rowvar=0)*25
            print "ep %(ep)d  new   cluster #  %(m)d" % {"ep" : epc, "m" : im}

        dmp = open(resFN("farClusts_%d.dmp" % epc, dir=oo.outdir), "wb")
        pickle.dump(farClusts, dmp, -1)
        dmp.close()


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

    if cmp2Existing and (M > 1):
        print "length of reassign_l8r   %d"  % len(reassign_l8r)
        for ii in xrange(len(reassign_l8r)):

            gz[it, reassign_l8r[ii][:, 0]] = False
            gz[it, reassign_l8r[ii][:, 0], reassign_l8r[ii][:, 1]] = True
    #     #  gz   is ITERS x N x M   (N # of spikes in epoch)
    #     gz[it, newNonHashSpks] = False   #  not a member of any of them
    #     gz[it, newNonHashSpks, newNonHashSpksMemClstr] = True

