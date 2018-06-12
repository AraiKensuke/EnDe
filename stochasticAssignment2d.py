import fastnum as _fm
import hc_bcast as _hcb
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
import pickle

twpi = 2*_N.pi
wdSpc = 1

def stochasticAssignment2d(oo, epc, it, M, K, l0, fx, fy, q2x, q2y, u, Sg, iSg, _fx_u, _fy_u, _u_u, _fx_q2, _fy_q2, _u_Sg, Asts, t0, mAS, xAS, yAS, rat, econt, gz, qdrMKS, freeClstr, hashthresh, m1stSignalClstr, cmp2Existing, diag_cov, clstsz):
    #  Msc   Msc signal clusters
    #  M     all clusters, including nz clstr.  M == Msc when not using nzclstr
    #  Gibbs sampling
    #  parameters l0, f, q2
    #  mASr, xASr   just the mark, position of spikes btwn t0 and t1
    #  qdrMKS   quadratic distance from all marks to the M cluster centers

    #ttt1       = _tm.time()
    nSpks = len(Asts)

    Kp1      = K+1
    pc       = _N.zeros(M)

    ur         = u.reshape((M, 1, K))
    #print q2
    iq2x        = 1./q2x
    iq2y        = 1./q2y

    pkFR       = _N.log(l0) - 0.5*_N.log(twpi*q2x*q2y)   #  M

    mkNrms = _N.log(1/_N.sqrt(twpi*_N.linalg.det(Sg)))
    mkNrms = mkNrms.reshape((M, 1))   #  M x 1

    #if epc == 1:
    #    print pkFR
    #ttt2       = _tm.time()
    rnds       = _N.random.rand(nSpks)
    #ttt3       = _tm.time()
    #pkFR      = pkFR.reshape((M, 1))
    #dmu        = (mASr - ur)     # mASr 1 x N x K,     ur  is M x 1 x K
    N          = mAS.shape[0]
    #dmu        = _N.empty((M, N, K))     # mASr 1 x N x K,     ur  is M x 1 x K
    #_hcb.hc_sub_2_vec_K4(mAS, u, dmu, M, N)
    #ttt4       = _tm.time()
    qdrSPC     = _N.empty((M, N))
    #_fm.multi_qdrtcs_par_func_sym(dmu, iSg, qdrMKS, M, N, K, nthrds=1)
    exp_arg = _N.empty((M, N))    
    #ttt5       = _tm.time()
    # print "1------------------"
    # print pkFR.shape
    # print mkNrms.shape
    # print "2------------------"
    # print exp_arg.shape
    # print fx.shape
    # print fy.shape
    # print "3------------------"
    # print xyAS.shape
    # print iq2x.shape
    # print iq2y.shape
    # print "4------------------"
    # print qdrSPC.shape
    # print mAS.shape
    # print "5------------------"
    # print u.shape
    # print iSg.shape
    # print qdrMKS.shape
    # print "6------------------"

    _fm.full_qdrtcs_K4_2d(pkFR, mkNrms, exp_arg, fx, fy, xAS, yAS, iq2x, iq2y, \
                          qdrSPC, mAS, u, iSg, qdrMKS, M, N, K)
    #ttt6       = _tm.time()

    cmp2Existing = False
    if cmp2Existing and (M > 1):   #  compare only non-hash spikes and non-hash clusters
        # realCl = _N.where(freeClstr == False)[0]
        # print freeClstr.shape
        # print realCl.shape

        #print "largest mark obs."
        maxMKS = _N.max(mAS, axis=0)  # mASr:  1 x N x K, mASr[0]:  N x K

        print hashthresh
        abvthrEachCh     = mAS > hashthresh    #  should be NxK of T,F
        abvthrAtLeast1Ch = _N.sum(abvthrEachCh, axis=1) > 0   # N x K
        belowthrAllCh    = _N.sum(abvthrEachCh, axis=1) == 0   # N x K
        newNonHashSpks   = _N.where(abvthrAtLeast1Ch)[0]

        #  newNonHashSpks inevitably will contain hash spks
        #  let's try to clean up some more.  If 
        #mASr[0, newNonHashSpks]
        
        #nonhash spks are far from H clstrs.  The closest one should be > 2 away

        abvthrEachCh = u > hashthresh  #  u:  M x K  (clstr centers)
        abvthrAtLeast1Ch = _N.sum(abvthrEachCh, axis=1) > 0
        belowthrAllCh    = _N.sum(abvthrEachCh, axis=1) == 0   # N x K
        
        #knownNonHclstrs  = _N.where(abvthrAtLeast1Ch & (freeClstr == False) & (q2 < wdSpc))[0]   
        allClstrs = _N.arange(M)
        #  looks like [2, 5, 6]
        knownNonHclstrs  = _N.where((freeClstr == False) & (allClstrs >= m1stSignalClstr))[0]
        #  looks like [0, 1] or [1]
        knownHclstrs     = _N.where((freeClstr == False) & (allClstrs < m1stSignalClstr))[0]

        freeClstrs      = _N.where(freeClstr == True)[0]     # clstr indices 
        knownClstrs       = _N.where(freeClstr == False)[0]  # clstr indices 
        freeNonHclstrs  = _N.where((freeClstr == True) & (allClstrs >= m1stSignalClstr))[0]

        newNonHashSpksMemClstr = _N.zeros(len(newNonHashSpks), dtype=_N.int)   #  initially, assign all of them to noise cluster


        print "known clusters"
        print knownClstrs
        #print "clusters not hash"

        #  Place prior for freeClstr near new non-hash spikes that are far 
        #  from known clusters that are not hash clusters 

        #  _N.min(qdrMKS[knownNonHclstrs], axis=0) is size N
        #  _N.min(qdrMKS[knownHclstrs], axis=0) is also size N
        fp = open(resFN("stochAss_cmp2Existing%d" % epc, dir=oo.outdir), "w")
        if len(knownNonHclstrs) > 0:   #  there are signal clusters available
            nNrstMKS_d = _N.sqrt(_N.min(qdrMKS[knownNonHclstrs], axis=0))  #  dim len(sts)  - distances
            nNrstSPC_d = _N.sqrt(_N.min(qdrSPC[knownNonHclstrs], axis=0))  #  dim len(sts)  - distances spatial
            closest_clust  = _N.sqrt(_N.min(qdrMKS[knownClstrs], axis=0))
            closest_clust_r= closest_clust.reshape((N, 1))
            #print _N.sqrt(qdrMKS[knownClstrs]).shape
            #print closest_clust_r.shape
            nNrstMKS_i     = knownClstrs[_N.where(_N.sqrt(qdrMKS[knownClstrs].T) == closest_clust_r)[1]]    #  index of closest cluster, regardless of hash or signal
        else:
            print "len(knownNonHclstrs)  is == 0   %d" % len(knownNonHclstrs)

        #newNonHashSpks_refnd = newNonHashSpks
        #  for each spike, distance to nearest non-hash cluster
        # print nNrstMKS_d
        # print nNrstSPC_d
        # print "=============="
        #  nNrstMKS_i  is index of closest cluster
        _N.savetxt("nNrstMKS_i", nNrstMKS_i, fmt="%d")
        closer2nonhash = _N.where(nNrstMKS_i >= m1stSignalClstr)[0]
        newNonHashSpks_r = _N.intersect1d(closer2nonhash, newNonHashSpks)

        dMK     = nNrstMKS_d[newNonHashSpks_r]
        dSP     = nNrstSPC_d[newNonHashSpks_r]

        alls = _N.empty((len(newNonHashSpks), 4+K))
        #  for each non-hash spike, distance to nearest cluster
        alls[:, 0] = 0
        alls[:, 1] = nNrstMKS_d[newNonHashSpks]
        alls[:, 2] = nNrstSPC_d[newNonHashSpks]
        alls[:, 3] = xASr[0, newNonHashSpks]
        alls[:, 4:4+K] = mAS[newNonHashSpks]
        xmkPrt = "%.3e " * (1+K)
        _N.savetxt(resFN("allnewspks%d" % epc, dir=oo.outdir), alls, fmt=("%d %.3e %.3e " + xmkPrt))
        #  looking for non-hash spikes whose closest cluster d
        clsnhs = _N.empty((len(newNonHashSpks_r), 4+K))   # close o non-hash
        #  for each non-hash spike, distance to nearest cluster
        clsnhs[:, 0] = 0
        clsnhs[:, 1] = nNrstMKS_d[newNonHashSpks_r]
        clsnhs[:, 2] = nNrstSPC_d[newNonHashSpks_r]
        clsnhs[:, 3] = xASr[0, newNonHashSpks_r]
        clsnhs[:, 4:4+K] = mAS[newNonHashSpks_r]
        xmkPrt = "%.3e " * (1+K)
        _N.savetxt(resFN("allnewspks_cls2nonhash%d" % epc, dir=oo.outdir), clsnhs, fmt=("%d %.3e %.3e " + xmkPrt))

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

        clsnhs = _N.empty((len(farSPinds), 4+K))   # close o non-hash
        #  for each non-hash spike, distance to nearest cluster
        clsnhs[:, 0] = 0
        clsnhs[:, 1] = nNrstMKS_d[newNonHashSpks_r[farSPinds]]
        clsnhs[:, 2] = nNrstSPC_d[newNonHashSpks_r[farSPinds]]
        clsnhs[:, 3] = xASr[0, newNonHashSpks_r[farSPinds]]
        clsnhs[:, 4:4+K] = mAS[newNonHashSpks_r[farSPinds]]
        xmkPrt = "%.3e " * (1+K)
        _N.savetxt(resFN("farSPinds%d" % epc, dir=oo.outdir), clsnhs, fmt=("%d %.3e %.3e " + xmkPrt))

        clsnhs = _N.empty((len(farMKinds), 4+K))   # close o non-hash
        #  for each non-hash spike, distance to nearest cluster
        clsnhs[:, 0] = 0
        clsnhs[:, 1] = nNrstMKS_d[newNonHashSpks_r[farMKinds]]
        clsnhs[:, 2] = nNrstSPC_d[newNonHashSpks_r[farMKinds]]
        clsnhs[:, 3] = xASr[0, newNonHashSpks_r[farMKinds]]
        clsnhs[:, 4:4+K] = mAS[newNonHashSpks_r[farMKinds]]
        xmkPrt = "%.3e " * (1+K)
        _N.savetxt(resFN("farMKinds%d" % epc, dir=oo.outdir), clsnhs, fmt=("%d %.3e %.3e " + xmkPrt))


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
        print "free clusters ----------------------------------------------"
        reassign_l8r = []  
        farClusts = []
        print freeClstrs

        #print "farInSpcOnly"
        #len(farInSpcOnly)
        #print "farInMkOnly"
        #len(farInMkOnly)
        #print "farInBoth"
        #len(farInBoth)

        while ii < len(freeNonHclstrs):  #  keep one fat
            farClusts.append([])
            im = freeNonHclstrs[ii]   # Asts + t0 gives absolute time
            ii += 1

            ths = newNonHashSpks_r
            if (ii % 3 == 0):
                if len(farInBoth) > 2:
                    print "1a"
                    ths = newNonHashSpks_r[farInBoth]
                elif len(farInSpcOnly) > 2:
                    print "1b"
                    ths = newNonHashSpks_r[farInSpcOnly]
                elif len(farInMkOnly) > 2:
                    print "1c"
                    ths = newNonHashSpks_r[farInMkOnly]
            elif (ii % 3 == 1):
                if len(farInSpcOnly) > 2:
                    print "2a"
                    ths = newNonHashSpks_r[farInSpcOnly]
                elif len(farInMkOnly) > 2:
                    print "2b"
                    ths = newNonHashSpks_r[farInMkOnly]
                elif len(farInBoth) > 2:
                    print "2c"
                    ths = newNonHashSpks_r[farInBoth]
            if (ii % 3 == 2):
                if len(farInMkOnly) > 2:
                    print "3a"
                    ths = newNonHashSpks_r[farInMkOnly]
                elif len(farInBoth) > 2:
                    print "3b"
                    ths = newNonHashSpks_r[farInBoth]
                elif len(farInSpcOnly) > 2:
                    print "3c"
                    ths = newNonHashSpks_r[farInSpcOnly]
            
            if len(ths) > 2:  # more than 2 spikes
                fp.write("se#tting priors for cluster %d\n" % im)
                #_u_u[im]  = _N.mean(mASr[0, ths], axis=0)
                u[im]     = _N.mean(mAS[ths], axis=0)
                Sg[im]     = _N.cov(mAS[ths], rowvar=0)
                fp.write("se#tting u to %s\n" % str(u[im]))
                xmk = _N.empty((len(ths), K+1))
                xmk[:, 0] = xASr[0, ths]
                xmk[:, 1:] = mAS[ths]
                #print "ii is %(ii)d     len farclusts %(l)d" % {"ii" : ii, "l" : len(farClusts)}
                farClusts[ii-1].append(xmk)   #  ii-1 becaus ii incremented

                #_f_u[im]  = 3#_N.mean(xASr[0, ths])
                _f_u[im]  = _N.median(xASr[0, ths])
                f[im]     = _N.median(xASr[0, ths])
                fp.write("se#tting f to %s\n" % str(f[im]))
                #q2[im]     = _N.std(xASr[0, ths])**2/10  #  bound to contain noise
                q2[im]     = (_N.std(xASr[0, ths])**2)  #  bound to contain noise
                l0[im]     = 10*_N.sqrt(q2[im])
                #_f_q2[im] = 0.5**2#2**2
                _f_q2[im] = 5**2
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
        fp.close()

    mx_exp_argr     = _N.max(exp_arg, axis=0).reshape((1, nSpks))  #  1 x N
    exp_arg       -= mx_exp_argr     #  M x N
    #ttt7       = _tm.time()
    _fm.exp_on_arr(exp_arg, econt, M, N)   # _N.exp(cont, out=econt)  
    #ttt8       = _tm.time()
    for m in xrange(M):  #  rat is (M x N)
        rat[m+1] = rat[m] + econt[m]

    rnds *= rat[M]   #  used to be     #rat /= rat[M] (more # of computations)
    # if (epc == 1) and (it > 2):
    #     qdr = qdrSPC + qdrMKS
    #     for n in xrange(rat.shape[1]):
    #         print qdr[:, n]
        
    #  want to see rat[:, 158]

    """
    # print f
    # print u
    # print q2
    # print Sg
    # print l0
    """

    # print rat
    #ttt9       = _tm.time()
    # M1 = rat[1:] >= rnds
    # M2 = rat[0:-1] <= rnds
    # gz[it] = (M1&M2).T
    _fm.set_occ(clstsz, rat, rnds, gz[it], M, N)            
    # print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
    # print gz[it, :, 3]
    # print gz[it, :, 8]
    # print gz[it, :, 10]
    # print "--------------------"
    # M1 = rat[1:] >= rnds
    # M2 = rat[0:-1] <= rnds
    # gz[it] = (M1&M2).T
    # print gz[it, :, 3]
    # print gz[it, :, 8]
    # print gz[it, :, 10]



    # if it % 1000 == 0:
    #     print "iter %d" % it
        # print rat[:, 158]
        # print gz[it, 158]

    #ttt10       = _tm.time()


    # print "#st timing start"
    # print "nt+= 1"
    # print "it2t1+=%.4e" % (#ttt2-#ttt1)
    # print "it3t2+=%.4e" % (#ttt3-#ttt2) 
    # print "it4t3+=%.4e" % (#ttt4-#ttt3) # slowest 0.12
    # print "it5t4+=%.4e" % (#ttt5-#ttt4) # slow  0.03
    # print "it6t5+=%.4e" % (#ttt6-#ttt5)
    # print "it7t6+=%.4e" % (#ttt7-#ttt6)  
    # print "it8t7+=%.4e" % (#ttt8-#ttt7)  
    # print "it9t8+=%.4e" % (#ttt9-#ttt8)  # slow 0.02
    # print "it10t9+=%.4e" % (#ttt10-#ttt9)  # slow 0.08
    # print "#st timing end"

    if cmp2Existing and (M > 1):
        print "length of reassign_l8r   %d"  % len(reassign_l8r)
        for ii in xrange(len(reassign_l8r)):

            gz[it, reassign_l8r[ii][:, 0]] = 0
            gz[it, reassign_l8r[ii][:, 0], reassign_l8r[ii][:, 1]] = 1
    #     #  gz   is ITERS x N x M   (N # of spikes in epoch)
    #     gz[it, newNonHashSpks] = False   #  not a member of any of them
    #     gz[it, newNonHashSpks, newNonHashSpksMemClstr] = True

