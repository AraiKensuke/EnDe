import numpy as _N
from filter import gauKer
import matplotlib.pyplot as _plt

def MAPvalues(epc, smp_prms, postMode, frm, ITERS, M, nprms, occ, gk, l_trlsNearMAP):

    for m in xrange(M):
        fig = _plt.figure(figsize=(11, 4))
        trlsNearMAP = _N.arange(0, ITERS-frm)
        for ip in xrange(nprms):  # params
            fig.add_subplot(1, nprms, ip+1)
            L     = _N.min(smp_prms[ip, frm:, m]);   H     = _N.max(smp_prms[ip, frm:, m])
            AMP   = H-L
            L     -= 0.1*AMP
            H     += 0.1*AMP
            cnts, bns = _N.histogram(smp_prms[ip, frm:, m], bins=_N.linspace(L, H, 60))
            ###  take the iters that gave 25% top counts
            ###  intersect them for each param.
            ###  
            col = nprms*m+ip
            
            xfit = 0.5*(bns[0:-1] + bns[1:])
            yfit = cnts

            fcnts = _N.convolve(cnts, gk, mode="same")
            _plt.plot(xfit, fcnts)
            ib  = _N.where(fcnts == _N.max(fcnts))[0][0]

            xLo  = xfit[_N.where(fcnts > fcnts[ib]*0.6)[0][0]]
            xHi  = xfit[_N.where(fcnts > fcnts[ib]*0.6)[0][-1]]

            if occ[m] > 0:
                these=_N.where((smp_prms[ip, frm:, m] > xLo) & (smp_prms[ip, frm:, m] < xHi))[0]
                trlsNearMAP = _N.intersect1d(these, trlsNearMAP)

            xMAP  = bns[ib]                        
            postMode[epc, col] = xMAP

        if l_trlsNearMAP is not None:
            trlsNearMAP += frm
            l_trlsNearMAP.append(trlsNearMAP)




def MAPvalues2(epc, smp_prms, postMode, frm, ITERS, M, nprms, occ, gk, l_trlsNearMAP, alltrials=False):
    print "IN MAPvalues2"
    N   = smp_prms.shape[1] - frm
    for m in xrange(M):
        #fig = _plt.figure(figsize=(11, 4))
        trlsNearMAP = _N.arange(0, ITERS-frm)
        if alltrials:
            l_trlsNearMAP.append(_N.arange(ITERS))
        else:
            for ip in xrange(nprms):  # params
                col = nprms*m+ip
                #fig.add_subplot(1, nprms, ip+1)

                smps  = smp_prms[ip, frm:, m]
                ismps = smps.argsort()
                spcng = _N.diff(smps[ismps])
                ispcng= spcng.argsort()

                if int(0.1*N) > 1:
                    hstSpc= _N.mean(spcng[ispcng[0:int(0.1*N)]])
                else:
                    hstSpc= 1

                thr_ispc = _N.where(spcng[ispcng] < 100*hstSpc)[0]

                if len(thr_ispc) > 0:   #  this param not all the same
                    L     = _N.min(smps);  H     = _N.max(smps[ismps[thr_ispc]])
                    A     = H-L
                    cnts, bns = _N.histogram(smps, bins=_N.linspace(L-0.1*A, H+0.1*A, 60))

                    xfit    = 0.5*(bns[1:] + bns[0:-1])

                    fcnts = _N.convolve(cnts, gk, mode="same")
                    ib  = int(_N.mean(_N.where(fcnts == _N.max(fcnts))[0]))
                    xMAP = bns[ib]
                    #_plt.plot(xfit, fcnts)
                    #_plt.axvline(x=xMAP)


                    xLo  = xfit[_N.where(fcnts > fcnts[ib]*0.6)[0][0]]
                    xHi  = xfit[_N.where(fcnts > fcnts[ib]*0.6)[0][-1]]

                    if occ[m] > 0:
                        these=_N.where((smps > xLo) & (smps < xHi))[0]
                        trlsNearMAP = _N.intersect1d(these, trlsNearMAP)


                    postMode[epc, col] = xMAP
                else:
                    postMode[epc, col] = smps[0]
                    trlsNearMAP = _N.arange(N)+frm    

            if l_trlsNearMAP is not None:
                #trlsNearMAP += frm
                l_trlsNearMAP.append(trlsNearMAP)




"""
smps  = _ss.invgamma.rvs(a, scale=B, size=N)
ismps = smps.argsort()
spcng = _N.diff(smps[ismps])   #  
ispcng= spcng.argsort()

hstSpc= _N.mean(spcng[ispcng[0:int(0.1*N)]])

thr_ispc = _N.where(spcng[ispcng] < 100*hstSpc)

L     = _N.min(smps)
H     = _N.max(smps[ismps[thr_ispc[0]]])
A     = H-L
cnts, bns = _N.histogram(smps, bins=_N.linspace(L-0.1*A, H+0.1*A, 60))

gk    = gauKer(3)
xb    = 0.5*(bns[1:] + bns[0:-1])
fcnts = _N.convolve(cnts, gk, mode="same")
iMax  = int(_N.mean(_N.where(fcnts == _N.max(fcnts))[0]))
"""
