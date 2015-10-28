import scipy.io as _sio
import pickle
import ExperimentalMarkContainer as EMC
import filter as _flt

#  experimental data mark, position container

mLp = _sio.loadmat('../SpkSrtd/bonlinpos04.mat')
mRp = _sio.loadmat('../SpkSrtd/bonrawpos04.mat')

ex=4-1; ep=2-1;

#%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%            Linearization           %%%% 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a = mLp["linpos"][0,ex][0,ep]["statematrix"][0,0]["segmentIndex"]

seg1 = _N.where(a[0][0].T[0] == 1)[0]
seg2 = _N.where(a[0][0].T[0] == 2)[0]
seg3 = _N.where(a[0][0].T[0] == 3)[0]
seg4 = _N.where(a[0][0].T[0] == 4)[0]
seg5 = _N.where(a[0][0].T[0] == 5)[0]


rp = mRp["rawpos"][0, ex][0,ep][0,0][0]
rpx = rp[:, 1]
rpy = rp[:, 2]

time=mLp["linpos"][0,ex][0,ep]["statematrix"][0][0]["time"][0,0].T[0]
lindist=mLp["linpos"][0,ex][0,ep]["statematrix"][0][0]["lindist"][0,0].T[0]

"""
_plt.plot(time[seg1[0]], lindist[seg1[0]], ls="", marker=".")
_plt.plot(time[seg4[0]], lindist[seg4[0]], ls="", marker=".")
_plt.plot(time[seg5[0]], lindist[seg5[0]], ls="", marker=".")
_plt.plot(time[seg2[0]], -1*lindist[seg2[0]], ls="", marker=".")
_plt.plot(time[seg3[0]], -1*lindist[seg3[0]], ls="", marker=".")
"""

# % figure
# % plot(time(seg1),lindist(seg1),'b.');
# % hold on
# % plot(time(seg4),lindist(seg4),'k.');
# % plot(time(seg5),lindist(seg5),'c.');
# % plot(time(seg2),-lindist(seg2),'r.');
# % plot(time(seg3),-lindist(seg3),'m.');
# % hold off

posT1=time[seg1]
posL1=lindist[seg1]/max(lindist[seg1])   #  posL1 is between 0 and 1
posL1b=_N.empty(len(posL1))

##  
#for k=1:length(posL1)

lseg1  = []   #  0 -> -1   (outbound)
lseg1.append(_N.where((posT1 >= 2460) & (posT1 <=2545.681))[0])
lseg1.append(_N.where((posT1 >= 2744.2885) & (posT1 <=2765.212))[0])
lseg1.append(_N.where((posT1 >= 2836.1515) & (posT1 <=2849.932))[0])
lseg1.append(_N.where((posT1 >= 2951.1645) & (posT1  <=2965.08))[0])
lseg1.append(_N.where((posT1 >= 3077.7625) & (posT1  <=3091.307))[0])
lseg1.append(_N.where((posT1 >= 3240.9455) & (posT1 <=3249.698))[0])

lseg2  = []   #  -5 -> -6 (inbound)
lseg2.append(_N.where((posT1 >= 2616.645) & (posT1<=2649.484))[0])
lseg2.append(_N.where((posT1 >= 2816.852) & (posT1<=2836.1514))[0])
lseg2.append(_N.where((posT1 >= 3124.28)  & (posT1<=3139.592))[0])
lseg2.append(_N.where((posT1 >= 3297.737) & (posT1<=3315.681))[0])

lseg3  = []   #  5  ->  6 (inbound)
lseg3.append(_N.where((posT1 >= 2726.31)  & (posT1<=2744.2884))[0])
lseg3.append(_N.where((posT1 >= 2931.63)  & (posT1<=2951.1644))[0])
lseg3.append(_N.where((posT1 >= 3059.031) & (posT1<=3077.7624))[0])
lseg3.append(_N.where((posT1 >= 3225.24)  & (posT1<=3240.9454))[0])
lseg3.append(_N.where((posT1 >= 3385.306) & (posT1<=3405))[0])

lseg4 = []

"""
ots = seg1[lseg1[0]]
_plt.scatter(rpx[ots], rpy[ots], color="red")
ots = seg1[lseg2[0]]
_plt.scatter(rpx[ots], rpy[ots], color="blue")

ots = seg1[lseg1[1]]
_plt.scatter(rpx[ots], rpy[ots], color="red")
ots = seg1[lseg2[1]]
_plt.scatter(rpx[ots], rpy[ots], color="blue")
"""

#_plt.scatter(rpx[seg1[lseg2[1]]], rpy[seg1[lseg2[1]]], color="red")
#_plt.scatter(rpx[seg1[lseg2[2]]], rpy[seg1[lseg2[2]]], color="red")

for k in xrange(len(posL1)):
    if (posT1[k] >= 2460) and (posT1[k]<=2545.681):
        posL1b[k]=-posL1[k] #%0=-1
    elif posT1[k]>=2744.2885 and posT1[k]<=2765.212:
        posL1b[k]=-posL1[k] #%0=-1
    elif posT1[k]>=2836.1515  and posT1[k]<=2849.932:
        posL1b[k]=-posL1[k] #%0=-1
    elif posT1[k]>=2951.1645 and posT1[k]<=2965.08:
        posL1b[k]=-posL1[k] #%0=-1
    elif posT1[k]>=3077.7625 and posT1[k]<=3091.307:
        posL1b[k]=-posL1[k] #%0=-1
    elif posT1[k]>=3240.9455 and posT1[k]<=3249.698:
        posL1b[k]=-posL1[k] #%0=-1




    elif (posT1[k]>=2616.645) and (posT1[k]<=2649.484):
        posL1b[k]=-(posL1[k]+2*(3-posL1[k])) #%#-5=-6
    elif posT1[k]>=2726.31 and posT1[k]<=2744.2884:
        posL1b[k]=posL1[k]+2*(3-posL1[k]) #%5=6
    elif posT1[k]>=2816.852 and posT1[k]<=2836.1514:
        posL1b[k]=-(posL1[k]+2*(3-posL1[k])) #%-5=-6
    elif posT1[k]>=2931.63 and posT1[k]<=2951.1644:
        posL1b[k]=posL1[k]+2*(3-posL1[k]) #%5=6
    elif posT1[k]>=3059.031 and posT1[k]<=3077.7624:
        posL1b[k]=posL1[k]+2*(3-posL1[k]) #%5=6
    elif posT1[k]>=3124.28 and posT1[k]<=3139.592:
        posL1b[k]=-(posL1[k]+2*(3-posL1[k])) #%-5=-6
    elif posT1[k]>=3225.24 and posT1[k]<=3240.9454:
        posL1b[k]=posL1[k]+2*(3-posL1[k]) #%5=6
    elif posT1[k]>=3297.737 and posT1[k]<=3315.681:
        posL1b[k]=-(posL1[k]+2*(3-posL1[k])) #%-5=-6
    elif posT1[k]>=3385.306 and posT1[k]<=3405:
        posL1b[k]=posL1[k]+2*(3-posL1[k]) #%5=6



    else:   #  outbound 0->1
        posL1b[k]=posL1[k]

# #%0-ACE
# posT4=time[seg4]
# posL4=1+(lindist[seg4] - min(lindist[seg4]))/(max(lindist[seg4]) - min(lindist[seg4]))
# posL4b=_N.empty(len(posL4))

# for k in xrange(len(posL4)):
#     if posT4[k]>=2707.462 and posT4[k]<=2765.212:
#         posL4b[k]=posL4[k]+2*(3-posL4[k]) #%3=6
#     elif posT4[k]>=2908.879 and posT4[k]<=2965.08:
#         posL4b[k]=posL4[k]+2*(3-posL4[k])
#     elif posT4[k]>=3040.688 and posT4[k]<=3091.307:
#         posL4b[k]=posL4[k]+2*(3-posL4[k])
#     elif posT4[k]>=3193.29 and posT4[k]<=3249.698:
#         posL4b[k]=posL4[k]+2*(3-posL4[k])
#     elif posT4[k]>=3347.402 and posT4[k]<=3405:
#         posL4b[k]=posL4[k]+2*(3-posL4[k])
#     else:
#         posL4b[k]=posL4[k]

# posT5=time[seg5]
# posL5=2+(lindist[seg5] - min(lindist[seg5]))/(max(lindist[seg5]) - min(lindist[seg5]))
# posL5b=_N.empty(len(posL5))

# for k in xrange(len(posL5)):
#     if posT5[k]>=2707.462 and posT5[k]<=2765.212:
#         posL5b[k]=posL5[k]+2*(3-posL5[k])
#     elif posT5[k]>=2908.879 and posT5[k]<=2965.08:
#         posL5b[k]=posL5[k]+2*(3-posL5[k])
#     elif posT5[k]>=3040.688 and posT5[k]<=3091.307:
#         posL5b[k]=posL5[k]+2*(3-posL5[k])
#     elif posT5[k]>=3193.29 and posT5[k]<=3249.698:
#         posL5b[k]=posL5[k]+2*(3-posL5[k])
#     elif posT5[k]>=3347.402 and posT5[k]<=3405:
#         posL5b[k]=posL5[k]+2*(3-posL5[k])
#     else:
#         posL5b[k]=posL5[k]

# posT2=time[seg2]
# posL2=1+(lindist[seg2] - min(lindist[seg2]))/(max(lindist[seg2]) - min(lindist[seg2]))
# posL2b=_N.empty(len(posL2))

# for k in xrange(len(posL2)):
#     if posT2[k]>=2580.971 and posT2[k]<=2680.835:
#         posL2b[k]=-(posL2[k]+2*(3-posL2[k])) #%-3=-6
#     elif posT2[k]>=2790.035 and posT2[k]<=2849.932:
#         posL2b[k]=-(posL2[k]+2*(3-posL2[k]))
#     elif posT2[k]>=2858.031 and posT2[k]<=2888.331:
#         posL2b[k]=-(posL2[k]+2*(3-posL2[k]))
#     elif posT2[k]>=2991.458 and posT2[k]<=3025.768:
#         posL2b[k]=-(posL2[k]+2*(3-posL2[k]))
#     elif posT2[k]>=3112.713 and posT2[k]<=3152.636:
#         posL2b[k]=-(posL2[k]+2*(3-posL2[k]))
#     elif posT2[k]>=3274.19 and posT2[k]<=3324.589:
#         posL2b[k]=-(posL2[k]+2*(3-posL2[k]))
#     else:
#         posL2b[k]=-posL2[k]
    

# posT3=time[seg3]
# posL3=2+(lindist[seg3] - min(lindist[seg3]))/(max(lindist[seg3]) - min(lindist[seg3]))
# posL3b=_N.empty(len(posL3))

# #posT3=time(seg3);
# #posL3=2+(lindist(seg3)-min(lindist(seg3)))/(max(lindist(seg3))-min(lindist(seg3)));
# for k in xrange(len(posL3)):
#     if posT3[k]>=2580.971 and posT3[k]<=2680.835:
#         posL3b[k]=-(posL3[k]+2*(3-posL3[k]))
#     elif posT3[k]>=2790.035 and posT3[k]<=2849.932:
#         posL3b[k]=-(posL3[k]+2*(3-posL3[k]))
#     elif posT3[k]>=2858.031 and posT3[k]<=2888.331:
#         posL3b[k]=-(posL3[k]+2*(3-posL3[k]))
#     elif posT3[k]>=2991.458 and posT3[k]<=3025.768:
#         posL3b[k]=-(posL3[k]+2*(3-posL3[k]))
#     elif posT3[k]>=3112.713 and posT3[k]<=3152.636:
#         posL3b[k]=-(posL3[k]+2*(3-posL3[k]))
#     elif posT3[k]>=3274.19 and posT3[k]<=3324.589:
#         posL3b[k]=-(posL3[k]+2*(3-posL3[k]))
#     else:
#         posL3b[k]=-posL3[k]

# vecL0=_N.array(posL1b.tolist() + posL4b.tolist() + posL5b.tolist() + posL2b.tolist() +  posL3b.tolist())
# vecT=_N.array(posT1.tolist() + posT4.tolist() + posT5.tolist() + posT2.tolist() + posT3.tolist())
# #  our neural recordings are over a longer period of time than our behavior

# for i in xrange(len(vecL0)):
#     if vecT[i]>=2540 and vecT[i]<=2550 and vecL0[i]>0.5:
#         vecL0[i]=-vecL0[i]
#     elif vecT[i]>=2610 and vecT[i]<=2620 and vecL0[i]>0.5:
#         vecL0[i]=-6+vecL0[i];
#     elif vecT[i]>=2670 and vecT[i]<=2690 and vecL0[i]<-0.5 and vecL0[i]>-2:
#         vecL0[i]=-vecL0[i]
#     elif vecT[i]>=2670 and vecT[i]<=2690 and vecL0[i]<-4 and vecL0[i]>-6:
#         vecL0[i]=6+vecL0[i]
#     elif vecT[i]>=2720 and vecT[i]<=2730 and vecL0[i]>0 and vecL0[i]<2:
#         vecL0[i]=6-vecL0[i]
#     elif vecT[i]>=2720 and vecT[i]<=2730 and vecL0[i]>-2 and vecL0[i]<0:
#         vecL0[i]=6+vecL0[i]
#     elif vecT[i]>=2760 and vecT[i]<=2770 and vecL0[i]>0.5:
#         vecL0[i]=-vecL0[i]
#     elif vecT[i]>=2810 and vecT[i]<=2820 and vecL0[i]>0.5:
#         vecL0[i]=-6+vecL0[i]
#     elif vecT[i]>=2845 and vecT[i]<=2855 and vecL0[i]>0.5 and vecL0[i]<2:
#         vecL0[i]=-vecL0[i]
#     elif vecT[i]>=2845 and vecT[i]<=2855 and vecL0[i]<-4 and vecL0[i]>-6:
#         vecL0[i]=-6-vecL0[i]    
#     elif vecT[i]>=2925 and vecT[i]<=2935 and vecL0[i]<-0.5:
#         vecL0[i]=6+vecL0[i]
#     elif vecT[i]>=2960 and vecT[i]<=2970 and vecL0[i]>0.5:
#         vecL0[i]=-vecL0[i]
#     elif vecT[i]>=3055 and vecT[i]<=3065 and vecL0[i]<-0.5:
#         vecL0[i]=6+vecL0[i]
#     elif vecT[i]>=3090 and vecT[i]<=3100 and vecL0[i]>0.5:
#         vecL0[i]=-vecL0[i]
#     elif vecT[i]>=3120 and vecT[i]<=3130 and vecL0[i]>0.5:
#         vecL0[i]=-6+vecL0[i]
#     elif vecT[i]>=3145 and vecT[i]<=3155 and vecL0[i]<-0.5 and vecL0[i]>-2:
#         vecL0[i]=-vecL0[i]
#     elif vecT[i]>=3145 and vecT[i]<=3155 and vecL0[i]<-4 and vecL0[i]>-6:
#         vecL0[i]=6+vecL0[i]
#     elif vecT[i]>=3220 and vecT[i]<=3230 and vecL0[i]<-0.5:
#         vecL0[i]=6+vecL0[i]
#     elif vecT[i]>=3245 and vecT[i]<=3255 and vecL0[i]<2 and vecL0[i]>0.5:
#         vecL0[i]=-vecL0[i]
#     elif vecT[i]>=3245 and vecT[i]<=3255 and vecL0[i]<6 and vecL0[i]>4:
#         vecL0[i]=-6+vecL0[i]
#     elif vecT[i]>=3290 and vecT[i]<=3300 and vecL0[i]>0.5:
#         vecL0[i]=-6+vecL0[i]
#     elif vecT[i]>=3320 and vecT[i]<=3330 and vecL0[i]<-0.5 and vecL0[i]>-2:
#         vecL0[i]=-vecL0[i]
#     elif vecT[i]>=3320 and vecT[i]<=3330 and vecL0[i]<-4 and vecL0[i]>-6:
#         vecL0[i]=6+vecL0[i]
#     elif vecT[i]>=3380 and vecT[i]<=3390 and vecL0[i]<-0.5:
#         vecL0[i]=6+vecL0[i]

# """

# A = _scio.loadmat("bond_data_day4/bond04/01-149/bond04-01_params.mat")
# tm = A["filedata"][0, 0]["params"][:, 0]
# ch1amp = A["filedata"][0, 0]["params"][:, 1]
# ch2amp = A["filedata"][0, 0]["params"][:, 2]
# ch3amp = A["filedata"][0, 0]["params"][:, 3]
# ch4amp = A["filedata"][0, 0]["params"][:, 4]

# _plt.scatter(ch1amp[0:2000], ch3amp[0:2000])
# """

# sinds = [i[0] for i in sorted(enumerate(vecT), key=lambda x:x[1])]
# svecT  = vecT[sinds]   #  this creates a new instance 
# svecL0 = vecL0[sinds]    #  ordered in time
# spd   = _N.diff(svecL0)
# pj    = _N.where(spd > 4)[0]   #  -6-->0
# nj    = _N.where(spd < -4)[0]  #   6-->0
# spd[pj] = 0.05
# spd[nj] = -0.05

# gk    = _flt.gauKer(10)  #  kernel of about 300ms.
# gk    /= _N.sum(gk)

# kspd  = _N.convolve(spd, gk, mode="same")
# fastTs = _N.where(_N.abs(kspd) >  0.003)[0]
# #_plt.plot(svecT[fastTs], svecL0[fastTs], marker=".", ls="", ms=2)

# svecT[fastTs]

# crgns  = []
# iStart = 0
# bInMvt = False


# for c in xrange(0, len(fastTs)-1):
#     if fastTs[c+1] - fastTs[c] == 1:
#         if not bInMvt:
#             iStart = fastTs[c]
#             bInMvt = True
#     else:
#         if bInMvt:
#             bInMvt = False
#             iStop = fastTs[c]
#             crgns.append([iStart, iStop])

# fastMsk = []
# for itv in xrange(len(crgns)-2, -1, -1):
#     if crgns[itv+1][0] - crgns[itv][1] < 100:   # 100 ms. of stop is not a stop
#         crgns[itv][1] = crgns[itv+1][1]
#         crgns.pop(itv+1)


# #  show detected moving states
# _plt.plot(svecT, svecL0, color="blue")
# vrgns = _N.array(crgns)

# for itv in xrange(vrgns.shape[0]):
#     indxs = range(vrgns[itv,0], vrgns[itv,1])
#     fastMsk.extend(indxs)
#     #_plt.plot(svecT[indxs], svecL0[indxs], lw=4, color="black")
# trgns = svecT[vrgns]     #  

# #  we are going to use 1 ms bins
# #t0 = int((svecT[0] - 0.5) * 1000)
# #t1 = int((svecT[-1] + 0.5) * 1000)
# t0 = int(svecT[0]  * 1000)
# t1 = int(svecT[-1] * 1000)

# k       =  1
# pos     =  _N.empty(t1-t0)


# svecT_ms = _N.linspace(t0, t1, t1-t0, endpoint=False)    #  
# svecL0_ms = _N.interp(svecT_ms, svecT*1000, svecL0)

# tetlist = ["01", "02", "03", "04", "05", "07", "08", "10", "11", "12", "13", "14", "17", "18", "19", "20", "21", "22", "23", "24", "25", "27", "28", "29"]
# marks   =  _N.empty((t1-t0, len(tetlist)), dtype=list)

# it      = -1
# for tet in tetlist:
#     it += 1
#     A = _sio.loadmat("../SpkSrtd/bond_data_day4/bond04/%(tt)s/bond04-%(tt)s_params.mat" % {"tt" : tet})
#     t_champs = A["filedata"][0,0]["params"][:, 0:5]  # time and amplitudes
#     t_champs[:, 1:5] /= 50.

#     #  tm/10000.    -->  seconds
#     #  vecT  is sampled every 33.4 ms?     #  let's intrapolate this 

#     #   we need  Nx, Nm, xA, mA, k, dt
#     #   pos, marks

#     ##  
#     #  times -0.5 before + 0.5 seconds after last position


#     #  svecT, svecL0    (time and position)  33Hz   (short period within expt.)
#     #  tm  chXamp       (time and mark)     10kHz

#     #  need to match start times
#     x  = []
#     xt = []
#     y  = []
#     #  t = 0 for the marks is 

#     rngT = []
#     rngX = []

#     for imk in xrange(t_champs.shape[0]):  #  more marks than there is behavioral data
#         now_s  = t_champs[imk, 0]/10000.
#         now_ms = t_champs[imk, 0]/10.
#         #ind = int(500 + (now_ms - svecT[0]*1000))
#         ind = int(now_ms - svecT[0]*1000)
#         if (now_s > svecT[0]) and (now_s < svecT[-1]):
#             for nr in xrange(trgns.shape[0]):
#                 if (now_s >= trgns[nr, 0]) and (now_s <= trgns[nr, 1]):
#                     fd = _N.where((now_s >= svecT[0:-1]) & (now_s <= svecT[1:]))[0]
#                     x.append(svecL0[fd[0]])   #  position
#                     xt.append(ind)
#                     y.append(t_champs[imk, 1])
#                     rngT.append(now_s)
#                     rngX.append(svecL0[fd[0]])
#                     marks[ind, it] = [t_champs[imk, 1:]]
        
#     fig = _plt.figure()
#     _plt.scatter(x, y, s=2)
#     _plt.suptitle("tet  %s" % tet)
#     _plt.savefig("tet%s" % tet)
#     _plt.close()


# emc = EMC.ExperimentalMarkContainer()
# emc.pos   = svecL0_ms   #  marks and position are not aligned
# emc.mvpos = x
# emc.mvpos_t = xt
# emc.mvpost = x
# emc.marks = marks
# emc.tetlist = tetlist

# emc.xA    = 6
# emc.mA    = 8
# emc.k     = 4
# emc.dt    = 0.001
# emc.Nx    = 50
# emc.Nm    = 50

# emc.save()

