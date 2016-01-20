import scipy.io as _sio
import pickle
import ExperimentalMarkContainer as EMC
import filter as _flt


#  The animal tends to spend much of its time in arms 1, 3, 5
#  At least for well-trained animals, animals also do not turn around in 
#  arms 1, 3, 5 very much.  We use

day    = 4

anim1 = "bon"
anim2 = "bond"
anim3 = "Bon"
"""
anim1 = "Cha"
anim2 = "bon"
anim3 = "bond"
"""

#  experimental data mark, position container

rip = _sio.loadmat("../SpkSrtd/%(s2)s_data_day%(dy)d/%(s3)s/%(s1)sripplescons0%(dy)d.mat" % {"s1" : anim1, "s2" : anim2, "s3" : anim3, "dy" : day})

#  these are in seconds
strt = rip["ripplescons"][0, ex][0, ep][0, 0]["starttime"][0,0]
endt = rip["ripplescons"][0, ex][0, ep][0, 0]["endtime"][0,0]

mLp = _sio.loadmat("../SpkSrtd/%(s2)s_data_day%(dy)d/%(s3)s/%(s1)slinpos0%(dy)d.mat" % {"s1" : anim1, "s2" : anim2, "s3" : anim3, "dy" : day})
mRp = _sio.loadmat("../SpkSrtd/%(s2)s_data_day%(dy)d/%(s3)s/%(s1)srawpos0%(dy)d.mat" % {"s1" : anim1, "s2" : anim2, "s3" : anim3, "dy" : day})

ex=4-1; ep=4-1;

#  episodes 2, 4, 6


#  seg 1->2->3
#  seg 1->4->5
#%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%            Linearization           %%%% 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pts=mLp["linpos"][0,ex][0,ep]["statematrix"][0][0]["time"][0,0].T[0]
a = mLp["linpos"][0,ex][0,ep]["statematrix"][0,0]["segmentIndex"]
r = mRp["rawpos"][0,ex][0,ep]["data"][0,0]

nzinds = _N.where(r[:, 1] > 0)[0]
scxAmp = _N.max(r[nzinds, 1]) - _N.min(r[nzinds, 1])
scxMin = _N.min(r[nzinds, 1]) - scxAmp*0.05
scxMax = _N.max(r[nzinds, 1]) + scxAmp*0.05
scyAmp = _N.max(r[nzinds, 2]) - _N.min(r[nzinds, 2])
scyMin = _N.min(r[nzinds, 2]) - scyAmp*0.05
scyMax = _N.max(r[nzinds, 2]) + scyAmp*0.05

seg_ts = a[0,0].T[0]
seg1 = _N.where(seg_ts == 1)
seg2 = _N.where(seg_ts == 2)
seg3 = _N.where(seg_ts == 3)
seg4 = _N.where(seg_ts == 4)
seg5 = _N.where(seg_ts == 5)

time=mLp["linpos"][0,ex][0,ep]["statematrix"][0][0]["time"][0,0].T[0]
lindist=mLp["linpos"][0,ex][0,ep]["statematrix"][0][0]["lindist"][0,0].T[0]

# _plt.plot(time[seg1[0]], lindist[seg1[0]], ls="", marker=".")
# _plt.plot(time[seg4[0]], lindist[seg4[0]], ls="", marker=".")
# _plt.plot(time[seg5[0]], lindist[seg5[0]], ls="", marker=".")
# _plt.plot(time[seg2[0]], -1*lindist[seg2[0]], ls="", marker=".")
# _plt.plot(time[seg3[0]], -1*lindist[seg3[0]], ls="", marker=".")


#  1 3 5     we're there long enough

#  t135  times when I am in arm 1 3 or 5
t135 = _N.where((seg_ts == 1) | (seg_ts == 3) | (seg_ts == 5))[0]
dt135= _N.diff(t135)
tFilled  = _N.ones(seg_ts.shape[0], dtype=_N.int8) * -1
tsFilled = _N.ones(seg_ts.shape[0], dtype=_N.int8) * -1   #  segment + direction

#  dt135 
#  if armChg[0] == 100, that means t135[101] - t135[100] > 1
#  in our case, armChg[0] = 2511    t135[2512] - t135[2511]
#  t135[armChg[i]+1] - t135[armChg[i]]

#  now we're interested in long stretches where dt135 == 1 
l_intv_cnt1s = []   #  continuous intervals where dt135 == 1
bIn        = dt135[0] == 1
if bIn:
    l_intv_cnt1s.append([0, -1])   #  dt135[0] means t135
for i in xrange(1, len(dt135)):
    if (dt135[i] != 1) and (bIn):
        l_intv_cnt1s[-1][1] = i  
        bIn = False   #  closing
    elif (dt135[i] == 1) and (not bIn):
        l_intv_cnt1s.append([i, -1])   #  dt135[0] means t135
        bIn = True

if bIn:     #  long stretch of 1,3 or 5 till end.
    l_intv_cnt1s[-1][1] = len(dt135)

"""
print len(l_intv_cnt1s)
#  now look at the 

for i in xrange(len(l_intv_cnt1s)-1, -1, -1):
    t0 = l_intv_cnt1s[i][0];    t1 = l_intv_cnt1s[i][1]
    if (_N.abs(lindist[t1] - lindist[t0] < 5)) and (t1 - t0 < 100):
        l_intv_cnt1s.pop(i)
        print "popped"
print len(l_intv_cnt1s)
"""

intv_cnt1s = _N.array(l_intv_cnt1s)

real_arm_events  = []

fig = _plt.figure()
for i in xrange(intv_cnt1s.shape[0]):
    if (intv_cnt1s[i, 1]) - (intv_cnt1s[i, 0]) > 100:
        real_arm_events.append(i)
        t0 = t135[intv_cnt1s[i, 0]]
        t1 = t135[intv_cnt1s[i, 1]]+1   #  to get the end correct
        _plt.plot(range(t0, t1), seg_ts[t0:t1], color="black", lw=4)
        tFilled[t0:t1] = seg_ts[t0:t1]

"""
   run hard-coded file.  Outside of "real" arm events.  
"""

hfn = "exceptions-%(ex)d-%(ep)d.py" % {"ex" : (ex+1), "ep" : (ep+1)}

if os.access(hfn, os.F_OK):
    exf(hfn)

x = 1; y = 2

#  now look at what happens between the stable time in the arm.  
for ri in xrange(1, len(real_arm_events)):
    i = real_arm_events[ri]
    ip = real_arm_events[ri-1]   #  previous
    fig = _plt.figure(figsize=(9, 4))
    t0 = t135[intv_cnt1s[ip, 1]]
    t1 = t135[intv_cnt1s[i,   0]]
    if (seg_ts[t0-1] == 1) and (seg_ts[t1+1] == 1):
        tFilled[t0:t1] = 1
    elif (seg_ts[t0-1] == 1) and (seg_ts[t1+1] == 3):
        tFilled[t0:t1] = 2
    elif (seg_ts[t0-1] == 1) and (seg_ts[t1+1] == 5):
        tFilled[t0:t1] = 4
    elif (seg_ts[t0-1] == 3) and (seg_ts[t1+1] == 3):
        tFilled[t0:t1] = 3
    elif (seg_ts[t0-1] == 3) and (seg_ts[t1+1] == 1):
        tFilled[t0:t1] = 2
    elif (seg_ts[t0-1] == 3) and (seg_ts[t1+1] == 5):
        tFilled[t0:t1] = seg_ts[t0:t1]
        ##  assumption in going 3->5  went 2 -> 4 without returning to 
        is3 = _N.where(tFilled[t0:t1] == 3)[0]
        tFilled[t0+is3] = 2
        is5 = _N.where(tFilled[t0:t1] == 5)[0]
        tFilled[t0+is5] = 4
    elif (seg_ts[t0-1] == 5) and (seg_ts[t1+1] == 5):
        tFilled[t0:t1] = 3
    elif (seg_ts[t0-1] == 5) and (seg_ts[t1+1] == 1):
        tFilled[t0:t1] = 4
    elif (seg_ts[t0-1] == 5) and (seg_ts[t1+1] == 3):
        tFilled[t0:t1] = 4

    fig.add_subplot(2, 1, 1)

    _plt.plot(range(t0-20, t1+20), seg_ts[t0-20:t1+20], color="black", lw=4)
    _plt.ylim(0.5, 5.5)

    fig.add_subplot(2, 1, 2)
    _plt.scatter(r[:, x], r[:, y], s=5, color="grey")
    _plt.scatter(r[t0:t1, x], r[t0:t1, y], s=9, color="black")
    _plt.plot(r[t0, x], r[t0, y], ms=20, color="blue", marker=".")
    _plt.plot(r[t1, x], r[t1, y], ms=20, color="red", marker=".")
    _plt.xlim(10, 150)
    _plt.ylim(50, 180)
    _plt.savefig("btwn_%(ep)d_%(i)d" % {"ep" : (ep+1), "i" : i})
    _plt.close()


_plt.ylim(0.5, 5.5)
_plt.plot(seg_ts, lw=2, color="red")


segN_mm = _N.empty((5, 2))
seg1 = _N.where(tFilled == 1)[0]
segN_mm[0] = _N.array([_N.min(lindist[seg1]), _N.max(lindist[seg1])])
seg2 = _N.where(tFilled == 2)[0]
segN_mm[1] = _N.array([_N.min(lindist[seg2]), _N.max(lindist[seg2])])
seg3 = _N.where(tFilled == 3)[0]
segN_mm[2] = _N.array([_N.min(lindist[seg3]), _N.max(lindist[seg3])])
seg4 = _N.where(tFilled == 4)[0]
segN_mm[3] = _N.array([_N.min(lindist[seg4]), _N.max(lindist[seg4])])
seg5 = _N.where(tFilled == 5)[0]
segN_mm[4] = _N.array([_N.min(lindist[seg5]), _N.max(lindist[seg5])])

trans =  _N.array([-1] + (_N.where(_N.diff(tFilled) != 0)[0]).tolist())+1

#  1, 2, 3, 4, 5     
#  (seg)0, (seg)1       10, 11    20, 21   30, 31   40, 41   50, 51


#  10  outbound  11  inbound
gk = _flt.gauKer(20)
gk /= _N.sum(gk)
#  first, break it into inbound, outbound.  
flindist = _N.convolve(lindist, gk, mode="same")

for isg in xrange(len(trans)-1):
    tSeg0 = trans[isg]    #  time when we moved into this segment
    tSeg1 = trans[isg+1]    #  time when we moved into this segment
    iseg = tFilled[tSeg0]-1  #
    outb = tFilled[tSeg0]*10
    inb  = outb + 1
    av   = _N.mean(segN_mm[iseg])
    if flindist[tSeg0] < av and flindist[tSeg1] > av:   #OUTBOUND
        tsFilled[tSeg0:tSeg1] = outb
    elif flindist[tSeg0] > av and flindist[tSeg1] < av:   #INBOUND
        tsFilled[tSeg0:tSeg1] = inb
    elif flindist[tSeg0] > av and flindist[tSeg1] > av:  #INBOUND->OUTBOUND
        minInd = _N.where(flindist[tSeg0:tSeg1] == _N.min(flindist[tSeg0:tSeg1]))[0][0]
        tsFilled[tSeg0:tSeg0+minInd]   = inb
        tsFilled[tSeg0+minInd:tSeg1] = outb
    elif flindist[tSeg0] < av and flindist[tSeg1] < av:   #OUTBOUND->INBOUND
        maxInd = _N.where(flindist[trans[isg]:trans[isg+1]] == _N.max(flindist[trans[isg]:trans[isg+1]]))[0][0]
        tsFilled[tSeg0:tSeg0+maxInd]   = outb
        tsFilled[tSeg0+maxInd:tSeg1] = inb

if lindist[tSeg1] > lindist[-1]:   #  far -> near   INBOUND
    tsFilled[tSeg1:] = tFilled[tSeg1:]*10 + 1
elif lindist[tSeg1] < lindist[-1]:
    tsFilled[tSeg1:] = tFilled[tSeg1:]*10  #  far -> near   OUTBOUND

tsrans =  _N.array([-1] + (_N.where(_N.diff(tsFilled) != 0)[0]).tolist())+1

ipg   = -1
iPPG  = 8
fig = _plt.figure(figsize=(13, 8))
pg    = 1
for ts in xrange(len(tsrans)-1):
    ipg += 1
    if (ipg % iPPG == 0) and (ipg != 0):
        if ts > 0:
            _plt.savefig("pieces_%(ep)d_%(pg)d" % {"ep" : (ep+1), "pg" : pg})
            _plt.close()
            pg += 1
        fig = _plt.figure(figsize=(13, 8))
        ipg = 0
    fig.add_subplot(4, 4, 2*ipg + 1)
    t0 = tsrans[ts];     t1 = tsrans[ts+1]
    _plt.plot(lindist[t0:t1])
    fig.add_subplot(4, 4, 2*ipg + 2)
    _plt.scatter(r[::40, 1], r[::40, 2], s=5, color="grey")
    _plt.scatter(r[t0:t1, 1], r[tsrans[ts]:tsrans[ts+1], 2], s=9, color="black")
    _plt.plot(r[t0, 1], r[t0, 2], ms=20, color="blue", marker=".")
    _plt.plot(r[t1, 1], r[t1, 2], ms=20, color="red", marker=".")
    _plt.xlim(scxMin, scxMax)
    _plt.ylim(scyMin, scyMax)

_plt.savefig("pieces_%(ep)d_%(pg)d" % {"ep" : (ep+1), "pg" : pg})
_plt.close()

oneDdist=_N.empty(len(lindist))

for isg in xrange(len(tsrans)-1):
    t0 = tsrans[isg]
    t1 = tsrans[isg+1]
    if tsFilled[t0] == 10:  #CENTER ARM - OUTBOUND
        if tsFilled[t1] == 20:      #  0 -> 1
            oneDdist[t0:t1] = (lindist[t0:t1] - min(lindist[seg1]))/(max(lindist[seg1]) - min(lindist[seg1]))
        elif tsFilled[t1] == 40:    #  0 -> -1
            oneDdist[t0:t1] = -(lindist[t0:t1] - min(lindist[seg1]))/(max(lindist[seg1]) - min(lindist[seg1]))
    elif tsFilled[t0] == 20:  #OUTBOUND segment 2
        oneDdist[t0:t1] = 1 + (lindist[t0:t1] - min(lindist[seg2]))/(max(lindist[seg2]) - min(lindist[seg2]))
    elif tsFilled[t0] == 30:  #OUTBOUND segment 3
        oneDdist[t0:t1] = 2 + (lindist[t0:t1] - min(lindist[seg3]))/(max(lindist[seg3]) - min(lindist[seg3]))
    elif tsFilled[t0] == 40:  #OUTBOUND segment 4
        oneDdist[t0:t1] = -(1 + (lindist[t0:t1] - min(lindist[seg4]))/(max(lindist[seg4]) - min(lindist[seg4])))
    elif tsFilled[t0] == 50:  #OUTBOUND segment 5
        oneDdist[t0:t1] = -(2 + (lindist[t0:t1] - min(lindist[seg5]))/(max(lindist[seg5]) - min(lindist[seg5])))


    elif tsFilled[t0] == 11:  #CENTER ARM - INBOUND
        tp0 = tsrans[isg-1]
        if (tsFilled[tp0] == 20) or (tsFilled[tp0] == 21):  #  1 -> 2
            oneDdist[t0:t1] = 6 - (lindist[t0:t1] - min(lindist[seg1]))/(max(lindist[seg1]) - min(lindist[seg1]))
        if (tsFilled[tp0] == 40) or (tsFilled[tp0] == 41):  #  -1 -> 0
            oneDdist[t0:t1] = -6 + (lindist[t0:t1] - min(lindist[seg1]))/(max(lindist[seg1]) - min(lindist[seg1]))
    elif tsFilled[t0] == 21:  #CENTER ARM - INBOUND   #  4->5
        oneDdist[t0:t1] = 5 - (lindist[t0:t1] - min(lindist[seg2]))/(max(lindist[seg2]) - min(lindist[seg2]))
    elif tsFilled[t0] == 31:  #CENTER ARM - INBOUND   #  3->4
        oneDdist[t0:t1] = 4 - (lindist[t0:t1] - min(lindist[seg3]))/(max(lindist[seg3]) - min(lindist[seg3]))
    elif tsFilled[t0] == 41:  #CENTER ARM - INBOUND   #  -4->-5
        oneDdist[t0:t1] = -5 + (lindist[t0:t1] - min(lindist[seg4]))/(max(lindist[seg4]) - min(lindist[seg4]))
    elif tsFilled[t0] == 51:  #CENTER ARM - INBOUND   #  -3->-4
        oneDdist[t0:t1] = -4 + (lindist[t0:t1] - min(lindist[seg5]))/(max(lindist[seg5]) - min(lindist[seg5]))

    else:
        print "no option here  t0 %(0)d  t1 %(1)d" % {"0" : t0, "1" : t1}

isg = len(tsrans)-1
t0 = tsrans[isg]
if tsFilled[t0] == 10:  #CENTER ARM - OUTBOUND
    #  we need to look ahead in to future.  Discard, since no more data
    print "Don't know what we should do"
    t1 = t0
    pass
else:
    t1 = len(lindist)

    if tsFilled[t0] == 20:  #OUTBOUND segment 2
        oneDdist[t0:t1] = 1 + (lindist[t0:t1] - min(lindist[seg2]))/(max(lindist[seg2]) - min(lindist[seg2]))
    elif tsFilled[t0] == 30:  #OUTBOUND segment 3
        oneDdist[t0:t1] = 2 + (lindist[t0:t1] - min(lindist[seg3]))/(max(lindist[seg3]) - min(lindist[seg3]))
    elif tsFilled[t0] == 40:  #OUTBOUND segment 4
        oneDdist[t0:t1] = -(1 + (lindist[t0:t1] - min(lindist[seg4]))/(max(lindist[seg4]) - min(lindist[seg4])))
    elif tsFilled[t0] == 50:  #OUTBOUND segment 5
        oneDdist[t0:t1] = -(2 + (lindist[t0:t1] - min(lindist[seg5]))/(max(lindist[seg5]) - min(lindist[seg5])))


    if tsFilled[t0] == 11:  #CENTER ARM - INBOUND
        tp0 = tsrans[isg-1]
        if (tsFilled[tp0] == 20) or (tsFilled[tp0] == 21):  #  1 -> 2
            oneDdist[t0:t1] = 6 - (lindist[t0:t1] - min(lindist[seg1]))/(max(lindist[seg1]) - min(lindist[seg1]))
        if (tsFilled[tp0] == 40) or (tsFilled[tp0] == 41):  #  -1 -> 0
            oneDdist[t0:t1] = -6 + (lindist[t0:t1] - min(lindist[seg1]))/(max(lindist[seg1]) - min(lindist[seg1]))
    elif tsFilled[t0] == 21:  #CENTER ARM - INBOUND   #  4->5
        oneDdist[t0:t1] = 5 - (lindist[t0:t1] - min(lindist[seg2]))/(max(lindist[seg2]) - min(lindist[seg2]))
    elif tsFilled[t0] == 31:  #CENTER ARM - INBOUND   #  3->4
        oneDdist[t0:t1] = 4 - (lindist[t0:t1] - min(lindist[seg3]))/(max(lindist[seg3]) - min(lindist[seg3]))
    elif tsFilled[t0] == 41:  #CENTER ARM - INBOUND   #  -4->-5
        oneDdist[t0:t1] = -5 + (lindist[t0:t1] - min(lindist[seg4]))/(max(lindist[seg4]) - min(lindist[seg4]))
    elif tsFilled[t0] == 51:  #CENTER ARM - INBOUND   #  -3->-4
        oneDdist[t0:t1] = -4 + (lindist[t0:t1] - min(lindist[seg5]))/(max(lindist[seg5]) - min(lindist[seg5]))





#########



#  t_champs[-1, 0] - t_champs[0, 0]       neural recording length, 2 10kHz
#  svecT[0]...svecT[-1]                   sorted time of behavioral observation

svecT  = pts
svecL0 = oneDdist
spd   = _N.diff(svecL0)
pj    = _N.where(spd > 4)[0]   #  -6-->0
nj    = _N.where(spd < -4)[0]  #   6-->0
spd[pj] = 0.05
spd[nj] = -0.05





gk    = _flt.gauKer(10)  #  kernel of about 300ms.
gk    /= _N.sum(gk)

kspd  = _N.convolve(spd, gk, mode="same")
fastTs = _N.where(_N.abs(kspd) >  0.003)[0]

svecT[fastTs]

crgns  = []
iStart = 0
bInMvt = False


for c in xrange(0, len(fastTs)-1):
    if fastTs[c+1] - fastTs[c] == 1:
        if not bInMvt:
            iStart = fastTs[c]
            bInMvt = True
    else:
        if bInMvt:
            bInMvt = False
            iStop = fastTs[c]
            crgns.append([iStart, iStop])

fastMsk = []
for itv in xrange(len(crgns)-2, -1, -1):
    if crgns[itv+1][0] - crgns[itv][1] < 100:   # 100 ms. of stop is not a stop
        crgns[itv][1] = crgns[itv+1][1]
        crgns.pop(itv+1)

#  show detected moving states
#_plt.plot(svecT, svecL0, color="blue")
vrgns = _N.array(crgns)

for itv in xrange(vrgns.shape[0]):
    indxs = range(vrgns[itv,0], vrgns[itv,1])
    fastMsk.extend(indxs)
    #_plt.plot(svecT[indxs], svecL0[indxs], lw=4, color="black")
trgns = svecT[vrgns]     #  

#  we are going to use 1 ms bins
#t0 = int((svecT[0] - 0.5) * 1000)
#t1 = int((svecT[-1] + 0.5) * 1000)
t0 = int(svecT[0]  * 1000)
t1 = int(svecT[-1] * 1000)

k       =  1
pos     =  _N.empty(t1-t0)


svecT_ms = _N.linspace(t0, t1, t1-t0, endpoint=False)    #  
svecL0_ms = _N.interp(svecT_ms, svecT*1000, svecL0)

tetlist = ["01", "02", "03", "04", "05", "07", "08", "10", "11", "12", "13", "14", "17", "18", "19", "20", "21", "22", "23", "24", "25", "27", "28", "29"]

marks   =  _N.empty((t1-t0, len(tetlist)), dtype=list)

it      = -1
for tet in tetlist:
    it += 1
    A = _sio.loadmat("../SpkSrtd/bond_data_day4/bond04/%(tt)s/bond04-%(tt)s_params.mat" % {"tt" : tet})
    t_champs = A["filedata"][0,0]["params"][:, 0:5]  # time and amplitudes
    t_champs[:, 1:5] /= 50.

    #  tm/10000.    -->  seconds
    #  vecT  is sampled every 33.4 ms?     #  let's intrapolate this 

    #   we need  Nx, Nm, xA, mA, k, dt
    #   pos, marks

    ##  
    #  times -0.5 before + 0.5 seconds after last position


    #  svecT, svecL0    (time and position)  33Hz   (short period within expt.)
    #  tm  chXamp       (time and mark)     10kHz

    #  need to match start times
    y  = []
    #  t = 0 for the marks is 

    rngT = []
    rngX = []

    for imk in xrange(t_champs.shape[0]):  #  more marks than there is behavioral data
        now_s  = t_champs[imk, 0]/10000.    #  now_s
        now_ms = t_champs[imk, 0]/10.
        ind = int(now_ms - svecT[0]*1000)  # svecT[0] is start of data we use
        if (now_s > svecT[0]) and (now_s < svecT[-1]):
            #for nr in xrange(trgns.shape[0]):
            #    if (now_s >= trgns[nr, 0]) and (now_s <= trgns[nr, 1]):
            fd = _N.where((now_s >= svecT[0:-1]) & (now_s <= svecT[1:]))[0]
            y.append(t_champs[imk, 1])
            rngT.append(now_s)
            rngX.append(svecL0[fd[0]])
            marks[ind, it] = [t_champs[imk, 1:]]

x  = []
xt = []
minds = _N.empty((trgns.shape[0], 2), dtype=_N.int)

for nr in xrange(trgns.shape[0]):
    t0 = vrgns[nr, 0]   #  @ 33.4 Hz
    t1 = vrgns[nr, 1]  

    minds[nr] = int((svecT[t0] - svecT[0]) * 1000), int((svecT[t1] - svecT[0]) * 1000)
    t0 = minds[nr, 0]
    t1 = minds[nr, 1]

    #x.extend(svecL0_ms[t0:t1].tolist())   #  position
    #xt.extend(range(t0, t1))

    _plt.plot(_N.arange(t0, t1), svecL0_ms[t0:t1])

emc = EMC.ExperimentalMarkContainer()
emc.pos   = svecL0_ms   #  marks and position are not aligned
emc.marks = marks
emc.tetlist = tetlist
emc.minds = minds

emc.xA    = 6
emc.mA    = 8
emc.k     = 4
emc.dt    = 0.001
emc.Nx    = 50
emc.Nm    = 50

emc.save(ep+1)




