from filter import gauKer
def get_boundaries(rawpos):
    """
    from rawpos which aren't occluded, find the corners of range of motion.
    """
    nzinds = _N.where(rawpos[:, 1] > 0)[0]   #  not occluded data
    scxAmp = _N.max(rawpos[nzinds, 1]) - _N.min(rawpos[nzinds, 1])
    scxMin = _N.min(rawpos[nzinds, 1]) - scxAmp*0.05
    scxMax = _N.max(rawpos[nzinds, 1]) + scxAmp*0.05
    scyAmp = _N.max(rawpos[nzinds, 2]) - _N.min(rawpos[nzinds, 2])
    scyMin = _N.min(rawpos[nzinds, 2]) - scyAmp*0.05
    scyMax = _N.max(rawpos[nzinds, 2]) + scyAmp*0.05

    return scxMin, scxMax, scyMin, scyMax, nzinds

def continuous(seg_ts, r, time, lindist):
    """
    segments where we're continuously in one of the 3 arms, L, C, R

    Returns
    t135      times when in arms 1, 3 or 5.  t135[i] = 3414, (!= i).  t135 not linear
    sum(t135) == total time spent in ams 1, 3 and 5

    t135 = 0 1 2 3 4 5 6 7  30 31 32
    dt135=  1 1 1 1 1 1 1 23  1  1
    t135[where dt135 > 1]  == 7
    """
    seg1 = _N.where(seg_ts == 1)
    seg2 = _N.where(seg_ts == 2)
    seg3 = _N.where(seg_ts == 3)
    seg4 = _N.where(seg_ts == 4)
    seg5 = _N.where(seg_ts == 5)

    ###  quick show
    zrrp = _N.where((r[:, 1] > 0) & (r[:, 2] > 0) & (r[:, 3] > 0) & (r[:, 4] > 0))[0]
    # _plt.scatter(r[zrrp, 1], r[zrrp, 2], color="black")
    # _plt.savefig(_edd.resFN("rawpos.png", dir="%(a)s%(sd)s0%(e)d" % {"a" : anim2, "sd" : sdy, "e" : ep+1}, create=True))
    # _plt.close()
    # _plt.plot(lindist, color="black")
    # _plt.savefig(_edd.resFN("lindist.png", dir="%(a)s%(sd)s0%(e)d" % {"a" : anim2, "sd" : sdy, "e" : ep+1}))
    # _plt.close()

    #  1 3 5     we're there long enough

    #  t135  times when I am in arm 1 3 or 5
    #  t135 looks like line y=x with occassional jumps.
    t135 = _N.where((seg_ts == 1) | (seg_ts == 3) | (seg_ts == 5))[0]
    dt135= _N.diff(t135)
    tFilled  = _N.ones(seg_ts.shape[0], dtype=_N.int8) * -1
    #  segment + direction
    tsFilled = _N.ones(seg_ts.shape[0], dtype=_N.int8) * -1   

    #  dt135 
    #  if armChg[0] == 100, that means t135[101] - t135[100] > 1
    #  in our case, armChg[0] = 2511    t135[2512] - t135[2511]
    #  t135[armChg[i]+1] - t135[armChg[i]]

    #  now we're interested in long stretches where dt135 == 1 
    l_intv_cnt1s = []   #  continuous intervals where dt135 == 1
    bIn        = (dt135[0] == 1)
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
s            l_intv_cnt1s.pop(i)
            print "popped"
    print len(l_intv_cnt1s)
    """

    intv_cnt1s = _N.array(l_intv_cnt1s)

    return t135, intv_cnt1s, tFilled, tsFilled, seg1, seg2, seg3, seg4, seg5


def find_actual_arm_events(intv_cnt1s, t135, tFilled):
    """
    when between arms  (seg 4 and 5)
    """

    real_arm_events  = []

    #fig = _plt.figure()
    for i in xrange(intv_cnt1s.shape[0]):
        #if (intv_cnt1s[i, 1]) - (intv_cnt1s[i, 0]) > 100:
        if (t135[intv_cnt1s[i, 1]]) - (t135[intv_cnt1s[i, 0]]) > 20:
            real_arm_events.append(i)
            t0 = t135[intv_cnt1s[i, 0]]
            t1 = t135[intv_cnt1s[i, 1]]+1   #  to get the end correct
            #_plt.plot(range(t0, t1), seg_ts[t0:t1], color="black", lw=4)
            print "real   %(0)d %(1)d" % {"0" : t0, "1" : t1}
            tFilled[t0:t1] = seg_ts[t0:t1]
            #  What if the last bit isn't in arm 1 3 or 5?
    if t135[intv_cnt1s[-1, 1]] < len(seg_ts)-1:
        #  I should be in either segment 2, 4 or crossed over from one or the other
        t0 = t135[intv_cnt1s[-1, 1]]+1
        t1 = len(seg_ts)
        uvals = _N.unique(seg_ts[t0:t1])
        nunq  = len(uvals)
        cnts  = _N.empty(len(uvals))
        for i in xrange(nunq):
            cnts[i] = len(_N.where(seg_ts[t0:t1] == uvals[i])[0])
        cnts /= _N.sum(cnts)

        if _N.max(cnts) > 0.98:  #  if I spent > most of my time in 1 segment
            mostInThisSeg = uvals[_N.where(cnts == _N.max(cnts))[0][0]]
            tFilled[t135[intv_cnt1s[-1, 1]]:] = mostInThisSeg
        else:
            print "----------  last bit not in 135, and spent in > 2 regions"
            print cnts
            print "may want to consider find_actual_arm_events() more"

    return real_arm_events



def smooth_over_deviations(day, ep, real_arm_events, seg_ts, t135, tFilled):
    """
    In between being in arms 1, 3 or 5  - we get spurious transitions between
    for example, 5->4->5->4->1.  Smooth over this so it looks like 5->4->1
    Write result in tFilled.  tFilled has no information about in- or outbound.
    """
    for ri in xrange(1, len(real_arm_events)):
        i = real_arm_events[ri]
        ip = real_arm_events[ri-1]   #  previous

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
            #btwnfigs(day, ep, t0, t1, seg_ts, tFilled, r, 1, 2, scxMin, scxMax, scyMin, scyMax)
        elif (seg_ts[t0-1] == 5) and (seg_ts[t1+1] == 5):
            tFilled[t0:t1] = 3
        elif (seg_ts[t0-1] == 5) and (seg_ts[t1+1] == 1):
            tFilled[t0:t1] = 4
        elif (seg_ts[t0-1] == 5) and (seg_ts[t1+1] == 3):
            tFilled[t0:t1] = seg_ts[t0:t1]
            is3 = _N.where(tFilled[t0:t1] == 3)[0]
            tFilled[t0+is3] = 2
            is5 = _N.where(tFilled[t0:t1] == 5)[0]
            tFilled[t0+is5] = 4
            #btwnfigs(day, ep, t0, t1, seg_ts, tFilled, r, 1, 2, scxMin, scxMax, scyMin, scyMax)
        else:
            print "came to else"


    #  now take a look at tFilled.  Are there any segments that are short lived?
    #  filter tFilled
    
    gk = gauKer(30)   #  1 second kernel
    gk /= _N.sum(gk)  #  
    ftFilled = _N.convolve(tFilled, gk, mode="same")

    #########  ONE MORE PASS.  Fix any remaining jumpiness
    for ri in xrange(1, len(real_arm_events)):
        i = real_arm_events[ri]
        ip = real_arm_events[ri-1]   #  previous

        t0 = t135[intv_cnt1s[ip, 1]]
        t1 = t135[intv_cnt1s[i,   0]]
        if ((seg_ts[t0-1] == 3) and (seg_ts[t1+1] == 5)) or \
           ((seg_ts[t0-1] == 5) and (seg_ts[t1+1] == 3)):
            #  all points in ftFilled closer to 2 shall be 2
            shldb2 = _N.where(_N.abs(ftFilled[t0:t1]-2) < 1)[0]
            shldb4 = _N.where(_N.abs(ftFilled[t0:t1]-4) < 1)[0]
            tFilled[t0+shldb2] = 2
            tFilled[t0+shldb4] = 4
            
            btwnfigs(day, ep, t0, t1, seg_ts, tFilled, r, 1, 2, scxMin, scxMax, scyMin, scyMax)


def min_max_lindist_for_segments(tFilled, segN_mm, ):
    """
    what is the min and max of the lindist over each segment?
    """
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

    #  times when maze state changes (ie from left arm to top left)
    #  [-1] because that



def assign_inb_or_outb(lindist, segN_mm, tsFilled, trans):
    """
    Decide whether a given segment is inbound or outbound movement.
    """
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
        iseg = tFilled[tSeg0]-1  # index for segN_mm
        outb = tFilled[tSeg0]*10
        inb  = outb + 1
        av   = _N.mean(segN_mm[iseg])  # av lindist
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

def linearize_inb_oub_segments(lindist, tsrans, tsFilled):
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

        if _N.max(oneDdist[t0:t1]) > 6:
            print "t0  %(0)d   t1  %(1)d" % {"0" : t0, "1" : t1}
            print tsFilled[t0]


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

    return oneDdist
