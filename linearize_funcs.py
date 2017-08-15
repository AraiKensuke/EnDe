__HW = 0   # home well
__CP = 1   # choice point

def segs_from_landmarks(segs, landmarks, length):
    #  e unit vector (outbound) direction
    segs[0, 0] = landmarks[0]     
    segs[0, 1] = landmarks[1]    
    length[0]  = _N.sqrt(_N.sum((landmarks[1]-landmarks[0])*(landmarks[1]-landmarks[0])))

    segs[1, 0] = landmarks[1]
    segs[1, 1] = landmarks[2]
    length[1]  = _N.sqrt(_N.sum((landmarks[2]-landmarks[1])*(landmarks[2]-landmarks[1])))

    segs[2, 0] = landmarks[2]
    segs[2, 1] = landmarks[3]
    length[2]  = _N.sqrt(_N.sum((landmarks[3]-landmarks[2])*(landmarks[3]-landmarks[2])))

    segs[3, 0] = landmarks[1]
    segs[3, 1] = landmarks[4]
    length[3]  = _N.sqrt(_N.sum((landmarks[4]-landmarks[1])*(landmarks[4]-landmarks[1])))

    segs[4, 0] = landmarks[4]
    segs[4, 1] = landmarks[5]
    length[4]  = _N.sqrt(_N.sum((landmarks[5]-landmarks[4])*(landmarks[5]-landmarks[4])))

def inout_dir(segs, Nsgs):
    e = _N.diff(segs, axis=1)
    a = _N.sqrt(_N.sum(e*e, axis=2))
    ar = a.reshape((Nsgs, 1, 1))
    e /= ar   #  unit vector
    return e

def slopes_of_segs(segs):
    #  a_s  slope
    a_s     = (segs[:, 1, 1] - segs[:, 0, 1]) / (segs[:, 1, 0] - segs[:, 0, 0]) 
    b_s     = -1
    c_s     = segs[:, 0, 1] - a_s*segs[:, 0, 0] 
    return a_s, b_s, c_s

def fillin_unobsvd(r):
    zrrp = _N.where((r[:, 1] == 0) | (r[:, 2] == 0) | (r[:, 3] == 0) & (r[:, 4] == 0))[0]    


    for iz in xrange(len(zrrp)):
        i = zrrp[iz]
        for ic in xrange(1, 5):
            if r[i, ic] == 0:
                r[i, ic] = r[i-1, ic]

def find_clsest(n, x0, y0, segs, rdists, seg_ts, Nsgs, online, offset, xcs, ycs, mins, linp):
    # Find the closest segment to point (rawpos) x0, y0 

    for ns in xrange(Nsgs):
        sns = segs[ns]
        onlinex = (((xcs[ns] >= sns[0, 0]) and (xcs[ns] <= sns[1, 0])) or 
                   ((xcs[ns] <= sns[0, 0]) and (xcs[ns] >= sns[1, 0])))
        onliney = (((ycs[ns] >= sns[0, 1]) and (ycs[ns] <= sns[1, 1])) or 
                   ((ycs[ns] <= sns[0, 1]) and (ycs[ns] >= sns[1, 1])))
        online[ns] = onlinex and onliney  #  closest point is on line

        if online[ns]:   #  closest point (x0, y0) on segment, not endpts
            mins[ns] = _N.min([(x0-xcs[ns])**2 + (y0-ycs[ns])**2, _N.min(rdists[n, ns])])
        else:
            mins[ns] = _N.min(rdists[n, ns])
    clsest = _N.where(mins == _N.min(mins))[0]
    iclsest= clsest[0]        #  segment ID that is closest to x0, y0
    seg_ts[n] = clsest[0]
    return iclsest


def lindist_x0y0(N, xpyp, segs, rdists, seg_ts, Nsgs, online, offset, a_s, b_s, c_s, mins, linp):

    for n in xrange(N):
        x0 = xpyp[n, 0]
        y0 = xpyp[n, 1]

        xcs = (b_s*(b_s*x0 - a_s*y0) - a_s*c_s) / (a_s*a_s + b_s*b_s)
        ycs = (-a_s*(b_s*x0 - a_s*y0) - b_s*c_s) / (a_s*a_s + b_s*b_s)

        ns= seg_ts[n]        #  segment ID that is closest to x0, y0

        sns = segs[ns]
        onlinex = (((xcs[ns] >= sns[0, 0]) and (xcs[ns] <= sns[1, 0])) or 
                   ((xcs[ns] <= sns[0, 0]) and (xcs[ns] >= sns[1, 0])))
        onliney = (((ycs[ns] >= sns[0, 1]) and (ycs[ns] <= sns[1, 1])) or 
                   ((ycs[ns] <= sns[0, 1]) and (ycs[ns] >= sns[1, 1])))
        online = onlinex and onliney  #  closest point is on line

        if online:  #  seg_ts will have been modified
            linp[0] = xcs[ns]
            linp[1] = ycs[ns]
            x = segs[ns, 0, 0]
            y = segs[ns, 0, 1]
            lindist[n] = _N.sqrt((linp[0] - x)*(linp[0] - x) + (linp[1] - y)*(linp[1] - y)) / length[ns]

        addone = 0
        if (not online) and (rdists[n, ns, 1] < rdists[n, ns, 0]):
            addone = 1
            lindist[n] = 0
        lindist[n] += offset[ns]+addone

def a_inout_x0y0(N, a_inout, inout, r, seg_ts, thr, e):
    gk = gauKer(5)
    gk /= _N.sum(gk)
    x = 0.5*(r[:, 1]+r[:, 3])
    y = 0.5*(r[:, 2]+r[:, 4])
    fx= _N.convolve(x, gk, mode="same")
    fy= _N.convolve(y, gk, mode="same")
    dfx = fx[1:]-fx[:-1]
    dfy = fy[1:]-fy[:-1]
    fspd = _N.sqrt(dfx*dfx + dfy*dfy)
    
    mvg = _N.where(fspd > thr)[0]
    rst = _N.where(fspd <= thr)[0]

    vngtv      = -100000
    a_inout[:] = vngtv
    vdir = _N.empty((N-1, 2))
    vdir[:, 0] = dfx
    vdir[:, 1] = dfy
    
    for n in mvg:
        iclsest = seg_ts[n]
        a_inout[n] = _N.dot(vdir[n], e[iclsest, 0])

    for n in xrange(len(mvg)-1):
        if mvg[n+1]-mvg[n] > 1:
            for fn in xrange(mvg[n]+1, mvg[n+1]):
                a_inout[fn] = a_inout[mvg[n]]
    for n in xrange(mvg[-1], N):
        a_inout[n] = a_inout[mvg[-1]]

    gk = gauKer(20)
    gk /= _N.sum(gk)
    finout = _N.convolve(a_inout, gk, mode="same")
    gt0     = _N.where(finout > 0)[0]
    lt0     = _N.where(finout <= 0)[0]
    inout[gt0] = 1
    inout[lt0] = -1

    #  For all points where inout switched, if it was during slow movement, 
    #  don't allow switch.  switch_times are when in/out changes
    switch_times = _N.where(_N.diff(inout) != 0)[0]
    
    nSTs = len(switch_times)
    throw_out = []
    for ist in xrange(0, nSTs-1):
        st1 = switch_times[ist]+1
        st2 = switch_times[ist+1]+1
        #  st+1 is new value
        blw = _N.where(fspd[st1-5:st2+6] < thr)[0]
        #if (len(blw) > 7) and (st2-st1) < 60:
        #    throw_out.append(ist)
        Lintv = st2+5-(st1-5)
        if (len(blw) > Lintv*0.8):# and (st2-st1) < 60:
            throw_out.append(ist)

    #print "throwing out"
    #print "original length switch_times is %d" % len(switch_times)
    for ito in throw_out:
        it_st1 = switch_times[ito]+1
        it_st2 = switch_times[ito+1]+1
        #print inout[it_st1-1:it_st2+1]
        inout[it_st1:it_st2] = 1 if (inout[it_st1] == -1) else -1
    

# def a_inout_x0y0(N, a_inout, r, hdir, vdir, seg_ts, e):
#     vx = 0.5*((r[1:, 1]+r[1:, 3]) - (r[0:-1, 1]+r[0:-1, 3]))
#     vy = 0.5*((r[1:, 2]+r[1:, 4]) - (r[0:-1, 2]+r[0:-1, 4]))
#     gk = gauKer(15)
#     gk /= _N.sum(gk)
#     fvx = _N.convolve(vx, gk, mode="same")
#     fvy = _N.convolve(vy, gk, mode="same")

#     for n in xrange(N):
#         iclsest = seg_ts[n]
        
#         # hdir[0]  = r[n, 3]-r[n, 1]
#         # hdir[1]  = r[n, 4]-r[n, 2]
#         # hdir     /= _N.sqrt(_N.sum(hdir*hdir))

#         vdir[0]  = fvx[n-1]
#         vdir[1]  = fvy[n-1]
#         amp      = _N.sqrt(_N.sum(vdir*vdir))
#         vdir[:]     /= amp

#         #if n == 0:
#         a_inout[n] = _N.dot(vdir, e[iclsest, 0])
#         #else:
#         #    a_inout[n] = _N.dot(0.5*(hdir+vdir), e[iclsest, 0])

def btwnfigs(day, ep, t0, t1, someplt1, lims1, someplt2, lims2, r, x, y, scxMin, scxMax, scyMin, scyMax):
    print "btwnfigs   %(ep)d  %(1)d %(2)d" % {"ep" : ep, "1" : t0, "2" : t1}
    fig = _plt.figure(figsize=(13, 4))
    fig.add_subplot(1, 3, 1)
    _plt.scatter(r[::10, x], r[::10, y], s=5, color="grey")
    _plt.scatter(r[t0:t1:2, x], r[t0:t1:2, y], s=9, color="black")
    _plt.plot(r[t0, x], r[t0, y], ms=40, color="blue", marker=".")
    _plt.plot(r[t1, x], r[t1, y], ms=40, color="red", marker=".")
    _plt.xlim(scxMin, scxMax)
    _plt.ylim(scyMin, scyMax)
    fig.add_subplot(1, 3, 2)
    _plt.plot(range(t0-20, t1+20), someplt1[t0-20:t1+20], color="black", lw=4)
    _plt.xlim(t0-20, t1+20)
    _plt.xticks(_N.arange(t0, t1, (t1-t0)/5))
    _plt.ylim(lims1[0], lims1[1])
    _plt.grid()
    fig.add_subplot(1, 3, 3)
    _plt.plot(range(t0-20, t1+20), someplt2[t0-20:t1+20], color="black", lw=4)
    _plt.xlim(t0-20, t1+20)
    _plt.ylim(lims2[0], lims2[1])
    _plt.xticks(_N.arange(t0, t1, (t1-t0)/5))
    _plt.grid()
    

    fig.subplots_adjust(left=0.04, bottom=0.1, top=0.95, right=0.99)

    _plt.savefig("btwn_%(dy)d_%(ep)d_%(1)d_%(2)d" % {"dy" : day, "ep" : (ep+1), "1" : t0, "2" : t1})
    _plt.close()

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

    return scxMin, scxMax, scyMin, scyMax

def clean_seg_ts(seg_ts):
    gk = gauKer(2)
    gk /= _N.sum(gk)
    fseg_ts = _N.convolve(seg_ts, gk, mode="same")
    seg_ts[:] = _N.round(fseg_ts)

def make_lin_inout(N, lindist, inout, lin_inout):
    #  lindist is 0 to 3

    for n in xrange(N):
        lin_inout[n] = lindist[n] if (inout[n] == 1) else 6 - lindist[n]

    v = r[1:] - r[0:-1]

def make_lin_lr(N, lr, lindist, seg_ts, r):
    global __HW, __CP
    #  find points where lindist near 0 and lindist near 1.
    #  if we're near lindist == 1, if it starts increasing afterwards, 
    nearHW = _N.where(lindist < 0.2)[0]
    nearCP = _N.where((lindist > 0.8) & (lindist < 1.2))[0]
    dnearHW = _N.diff(nearHW)
    dnearCP = _N.diff(nearCP)
    hw_visits = _N.array(nearHW[_N.where(dnearHW > 30)[0]].tolist()  + [nearHW[-1]])
    cp_visits = _N.array(nearCP[_N.where(dnearCP > 30)[0]].tolist()  + [nearCP[-1]])

    #  If for each point nearHW 
    #  its possible f

    fig = _plt.figure()
    
    for isv in xrange(len(cp_visits)):  #  index that would sort array visits
        iv0 = cp_visits[isv]
        if isv < len(cp_visits)-1:
            iv1 = cp_visits[isv+1]
        else:
            iv1 = N-1

        #  seg_ts goes from 0 to 4
        left  = _N.where((seg_ts[iv0:iv1] == 1) | (seg_ts[iv0:iv1] == 2) )[0]
        right = _N.where((seg_ts[iv0:iv1] == 3) | (seg_ts[iv0:iv1] == 4) )[0]

        #  closest2HW
        min_lindist = _N.min(lindist[iv0:iv1])
        max_lindist = _N.max(lindist[iv0:iv1])
        #fig = _plt.figure(figsize=(6, 10))
        #fig.add_subplot(3, 1, 1)
        if max_lindist > 1.5:  #
            if len(left) > len(right):
                _plt.scatter(r[iv0:iv1, 1], r[iv0:iv1, 2], color="red")
                lr[iv0:iv1] = 1
                if lr[iv0-1] == 0:
                    #  we came from HW.  Trace back while lr == 0
                    bcktrck = _N.where(hw_visits < iv0)[0]
                    lr[hw_visits[bcktrck[-1]]:iv0] = 1
                    print "bcktrck[-1]  %(bt)d  iv0  %(iv0)d" % {"bt" : hw_visits[bcktrck[-1]], "iv0" : iv0}
            else:
                _plt.scatter(r[iv0:iv1, 1], r[iv0:iv1, 2], color="blue")
                lr[iv0:iv1] = -1
                if lr[iv0-1] == 0:
                    #  we came from HW.  Trace back while lr == 0
                    bcktrck = _N.where(hw_visits < iv0)[0]
                    print "bcktrck[-1]  %(bt)d  iv0  %(iv0)d" % {"bt" : hw_visits[bcktrck[-1]], "iv0" : iv0}
                    lr[hw_visits[bcktrck[-1]]:iv0] = -1

        if min_lindist < 0.3:
            _plt.scatter(r[iv0:iv1, 1], r[iv0:iv1, 2], color="black")
            lr[iv0:iv1] = 0
        _plt.suptitle("%(ll)d  %(lr)d" % {"ll" : len(left), "lr" : len(right)})

    #  look forward following visit to CP.  
    for isv in xrange(len(cp_visits)-1):  #  index that would sort array visits
        iv0 = cp_visits[isv]
        iv1 = cp_visits[isv+1]

        if lr[iv0+1] == 0:
            #  we're at CP, going back to HW
            fwdtrck = _N.where((hw_visits > iv0) & (hw_visits < iv1))[0]

            lr[iv0:hw_visits[fwdtrck[0]]] = lr[iv0-1]
            print "iv0  %(iv0)d  fwdtrck[0]  %(ft)d    lr[iv0-1] %(l0)d" % {"ft" : hw_visits[fwdtrck[0]], "iv0" : iv0, "l0" : lr[iv0-1]}

    #  beginning.
    i = -1
    i1 = 0
    i2 = 0
    while i < N:
        i += 1

        if (lr[i] == -3) or (lr[i] == 0):
            print "come here   %d" % i
            i1 = i
            while (i < N) and ((lr[i] == -3) or (lr[i] == 0)):
                i += 1 
            i2 = i

            if i2 < N:
                print "here   %(i1)d  %(i2)d == %(lr)d" % {"i1" : i1, "i2" : i2, "lr" : lr[i2+2]}
                
                lr[i1:i2] = lr[i2+1]


    """
    #  if rat spends time near HW, may get consecutive HW visits b4 hitting CP
    for isv in xrange(len(hw_visits)-1):  #  index that would sort array visits
        iv0 = hw_visits[isv]
        iv1 = hw_visits[isv+1]
        if _N.sum(lr[iv0:iv1]) == 0:
            lr[iv0:iv1] = lr[iv1+1]   #  
    """


def cohrnt_mv(fx, fy, rst, mvg):
    """
    rest periods where movement is coherent and directional
    """

    dfx  = _N.diff(fx)
    dfy  = _N.diff(fy)
    lrst = rst.tolist()
    lmvg = mvg.tolist()

    #  consecutive 1s 
    dmvg = _N.diff(mvg)  # mvg 30 47 48 49 50  60        # duration of interval btwn consecutive rest intervals
    #  mvg are non-consecutive time points where spd is high
    longNoMvg = _N.where(dmvg > 40)[0]   #  mvg[longNoMvg]+1
    #dmvg[longNoMvg[0]] = 12
    #dmvg[longNoMvg[0]-1] = 1
    # mvg[longNoMvg[0]-1] = 11
    # mvg[longNoMvg[0]]   = 12   rstBeg, restEnd = mvg[longNoMvg:longNoMvg+2]
    # mvg[longNoMvg[0]+1] = 91
    #

    drst = _N.diff(rst)  # rest 30 47 48 49 50  60        # duration of interval btwn consecutive  intervals
    #  mvg are non-consecutive time points where spd is high

    longMvg = _N.where(drst > 40)[0]   #  rst[longNoMvg]+1
    mvBeg, mvEnd = rst[longMvg:longMvg+2]


    coherent = []
    for lnm in longNoMvg:
        restBeg, restEnd = mvg[lnm:lnm+2]

        #print "%(ux).4f  %(dx).4f %(cvx).4f        %(uy).4f  %(dy).4f  %(cvy).4f" % {"ux" : ux, "dx" : sdx, "uy" : uy, "dy" : sdy, "cvx" : (sdx/ux), "cvy" : (sdy/uy)}
        #  sum(dfx) / sum(abs(dfx))
        sdfx = dfx[restBeg:restEnd]
        sdfy = dfy[restBeg:restEnd]
        rx   = _N.sum(sdfx)/_N.sum(_N.abs(sdfx))
        ry   = _N.sum(sdfy)/_N.sum(_N.abs(sdfy))

        if (_N.abs(rx) > 0.9) or (_N.abs(ry) > 0.9):
            coherent.append(lnm)
            print "%(rx).4f   %(ry).4f     %(lngth)d" % {"rx" : rx, "ry" : ry, "lngth" : (restEnd - restBeg)}
            fig = _plt.figure()
            _plt.scatter(fx[restBeg:restEnd], fy[restBeg:restEnd])
        
    
    print len(coherent)

