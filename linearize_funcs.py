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

def lindist_x0y0(n, x0, y0, segs, rdists, seg_ts, Nsgs, online, offset, xcs, ycs, clrs, mins, linp):
    # Find the closest segment to point (rawpos) x0, y0 
    for ns in xrange(Nsgs):
        onlinex = (((xcs[ns] >= segs[ns, 0, 0]) and (xcs[ns] <= segs[ns, 1, 0])) or 
                   ((xcs[ns] <= segs[ns, 0, 0]) and (xcs[ns] >= segs[ns, 1, 0])))
        onliney = (((ycs[ns] >= segs[ns, 0, 1]) and (ycs[ns] <= segs[ns, 1, 1])) or 
                   ((ycs[ns] <= segs[ns, 0, 1]) and (ycs[ns] >= segs[ns, 1, 1])))
        online[ns] = onlinex and onliney  #  closest point is on line

    for ns in xrange(Nsgs):
        if online[ns]:   #  closest point (x0, y0) on segment, not endpts
            mins[ns] = _N.min([(x0-xcs[ns])**2 + (y0-ycs[ns])**2, _N.min(rdists[n, ns])])
        else:
            mins[ns] = _N.min(rdists[n, ns])
    clsest = _N.where(mins == _N.min(mins))[0]
    iclsest= clsest[0]        #  segment ID that is closest to x0, y0
    if online[iclsest]:
        linp[0] = xcs[iclsest]
        linp[1] = ycs[iclsest]
        x = segs[iclsest, 0, 0]
        y = segs[iclsest, 0, 1]
        lindist[n] = _N.sqrt((linp[0] - x)*(linp[0] - x) + (linp[1] - y)*(linp[1] - y)) / length[iclsest]

    addone = 0
    if (not online[iclsest]) and (rdists[n, iclsest, 1] < rdists[n, iclsest, 0]):
        addone = 1
        lindist[n] = 0
    lindist[n] += offset[iclsest]+addone

    if len(clsest) == 2:   #  
        clr = "black"
    else:
        clr = clrs[clsest[0]]
    seg_ts[n] = clsest[0]
    return iclsest


def a_inout_x0y0(n, r, hdir, vdir, iclsest, e):
    hdir[0]  = r[n, 3]-r[n, 1]
    hdir[1]  = r[n, 4]-r[n, 2]
    hdir     /= _N.sqrt(_N.sum(hdir*hdir))
    if n > 0:
        vdir[0]  = 0.5*((r[n, 1]-r[n-1, 1]) + (r[n, 3]-r[n-1, 3]))
        vdir[1]  = 0.5*((r[n, 2]-r[n-1, 2]) + (r[n, 4]-r[n-1, 4]))
        amp      = _N.sqrt(_N.sum(vdir*vdir))
        if amp > 0:
            vdir     /= _N.sqrt(_N.sum(vdir*vdir))
        else:
            vdir[:]     = hdir[:]

    if n == 0:
        a_inout[n] = _N.dot(hdir, e[iclsest, 0])
    else:
        a_inout[n] = _N.dot(0.5*(hdir+vdir), e[iclsest, 0])


def btwnfigs(day, ep, t0, t1, seg_ts, r, x, y, scxMin, scxMax, scyMin, scyMax):
    print "btwnfigs   %(ep)d  %(1)d %(2)d" % {"ep" : ep, "1" : t0, "2" : t1}
    fig = _plt.figure(figsize=(13, 4))
    fig.add_subplot(1, 3, 1)
    _plt.scatter(r[:, x], r[:, y], s=5, color="grey")
    _plt.scatter(r[t0:t1, x], r[t0:t1, y], s=9, color="black")
    _plt.plot(r[t0, x], r[t0, y], ms=40, color="blue", marker=".")
    _plt.plot(r[t1, x], r[t1, y], ms=40, color="red", marker=".")
    _plt.xlim(scxMin, scxMax)
    _plt.ylim(scyMin, scyMax)
    fig.add_subplot(1, 3, 2)
    _plt.plot(range(t0-20, t1+20), seg_ts[t0-20:t1+20], color="black", lw=4)
    _plt.xlim(t0-20, t1+20)
    _plt.xticks(_N.arange(t0, t1, (t1-t0)/5))
    _plt.ylim(0.5, 5.5)
    _plt.grid()
    fig.add_subplot(1, 3, 3)
    _plt.plot(range(t0-20, t1+20), tFilled[t0-20:t1+20], color="black", lw=4)
    _plt.xlim(t0-20, t1+20)
    _plt.ylim(0.5, 5.5)
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
