import scipy.io as _sio
import pickle
import ExperimentalMarkContainer as EMC
import filter as _flt
from os import listdir
from os.path import isdir, join
import EnDedirs as _edd

#  The animal tends to spend much of its time in arms 1, 3, 5
#  At least for well-trained animals, animals also do not turn around in 
#  arms 1, 3, 5 very much.  We use

#  for bond day4, ex = 2, 4, 6
#  for day3, ex 

animals = [["bon", "bond", "Bon"], ["fra", "frank", "Fra"], ["gov", "GovernmentData", "Gov"]]
#basedir = "/Volumes/Seagate Expansion Drive"
#basedir = "/Volumes/ExtraDisk/LorenData"
basedir = "/Users/arai/TEMP/LorenData"

exf("rawlin_debugFigs.py")
exf("rawlin_funcs.py")

#  home well, choice point    left well, left corner    right well, right corner
landmarks = _N.empty((6, 2))
Nsgs      = 5
segs      = _N.empty((Nsgs, 2, 2))  
length    = _N.empty(Nsgs)


ii     = 0

r      = None

def onclick(event):
    global ix, iy
    global ii
    ix, iy = event.xdata, event.ydata
    print 'x = %d, y = %d'%(
        ix, iy)

    global coords
    coords = [ix, iy]
    landmarks[ii, 0] = ix
    landmarks[ii, 1] = iy
    ii += 1

    if ii == 6:
        done()

seg_ts = None
inout  = None   # inbound - outbound
a_inout  = None   # inbound - outbound

lindist  = None        #  linearization with no left-right

def done():
    global r, seg_ts, segs, Nsgs, inout, a_inout, lindist
    hdir = _N.empty(2)
    vdir = _N.empty(2)
    linp = _N.empty(2)
    """
    L5       L0       L3
    ||       ||       ||
    ||       ||       ||
    5        1        3
    ||       ||       ||
    ||       ||       ||
    L4===4===L1===2===L2
    """
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

    e = _N.diff(segs, axis=1)
    a = _N.sqrt(_N.sum(e*e, axis=2))
    ar = a.reshape((Nsgs, 1, 1))
    e /= ar   #  unit vector


    #  a_s  slope
    a_s     = (segs[:, 1, 1] - segs[:, 0, 1]) / (segs[:, 1, 0] - segs[:, 0, 0]) 
    b_s     = -1
    c_s     = segs[:, 0, 1] - a_s*segs[:, 0, 0] 

    _plt.plot([segs[0, 0, 0], segs[0, 1, 0]], [segs[0, 0, 1], segs[0, 1, 1]], lw=3, color="black")
    _plt.plot([segs[1, 0, 0], segs[1, 1, 0]], [segs[1, 0, 1], segs[1, 1, 1]], lw=3, color="black")
    _plt.plot([segs[2, 0, 0], segs[2, 1, 0]], [segs[2, 0, 1], segs[2, 1, 1]], lw=3, color="black")
    _plt.plot([segs[3, 0, 0], segs[3, 1, 0]], [segs[3, 0, 1], segs[3, 1, 1]], lw=3, color="black")
    _plt.plot([segs[4, 0, 0], segs[4, 1, 0]], [segs[4, 0, 1], segs[4, 1, 1]], lw=3, color="black")

    segsr       = segs.reshape((10, 2))

    clrs = ["blue", "orange", "red", "green", "yellow", "black", "brown"]

    zrrp = _N.where((r[:, 1] == 0) | (r[:, 2] == 0) | (r[:, 3] == 0) & (r[:, 4] == 0))[0]    


    for iz in xrange(len(zrrp)):
        i = zrrp[iz]
        for ic in xrange(1, 5):
            if r[i, ic] == 0:
                r[i, ic] = r[i-1, ic]

    N           = r.shape[0]
    seg_ts        = _N.empty(N, dtype=_N.int)
    lindist         = _N.empty(N)
    inout         = _N.empty(N, dtype=_N.int)
    a_inout         = _N.empty(N)
    xp          = 0.5*(r[:, 1] + r[:, 3])
    yp          = 0.5*(r[:, 2] + r[:, 4])
    xpyp        = _N.empty((N, 2))
    xpyp[:, 0]  = xp
    xpyp[:, 1]  = yp

    _xpyp       = _N.repeat(xpyp, Nsgs*2, axis=0)
    rxpyp       = _xpyp.reshape((N, Nsgs*2, 2))

    dv          = segsr - rxpyp
    dists       = _N.sum(dv*dv, axis=2)   # closest point on maze from field points
    rdists      = dists.reshape((N, Nsgs, 2))

    online = _N.empty(Nsgs, dtype=bool)
    mins   = _N.empty(Nsgs)

    _plt.scatter(xp, yp)
    for n in xrange(0, N):
        x0 = xpyp[n, 0]
        y0 = xpyp[n, 1]

        xcs = (b_s*(b_s*x0 - a_s*y0) - a_s*c_s) / (a_s*a_s + b_s*b_s)
        ycs = (-a_s*(b_s*x0 - a_s*y0) - b_s*c_s) / (a_s*a_s + b_s*b_s)

        for ns in xrange(Nsgs):
            onlinex = (((xcs[ns] >= segs[ns, 0, 0]) and (xcs[ns] <= segs[ns, 1, 0])) or 
                       ((xcs[ns] <= segs[ns, 0, 0]) and (xcs[ns] >= segs[ns, 1, 0])))
            onliney = (((ycs[ns] >= segs[ns, 0, 1]) and (ycs[ns] <= segs[ns, 1, 1])) or 
                       ((ycs[ns] <= segs[ns, 0, 1]) and (ycs[ns] >= segs[ns, 1, 1])))
            online[ns] = onlinex and onliney  #  closest point is on line

        for ns in xrange(Nsgs):
            if online[ns]:   #  closest point on the segment
                mins[ns] = _N.min([(x0-xcs[ns])**2 + (y0-ycs[ns])**2, _N.min(rdists[n, ns])])
            else:
                mins[ns] = _N.min(rdists[n, ns])
        clsest = _N.where(mins == _N.min(mins))[0]
        iclsest= clsest[0]        #  segment ID
        if online[iclsest]:
            linp[0] = xcs[iclsest]
            linp[1] = ycs[iclsest]
            x = segs[iclsest, 0, 0]
            y = segs[iclsest, 0, 1]
            lindist[n] = _N.sqrt((linp[0] - x)*(linp[0] - x) + (linp[1] - y)*(linp[1] - y)) / length[iclsest]
            if iclsest == 1:
                lindist[n] += 1
            elif iclsest == 2:
                lindist[n] += 2
            elif iclsest == 3:
                lindist[n] += 1
            elif iclsest == 4:
                lindist[n] += 2
        else:
            if rdists[n, iclsest, 1] > rdists[n, iclsest, 0]:
                if iclsest == 0:
                    lindist[n] = 0
                elif iclsest == 1:
                    lindist[n] = 1
                elif iclsest == 2:
                    lindist[n] = 2
                elif iclsest == 3:
                    lindist[n] = 1
                elif iclsest == 4:
                    lindist[n] = 2
            else:
                if iclsest == 0:
                    lindist[n] = 1
                elif iclsest == 1:
                    lindist[n] = 2
                elif iclsest == 2:
                    lindist[n] = 3
                elif iclsest == 3:
                    lindist[n] = 2
                elif iclsest == 4:
                    lindist[n] = 3

        if len(clsest) == 2:   #  
            clr = "black"
        else:
            clr = clrs[clsest[0]]
        seg_ts[n] = clsest[0]
        
        #_plt.plot([x0, x0], [y0, y0], ms=10, marker=".", color=clr)

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

    gk = gauKer(10)
    gk /= _N.sum(gk)
    finout = _N.convolve(a_inout, gk, mode="same")
    gt0     = _N.where(finout > 0)[0]
    lt0     = _N.where(finout <= 0)[0]
    inout[gt0] = 1
    inout[lt0] = -1

    




debugFigs = False   #  plot figures for debugging

for an in animals[0:1]:
    anim1 = an[0]
    anim2 = an[1]
    anim3 = an[2]

    #for day in xrange(0, 12):
    for day in xrange(10, 11):
        sdy    = ("0%d" % day) if (day < 10) else "%d" % day

        frip = "%(bd)s/%(s3)s/%(s1)sripplescons%(sdy)s.mat" % {"s1" : anim1, "sdy" : sdy, "s3" : anim3, "bd" : basedir}
        flnp = "%(bd)s/%(s3)s/%(s1)slinpos%(sdy)s.mat" % {"s1" : anim1, "sdy" : sdy, "s3" : anim3, "bd" : basedir}
        frwp = "%(bd)s/%(s3)s/%(s1)srawpos%(sdy)s.mat" % {"s1" : anim1, "sdy" : sdy, "s3" : anim3, "bd" : basedir}

        if os.access("%s" % frip, os.F_OK):
            rip = _sio.loadmat(frip)    #  load matlab .mat files
            mLp = _sio.loadmat(flnp)
            mRp = _sio.loadmat(frwp)

            ex   = rip["ripplescons"].shape[1] - 1
            _pts=mLp["linpos"][0,ex]


            #for epc in range(0, _pts.shape[1], 2):
            for epc in range(0, 1):
                ep=epc+1;

            #  experimental data mark, position container

            # frip = "%(bd)s/Dropbox (EastWestSideHippos)/BostonData/%(s3)s/%(s1)sripplescons%(sdy)s.mat" % {"s1" : anim1, "sdy" : sdy, "s3" : anim3, "bd" : basedir}
            # flnp = "%(bd)s/Dropbox (EastWestSideHippos)/BostonData/%(s3)s/%(s1)slinpos%(sdy)s.mat" % {"s1" : anim1, "sdy" : sdy, "s3" : anim3, "bd" : basedir}
            # frwp = "%(bd)s/Dropbox (EastWestSideHippos)/BostonData/%(s3)s/%(s1)srawpos%(sdy)s.mat" % {"s1" : anim1, "sdy" : sdy, "s3" : anim3, "bd" : basedir}


                #  these are in seconds
                #  episodes 2, 4, 6

                #  seg 1->2->3
                #  seg 1->4->5
                #%%
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                #%%%%            Linearization           %%%% 
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                _pts=mLp["linpos"][0,ex][0,ep] 
                if (_pts.shape[1] > 0):     #  might be empty epoch
                    pts = _pts["statematrix"][0][0]["time"][0,0].T[0]
                    a = mLp["linpos"][0,ex][0,ep]["statematrix"][0,0]["segmentIndex"]
                    r = mRp["rawpos"][0,ex][0,ep]["data"][0,0]


                    zrrp = _N.where((r[:, 1] > 0) & (r[:, 2] > 0) & (r[:, 3] > 0) & (r[:, 4] > 0))[0]

                    szrrp = zrrp
                    fig  = _plt.figure()
                    _plt.scatter(0.5*(r[szrrp, 1]+r[szrrp, 3]), 0.5*(r[szrrp, 2] + r[szrrp, 4]), s=3, color="grey")

                    cid = fig.canvas.mpl_connect('button_press_event', onclick)
                
                # for each instant in time
                # calculate line closest, and the velocity vector
                # use lindist at point closest.
                # at each point on line closest, we have a unit vector for inbound, outbound
                # at each time point, we get a (x_closest, +/- 1)
                # 

                N = r.shape[0]
                #  landmarks is 
                #  segs

                #segs - 
                
                
                #cr[:, 0] = 0.5*(r[zrrp, 1]+r[zrrp, 3])
                #cr[:, 1] = 0.5*(r[zrrp, 2]+r[zrrp, 4])

                #  5 segments, 2 points
                
                
                #  crds = N x 2

                
