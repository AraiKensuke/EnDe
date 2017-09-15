import scipy.io as _sio
import pickle
import ExperimentalMarkContainer as EMC
import filter as _flt
from os import listdir
from os.path import isdir, join
import EnDedirs as _edd
from filter import gauKer
import utilities as _U

#  The animal tends to spend much of its time in arms 1, 3, 5
#  At least for well-trained animals, animals also do not turn around in 
#  arms 1, 3, 5 very much.  We use

#  for bond day4, ex = 2, 4, 6
#  for day3, ex 

animals = [["bon", "bond", "Bon"], ["fra", "frank", "Fra"], ["gov", "GovernmentData", "Gov"]]
#basedir = "/Volumes/Seagate Expansion Drive"
#basedir = "/Volumes/ExtraDisk/LorenData"
#basedir = "/Users/arai/TEMP/LorenData"
basedir = "/Users/arai/usb/nctc/Workspace/EnDe/LORENDATA"

exf("linearize_funcs.py")

#  home well, choice point    left well, left corner    right well, right corner
landmarks = _N.empty((6, 2))
Nsgs      = 5
segs      = _N.empty((Nsgs, 2, 2))  
length    = _N.empty(Nsgs)
offset    = _N.array([0, 1, 2, 1, 2])

gkRWD  = gauKer(5)
gkRWD  /= _N.sum(gkRWD)

#  regular lindist     #  0 to 3
#  lin_inout           #  inbound outbound  0 to 6
#  lin_lr              #  -3 to 3
#  lin_lr_inout        #  -3 to 3

ii     = 0

anim1     = None
anim2     = None
day    = None
ep     = None
r      = None

def onclick(event):
    global ix, iy, an, day, ep
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
lr       = None

fspd     = None

lindist  = None        #  linearization with no left-right
raw_lindist  = None        #  linearization with no left-right
lin_lr   = None
lin_inout= None
lin_lr_inout = None
scxMin = None
scxMax = None
scyMin = None 
scyMax = None

def done():
    """
    come here after 6 landmarks chosen
    """

    global r, seg_ts, segs, Nsgs, inout, a_inout, lindist, lin_lr, lin_inout, lin_lr_inout, lr, raw_lindist
    global scxMin, scxMax, scyMin, scyMax
    global an, day, ep

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
    scxMin, scxMax, scyMin, scyMax = get_boundaries(r)
    segs_from_landmarks(segs, landmarks, length)
    e = inout_dir(segs, Nsgs)
    a_s, b_s, c_s = slopes_of_segs(segs)

    _plt.plot([segs[0, 0, 0], segs[0, 1, 0]], [segs[0, 0, 1], segs[0, 1, 1]], lw=3, color="black")
    _plt.plot([segs[1, 0, 0], segs[1, 1, 0]], [segs[1, 0, 1], segs[1, 1, 1]], lw=3, color="black")
    _plt.plot([segs[2, 0, 0], segs[2, 1, 0]], [segs[2, 0, 1], segs[2, 1, 1]], lw=3, color="black")
    _plt.plot([segs[3, 0, 0], segs[3, 1, 0]], [segs[3, 0, 1], segs[3, 1, 1]], lw=3, color="black")
    _plt.plot([segs[4, 0, 0], segs[4, 1, 0]], [segs[4, 0, 1], segs[4, 1, 1]], lw=3, color="black")

    segsr       = segs.reshape((10, 2))

    clrs = ["blue", "orange", "red", "green", "yellow", "black", "brown"]
    fillin_unobsvd(r)

    N           = r.shape[0]
    seg_ts        = _N.empty(N, dtype=_N.int)
    lindist         = _N.empty(N)
    lin_lr          = _N.empty(N)
    lin_inout       = _N.empty(N)
    lin_lr_inout    = _N.empty(N)
    lr              = _N.ones(N, dtype=_N.int) * -3

    inout         = _N.empty(N, dtype=_N.int)
    a_inout         = _N.empty(N)
    gk          = gauKer(30)
    gk          /= _N.sum(gk)
    fx          = _N.convolve(0.5*(r[:, 1] + r[:, 3]), gk, mode="same")
    fy          = _N.convolve(0.5*(r[:, 2] + r[:, 4]), gk, mode="same")
    xp          = fx
    yp          = fy
    xpyp        = _N.empty((N, 2))
    xpyp[:, 0]  = xp
    xpyp[:, 1]  = yp

    _xpyp       = _N.repeat(xpyp, Nsgs*2, axis=0)
    rxpyp       = _xpyp.reshape((N, Nsgs*2, 2))

    dv          = segsr - rxpyp
    dists       = _N.sum(dv*dv, axis=2)   # closest point on maze from field points
    rdists      = dists.reshape((N, Nsgs, 2))
    print rdists.shape

    online = _N.empty(Nsgs, dtype=bool)
    mins   = _N.empty(Nsgs)

    for n in xrange(N):
        x0 = xpyp[n, 0]
        y0 = xpyp[n, 1]
        #  xcs, ycs: pt on all line segs closest to x0, y0 (may b byond endpts)
        xcs = (b_s*(b_s*x0 - a_s*y0) - a_s*c_s) / (a_s*a_s + b_s*b_s)
        ycs = (-a_s*(b_s*x0 - a_s*y0) - b_s*c_s) / (a_s*a_s + b_s*b_s)
        
        find_clsest(n, x0, y0, segs, rdists, seg_ts, Nsgs, online, offset, xcs, ycs, mins, linp)


    # fig = _plt.figure()
    # _plt.plot(seg_ts)
    # clean_seg_ts(seg_ts)

    # _plt.plot(seg_ts)

    raw_lindist        = _N.zeros(N)
    lindist_x0y0(N, xpyp, segs, rdists, seg_ts, Nsgs, online, offset, a_s, b_s, c_s, mins, linp, raw_lindist)
    
    smooth_lindist(raw_lindist, lindist)

    # fig = _plt.figure(figsize=(10, 4))
    # _plt.plot(lindist)
    # gk = gauKer(8)   #  don't want to make this too large.  if we just pass  through the choice point, we can miss it.
    # gk /= _N.sum(gk)
    # flindist = _N.convolve(lindist, gk, mode="same")
    # lindist  = flindist
    
    # rm_lindist_jumps(N, lindist, seg_ts)
    fig = _plt.figure(figsize=(10, 4))
    _plt.plot(lindist)



    spd_thr = 0.35
    a_inout_x0y0(N, a_inout, inout, r, seg_ts, spd_thr, e)
        #_plt.plot([x0, x0], [y0, y0], ms=10, marker=".", color=clr)

    make_lin_inout(N, lindist, inout, lin_inout)
    make_lin_lr(N, lr, lindist, seg_ts, r)
    build_lin_lr_inout(N, lin_lr_inout, lindist, lr, inout, gkRWD)

    #  inout 
    cp_lr, cp_inout = cpify_LR_inout(lr, inout)

    sday = ("0%d" % day) if (day < 10) else ("%d" % day)
    fn = _edd.datFN("lindist.dat", dir="linearize/%(an)s%(dy)s0%(ep)d" % {"dy" : sday, "ep" : (ep+1), "an" : anim2}, create=True)
    _N.savetxt(fn, lindist, fmt="%.3f")
    fn = _edd.datFN("cp_lr.dat", dir="linearize/%(an)s%(dy)s0%(ep)d" % {"dy" : sday, "ep" : (ep+1), "an" : anim2})
    _U.savetxtWCom(fn, cp_lr, fmt="%d %d", com=("# N=%d.  1st column time, 2nd column  - inout value from this time until time in next row" % N))
    fn = _edd.datFN("cp_inout.dat", dir="linearize/%(an)s%(dy)s0%(ep)d" % {"dy" : sday, "ep" : (ep+1), "an" : anim2})
    _U.savetxtWCom(fn, cp_inout, fmt="%d %d", com=("# N=%d.  1st column time, 2nd column  - inout value from this time until time in next row" % N))
    fn = _edd.datFN("lin_lr_inout.dat", dir="linearize/%(an)s%(dy)s0%(ep)d" % {"dy" : sday, "ep" : (ep+1), "an" : anim2})
    _N.savetxt(fn, lin_lr_inout, fmt="%.3f")
    
    """
    """
    t0 = 0
    winsz = 1000
    t1 = 0
    iw    = -1
    while t1 < N:
        iw += 1
        t0 = iw*winsz
        t1 = (iw+1)*winsz if (iw+1)*winsz < N else N-1
        #btwnfigs(anim2, day, ep, t0, t1, inout, [-1.1, 1.1], seg_ts+1, [0.9, 5.1], r, 1, 2, scxMin, scxMax, scyMin, scyMax)
        btwnfigs(anim2, day, ep, t0, t1, inout, "INOUT", [-1.1, 1.1], lr, "LR", [-1.1, 1.1], lin_lr_inout, "lin_lr_inout", [-6.1, 6.1], r, 1, 2, scxMin, scxMax, scyMin, scyMax)
        #btwnfigs(anim2, day, ep, t0, t1, lindist, [-0.1, 3.1], seg_ts+1, [0.9, 5.1], r, 1, 2, scxMin, scxMax, scyMin, scyMax)


    
############################################
for an in animals[0:1]:
    anim1 = an[0]
    anim2 = an[1]
    anim3 = an[2]

    #for day in xrange(0, 12):
    #for day in xrange(10, 11):
    for day in xrange(5, 6):
        sdy    = ("0%d" % day) if (day < 10) else "%d" % day

        frip = "%(bd)s/%(s3)s/%(s1)sripplescons%(sdy)s.mat" % {"s1" : anim1, "sdy" : sdy, "s3" : anim3, "bd" : basedir}
        flnp = "%(bd)s/%(s3)s/%(s1)slinpos%(sdy)s.mat" % {"s1" : anim1, "sdy" : sdy, "s3" : anim3, "bd" : basedir}
        frwp = "%(bd)s/%(s3)s/%(s1)srawpos%(sdy)s.mat" % {"s1" : anim1, "sdy" : sdy, "s3" : anim3, "bd" : basedir}

        if os.access("%s" % frip, os.F_OK):
            rip = _sio.loadmat(frip)    #  load matlab .mat files
            mLp = _sio.loadmat(flnp)
            mRp = _sio.loadmat(frwp)

            ex   = rip["ripplescons"].shape[1] - 1
            _pts = mLp["linpos"][0,ex]

            #for epc in range(0, _pts.shape[1], 2):
            for epc in range(4, 5):
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

                    szrrp = zrrp[::4]
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

