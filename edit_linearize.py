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

minINOUT  = 100

#  regular lindist     #  0 to 3
#  lin_inout           #  inbound outbound  0 to 6
#  lin_lr              #  -3 to 3
#  lin_lr_inout        #  -3 to 3

ii     = 0

gkRWD  = gauKer(5)
gkRWD  /= _N.sum(gkRWD)

anim1     = None
anim2     = None
day    = None
ep     = None
r      = None


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

    
############################################
for an in animals[0:1]:
    anim1 = an[0]
    anim2 = an[1]
    anim3 = an[2]

    #for day in xrange(0, 12):
    #for day in xrange(10, 11):
    #for day in xrange(3, 4):
    for day in xrange(5, 6):
        sdy    = ("0%d" % day) if (day < 10) else "%d" % day

        frip = "%(bd)s/%(s3)s/%(s1)sripplescons%(sdy)s.mat" % {"s1" : anim1, "sdy" : sdy, "s3" : anim3, "bd" : basedir}
        frwp = "%(bd)s/%(s3)s/%(s1)srawpos%(sdy)s.mat" % {"s1" : anim1, "sdy" : sdy, "s3" : anim3, "bd" : basedir}
        flnp = "%(bd)s/%(s3)s/%(s1)slinpos%(sdy)s.mat" % {"s1" : anim1, "sdy" : sdy, "s3" : anim3, "bd" : basedir}

        if os.access("%s" % frip, os.F_OK):
            rip = _sio.loadmat(frip)    #  load matlab .mat files
            mLp = _sio.loadmat(flnp)
            mRp = _sio.loadmat(frwp)

            ex   = rip["ripplescons"].shape[1] - 1



        #for epc in range(0, _pts.shape[1], 2):
        for epc in range(4, 5):
            ep=epc+1;

            _pts=mLp["linpos"][0,ex][0,ep] 
            if (_pts.shape[1] > 0):     #  might be empty epoch
                pts = _pts["statematrix"][0][0]["time"][0,0].T[0]
                a = mLp["linpos"][0,ex][0,ep]["statematrix"][0,0]["segmentIndex"]
                r = mRp["rawpos"][0,ex][0,ep]["data"][0,0]

                fillin_unobsvd(r)

                scxMin, scxMax, scyMin, scyMax = get_boundaries(r)

            sday = ("0%d" % day) if (day < 10) else ("%d" % day)

            fn = _edd.datFN("cp_lr.dat", dir="linearize/%(an)s%(dy)s0%(ep)d" % {"dy" : sday, "ep" : (ep+1), "an" : anim2})
            cp_lr = _N.loadtxt(fn, dtype=_N.int)
            fn = _edd.datFN("cp_inout.dat", dir="linearize/%(an)s%(dy)s0%(ep)d" % {"dy" : sday, "ep" : (ep+1), "an" : anim2})
            cp_inout = _N.loadtxt(fn, dtype=_N.int)

            lr, inout = thaw_LR_inout(27901, cp_lr, cp_inout)

            fn = _edd.datFN("lindist.dat", dir="linearize/%(an)s%(dy)s0%(ep)d" % {"dy" : sday, "ep" : (ep+1), "an" : anim2})
            lindist = _N.loadtxt(fn)

            N  = len(lindist)
            lin_lr_inout = _N.empty(N)
            build_lin_lr_inout(N, lin_lr_inout, lindist, lr, inout, gkRWD)

            t0 = 0
            winsz = 1000
            t1 = 0
            iw    = -1
            while t1 < N:
                iw += 1
                t0 = iw*winsz
                t1 = (iw+1)*winsz if (iw+1)*winsz < N else N-1
                btwnfigs(anim2, day, ep, t0, t1, inout, "INOUT", [-1.1, 1.1], lr, "LR", [-1.1, 1.1], lin_lr_inout, "lin_lr_inout", [-6.1, 6.1], r, 1, 2, scxMin, scxMax, scyMin, scyMax)
