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
basedir = "/Users/arai/TEMP/LorenData"

exf("linearize_funcs.py")

#  home well, choice point    left well, left corner    right well, right corner
landmarks = _N.empty((6, 2))
Nsgs      = 5
segs      = _N.empty((Nsgs, 2, 2))  
length    = _N.empty(Nsgs)
offset    = _N.array([0, 1, 2, 1, 2])

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
    for day in xrange(3, 4):
        sdy    = ("0%d" % day) if (day < 10) else "%d" % day

        #for epc in range(0, _pts.shape[1], 2):
        for epc in range(0, 1):
            ep=epc+1;

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
            build_lin_lr_inout(N, lin_lr_inout, lindist, lr, inout)
