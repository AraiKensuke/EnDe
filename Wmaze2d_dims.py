import scipy.io as _sio
import pickle
import ExperimentalMarkContainer2d as EMC2d
import filter as _flt
from os import listdir
from os.path import isdir, join
import EnDedirs as _edd

#  The animal tends to spend much of its time in arms 1, 3, 5
#  At least for well-trained animals, animals also do not turn around in 
#  arms 1, 3, 5 very much.  We use

#  for bond day4, ex = 2, 4, 6
#  for day3, ex 

#animals = [["bon", "bond", "Bon"], ["fra", "frank", "Fra"], ["gov", "GovernmentData", "Gov"]]
animals = [["bon", "bond", "Bon"]]
#basedir = "/Volumes/Seagate Expansion Drive"
#basedir = "/Volumes/ExtraDisk/LorenData"
basedir = "/Users/arai/usb/nctc/Workspace/EnDe/LORENDATA"


exf("rawlin_debugFigs.py")
exf("rawlin_funcs.py")

debugFigs = False   #  plot figures for debugging

gk2d = _flt.gauKer(2)   #  use on intrapolated raw 2D position @ 1kHz
gk2d /= _N.sum(gk2d)

gkspd = _flt.gauKer(200)   # 
gkspd /= _N.sum(gkspd)

gkpth = _flt.gauKer(100)   # 
gkpth /= _N.sum(gkpth)

for an in animals[0:1]:
    anim1 = an[0]
    anim2 = an[1]
    anim3 = an[2]

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
            #for epc in range(0, 1):
            for epc in range(2, 3):
                ep=epc+1;

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
                    for it in xrange(r.shape[0]):
                        if r[it, 1] == 0:
                            r[it, 1] = r[it-1, 1]
                        if r[it, 2] == 0:
                            r[it, 2] = r[it-1, 2]

                    tmp = _N.convolve(gk2d, r[:, 1], mode="same")
                    r[:, 1] = tmp
                    tmp = _N.convolve(gk2d, r[:, 2], mode="same")
                    r[:, 2] = tmp
                    
__t0 = 0
__t1 = r.shape[0]

_plt.plot(r[:, 1], r[:, 2])


print "t0=0"
print "t1=%d" % __t1
print "_plt.plot(path2D[t0:t1, 0], path2D[t0:t1, 1])"
print "now run something like trim(10, 20) to see what starting (10x33)ms later and ending (20x33)ms earlier does to the path"

def trim(tr_t0, tr_t1):
    _plt.plot(r[__t0+tr_t0:__t1-tr_t1, 1], r[__t0+tr_t0:__t1-tr_t1, 2])

def save_trim(tr_t0, tr_t1):
    #  bon%(day)d%(epc)d 
    fn = "toff_%(an)s_%(d)d_%(e)d.txt" % {"an" : an[0], "d" : day, "e" : epc}

    fp = open(fn, "w")
    fp.write("#  start later by  and end earlier by (33ms units)\n")
    fp.write("%(1)d %(2)d\n" % {"1" : tr_t0, "2" : tr_t1})
    fp.close()
