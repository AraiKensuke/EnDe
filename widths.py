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
            for epc in range(2, 3):
                ep=epc+1;

                prmfilepath = "%(bd)s/%(s2)s/%(s2)s%(dy)s" %  {"s1" : anim1, "dy" : sdy, "s2" : anim2, "bd" : basedir}


                onlydirs = [f for f in listdir(prmfilepath) if isdir(join(prmfilepath, f))]
                srtdirs  = _N.sort(onlydirs)

                tetlist = []
                tetlistlen = 0
                for dir in srtdirs:
                    tet = dir.split("-")[0]
                    prmfn = "%(d)s/%(td)s/%(an)s%(sd)s-%(st)s_params.mat" % {"d" : prmfilepath, "an" : anim2, "sd" : sdy, "st" : tet, "td" : dir}
                    if os.access(prmfn, os.F_OK):
                        tetlistlen += 1
                        tetlist.append(tet)

                it      = -1

                for dir in srtdirs:   #  these look like "07-162", "14-096" etc.
                    tet = dir.split("-")[0]
                    print "dir   is %s" % dir
                    prmfn = "%(d)s/%(td)s/%(an)s%(sd)s-%(st)s_params.mat" % {"d" : prmfilepath, "an" : anim2, "sd" : sdy, "st" : tet, "td" : dir}
                    if os.access(prmfn, os.F_OK):
                        print "------------------------"
                        it += 1
                        A = _sio.loadmat(prmfn)
                        t_champsw = _N.array(A["filedata"][0,0]["params"][:, 0:6], dtype=_N.float32)  # time and amplitudes

                        fig = _plt.figure()
                        _plt.hist(t_champsw[:, 5], bins=_N.linspace(0, 50, 201))
