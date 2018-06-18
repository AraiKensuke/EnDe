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

    for day in xrange(3, 4):
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
                    print "here"
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
                    

                    time=mLp["linpos"][0,ex][0,ep]["statematrix"][0][0]["time"][0,0].T[0]   #  30Hz sampling of position
                    lindist=mLp["linpos"][0,ex][0,ep]["statematrix"][0][0]["lindist"][0,0].T[0]

                    svecT  = pts
                    t0 = int(svecT[0]  * 1000)
                    t1 = int(svecT[-1] * 1000)

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


                    marks   =  _N.empty((t1-t0+1, tetlistlen), dtype=list)

                    it      = -1

                    for dir in srtdirs:   #  these look like "07-162", "14-096" etc.
                        tet = dir.split("-")[0]
                        print "dir   is %s" % dir
                        prmfn = "%(d)s/%(td)s/%(an)s%(sd)s-%(st)s_params.mat" % {"d" : prmfilepath, "an" : anim2, "sd" : sdy, "st" : tet, "td" : dir}
                        if os.access(prmfn, os.F_OK):
                            print "------------------------"
                            it += 1
                            A = _sio.loadmat(prmfn)
                            t_champs = _N.array(A["filedata"][0,0]["params"][:, 0:5], dtype=_N.float32)  # time and amplitudes



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

                            path2D= _N.empty((t1-t0+1, 2), dtype=_N.float32)

                            for tt in xrange(len(svecT)):
                                t   = svecT[tt]
                                path2D[int(t*1000)-t0] = r[tt, 1:3]

                            dfnd = _N.array(svecT*1000-t0, dtype=_N.int)

                            path2D[:, 0] = _N.interp(_N.arange(t1-t0+1), dfnd, r[:, 1])
                            path2D[:, 1] = _N.interp(_N.arange(t1-t0+1), dfnd, r[:, 2])
                            path2D[:, 0] = _N.convolve(gkpth, path2D[:, 0], mode="same")
                            path2D[:, 1] = _N.convolve(gkpth, path2D[:, 1], mode="same")

                            dpath2D = _N.diff(path2D, axis=0)
                            spd = _N.sqrt(_N.sum(dpath2D*dpath2D, axis=1))
                            spd = _N.convolve(gkspd, spd, mode="same")

                            AMPS = _N.max(path2D, axis=0) - _N.min(path2D, axis=0)
                            spdthr = AMPS[0] / 30000.  #  spd where it takes 30 seconds to move 1 arm length is considered very slow
                            #spdthresh = AMPS

                            #  recording goes from svecT[0] to svecT[-1]
                            for imk in xrange(t_champs.shape[0]):  #  more marks than there is behavioral data
                                now_s  = t_champs[imk, 0]/10000.    #  now_s
                                now_ms = t_champs[imk, 0]/10.
                                ind = int(now_ms - svecT[0]*1000)  # svecT[0] is start of data we use
                                if (now_s > svecT[0]) and (now_s < svecT[-1]):
                                    #for nr in xrange(trgns.shape[0]):
                                    #    if (now_s >= trgns[nr, 0]) and (now_s <= trgns[nr, 1]):
                                    fd = _N.where((now_s >= svecT[0:-1]) & (now_s <= svecT[1:]))[0]
                                    rngT.append(now_s)
                                    #rngX.append(svecL0[fd[0]])
                                    if marks[ind, it] is None:  #  1st spike in this time bin
                                        marks[ind, it] = [_N.array(t_champs[imk, 1:], dtype=_N.float32)]
                                    else:
                                        #  2nd or more spikes in this time bin
                                        print "more than 2 spikes in this time bin"
                                        marks[ind, it].append(_N.array(t_champs[imk, 1:], dtype=_N.float32))

                    x  = []
                    xt = []

                    minds = _N.where(spd > spdthr)[0]
                    emc = EMC2d.ExperimentalMarkContainer2d(anim2, day, ep+1)
                    emc.pos2d   = path2D
                    emc.minds   = minds
                    emc.marks = marks
                    emc.tetlist = tetlist
                    emc.xA    = 6
                    emc.mA    = 8
                    emc.k     = 4
                    emc.dt    = 0.001
                    emc.Nx    = 50
                    emc.Nm    = 50
                    emc.save()
