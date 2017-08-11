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

debugFigs = False   #  plot figures for debugging

for an in animals[0:1]:
    anim1 = an[0]
    anim2 = an[1]
    anim3 = an[2]

    #for day in xrange(0, 12):
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
                    time=mLp["linpos"][0,ex][0,ep]["statematrix"][0][0]["time"][0,0].T[0]   #  30Hz sampling of position
                    lindist=mLp["linpos"][0,ex][0,ep]["statematrix"][0][0]["lindist"][0,0].T[0]

                    seg_ts = a[0,0].T[0]
                    force_24 = _N.ones(len(seg_ts), dtype=_N.int)*-1
                    ##  t135, l_intv_cnt1s, tFilled, tsFilled
                    scxMin, scxMax, scyMin, scyMax, nzinds = get_boundaries(r)

                    if day > 9:
                        hfn = "tFilled_hc-%(an)s%(dy)d0%(ep)d.py" % {"dy" : day, "ep" : (ep+1), "an" : anim2}
                    else:
                        hfn = "tFilled_hc-%(an)s0%(dy)d0%(ep)d.py" % {"dy" : day, "ep" : (ep+1), "an" : anim2}

                    if os.access(hfn, os.F_OK):
                        print "found %s" % hfn
                        exf(hfn)

                    #  tFilled only created, but is in initial state of -1
                    #  intv_cnt1s, t135 created using seg_ts
                    t135, intv_cnt1s, tFilled, tsFilled, seg1, seg2, seg3, seg4, seg5 = continuous(seg_ts, r, time, lindist)  #  tFilled

                    """
                    run hard-coded file.  Outside of "real" arm events.  
                    hard-coded file should modify tFilled.
                    look at linpos.png output, and 
                    btwnfigs(day, ep, 22500, 24000, seg_ts, tFilled, r, 1, 2, scxMin, scxMax, scyMin, scyMax)
                    """


                    # #  tFilled at this point has vals 1, 3, 5 or -1 only
                    real_arm_events = find_actual_arm_events(intv_cnt1s, t135, tFilled)

                    # x = 1; y = 2

                    # ###  debug by _plt.plot(seg_ts)
                    # ###  btwnfigs(0, 1000, 2000, seg_ts, r, 1, 2, scxMin, scxMax, scyMin, scyMax)
                    # #  now look at what happens between the stable time in the arm.  
                    fig = _plt.figure()
                    _plt.plot(tFilled)
                    smooth_over_deviations(day, ep, real_arm_events, seg_ts, t135, tFilled, force_24)

                    fig = _plt.figure()
                    _plt.plot(tFilled)
                    _plt.plot(seg_ts)

                    segN_mm = _N.empty((5, 2))  #  min, max lin distance for this segment
                    min_max_lindist_for_segments(tFilled, segN_mm)

                    trans =  _N.array([-1] + (_N.where(_N.diff(tFilled) != 0)[0]).tolist())+1
                    assign_inb_or_outb(lindist, segN_mm, tsFilled, trans)

                    tsrans =  _N.array([-1] + (_N.where(_N.diff(tsFilled) != 0)[0]).tolist())+1


                    oneDdist = linearize_inb_oub_segments(lindist, tsrans, tsFilled)

                    #########
                    #  t_champs[-1, 0] - t_champs[0, 0]       neural recording length, 2 10kHz
                    #  svecT[0]...svecT[-1]                   sorted time of behavioral observation

                    svecT  = pts
                    svecL0 = oneDdist
                    spd   = _N.diff(svecL0)
                    pj    = _N.where(spd > 4)[0]   #  -6-->0
                    nj    = _N.where(spd < -4)[0]  #   6-->0
                    spd[pj] = 0.05
                    spd[nj] = -0.05

                    gk    = _flt.gauKer(10)  #  kernel of about 300ms.
                    gk    /= _N.sum(gk)

                    kspd  = _N.convolve(spd, gk, mode="same")
                    fastTs = _N.where(_N.abs(kspd) >  0.003)[0]

                    crgns  = []
                    iStart = 0
                    bInMvt = False

                    for c in xrange(0, len(fastTs)-1):
                        if fastTs[c+1] - fastTs[c] == 1:
                            if not bInMvt:
                                iStart = fastTs[c]
                                bInMvt = True
                        else:
                            if bInMvt:
                                bInMvt = False
                                iStop = fastTs[c]
                                crgns.append([iStart, iStop])

                    fastMsk = []
                    for itv in xrange(len(crgns)-2, -1, -1):
                        if crgns[itv+1][0] - crgns[itv][1] < 100:   # 100 ms. of stop is not a stop
                            crgns[itv][1] = crgns[itv+1][1]
                            crgns.pop(itv+1)

                    #  show detected moving states
                    #_plt.plot(svecT, svecL0, color="blue")
                    vrgns = _N.array(crgns)

                    for itv in xrange(vrgns.shape[0]):
                        indxs = range(vrgns[itv,0], vrgns[itv,1])
                        fastMsk.extend(indxs)
                        #_plt.plot(svecT[indxs], svecL0[indxs], lw=4, color="black")
                    trgns = svecT[vrgns]     #  

                    #  we are going to use 1 ms bins
                    #t0 = int((svecT[0] - 0.5) * 1000)
                    #t1 = int((svecT[-1] + 0.5) * 1000)
                    t0 = int(svecT[0]  * 1000)
                    t1 = int(svecT[-1] * 1000)

                    k       =  1
                    pos     =  _N.empty(t1-t0)


                    svecT_ms = _N.linspace(t0, t1, t1-t0, endpoint=False)    #  
                    svecL0_ms = _N.interp(svecT_ms, svecT*1000, svecL0)

                    #prmfilepath = "%(bd)s/Dropbox (EastWestSideHippos)/BostonData/%(s2)s%(dy)s" %  {"s1" : anim1, "dy" : sdy, "s2" : anim2, "bd" : basedir}
                    prmfilepath = "%(bd)s/%(s2)s/%(s2)s%(dy)s" %  {"s1" : anim1, "dy" : sdy, "s2" : anim2, "bd" : basedir}

                    fig = _plt.figure(figsize=(11, 7))
                    fig.add_subplot(3, 1, 1)
                    _plt.plot(svecT, svecL0, color="black")
                    _plt.xlim(svecT[0], svecT[-1])
                    _plt.ylim(-6.2, 6.2)
                    fig.add_subplot(3, 1, 2)
                    _plt.plot(svecL0, color="black")
                    _plt.xlim(0, len(svecL0))
                    _plt.ylim(-6.2, 6.2)
                    fig.add_subplot(3, 1, 3)
                    _plt.plot(tFilled, color="black")
                    _plt.xlim(0, len(tFilled))
                    _plt.yticks(range(1, 6))
                    _plt.ylim(0.9, 5.1)

                    png = _edd.resFN("%(anim)s%(sdy)s0%(ep)s_linpos.png" % {"anim" : anim2, "sdy" : sdy, "ep" : (ep+1)})
                    print png
                    _plt.savefig(png)
                    _plt.close()
                    tp  = _N.empty((svecT.shape[0], 2))
                    tp[:, 0] = svecT
                    tp[:, 1] = svecL0
                    datfn = _edd.resFN("%(anim)s%(sdy)s0%(ep)s_tpos.dat" % {"anim" : anim2, "sdy" : sdy, "ep" : (ep+1)})
                    _N.savetxt(datfn, tp, fmt="%.4f %.4f")


                    """
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


                    marks   =  _N.empty((t1-t0, tetlistlen), dtype=list)

                    it      = -1

                    
                    for dir in srtdirs:   #  these look like "07-162", "14-096" etc.
                        tet = dir.split("-")[0]
                        print "dir   is %s" % dir
                        prmfn = "%(d)s/%(td)s/%(an)s%(sd)s-%(st)s_params.mat" % {"d" : prmfilepath, "an" : anim2, "sd" : sdy, "st" : tet, "td" : dir}
                        if os.access(prmfn, os.F_OK):
                            it += 1
                            A = _sio.loadmat(prmfn)
                            t_champs = _N.array(A["filedata"][0,0]["params"][:, 0:5], dtype=_N.float32)  # time and amplitudes
                            t_champs[:, 1:5] /= 50.



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
                            rngX = []

                            for imk in xrange(t_champs.shape[0]):  #  more marks than there is behavioral data
                                now_s  = t_champs[imk, 0]/10000.    #  now_s
                                now_ms = t_champs[imk, 0]/10.
                                ind = int(now_ms - svecT[0]*1000)  # svecT[0] is start of data we use
                                if (now_s > svecT[0]) and (now_s < svecT[-1]):
                                    #for nr in xrange(trgns.shape[0]):
                                    #    if (now_s >= trgns[nr, 0]) and (now_s <= trgns[nr, 1]):
                                    fd = _N.where((now_s >= svecT[0:-1]) & (now_s <= svecT[1:]))[0]
                                    y.append(t_champs[imk, 1])
                                    rngT.append(now_s)
                                    rngX.append(svecL0[fd[0]])
                                    if marks[ind, it] is None:  #  1st spike in this time bin
                                        marks[ind, it] = [_N.array(t_champs[imk, 1:], dtype=_N.float32)]
                                    else:
                                        #  2nd or more spikes in this time bin
                                        print "more than 2 spikes in this time bin"
                                        marks[ind, it].append(_N.array(t_champs[imk, 1:], dtype=_N.float32))

                    x  = []
                    xt = []
                    minds = _N.empty((trgns.shape[0], 2), dtype=_N.int)

                    for nr in xrange(trgns.shape[0]):
                        t0 = vrgns[nr, 0]   #  @ 33.4 Hz
                        t1 = vrgns[nr, 1]  

                        minds[nr] = int((svecT[t0] - svecT[0]) * 1000), int((svecT[t1] - svecT[0]) * 1000)
                        t0 = minds[nr, 0]
                        t1 = minds[nr, 1]

                        #x.extend(svecL0_ms[t0:t1].tolist())   #  position
                        #xt.extend(range(t0, t1))

                        _plt.plot(_N.arange(t0, t1), svecL0_ms[t0:t1])

                    emc = EMC.ExperimentalMarkContainer(anim2, day, ep+1)
                    emc.pos   = _N.array(svecL0_ms, dtype=_N.float16)   #  marks and position are not aligned
                    emc.marks = marks
                    emc.tetlist = tetlist
                    emc.minds = minds
                    emc.xA    = 6
                    emc.mA    = 8
                    emc.k     = 4
                    emc.dt    = 0.001
                    emc.Nx    = 50
                    emc.Nm    = 50
                    emc.save()





                    """
