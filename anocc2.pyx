from filter import gauKer
import numpy as _N

def desort(arr, rat):
    N = arr.shape[0]
    indprs = _N.array(N*_N.random.rand(int(rat*N), 2), dtype=_N.int)
    for n in xrange(int(rat*N)):
        tmp = arr[indprs[n, 0]]
        arr[indprs[n, 0]] = arr[indprs[n, 1]]
        arr[indprs[n, 1]] = tmp

twpi = 2*_N.pi

def approx_Wmaze(blur, xt0t1, yt0t1, smth_krnl_x=500, smth_krnl_y=500):
    """
    eventually accommodate different orientation of the maze
    """
    global twpi

    blur = False   #  blur path when Gibbs sampling, but not when calculating approx path
    N   = xt0t1.shape[0]

    raw_path_xy  = _N.empty((2, N))
    raw_path_xy[0]  = xt0t1
    raw_path_xy[1]  = yt0t1

    gkx        = gauKer(smth_krnl_x) 
    gkx        /= _N.sum(gkx)
    gky        = gauKer(smth_krnl_y) 
    gky        /= _N.sum(gky)

    path_xy     = _N.empty((2, N))
    path_xy[0] = _N.convolve(raw_path_xy[0], gkx, mode="same")
    path_xy[1] = _N.convolve(raw_path_xy[1], gky, mode="same")

    dpth0      = _N.diff(path_xy[0])
    dpth1      = _N.diff(path_xy[1])
    #segs = _N.where(((dpth1[0:-1] <= 0) & (dpth1[1:] > 0)) | ((dpth1[0:-1] >= 0) & (dpth1[1:] < 0)))[0]

    blendt = 200
    enter_wells = _N.where((path_xy[1, 0:-1] < 0.74) & (path_xy[1, 1:] >= 0.74))[0]
    enter_wells += _N.array(blendt*_N.random.randn(len(enter_wells)), dtype=_N.int)
    leave_wells = _N.where((path_xy[1, 0:-1] > 0.74) & (path_xy[1, 1:] <= 0.74))[0]
    leave_wells += _N.array(blendt*_N.random.randn(len(leave_wells)), dtype=_N.int)
    enter_Ts = _N.where((path_xy[1, 0:-1] > 0.25) & (path_xy[1, 1:] <= 0.25))[0]
    enter_Ts += _N.array(blendt*_N.random.randn(len(enter_Ts)), dtype=_N.int)
    leave_Ts = _N.where((path_xy[1, 0:-1] < 0.25) & (path_xy[1, 1:] >= 0.25))[0]
    leave_Ts += _N.array(blendt*_N.random.randn(len(leave_Ts)), dtype=_N.int)

    allsegs_l  = enter_wells.tolist() + leave_wells.tolist() + enter_Ts.tolist() + leave_Ts.tolist()

    allsegs        = _N.array(allsegs_l)
    allsegs.sort()

    segID          = _N.empty(len(allsegs)-1, dtype=_N.int)
    segmns         = _N.empty((len(allsegs)-1, 2))

    for s in xrange(len(allsegs)-1):
        mx = _N.mean(raw_path_xy[0, allsegs[s]:allsegs[s+1]])
        my = _N.mean(raw_path_xy[1, allsegs[s]:allsegs[s+1]])
        segmns[s, 0] = mx
        segmns[s, 1] = my

        print "%(1).2f  %(2).2f" % {"1" : mx, "2" : my}
        if (my > 0.7):
            if (mx < 0.25):
                segID[s]  = -2
            elif (mx < 0.6):
                segID[s]  = -1
            else:
                segID[s]  = -3
        elif (my > 0.3) and (my < 0.7):
            if (mx < 0.25):
                segID[s]  = 3
            elif (mx < 0.6):
                segID[s]  = 1
            else:
                segID[s]  = 5
        elif (my < 0.3):
            if (mx < 0.5):
                segID[s]  = 2
            else:
                segID[s]  = 4


    pcsW      = 4
    pcsA      = 7
    pcsT      = 4
    totalpcs = 3*pcsW + 2*pcsT + 3*pcsA

    pc0 = 0
    Ns     = _N.empty(totalpcs)
    mns    = _N.empty((2, totalpcs))
    sd2s   = _N.empty((2, totalpcs))
    isd2s  = _N.empty((2, totalpcs))

    for sid in [-3, -2, -1, 1, 2, 3, 4, 5]:
        segs = _N.where(segID == sid)[0]
        segxl= []
        segyl= []

        for seg in segs:
            t0 = allsegs[seg]
            t1 = allsegs[seg+1]
            segxl.extend(raw_path_xy[0, t0:t1])
            segyl.extend(raw_path_xy[1, t0:t1])

        segx = _N.array(segxl)
        segy = _N.array(segyl)

        #  we don't necessarily want the srtdinds to PERFECTLY sort the array
        if (sid < 0):
            pcs = pcsW
            srtdinds = segy.argsort()  
            desort(srtdinds, 0.01)
            ssegx = segx[srtdinds]
            ssegy = segy[srtdinds]
        elif (sid == 1) or (sid == 3) or (sid == 5):
            srtdinds = segy.argsort()  
            desort(srtdinds, 0.01)
            ssegx = segx[srtdinds]
            ssegy = segy[srtdinds]
            pcs = pcsA
        elif (sid == 2) or (sid == 4):
            pcs = pcsT
            srtdinds = segx.argsort()
            desort(srtdinds, 0.01)
            ssegx = segx[srtdinds]
            ssegy = segy[srtdinds]
        pc1 = pc0 + pcs

        for pc in xrange(pc0, pc1):
            ipc0 = int((float(pc-pc0)/pcs)*len(ssegx))
            ipc1 = int((float(pc-pc0 + 1)/pcs)*len(ssegx))
            Ns[pc]     = ipc1-ipc0
            mns[0, pc] = _N.mean(ssegx[ipc0:ipc1])
            mns[1, pc] = _N.mean(ssegy[ipc0:ipc1])
            sd2s[0, pc] = _N.std(ssegx[ipc0:ipc1])**2
            sd2s[1, pc] = _N.std(ssegy[ipc0:ipc1])**2
            isd2s[0, pc]= 1./sd2s[0, pc]
            isd2s[1, pc]= 1./sd2s[1, pc]
        #print "%(0)d  %(1)d" % {"0" : pc0, "1" : pc1}
        pc0 = pc1

    return totalpcs, Ns, mns, sd2s, isd2s

