import scipy.io as _sio
import pickle
import ExperimentalMarkContainer as EMC
import filter as _flt
from os import listdir
from os.path import isdir, join
import EnDedirs as _edd
from filter import gauKer

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

pos_gk = gauKer(5)

thr    = 0.33
pos_gk /= _N.sum(pos_gk)    
############################################
for an in animals[0:1]:
    anim1 = an[0]
    anim2 = an[1]
    anim3 = an[2]

    #for day in xrange(0, 12):
    #for day in xrange(10, 11):
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

                _pts=mLp["linpos"][0,ex][0,ep] 
                if (_pts.shape[1] > 0):     #  might be empty epoch
                    pts = _pts["statematrix"][0][0]["time"][0,0].T[0]
                    a = mLp["linpos"][0,ex][0,ep]["statematrix"][0,0]["segmentIndex"]
                    r = mRp["rawpos"][0,ex][0,ep]["data"][0,0]




                    zrrp = _N.where((r[:, 1] == 0) | (r[:, 2] == 0) | (r[:, 3] == 0) & (r[:, 4] == 0))[0]    

                    for iz in xrange(len(zrrp)):
                        i = zrrp[iz]
                        for ic in xrange(1, 5):
                            if r[i, ic] == 0:
                                r[i, ic] = r[i-1, ic]


                x = 0.5*(r[1:, 1]+r[1:, 3])
                y = 0.5*(r[1:, 2]+r[1:, 4])
                fx= _N.convolve(x, pos_gk, mode="same")
                fy= _N.convolve(y, pos_gk, mode="same")
                dfx = fx[1:]-fx[:-1]
                dfy = fy[1:]-fy[:-1]
                fspd = _N.sqrt(dfx*dfx + dfy*dfy)


                #_N.where(fspd > 0.1)
                rst = _N.where(fspd <= thr)[0]
                mvg = _N.where(fspd > thr)[0]

                #  for all continuous rst segments, if 
                # fig = _plt.figure(figsize=(13, 7))
                # fig.add_subplot(2, 1, 1)
                # _plt.scatter(rst, fx[rst], marker=".", color="black", s=3)
                # _plt.scatter(mvg, fx[mvg], marker=".", color="orange", s=3)
                # fig.add_subplot(2, 1, 2)
                # _plt.scatter(rst, fy[rst], marker=".", color="black", s=3)
                # _plt.scatter(mvg, fy[mvg], marker=".", color="orange", s=3)

                cohrnt_mv(fx, fy, rst, mvg)

                fig = _plt.figure(figsize=(13, 7))
                fig.add_subplot(2, 1, 1)
                _plt.scatter(rst, fx[rst], marker=".", color="black", s=3)
                _plt.scatter(mvg, fx[mvg], marker=".", color="orange", s=3)
                fig.add_subplot(2, 1, 2)
                _plt.scatter(rst, fy[rst], marker=".", color="black", s=3)
                _plt.scatter(mvg, fy[mvg], marker=".", color="orange", s=3)
                
