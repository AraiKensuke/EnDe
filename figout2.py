import scipy.io as _sio
import pickle
import ExperimentalMarkContainer as EMC
import filter as _flt

#  experimental data mark, position container

mLp = _sio.loadmat('../SpkSrtd/bonlinpos04.mat')
mRp = _sio.loadmat('../SpkSrtd/bonrawpos04.mat')

ex=4-1; ep=2-1;

#%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%            Linearization           %%%% 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a = mLp["linpos"][0,ex][0,ep]["statematrix"][0,0]["segmentIndex"][0][0].T[0]

seg1 = _N.where(a == 1)[0]
seg2 = _N.where(a == 2)[0]
seg3 = _N.where(a == 3)[0]
seg4 = _N.where(a == 4)[0]
seg5 = _N.where(a == 5)[0]


#rp = mRp["rawpos"][0, ex][0,ep][0,0][0]
rp = mRp["rawpos"][0, ex][0,ep][0,0][0]

#unobsH = _N.where(rp[:, 1] == 0)[0]
#unobsB = _N.where(rp[:, 3] == 0)[0]

ln0h = 0   # last non-zero head pos
ln0b = 0   # last non-zero butt pos
for i in xrange(1, rp.shape[0]):
    if rp[i, 1] == 0:   # head
        rp[i, 1:3] =         rp[i-1, 1:3] 
        ln0h = i
    if rp[i, 1] == 0:   # head
        rp[i, 3:5] =         rp[i-1, 3:5] 
        ln0b = i


hpx = rp[:, 1]   # head position
hpy = rp[:, 2]

bpx = rp[:, 3]  # butt position
bpy = rp[:, 4]

hdx  = hpx-bpx
hdy  = hpy-bpy
vhx   = _N.array([0] + _N.diff(hpx).tolist())
vhy   = _N.array([0] + _N.diff(hpy).tolist())

vbx   = _N.array([0] + _N.diff(bpx).tolist())
vby   = _N.array([0] + _N.diff(bpy).tolist())

#  smoothed velocities.  
svhy = _N.convolve(vhy, gk, mode="same")
svhx = _N.convolve(vhx, gk, mode="same")
svbx = _N.convolve(vbx, gk, mode="same")
svby = _N.convolve(vby, gk, mode="same")

#  head direction is positively correlated with velocities  (esp. smoothed vel)
# _ss.pearsonr(hdC, svhC), _ss.pearsonr(hdC, svbC)   # C == x, y

time=mLp["linpos"][0,ex][0,ep]["statematrix"][0][0]["time"][0,0].T[0]
lindist=mLp["linpos"][0,ex][0,ep]["statematrix"][0][0]["lindist"][0,0].T[0]

#  head direction is robust

startT = 3000
for i in xrange(startT+1, 4000):
    if a[i-1] != a[i]:
        endT = i
        _plt.figure()
        _plt.scatter(rpx[startT:endT], rpy[startT:endT])
        startT = i


