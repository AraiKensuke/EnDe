#  position dependent firing rate
#  discontinuous change of place field params
import os
import utilities as _U
from utils import createSmoothedPath

Ns     = _N.empty(RTs, dtype=_N.int)
if mvPat == NUNIF:

    for rt in xrange(RTs):
        Ns[rt] = N*((1-pT) + pT*_N.random.rand())
else:
    Ns[:] = N

NT     = _N.sum(Ns)
pths    = _N.empty(NT)

x01    = _N.linspace(0, 1, len(pths))
x01    = x01.reshape((1, NT))
plastic = False
##########  nonstationary center width
#  sxt  should be (M x NT)
sxt   = _N.empty((M, NT))
for m in xrange(M):  # sxts time scale
    sxt[m] = createSmoothedPath(sx_chpts[m], NT, sxts)
    if len(sx_chpts[m]) > 1:  plastic = True
    
sx    = sxt**2     #  var of firing rate function

##########  nonstationary center height l0
#  f is NT x M
l0   = _N.empty((M, NT))
for m in xrange(M):
    l0[m] = createSmoothedPath(l0_chpts[m], NT, l0ts)
    if len(l0_chpts[m]) > 1:  plastic = True

f     = l0/_N.sqrt(2*_N.pi*sx)   #  f*dt

##########  nonstationary center location
ctr  = _N.empty((M, NT))
for m in xrange(M):
    ctr[m] = createSmoothedPath(ctr_chpts[m], NT, ctrts)
    if len(ctr_chpts[m]) > 1:  plastic = True

if mvPat == NUNIF:
    now = 0
    for rt in xrange(RTs):
        N = Ns[rt]    #  each traverse slightly different duration
        rp  = _N.random.rand(N/100)
        x     = _N.linspace(Lx, Hx, N)
        xp     = _N.linspace(Lx, Hx, N/100)

        r   = _N.interp(x, xp, rp)       #  creates a velocity vector
        #  create movement without regard for place field
        r += Amx*(1.1+_N.sin(2*_N.pi*_N.linspace(0, 1, N, endpoint=False)*frqmx))
        pth = _N.zeros(N+1)
        for n in xrange(1, N+1):
            pth[n] = pth[n-1] + r[n-1]

        pth   /= (pth[-1] - pth[0])
        pth   *= (Hx-Lx)
        pth   += Lx

        pths[now:now+N]     = pth[0:N]
        now += N
else:
    now = 0
    x = _N.linspace(Lx, Hx, N)
    for rt in xrange(RTs):
        N = Ns[rt]
        pths[now:now+N]     = x
        now += N

###  now calculate firing rates
dt   = 0.001
fdt  = f*dt
#  change place field location
Lam   = f*dt*_N.exp(-0.5*(pths-ctr)**2 / sx)

rnds = _N.random.rand(NT)

#dat = _N.zeros((NT, 2 + K))
dat = _N.zeros((NT, 2))
dat[:, 0] = pths

for m in xrange(M):
    sts  = _N.where(rnds < Lam[m])[0]
    dat[sts, 1] = 1
    

bFnd  = False

##  us un   uniform sampling of space, stationary or non-stationary place field
##  ns nn   non-uni sampling of space, stationary or non-stationary place field
##  bs bb   biased and non-uni sampling of space

bfn     = "" if (M == 1) else ("%d" % M)

if mvPat == UNIF:
    bfn += "u"
else:
    bfn += "b" if (Amx > 0) else "n"

bfn += "n" if plastic else "s"

iInd = 0
while not bFnd:
    iInd += 1
    fn = "../DATA/%(bfn)s%(iI)d.dat" % {"bfn" : bfn, "iI" : iInd}
    fnocc="../DATA/%(bfn)s%(iI)docc.png" % {"bfn" : bfn, "iI" : iInd}
    fnprm = "../DATA/%(bfn)s%(iI)d_prms.dat" % {"bfn" : bfn, "iI" : iInd}

    if not os.access(fn, os.F_OK):  # file exists
        bFnd = True
        
_N.savetxt("%s" % fn, dat, fmt="%.4f %d", delimiter=" ")

prms = _N.zeros((NT, M*3))

for m in xrange(M):
    prms[:, m*3]   = ctr[m]
    prms[:, m*3+1] = sx[m]
    prms[:, m*3+2] = l0[m]

_N.savetxt("%s" % fnprm, prms, fmt=("%.4f %.4f %.4f " * M), delimiter=" ")

print "created %s" % fn

fig = _plt.figure()
_plt.hist(dat[:, 0], bins=_N.linspace(0, 3, 61), color="black")
_plt.savefig(fnocc)
_plt.close()

