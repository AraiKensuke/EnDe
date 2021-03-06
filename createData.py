#  position dependent firing rate
import os
import utilities as _U

#  Let's assume space goes from [0, 3]

Lx     = 0
Hx     = 3

UNIF   = 1
NUNIF  = 2

mvPat = NUNIF

#  movement patterns
#  non-uniform
#  uniform

N      = 4500

RTs   = 30      # round trips
Ns     = _N.empty(RTs, dtype=_N.int)
if mvPat == NUNIF:
    for rt in xrange(RTs):
        #Ns[rt] = int(N*(1 + 0.25*_N.random.randn()))
        Ns[rt] = N*(0.15 + 0.85*_N.random.rand())
else:
    Ns[:] = N

NT     = _N.sum(Ns)
pths    = _N.empty(NT)

##########  PLACE FIELD PARAMS
mx    = 1.1   #  mean of location
#  for simplicity, we allow the mean of location to vary sinusoidally
frqmx = 2.3
Amx   = 0.4#  mx + Amx*_N.sin(2*_N.pi*frqmx*_N.linspace(0, 1, len(pth)))

##########  POSITION BIAS PARAMS
frqsx = 1.3
msx   = 0.
sxt   = 0.15*(1 + msx*_N.cos(2*_N.pi*frqsx*_N.linspace(0, 1, len(pths))))
sx    = sxt**2     #  var of firing rate function

frql0 = 1.2
ml0   = 0.
l0   = 15*(1 + ml0*_N.cos(2*_N.pi*frql0*_N.linspace(0, 1, len(pths))))
f     = l0/_N.sqrt(2*_N.pi*sx)


#  for NUNIF, the mean makes sinusoidal drift movement
mSM   = 0.4     #  
frq   = 0.4

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
ctr  = mx + mSM*_N.sin(2*_N.pi*frq*_N.linspace(0, 1, len(pths)))
Lam   = f*dt*_N.exp(-0.5*(pths-ctr)**2 / sx)

rnds = _N.random.rand(NT)

sts  = _N.where(rnds < Lam)[0]
nts  = _N.setdiff1d(_N.arange(NT), sts)

bFnd  = False

##  us un   uniform sampling of space, stationary or non-stationary place field
##  ns nn   non-uni sampling of space, stationary or non-stationary place field
##  bs bb   biased and non-uni sampling of space
bfn  = "u" if (mvPat == UNIF) else "n"
if (mvPat == NUNIF) and (Amx > 0):
    bfn = "b"

bfn += "n" if (mSM > 0) else "s"

iInd = 0
while not bFnd:
    iInd += 1
    fn = "../DATA/%(bfn)s%(iI)d.dat" % {"bfn" : bfn, "iI" : iInd}
    fnocc="../DATA/%(bfn)s%(iI)docc.png" % {"bfn" : bfn, "iI" : iInd}

    if not os.access(fn, os.F_OK):  # file exists
        bFnd = True
        
dat = _N.zeros((NT, 5))
dat[:, 0] = pths
dat[sts, 1] = 1
dat[:, 2] = ctr
dat[:, 3] = sx
dat[:, 4] = l0

com = "#  f=%(mx).3f   q2=%(q2).4f   l0=%(l0).4f" % {"mx" : mx, "q2" : _N.mean(sx), "l0" : _N.mean(l0)}
_U.savetxtWCom("%s" % fn, dat, fmt="%.4f %d %.4f %.4f %.4f", delimiter=" ", com=com)
print "created %s" % fn

fig = _plt.figure()
_plt.hist(dat[:, 0], bins=_N.linspace(0, 3, 61), color="black")
_plt.savefig(fnocc)
_plt.close()
