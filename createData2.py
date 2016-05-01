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
M      = 2    # # of place fields

##########  nonuniformity in movement
#  for simplicity, we allow the mean of location to vary sinusoidally
frqmx = 2.3
Amx   = 0.4
pT    = 0.85  # variability in sweep time

RTs   = 30      # round trips
Ns     = _N.empty(RTs, dtype=_N.int)
if mvPat == NUNIF:

    for rt in xrange(RTs):
        #Ns[rt] = int(N*(1 + 0.25*_N.random.randn()))
        Ns[rt] = N*((1-pT) + pT*_N.random.rand())
else:
    Ns[:] = N

NT     = _N.sum(Ns)
pths    = _N.empty(NT)

x01    = _N.linspace(0, 1, len(pths))
x01    = x01.reshape((1, NT))
##########  nonstationary center width
frqsx = _N.array([[1.3], [0.8]])     # frequency of variation
asx   = _N.array([[0.1], [0.1]])     # amplitude of variation
msx   = _N.array([[0.15], [0.19]])   # mean value
#  sxt  should be (M x NT)
sxt   = msx*(1 + asx*_N.cos(2*_N.pi*frqsx*x01))
sx    = sxt**2     #  var of firing rate function


##########  nonstationary center height l0
frql0 = _N.array([[0.8], [0.7]])
al0   = _N.array([[0.15], [0.15]])
ml0   = _N.array([[15.], [13.]])
l0    = ml0*(1 + al0*_N.cos(2*_N.pi*frql0*x01))
#  f is NT x M
f     = l0/_N.sqrt(2*_N.pi*sx)


##########  nonstationary center location
frq   = _N.array([[0.8], [0.7]])
aCL   = _N.array([[0.2], [0.4]])   #  mean of location
mCL   = _N.array([[0.4], [1.8]])   #  mean of location
ctr  = mCL*(1 + aCL*_N.sin(2*_N.pi*frq*x01))

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
Lam   = _N.sum(f*dt*_N.exp(-0.5*(pths-ctr)**2 / sx), axis=0)

rnds = _N.random.rand(NT)

sts  = _N.where(rnds < Lam)[0]
nts  = _N.setdiff1d(_N.arange(NT), sts)

bFnd  = False

##  us un   uniform sampling of space, stationary or non-stationary place field
##  ns nn   non-uni sampling of space, stationary or non-stationary place field
##  bs bb   biased and non-uni sampling of space

bfn     = "" if (M == 1) else ("%d" % M)

if mvPat == UNIF:
    bfn += "u"
else:
    bfn += "b" if (Amx > 0) else "n"

bfn += "n" if ((_N.sum(aCL) > 0) or (_N.sum(asx) > 0) or (_N.sum(al0) > 0)) else "s"

iInd = 0
while not bFnd:
    iInd += 1
    fn = "../DATA/%(bfn)s%(iI)d.dat" % {"bfn" : bfn, "iI" : iInd}
    fnocc="../DATA/%(bfn)s%(iI)docc.png" % {"bfn" : bfn, "iI" : iInd}
    fnprm = "../DATA/prms%(iI)d.dat" % {"bfn" : bfn, "iI" : iInd}

    if not os.access(fn, os.F_OK):  # file exists
        bFnd = True
        
dat = _N.zeros((NT, 2))
dat[:, 0] = pths
dat[sts, 1] = 1

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

