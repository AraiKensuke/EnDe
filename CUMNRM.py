#  for a unit 0-mean normal distribution

import pickle 
W      = 6
Nu     = 200    #  how many to split unit space into
N      = 2*Nu*W+1   #  row are reserved for 0.
NRM    = _N.empty(N)
CUMNRM = _N.zeros((N, N))
xls = _N.linspace(-W, W, N)
xrs = _N.linspace(-W, W, N)

NRM    = (1./_N.sqrt(2*_N.pi))*_N.exp(-0.5*xls*xls)

Nuf    = float(Nu)
dx     = 1./Nu
il  = -1
for xl in xls:
    il += 1
    for ir in xrange(0, il):
        #CUMNRM[il, ir] = Nuf /(ir-il)*_N.sum(NRM[ir:il])*dx
        CUMNRM[il, ir] = _N.sum(NRM[ir:il])*dx
    for ir in xrange(il+1, N):
        #CUMNRM[il, ir] = Nuf/(il-ir)*_N.sum(NRM[il:ir])*dx
        CUMNRM[il, ir] = _N.sum(NRM[il:ir])*dx

dmp = open("CUMNRM.dump", "wb")
pcklme= {}
pcklme["mat"]   = CUMNRM
pcklme["width"] = W
pcklme["Nu"]    = Nu
pickle.dump(pcklme, dmp, -1)
dmp.close()

