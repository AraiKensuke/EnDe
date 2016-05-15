"""
read an existing epoch file, 
"""
saveDir   = "../Results/sim2"

fn        = "epochsLL"
epochs    = _N.loadtxt("%(sd)s/%(fn)s.txt" % {"sd" : saveDir, "fn" : fn}, dtype=_N.int)

nSplit    = 15  #  split the encode epochs into smaller pieces
sepochs   = []
rats      = 1.5+_N.random.rand(nSplit)
crats     = _N.zeros(nSplit + 1)

for ii in xrange(0, epochs.shape[0]-1, 2):
    t0    = epochs[ii, 0]
    t1    = epochs[ii, 1]

    print "%(1)d   %(2)d" % {"1" : t0, "2" : t1}

    if ii > 0:
        for i in xrange(1, nSplit+1):
            crats[i] = crats[i-1] + rats[i-1]
        
        crats /= crats[nSplit]

        ts = (t1-t0)*crats

        for n in xrange(nSplit):
            sepochs.append([int(ts[n] + t0), int(ts[n+1] + t0)])

    else:
        sepochs.append([t0, t1])
    sepochs.append(epochs[ii+1])
    
_N.savetxt("%(sd)s/%(fn)s_%(ne)d.txt" % {"sd" : saveDir, "ne" : nSplit, "fn" : fn}, _N.array(sepochs), fmt="%d %d")
