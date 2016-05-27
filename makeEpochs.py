#  in terms of relative length
firstT   = 3.1
minT     = 0.5
maxT     = 1.5

epochs   = 18

intvs = _N.empty(epochs+1)

eqI   = False
t     = 0
intvs[0] = 0
for epc in xrange(epochs):
    if epc == 0:
        t += firstT
    else:
        t += firstT if eqI else (minT + (maxT - minT)*_N.random.rand())
    intvs[epc+1] = t

intvs /= intvs[-1]
iInd = 0
bFnd = False
while not bFnd:
    iInd += 1
    fn = "../DATA/itv%(e)d_%(iI)d.dat" % {"e" : epochs, "iI" : iInd}

    if not os.access(fn, os.F_OK):  # file exists
        bFnd = True
        _N.savetxt(fn, intvs, fmt="%.5f")

print "saved to %s" % fn
