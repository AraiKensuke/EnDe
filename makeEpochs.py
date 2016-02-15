outDir   = "../Results/wtrackS2"

firstT   = 120000
minT     = 45000
maxT     = 80000

totalT   = 1000000
mmT      = maxT - minT

t0        = 0
t1        = firstT + int(_N.random.rand()*mmT)

lencIntvs = []
lencIntvs.append([t0, t1])

while t1 < totalT:

    t0 = t1
    t1        = t0 + minT + int(_N.random.rand()*mmT)
    
    if t1 >= totalT:
        t1 = totalT
    lencIntvs.append([t0, t1])

_N.savetxt("%s/epochs.txt" % outDir, _N.array(lencIntvs), fmt="%d %d")
