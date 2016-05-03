import numpy as _N


def cellsSet1(mA=1):
    return [[[-3.5,],     #  POS
             [0.9,  2,   ],     #  MARK
             [0.27, ],  #  STDP
             [33*mA, ]],        #  ALPHA
    ]        #  ALPHA

def cellsSet2(mA=1):
    return [[[3.5,],     #  POS
             [0.9,  2,   ],     #  MARK
             [0.27, ],  #  STDP
             [33*mA, ]],        #  ALPHA
        ]        #  ALPHA

def cellsSet3(mA=1):
    return [[[-3.5,],     #  POS
             [0.9,  2,   ],     #  MARK
             [0.27, ],  #  STDP
             [33*mA, ]],        #  ALPHA
            [[4.0,],     #  POS
             [3.9, 0.8],     #  MARK
             [0.2,],  #  STDP
             [30*mA, ]],        #  ALPHA
            [[1.0,],     #  POS
             [2.9, 1.4],     #  MARK
             [0.4,],  #  STDP
             [23*mA, ]],        #  ALPHA
        ]        #  ALPHA

def buildSet(mND, cells, mA=1):
    #  M x Npf
    #  positions
    #  WF
    #  stdP
    #  alp

    alp  = []
    up0l = []
    mks  = []
    markIDs = []
    stdP = []
    M    = 0
    for cll in cells:
        up0l.extend(cll[0])
        stdP.extend(cll[2])
        alp.extend(cll[3])
        for m in xrange(len(cll[0])):
            markIDs.append(M)
            mks.append(cll[1])
        M += len(cll[0])

    mND.M          = M
    mND.uP0        = _N.array(up0l)
    mND.stdPMags   = _N.array(stdP)
    mND.alp0       = _N.array(alp)
    mND.um0        = _N.array(mks)
    mND.markIDs    = _N.array(markIDs)
    
