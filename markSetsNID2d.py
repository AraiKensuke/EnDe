import numpy as _N

def cellsSet1(mA=1):
    return [[[-3.5, ],     #  POS
             [0.9,  2, 5.5, 2.1  ],     #  MARK
             [0.12, ],  #  STDP
             [25*mA, ]],        #  ALPHA
            [[4.0,-4.1],     #  POS
             [5.9, 0.8, 1.3, 1.8],     #  MARK
             [0.12, 0.12],  #  STDP
             [25*mA, 25*mA]],        #  ALPHA
            [[-2.3, 2.2],     #  POS
             [4., 1.4, 2., 5.4],     #  MARK
             [0.12, 0.12],  #  STDP
             [25*mA, 25*mA]],        #  ALPHA
        ]        #  ALPHA

def cellsSet2(mA=1):
    return [[[3.5,],     #  POS
             [0.9,  2, 1.9, 2.6  ],     #  MARK
             [0.27, ],  #  STDP
             [33*mA, ]],        #  ALPHA
            [[-3.5,],     #  POS
             [1.9,  1, 3.9, 4.6  ],     #  MARK
             [0.27, ],  #  STDP
             [33*mA, ]],        #  ALPHA
        ]        #  ALPHA

def cellsSet3(mA=1):
    return [[[-3.5, ],     #  POS
             [0.9,  2, 5.5, 2.1  ],     #  MARK
             [0.18, ],  #  STDP
             [33*mA, ]],        #  ALPHA
            [[4.0,-4.1],     #  POS
             [5.9, 0.8, 1.3, 1.8],     #  MARK
             [0.12, 0.1],  #  STDP
             [30*mA, 22*mA]],        #  ALPHA
            [[1.4, -1.2],     #  POS
             [2.9, 1.4, 2.3, 4.8],     #  MARK
             [0.12, 0.1],  #  STDP
             [23*mA, 18*mA]],        #  ALPHA
            [[-1.0, 1.1],     #  POS
             [3.9, 5.4, 2.3, 1.9],     #  MARK
             [0.15, 0.14],  #  STDP
             [20*mA, 21*mA]],        #  ALPHA
            [[-0.7, ],     #  POS
             [1.9, 1.4, 4.3, 4.9],     #  MARK
             [0.11,],  #  STDP
             [22*mA, ]],        #  ALPHA
            [[1.7, ],     #  POS
             [4.9, 4.4, 2.3, 2.9],     #  MARK
             [0.11,],  #  STDP
             [22*mA, ]],        #  ALPHA
            [[-2.0, 1.9],     #  POS
             [4., 1.4, 2., 5.4],     #  MARK
             [0.15, 0.12],  #  STDP
             [20*mA, 21*mA]],        #  ALPHA
            [[-5.0, 5.1],     #  POS
             [1., 1.9, 5.3, 3.4],     #  MARK
             [0.15, 0.12],  #  STDP
             [15*mA, 14*mA]],        #  ALPHA

        ]        #  ALPHA

def cellsSet4(mA=1):
    return [[[-5.5, ],     #  POS
             [5.5,  2, 0.9, 2.1  ],     #  MARK
             [0.13, ],  #  STDP
             [30*mA, ]],        #  ALPHA
            [[5.2, ],     #  POS
             [2.4,  1, 1.3, 5.1  ],     #  MARK
             [0.12, ],  #  STDP
             [30*mA, ]],        #  ALPHA
            [[4.0,-4.1],     #  POS
             [5.2, 0.8, 1.3, 1.8],     #  MARK
             [0.12, 0.1],  #  STDP
             [29*mA, 22*mA]],        #  ALPHA
            [[1.4, -1.2],     #  POS
             [2.9, 1.4, 2.3, 4.8],     #  MARK
             [0.12, 0.1],  #  STDP
             [23*mA, 18*mA]],        #  ALPHA
            [[-1.0, 1.1],     #  POS
             [3.9, 5.4, 2.3, 1.9],     #  MARK
             [0.15, 0.14],  #  STDP
             [20*mA, 21*mA]],        #  ALPHA
            [[-0.7, ],     #  POS
             [1.9, 1.4, 4.3, 4.9],     #  MARK
             [0.11,],  #  STDP
             [22*mA, ]],        #  ALPHA
            [[1.7, ],     #  POS
             [4.1, 4.9, 2.3, 2.9],     #  MARK
             [0.11,],  #  STDP
             [22*mA, ]],        #  ALPHA
            [[-2.0, 1.9],     #  POS
             [4., 1.4, 2., 5.4],     #  MARK
             [0.15, 0.12],  #  STDP
             [20*mA, 21*mA]],        #  ALPHA
            [[-5.0, 5.1],     #  POS
             [1., 1.9, 5.3, 3.4],     #  MARK
             [0.15, 0.12],  #  STDP
             [15*mA, 14*mA]],        #  ALPHA

        ]        #  ALPHA

def cellsSet5(mA=1):
    return [[[-5.5, ],     #  POS
             [5.9,  2, 0.9, 2.1  ],     #  MARK
             [0.13, ],  #  STDP
             [15*mA, ]],        #  ALPHA
            [[5.2, ],     #  POS
             [2.4,  1, 1.3, 5.4  ],     #  MARK
             [0.12, ],  #  STDP
             [33*mA, ]],        #  ALPHA
            [[4.0,-4.1],     #  POS
             [5.2, 0.8, 1.3, 1.8],     #  MARK
             [0.12, 0.1],  #  STDP
             [21*mA, 22*mA]],        #  ALPHA
            [[1.4, -1.2],     #  POS
             [2.9, 1.4, 2.3, 4.8],     #  MARK
             [0.12, 0.1],  #  STDP
             [21*mA, 15*mA]],        #  ALPHA
            [[-1.0, 1.1],     #  POS
             [3.9, 5.4, 2.3, 1.9],     #  MARK
             [0.15, 0.14],  #  STDP
             [18*mA, 21*mA]],        #  ALPHA
            [[-0.7, ],     #  POS
             [1.9, 1.4, 4.3, 4.9],     #  MARK
             [0.11,],  #  STDP
             [20*mA, ]],        #  ALPHA
            [[1.7, ],     #  POS
             [4.1, 4.9, 2.3, 2.9],     #  MARK
             [0.11,],  #  STDP
             [22*mA, ]],        #  ALPHA
            [[2.9, ],     #  POS
             [1.1, 4.9, 5.3, 5.9],     #  MARK
             [0.11,],  #  STDP
             [22*mA, ]],        #  ALPHA
            [[-5.0, 5.1],     #  POS
             [1., 1.9, 5.3, 3.4],     #  MARK
             [0.15, 0.12],  #  STDP
             [15*mA, 14*mA]],        #  ALPHA
            [[-3.0, 2.9],     #  POS
             [4., 3.4, 5., 6.4],     #  MARK
             [0.15, 0.12],  #  STDP
             [18*mA, 19*mA]],        #  ALPHA

        ]        #  ALPHA


def hashSet1(mA=1):
    return [[[2.5,],     #  POS
             [0.8,  1.2,   0.8, 0.9],     #  MARK
             [2.2,],  #  STDP
             [3*mA,]],        #  ALPHA
            [[5.1,],     #  POS
            [1.,  1.1,   0.3, 0.8],     #  MARK
             [2.2,],  #  STDP
             [3*mA,]],        #  ALPHA
            [[-3.1,],     #  POS
             [1.,  0.6,   0.9, 0.3],     #  MARK
             [2.2,],  #  STDP
             [3*mA,]],        #  ALPHA
            [[3.5, -3.8,],     #  POS
             [0.8,  1.1,   1.2, 1.1],     #  MARK
             [2.3, 2.8],  #  STDP
             [4*mA, 3*mA]],        #  ALPHA
            [[1.5, -2.9,],     #  POS
             [1.1,  0.8,   0.8, 1.0],     #  MARK
             [2.5, 2.1],  #  STDP
             [4*mA, 3*mA]]]        #  ALPHA

def hashSet2(mA=1):
    return [[[2.5,],     #  POS
             [0.8,  1.3,   0.8, 0.9],     #  MARK
             [3.3,],  #  STDP
             [4*mA,]],        #  ALPHA
            [[5.1,],     #  POS
             [1.3,  0.4,   0.3, 1.2],     #  MARK
             [2.4,],  #  STDP
             [4*mA,]],        #  ALPHA
            [[-3.1,],     #  POS
             [1.,  0.6,   0.9, 0.3],     #  MARK
             [3.2,],  #  STDP
             [4*mA,]]]        #  ALPHA

def hashSet3(mA=1):
    return [[[2.5,],     #  POS
             [0.8,  1.2,   0.8, 0.9],     #  MARK
             [3.3,],  #  STDP
             [4*mA,]],        #  ALPHA
            [[5.1,],     #  POS
            [1.,  1.1,   0.3, 0.8],     #  MARK
             [2.4,],  #  STDP
             [4*mA,]],        #  ALPHA
            [[-3.1,],     #  POS
             [1.,  0.6,   0.9, 0.3],     #  MARK
             [2.8,],  #  STDP
             [5*mA,]],        #  ALPHA
            [[3.5, -3.8,],     #  POS
             [0.8,  1.2,   1.2, 1.1],     #  MARK
             [2.3, 2.8],  #  STDP
             [4*mA, 3*mA]],        #  ALPHA
            [[1.5, -2.9,],     #  POS
             [1.1,  0.8,   0.8, 1.0],     #  MARK
             [2.5, 2.1],  #  STDP
             [4*mA, 3.4*mA]]]        #  ALPHA

def hashSet4(mA=1):
    return [[[2.5,],     #  POS
             [0.3,  1.4,   0.8, 0.9],     #  MARK
             [1.8,],  #  STDP
             [4.5*mA,]],        #  ALPHA
            [[4.1,],     #  POS
             [1.2,  0.5,   1.3, 0.3],     #  MARK
             [1.9,],  #  STDP
             [4.6*mA,]],        #  ALPHA
            [[-2.1,],     #  POS
             [0.3,  0.6,   0.3, 1.5],     #  MARK
             [2.1,],  #  STDP
             [5.4*mA,]],        #  ALPHA
            [[-4.9,],     #  POS
             [1.1,  1.4,   1.8, 1.0],     #  MARK
             [2.1],  #  STDP
             [4.4*mA,]]]        #  ALPHA
def buildSet(mND, cells_nh, cells_h, mA=1):
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
    Mh_strt  = 0

    for cll in cells_nh:
        up0l.extend(cll[0])
        stdP.extend(cll[2])
        alp.extend(cll[3])
        for m in xrange(len(cll[0])):
            markIDs.append(M)
            mks.append(cll[1])
        M += len(cll[0])

    Mh_strt  = M
    for cll in cells_h:
        up0l.extend(cll[0])
        stdP.extend(cll[2])
        alp.extend(cll[3])
        for m in xrange(len(cll[0])):
            markIDs.append(M)
            mks.append(cll[1])
        M += len(cll[0])

    mND.M          = M
    mND.Mh_strt    = Mh_strt
    mND.uP0        = _N.array(up0l)
    mND.stdPMags   = _N.array(stdP)
    mND.alp0       = _N.array(alp)
    mND.um0        = _N.array(mks)
    mND.markIDs    = _N.array(markIDs)
    

