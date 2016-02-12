import numpy as _N

####
#  Not many splitter cells, 
#  I want some cells that fire at 1.5 and -1,-5  same wf

def set2(mND, mA=1):
    #  M x Npf
    #  positions
    #  WF
    #  stdP
    #  alp
    cells          = [[[1.1, 5, -4.95, -1],     #  POS
                       [8,  3,   2.9, 2.3],     #  MARK
                       [0.1, 0.11, 0.12, 0.1],  #  STDP
                       [15*mA, 17*mA, 19*mA, 14*mA]],       #  ALPHA
                      [[0.8, -5.1],     #  POS
                       [3,  2,   5.9, 2.3],     #  MARK
                       [0.1, 0.11],  #  STDP
                       [19*mA, 18*mA]],       #  ALPHA
                      [[2.6, 3.4, -3.4, -2.6],     #  POS
                       [1.9,  3,   4.9, 6.8],     #  MARK
                       [0.17, 0.19, 0.19, 0.13],  #  STDP
                       [20*mA, 8*mA, 11*mA, 15*mA]],        #  ALPHA
                      [[2., -2.4,],     #  POS
                       [1.8,  7.5,   1.9, 2],     #  MARK
                       [0.17, 0.14],  #  STDP
                       [11*mA, 12*mA]],        #  ALPHA
                      [[-1.8],     #  POS
                       [4.3,  2,   5.3, 4.3],     #  MARK
                       [0.12],  #  STDP
                       [19*mA, ]],       #  ALPHA
                      [[1.1],     #  POS
                       [1.9,  7,   2.9, 3.3],     #  MARK
                       [0.1,],  #  STDP
                       [15*mA, ]],       #  ALPHA
                      #################  HASH
                      [[2.5,],     #  POS
                       [0.8,  1.3,   1.2, 1.8],     #  MARK
                       [2.2,],  #  STDP
                       [3*mA,]],        #  ALPHA
                      [[5.1,],     #  POS
                       [1.2,  1.8,   0.3, 0.8],     #  MARK
                       [2.2,],  #  STDP
                       [3*mA,]],        #  ALPHA
                      [[-3.1,],     #  POS
                       [1.8,  0.4,   0.9, 0.3],     #  MARK
                       [2.2,],  #  STDP
                       [3*mA,]],        #  ALPHA
                      [[3.5, -3.8,],     #  POS
                       [2.1,  1.8,   1.9, 1.1],     #  MARK
                       [2.3, 2.8],  #  STDP
                       [4*mA, 3*mA]],        #  ALPHA
                      [[1.5, -2.9,],     #  POS
                       [1.1,  2.8,   0.8, 1.0],     #  MARK
                       [2.5, 2.1],  #  STDP
                       [4*mA, 3*mA]]]        #  ALPHA

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

def set3(mND, mA=1):
    #  M x Npf
    #  positions
    #  WF
    #  stdP
    #  alp
    cells          = [[[1.1, 5, -4.95, -1],     #  POS
                       [6,  1,   1.9, 1.3],     #  MARK
                       [0.1, 0.11, 0.12, 0.1],  #  STDP
                       [15*mA, 17*mA, 19*mA, 14*mA]],       #  ALPHA
                      [[0.8, -5.1],     #  POS
                       [3,  2,   4.9, 2.3],     #  MARK
                       [0.1, 0.11],  #  STDP
                       [19*mA, 18*mA]],       #  ALPHA
                      [[2.6, 3.4, -3.4, -2.6],     #  POS
                       [1,  3,   4.9, 5.3],     #  MARK
                       [0.17, 0.19, 0.19, 0.13],  #  STDP
                       [20*mA, 8*mA, 11*mA, 15*mA]],        #  ALPHA
                      [[2., -2.4,],     #  POS
                       [0.8,  5,   1.9, 2],     #  MARK
                       [0.17, 0.14],  #  STDP
                       [11*mA, 12*mA]],        #  ALPHA
                      [[-1.8],     #  POS
                       [3,  2,   2.9, 4.3],     #  MARK
                       [0.12],  #  STDP
                       [19*mA, ]],       #  ALPHA
                      [[1.1],     #  POS
                       [1,  5,   2.9, 1.3],     #  MARK
                       [0.1,],  #  STDP
                       [15*mA, ]],       #  ALPHA
                      #################  HASH
                      [[2.5,],     #  POS
                       [0.8,  1.3,   1.2, 1.8],     #  MARK
                       [2.2,],  #  STDP
                       [3*mA,]],        #  ALPHA
                      [[5.1,],     #  POS
                       [1.2,  1.8,   0.3, 0.8],     #  MARK
                       [2.2,],  #  STDP
                       [3*mA,]],        #  ALPHA
                      [[-3.1,],     #  POS
                       [1.8,  0.4,   0.9, 0.3],     #  MARK
                       [2.2,],  #  STDP
                       [3*mA,]],        #  ALPHA
                      [[3.5, -3.8,],     #  POS
                       [2.1,  1.8,   1.9, 1.1],     #  MARK
                       [2.3, 2.8],  #  STDP
                       [4*mA, 3*mA]],        #  ALPHA
                      [[1.5, -2.9,],     #  POS
                       [1.1,  2.8,   0.8, 1.0],     #  MARK
                       [2.5, 2.1],  #  STDP
                       [4*mA, 3*mA]]]        #  ALPHA

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


def set4(mND, mA=1):
    #  M x Npf
    #  positions
    #  WF
    #  stdP
    #  alp
    cells          = [[[1.1, 5, -4.95, -1],     #  POS
                       [7,  1.5,   1.9, 1.3],     #  MARK
                       [0.1, 0.11, 0.12, 0.1],  #  STDP
                       [15*mA, 17*mA, 19*mA, 14*mA]],       #  ALPHA
                      [[0.8, -5.1],     #  POS
                       [3,  2,   5.9, 2.3],     #  MARK
                       [0.1, 0.11],  #  STDP
                       [19*mA, 18*mA]],       #  ALPHA
                      [[2.6, 3.4, -3.4, -2.6],     #  POS
                       [1,  3,   6.9, 5.3],     #  MARK
                       [0.17, 0.19, 0.19, 0.13],  #  STDP
                       [20*mA, 8*mA, 11*mA, 15*mA]],        #  ALPHA
                      [[2., -2.4,],     #  POS
                       [0.8,  6,   1.9, 2],     #  MARK
                       [0.17, 0.14],  #  STDP
                       [11*mA, 12*mA]],        #  ALPHA
                      [[-1.8],     #  POS
                       [3,  2,   2.9, 6.3],     #  MARK
                       [0.12],  #  STDP
                       [19*mA, ]],       #  ALPHA
                      [[1.1],     #  POS
                       [1,  6.5,   2.9, 1.3],     #  MARK
                       [0.1,],  #  STDP
                       [15*mA, ]],       #  ALPHA
                      #################  HASH
                      [[2.5,],     #  POS
                       [0.8,  1.3,   1.2, 1.8],     #  MARK
                       [2.2,],  #  STDP
                       [3*mA,]],        #  ALPHA
                      [[5.1,],     #  POS
                       [1.2,  1.8,   0.3, 0.8],     #  MARK
                       [2.2,],  #  STDP
                       [3*mA,]],        #  ALPHA
                      [[-3.1,],     #  POS
                       [1.8,  0.4,   0.9, 0.3],     #  MARK
                       [2.2,],  #  STDP
                       [3*mA,]],        #  ALPHA
                      [[3.5, -3.8,],     #  POS
                       [2.1,  1.8,   1.9, 1.1],     #  MARK
                       [2.3, 2.8],  #  STDP
                       [4*mA, 3*mA]],        #  ALPHA
                      [[1.5, -2.9,],     #  POS
                       [1.1,  2.8,   0.8, 1.0],     #  MARK
                       [2.5, 2.1],  #  STDP
                       [4*mA, 3*mA]]]        #  ALPHA

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



def set5(mND):   #  like set4, but with more spacing btwn cluster near x=0
    #  M x Npf
    #  positions
    #  WF
    #  stdP
    #  alp
    cells          = [[[1.5, 5, -4.95, -1.5],     #  POS
                       [7,  1.5,   1.9, 1.3],     #  MARK
                       [0.1, 0.11, 0.12, 0.1],  #  STDP
                       [19, 20, 19, 18]],       #  ALPHA
                      [[0.8, -5.1],     #  POS
                       [3,  2,   5.9, 2.3],     #  MARK
                       [0.1, 0.11],  #  STDP
                       [19, 18]],       #  ALPHA
                      [[2.6, 3.4, -3.4, -2.6],     #  POS
                       [1,  3,   6.9, 5.3],     #  MARK
                       [0.17, 0.19, 0.19, 0.13],  #  STDP
                       [20, 8, 11, 15]],        #  ALPHA
                      [[2., -2.4,],     #  POS
                       [0.8,  6,   1.9, 2],     #  MARK
                       [0.17, 0.14],  #  STDP
                       [11, 12]],        #  ALPHA
                      [[-2.1],     #  POS
                       [3,  2,   2.9, 6.3],     #  MARK
                       [0.12],  #  STDP
                       [19, ]],       #  ALPHA
                      [[0.3],     #  POS
                       [1,  6.5,   2.9, 1.3],     #  MARK
                       [0.1,],  #  STDP
                       [15, ]],       #  ALPHA
                      #################  HASH
                      [[2.5,],     #  POS
                       [0.8,  1.3,   1.2, 1.8],     #  MARK
                       [2.2,],  #  STDP
                       [3,]],        #  ALPHA
                      [[5.1,],     #  POS
                       [1.2,  1.8,   0.3, 0.8],     #  MARK
                       [2.2,],  #  STDP
                       [3,]],        #  ALPHA
                      [[-3.1,],     #  POS
                       [1.8,  0.4,   0.9, 0.3],     #  MARK
                       [2.2,],  #  STDP
                       [3,]],        #  ALPHA
                      [[3.5, -3.8,],     #  POS
                       [2.1,  1.8,   1.9, 1.1],     #  MARK
                       [2.3, 2.8],  #  STDP
                       [4, 3]],        #  ALPHA
                      [[1.5, -2.9,],     #  POS
                       [1.1,  2.8,   0.8, 1.0],     #  MARK
                       [2.5, 2.1],  #  STDP
                       [4, 3]]]        #  ALPHA

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





def set1(mND, mA=1):
    #  M x Npf
    #  positions
    #  WF
    #  stdP
    #  alp
    cells          = [[[1.1],     #  POS
                       [6,  1,   1.9, 1.3],     #  MARK
                       [0.1,],  #  STDP
                       [15*mA, ]],       #  ALPHA
                      [[-1.8],     #  POS
                       [3,  5,   2.9, 1.3],     #  MARK
                       [0.12],  #  STDP
                       [19*mA, ]],       #  ALPHA
                      [[0.8, -5.1],     #  POS
                       [3,  2,   4.9, 2.3],     #  MARK
                       [0.17, 0.11],  #  STDP
                       [19*mA, 18*mA]],       #  ALPHA
                      [[2.6, -3.4, -2.6],     #  POS
                       [4,  3,   4.9, 5.3],     #  MARK
                       [0.21, 0.19, 0.13],  #  STDP
                       [20*mA, 11*mA, 15*mA]],        #  ALPHA
                      [[2., -2.4,],     #  POS
                       [1.1,  5,   1.9, 2],     #  MARK
                       [0.19, 0.14],  #  STDP
                       [11*mA, 12*mA]],        #  ALPHA
                      [[-4.4,],     #  POS
                       [0.9,  2,   3.9, 4],     #  MARK
                       [0.17, ],  #  STDP
                       [13*mA, ]],        #  ALPHA

                      #################  HASH
                      [[2.5,],     #  POS
                       [0.8,  1.3,   1.2, 1.8],     #  MARK
                       [2.2,],  #  STDP
                       [3*mA,]],        #  ALPHA
                      [[5.1,],     #  POS
                       [1.2,  1.8,   0.3, 0.8],     #  MARK
                       [2.2,],  #  STDP
                       [3*mA,]],        #  ALPHA
                      [[-3.1,],     #  POS
                       [1.8,  0.4,   0.9, 0.3],     #  MARK
                       [2.2,],  #  STDP
                       [3*mA,]],        #  ALPHA
                      [[3.5, -3.8,],     #  POS
                       [2.1,  1.8,   1.9, 1.1],     #  MARK
                       [2.3, 2.8],  #  STDP
                       [4*mA, 3*mA]],        #  ALPHA
                      [[1.5, -2.9,],     #  POS
                       [1.1,  2.8,   0.8, 1.0],     #  MARK
                       [2.5, 2.1],  #  STDP
                       [4*mA, 3*mA]]]        #  ALPHA

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


def set1n(mND, mA=1):
    #  M x Npf
    #  positions
    #  WF
    #  stdP
    #  alp
    cells          = [[[1.1],     #  POS
                       [6,  1,   1.9, 1.3],     #  MARK
                       [0.15,],  #  STDP
                       [15*mA, ]],       #  ALPHA
                      [[-1.8],     #  POS
                       [3,  5,   2.9, 1.3],     #  MARK
                       [0.18],  #  STDP
                       [19*mA, ]],       #  ALPHA
                      [[0.8, -5.1],     #  POS
                       [3,  2,   4.9, 2.3],     #  MARK
                       [0.19, 0.18],  #  STDP
                       [19*mA, 18*mA]],       #  ALPHA
                      [[2.6, -3.4, -2.6],     #  POS
                       [4,  3,   4.9, 5.3],     #  MARK
                       [0.17, 0.19, 0.18],  #  STDP
                       [20*mA, 11*mA, 15*mA]],        #  ALPHA
                      [[2., -2.4,],     #  POS
                       [1.1,  5,   1.9, 2],     #  MARK
                       [0.19, 0.19],  #  STDP
                       [11*mA, 12*mA]],        #  ALPHA
                      [[-4.4,],     #  POS
                       [0.9,  2,   3.9, 4],     #  MARK
                       [0.18, ],  #  STDP
                       [13*mA, ]],        #  ALPHA

                      #################  HASH
                      [[2.5,],     #  POS
                       [0.8,  1.3,   1.2, 1.8],     #  MARK
                       [2.8,],  #  STDP
                       [3*mA,]],        #  ALPHA
                      [[5.1,],     #  POS
                       [1.2,  1.8,   0.3, 0.8],     #  MARK
                       [2.3,],  #  STDP
                       [3*mA,]],        #  ALPHA
                      [[-3.1,],     #  POS
                       [1.8,  0.4,   0.9, 0.3],     #  MARK
                       [2.8,],  #  STDP
                       [3*mA,]],        #  ALPHA
                      [[3.5, -3.8,],     #  POS
                       [2.1,  1.8,   1.9, 1.1],     #  MARK
                       [2.4, 2.8],  #  STDP
                       [4*mA, 3*mA]],        #  ALPHA
                      [[1.5, -2.9,],     #  POS
                       [1.1,  2.8,   0.8, 1.0],     #  MARK
                       [2.6, 2.1],  #  STDP
                       [4*mA, 3*mA]]]        #  ALPHA

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


def setSigOnly2(mND, mA=1):
    #  M x Npf
    #  positions
    #  WF
    #  stdP
    #  alp
    cells          = [[[-5.2,],     #  POS
                       [0.5,  0.2,   0.5, 1.8],     #  MARK
                       [0.27, ],  #  STDP
                       [13*mA, ]],
                      [[-4.4,],     #  POS
                       [0.9,  2,   3.9, 4],     #  MARK
                       [0.27, ],  #  STDP
                       [13*mA, ]],        #  ALPHA
                      [[-2.4,],     #  POS
                       [2.9,  1,   3.1, 0.8],     #  MARK
                       [0.27, ],  #  STDP
                       [13*mA, ]],        #  ALPHA
                      [[-1.8],     #  POS
                       [3,  5,   2.9, 1.3],     #  MARK
                       [0.27],  #  STDP
                       [13*mA, ]],       #  ALPHA
                      [[0.2,],     #  POS
                       [3,  2,   4.9, 2.3],     #  MARK
                       [0.27],  #  STDP
                       [13*mA,]],       #  ALPHA
                      [[1.1],     #  POS
                       [6,  1,   1.9, 1.3],     #  MARK
                       [0.27,],  #  STDP
                       [13*mA, ]],       #  ALPHA
                      [[2.,],     #  POS
                       [1.1,  5,   1.9, 2],     #  MARK
                       [0.27,],  #  STDP
                       [13*mA,]],        #  ALPHA
                      [[2.6,],     #  POS
                       [5,  1,   4.9, 1.3],     #  MARK
                       [0.27,],  #  STDP
                       [13*mA, ]],        #  ALPHA
                      [[3.6,],     #  POS
                       [1,  3,  2.9, 5.8],     #  MARK
                       [0.27,],  #  STDP
                       [13*mA, ]],        #  ALPHA
                      [[4.5,],     #  POS
                       [6,  6,   5.1, 4.8],     #  MARK
                       [0.27, ],  #  STDP
                       [13*mA, ]],
                      [[5.2,],     #  POS
                       [6,  1,   3.1, 5.8],     #  MARK
                       [0.27, ],  #  STDP
                       [13*mA, ]]]        #  ALPHA



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



def setSigOnly1(mND, mA=1):
    #  M x Npf
    #  positions
    #  WF
    #  stdP
    #  alp
    cells          = [[[-4.4,],     #  POS
                       [0.9,  2,   3.9, 4],     #  MARK
                       [0.27, ],  #  STDP
                       [13*mA, ]],        #  ALPHA
                      [[-2.9,],     #  POS
                       [2.9,  1,   3.1, 0.8],     #  MARK
                       [0.27, ],  #  STDP
                       [13*mA, ]],        #  ALPHA
                      [[-1.3],     #  POS
                       [3,  5,   2.9, 1.3],     #  MARK
                       [0.27],  #  STDP
                       [13*mA, ]],       #  ALPHA
                      [[1.3,],     #  POS
                       [1.1,  5,   1.9, 2],     #  MARK
                       [0.27,],  #  STDP
                       [13*mA,]],        #  ALPHA
                      [[2.8,],     #  POS
                       [5,  1,   4.9, 1.3],     #  MARK
                       [0.27,],  #  STDP
                       [13*mA, ]],        #  ALPHA
                      [[4.4,],     #  POS
                       [1,  3,  2.9, 5.8],     #  MARK
                       [0.27,],  #  STDP
                       [13*mA, ]],        #  ALPHA
    ]        #  ALPHA



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


