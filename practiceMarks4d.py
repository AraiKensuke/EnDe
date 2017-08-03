import numpy as _N
import simUtil 

def cellsSet1(mA=1):
    return [[[1,],     #  POS
             [0.9,  3.2, 1.9, 0.6  ],     #  MARK
             [0.27, ],  #  STDP
             [33*mA, ]],        #  ALPHA
            [[-1,],     #  POS
             [3.9,  1, 3.9, 4.6  ],     #  MARK
             [0.27, ],  #  STDP
             [33*mA, ]],        #  ALPHA
            # [[2,],     #  POS
            #  [2.9,  1.1, 1.2, 3.6  ],     #  MARK
            #  [0.27, ],  #  STDP
            #  [33*mA, ]],        #  ALPHA
            # [[-2,],     #  POS
            #  [5.9,  1.9, 5.9, 1.6  ],     #  MARK
            #  [0.27, ],  #  STDP
            #  [33*mA, ]],        #  ALPHA
            [[4,],     #  POS
             [1.9,  6.1, 5.2, 1.6  ],     #  MARK
             [0.27, ],  #  STDP
             [33*mA, ]],        #  ALPHA
            [[-4,],     #  POS
             [4.9,  0.9, 1.9, 6.6  ],     #  MARK
             [0.27, ],  #  STDP
             [33*mA, ]],        #  ALPHA
            # [[5,],     #  POS
            #  [4.9,  5.9, 1.1, 3.2  ],     #  MARK
            #  [0.27, ],  #  STDP
            #  [33*mA, ]],        #  ALPHA
            # [[-5,],     #  POS
            #  [0.9,  6.9, 3.3, 2.6  ],     #  MARK
            #  [0.27, ],  #  STDP
            #  [33*mA, ]],        #  ALPHA
        ]        #  ALPHA


def cellsSet2(mA=1):
    return [[[2.3,],     #  POS
             [0.9,  2, 1.9, 2.6  ],     #  MARK
             [1.3, ],  #  STDP
             [33*mA, ]],        #  ALPHA
            [[-2.3,],     #  POS
             [1.9,  1, 3.9, 4.6  ],     #  MARK
             [1.3, ],  #  STDP
             [33*mA, ]],        #  ALPHA
        ]        #  ALPHA

def cellsSet3(mA=1):
    return [[[2.3,],     #  POS
             [0.9,  2, 1.9, 2.6  ],     #  MARK
             [1.3, ],  #  STDP
             [33*mA, ]],        #  ALPHA
            [[-2.3,],     #  POS
             [1.9,  1, 3.9, 4.6  ],     #  MARK
             [1.3, ],  #  STDP
             [33*mA, ]],        #  ALPHA
            [[3.8,],     #  POS
             [0.3,  0.9, 1.2, 2.1  ],     #  MARK
             [1.4, ],  #  STDP
             [35*mA, ]],        #  ALPHA
            [[-3.8,],     #  POS
             [1.1,  1.3, 0.8, 0.6  ],     #  MARK
             [1.4, ],  #  STDP
             [35*mA, ]],        #  ALPHA
        ]        #  ALPHA

def cellsSet4(mA=1):
    return [[[-3.5, ],     #  POS
             [0.9,  2, 7.8, 2.1  ],     #  MARK
             [0.18, ],  #  STDP
             [33*mA, ]],        #  ALPHA
            [[1.4, -1.2],     #  POS
             [2.9, 1.4, 4.3, 7.3],     #  MARK
             [0.12, 0.1],  #  STDP
             [23*mA, 18*mA]],        #  ALPHA
            [[-1.0, -5.],     #  POS
             [4.9, 6.9, 2.3, 1.9],     #  MARK
             [0.15, 0.14],  #  STDP
             [20*mA, 21*mA]],        #  ALPHA
            [[-0.7, ],     #  POS
             [1.9, 1.4, 7.2, 5.9],     #  MARK
             [0.11,],  #  STDP
             [22*mA, ]],        #  ALPHA
            [[1.7, ],     #  POS
             [4.9, 6.2, 1.3, 3.9],     #  MARK
             [0.11,],  #  STDP
             [22*mA, ]],        #  ALPHA
            [[2.0, 4.1],     #  POS
             [1., 6.5, 7.5, 3.4],     #  MARK
             [0.15, 0.12],  #  STDP
             [15*mA, 14*mA]],        #  ALPHA
        ]        #  ALPHA

def cellsSet5(mA=1):
    return [[[3.5, ],     #  POS
             [6.9,  2, 2.8, 2.1  ],     #  MARK
             [0.18, ],  #  STDP
             [33*mA, ]],        #  ALPHA
            [[2.4, -2.2],     #  POS
             [5.9, 1.4, 7.3, 4.3],     #  MARK
             [0.12, 0.1],  #  STDP
             [23*mA, 18*mA]],        #  ALPHA
            [[-1.0, -5.],     #  POS
             [3.9, 7.2, 1.9, 3.3],     #  MARK
             [0.15, 0.14],  #  STDP
             [20*mA, 21*mA]],        #  ALPHA
            [[-0.7, ],     #  POS
             [1.9, 7.4, 4.2, 5.9],     #  MARK
             [0.11,],  #  STDP
             [22*mA, ]],        #  ALPHA
            [[2.3, ],     #  POS
             [4.9, 6.2, 0.3, 4.9],     #  MARK
             [0.11,],  #  STDP
             [22*mA, ]],        #  ALPHA
            [[2.0, 4.1],     #  POS
             [1., 3, 7.5, 6.4],     #  MARK
             [0.15, 0.12],  #  STDP
             [15*mA, 14*mA]],        #  ALPHA
        ]        #  ALPHA

def cellsSet6(mA=1):
    return [[[-3.5, ],     #  POS
             [6.9,  2, 2.8, 2.1  ],     #  MARK
             [0.18, ],  #  STDP
             [33*mA, ]],        #  ALPHA
            [[2.4, -2.2],     #  POS
             [5.9, 1.4, 7.3, 4.3],     #  MARK
             [0.12, 0.1],  #  STDP
             [23*mA, 18*mA]],        #  ALPHA
            [[-1.5, -4.4],     #  POS
             [3.9, 7.2, 1.9, 3.3],     #  MARK
             [0.15, 0.14],  #  STDP
             [20*mA, 21*mA]],        #  ALPHA
            [[-0.9, ],     #  POS
             [1.9, 7.4, 4.2, 5.9],     #  MARK
             [0.11,],  #  STDP
             [22*mA, ]],        #  ALPHA
            [[2.9, ],     #  POS
             [4.9, 6.2, 0.3, 4.9],     #  MARK
             [0.11,],  #  STDP
             [22*mA, ]],        #  ALPHA
            [[2.0, 4.1],     #  POS
             [1., 3, 7.5, 6.4],     #  MARK
             [0.15, 0.12],  #  STDP
             [15*mA, 14*mA]],        #  ALPHA
        ]        #  ALPHA

def hashSet1(mA=1):
    return [[[-2.3,],     #  POS
             [0.9,  0.3, 0.4, 0.3  ],     #  MARK
             [2.9, ],  #  STDP
             [33*mA, ]],        #  ALPHA
            [[3.8,],     #  POS
             [0.3,  0.4, 1., 0.5  ],     #  MARK
             [2.9, ],  #  STDP
             [35*mA, ]],        #  ALPHA
        ]        #  ALPHA


def hashSet2(mA=1):
    return [[[-3.3,],     #  POS
             [0.9,  0.3, 0.4, 0.3  ],     #  MARK
             [2.9, ],  #  STDP
             [33*mA, ]],        #  ALPHA
            [[2.8,],     #  POS
             [0.3,  0.4, 1., 0.5  ],     #  MARK
             [2.9, ],  #  STDP
             [35*mA, ]],        #  ALPHA
        ]        #  ALPHA

