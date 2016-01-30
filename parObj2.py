import numpy as np
import multiprocessing as mp

class Tester:
    num = 0.0
    def __init__(self, tnum=num):
        self.num  = tnum
        print self.num

    def modme(self, nn):
        self.num += nn
        print self.num
        #return self
        return (nn+5)

def modhelp(test, nn):
    test.modme(nn)
    #out_queue.put(test)
    return test


N = 2
p = mp.Pool(processes=N)

tts = _N.empty(N, dtype=object)
for nt in xrange(N):
    tts[nt] = Tester(tnum=nt)

results = _N.empty(N, dtype=object)
for nt in xrange(N):
    print "doing this"
    results[nt] = p.apply_async(modhelp, args=(tts[nt], nt+5,))
