import numpy as np
import multiprocessing as mp

class Tester:
    def __init__(self, tnum=-1):
        self.num   = tnum
        self.num2  = 10*tnum

    def modme(self, nn, val2=None):
        self.num += nn
        if val2 is not None:
            print "Got non-None value for val2"
            self.num2 = val2
        #return self
        return (nn+5)

def modhelp(test, name, *args, **kwargs):
    callme = getattr(test, name)
    callme(*args, **kwargs)#, kwargs)
    return test

def modhelpSP(test, nn, name, **kwargs):
    callme = getattr(test, name)
    callme(nn, **kwargs)#, kwargs)


N = 2
p = mp.Pool(processes=N)

tts = _N.empty(N, dtype=object)
for nt in xrange(N):
    tts[nt] = Tester(tnum=nt)
    #modhelpSP(tts[nt], nt+5, "modme", val2=(nt*5))


results = _N.empty(N, dtype=object)
for nt in xrange(N):
    kwds = {"val2" : (nt*5+1)}
    results[nt] = p.apply_async(modhelp, args=(tts[nt], "modme", nt+5, ), kwds=kwds)

