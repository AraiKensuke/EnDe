import EnDedirs as _edd
#import pickle
import cPickle as _pkl

class ExperimentalMarkContainer:
    tetlist  = None
    pos      = None
    marks    = None

    Nx       = None
    Nm       = None
    xA       = None 
    mA       = None 
    k        = None 
    dt       = None
    
    anim     = "bond"
    day      = "4"
    ep       = "2"

    def __init__(self, anim="bond", day=4, ep=2):
        self.anim = anim
        self.day  = day
        self.ep   = ep
        
    def save(self):
        oo   = self
        
        print _edd.resFN("alltetmarks.pkl", dir="%(a)s0%(d)d0%(e)d" % {"a" : oo.anim, "d" : oo.day, "e" : oo.ep})
        dmp = open(_edd.resFN("alltetmarks.pkl", dir="%(a)s0%(d)d0%(e)d" % {"a" : oo.anim, "d" : oo.day, "e" : oo.ep}, create=True), "wb")
        _pkl.dump(oo, dmp, -1)
        dmp.close()
