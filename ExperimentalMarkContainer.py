import EnDedirs as _edd
import pickle

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

    def save(self, ep):
        oo   = self
        
        dmp = open(_edd.resFN("alltetmarks.pkl", dir="bond040%d" % ep, create=True), "wb")
        pickle.dump(oo, dmp, -1)
        dmp.close()
