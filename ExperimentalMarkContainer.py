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

    def save(self):
        oo   = self
        dmp = open(_edd.resFN("marks.dump", dir="bond0402", create=True), "wb")
        pickle.dump(oo, dmp)
        dmp.close()
