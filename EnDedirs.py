import os

"""
def resFN(fn, dir=None, create=False):
    #  ${EnDedir}/Results/
    __EnDeResultDir__ = os.environ["__EnDeResultDir__"]
    rD = __EnDeResultDir__

    if dir != None:
        rD = "%(rd)s/%(ed)s" % {"rd" : __EnDeResultDir__, "ed" : dir}
        if not os.access("%s" % rD, os.F_OK) and create:
            os.mkdir(rD)
    return "%(rd)s/%(fn)s" % {"rd" : rD, "fn" : fn}
"""

def resFN(fn, dir=None, create=False):
    #  ${EnDedir}/Results/
    __EnDeResultDir__ = os.environ["__EnDeResultDir__"]
    rD = __EnDeResultDir__

    if dir != None:
        lvls = dir.split("/")
        for lvl in lvls:
            rD += "/%s" % lvl
            if not os.access("%s" % rD, os.F_OK) and create:
                os.mkdir(rD)
    return "%(rd)s/%(fn)s" % {"rd" : rD, "fn" : fn}

def resFN(fn, dir=None, create=False):
    #  ${EnDedir}/Results/
    __EnDeResultDir__ = os.environ["__EnDeResultDir__"]
    rD = __EnDeResultDir__

    if dir != None:
        lvls = dir.split("/")
        for lvl in lvls:
            rD += "/%s" % lvl
            if not os.access("%s" % rD, os.F_OK) and create:
                os.mkdir(rD)
    return "%(rd)s/%(fn)s" % {"rd" : rD, "fn" : fn}

def pracFN(fn, dir=None, create=False):
    #  ${EnDedir}/Results/
    __EnDePracDir__ = os.environ["__EnDePracDir__"]
    rD = __EnDePracDir__

    if dir != None:
        lvls = dir.split("/")
        for lvl in lvls:
            rD += "/%s" % lvl
            if not os.access("%s" % rD, os.F_OK) and create:
                os.mkdir(rD)
    return "%(rd)s/%(fn)s" % {"rd" : rD, "fn" : fn}

def prcmpFN(fn, dir=None, create=False):
    #  ${EnDedir}/Results/
    __EnDePrecompDir__ = os.environ["__EnDePrecompDir__"]
    pD = __EnDePrecompDir__

    if dir != None:
        pD = "%(rd)s/%(ed)s" % {"rd" : __EnDePrecompDir__, "ed" : dir}
        if not os.access("%s" % pD, os.F_OK) and create:
            os.mkdir(pD)
    return "%(rd)s/%(fn)s" % {"rd" : pD, "fn" : fn}

def datFN(fn, dir=None, create=False):
    #  ${EnDedir}/Results/
    __EnDeDataDir__ = os.environ["__EnDeDataDir__"]
    dD = __EnDeDataDir__

    if dir != None:
        lvls = dir.split("/")
        for lvl in lvls:
            dD += "/%s" % lvl
            if not os.access("%s" % dD, os.F_OK) and create:
                os.mkdir(dD)
    return "%(dd)s/%(fn)s" % {"dd" : dD, "fn" : fn}


