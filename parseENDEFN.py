import inspect
import re

def parseENDEFN(baseFN):
    """
    (np)/(wp)_tr0-tr1_Cn_R
    """
    FN         = baseFN.split("/")[-1]
    #p          = re.compile("(\w+)\-([\w\d]+)\-([\w\d]+)")
    p          = re.compile("([\w\d]+)\-([\w\d]+)")
    m = p.match(FN)
    
    return m.group(1), m.group(2), m.group(3)

