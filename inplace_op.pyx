cimport cython
from cython.parallel import parallel, prange
from libc.stdlib cimport abort, malloc, free
from libc.stdio cimport printf
from libc.math cimport exp
import numpy as _N
cimport numpy as _N

def multiply_all(ndarr, mlt):
    shp = ndarr.shape

    
