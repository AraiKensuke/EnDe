import numpy
import sys
import os
from distutils.core import setup, Extension
from distutils.sysconfig import get_python_inc
from distutils.extension import Extension
#from Cython.Build import cythonize      # cythonize compiles a pyx
from Cython.Distutils import build_ext   # Extension for a c-file, build_ext for cython file


###  import LogitWrapper 
###  LogitWrapper
#  -undefined dynamic_lookup
#  -lgsl
#  -fpic   #  build a shared
#  -bundle

#  OMP_THREAD_NUM

#  use --user  to install in
#  to specify compiler, maybe set CC environment variable
#  or python setup.py build --compiler=g++
incdir1 = [get_python_inc(plat_specific=1), numpy.get_include(), "pyPG/include/RNG"]
os.environ["CC"]  = "g++"
os.environ["CXX"] = "g++"

##  Handle OPENMP switch here

#  http://stackoverflow.com/questions/677577/distutils-how-to-pass-a-user-defined-parameter-to-setup-py
USE_OPENMP = False
#  -fPIC meaningless in osx
#extra_compile_args = ["-fPIC", "-bundle", "-undefined dynamic_lookup", "-shared"]
extra_compile_args = ["-undefined dynamic_lookup", "-shared"]
#extra_link_args    = ["-lblas", "-llapack", "-lgsl"]
#  didn't need -llapack on Ubuntu
#extra_link_args    = ["-lblas", "-lgsl"]
extra_link_args    = ["-fopenmp"]

if "--use_openmp" in sys.argv:
    USE_OPENMP = True
    extra_compile_args.extend(["-fopenmp", "-DUSE_OPEN_MP"])
    extra_link_args.append("-lgomp")
    iop = sys.argv.index("--use_openmp")
    sys.argv.pop(iop)

#  may also need to set $LD_LIBRARY_PATH in order to use shared libgsl

cmdclass = {'build_ext' : build_ext}
#  Output to be named _LogitWrapper.so
ext_modules = Extension('par_intgrls',
                    ["par_intgrls.pyx"],
                    #libraries = ['gsl', 'gslcblas'],
                    #include_dirs=incdir1,
                    extra_compile_args=extra_compile_args,
                    extra_link_args=extra_link_args)  #  linker args
setup(
    name='par_intgrls',
    cmdclass = cmdclass,
    #ext_modules = 
    ext_modules=[ext_modules],
)
