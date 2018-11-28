import numpy
import sys
import os
from distutils.core import setup, Extension
from distutils.sysconfig import get_python_inc
from distutils.extension import Extension
#from Cython.Build import cythonize      # cythonize compiles a pyx
from Cython.Distutils import build_ext   # Extension for a c-file, build_ext for cython file

#modules = ["hc_bcast", "par_intgrls_f", "par_intgrls_q2", "fastnum"]
#modules = ["raw_random_access"]
#modules = ["hc_bcast", "GoFfuncs_pyx", "GoF_util"]
modules = ["hc_bcast"]
modules = ["fastnum"]
#modules  = ["GoFfuncs_pyxKDE"]
#modules  = ["GoF_util", "GoF_util_2", "GoF_util_4"]
#modules  = ["cdf_smp_sumx_2d"]
#modules  = ["cdf_smp_2d_apprx"]
#modules  = ["anocc2"]
#modules  = ["compress_gz_pyx"]

#modules = ["fastnum", "hc_bcast", "GoFfuncs_pyxKDE", "GoF_util", "GoF_util_2", "GoF_util_4"]

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
incdir = [get_python_inc(plat_specific=1), numpy.get_include(), "pyPG/include/RNG"]
libdir = ['/usr/local/lib/gcc/6', '/usr/local/lib']
os.environ["CC"]  = "gcc-6"
os.environ["CXX"] = "gcc-6"

##  Handle OPENMP switch here

#  http://stackoverflow.com/questions/677577/distutils-how-to-pass-a-user-defined-parameter-to-setup-py
USE_OPENMP = False
#  -fPIC meaningless in osx
#extra_compile_args = ["-fPIC", "-bundle", "-undefined dynamic_lookup", "-shared"]
#extra_compile_args = ["-undefined dynamic_lookup", "-shared"]
extra_compile_args = []
#extra_link_args    = ["-lblas", "-llapack", "-lgsl"]
#  didn't need -llapack on Ubuntu
extra_link_args    = ["-lblas", "-lgsl"]
#extra_link_args    = ["-fopenmp"]

if "--use_openmp" in sys.argv:
    USE_OPENMP = True
    extra_compile_args.extend(["-fopenmp", "-DUSE_OPEN_MP"])
    extra_link_args.append("-lgomp")
    iop = sys.argv.index("--use_openmp")
    sys.argv.pop(iop)

#  may also need to set $LD_LIBRARY_PATH in order to use shared libgsl

cmdclass = {'build_ext' : build_ext}
#  Output to be named _LogitWrapper.so

for module in modules:
    ext_modules = Extension(module,
                            ["%s.pyx" % module],
                            #libraries = ['gsl', 'gslcblas'],
                            include_dirs=incdir,   #  include_dirs for Mac
                            library_dirs=libdir,
                            extra_compile_args=extra_compile_args,
                            extra_link_args=extra_link_args)  #  linker args

    setup(
        name=module,
        cmdclass = cmdclass,
        #ext_modules = 
        ext_modules=[ext_modules],
    )
