import time as _tm
import os

#from par_intgrls  import M_times_N_q2_intgrls, M_times_N_q2_intgrls_noOMP, M_times_N_q2_intgrls_noOMP_raw, M_times_N_q2_intgrls_pure, M_times_N_q2_intgrls_raw, M_times_N_q2_intgrls_raw_no_func
#from par_intgrls  import M_times_N_intgrls_noOMP

from par_intgrls_q2 import M_times_N_q2_intgrls, M_times_N_q2_intgrls_raw
M = 62

#  Fill the U

nThrds  = 16
Nupx  = 300
q2ss   = 300
# U     = _N.random.randn(M)

f         = _N.random.rand(M)*12-6
frr       = f.reshape((M, 1, 1))
ux    = _N.linspace(-6, 6, Nupx)   #  range over which to integrate over space
uxrr= ux.reshape((1, 1, Nupx))
q2x    = _N.exp(_N.linspace(_N.log(1e-7), _N.log(100), q2ss))  #  5 orders of
iiq2xs     = 1./q2x
iq2xrr= iiq2xs.reshape((1, q2ss, 1))

px    = _N.sin(2*_N.pi*_N.linspace(0, 1, Nupx)) + 0.3  #  occupation
pxrr     = px.reshape((1, 1, Nupx))
dSilenceX  = 0.1


t3 = _tm.time()
q2_exp_px_omp = _N.empty((M, q2ss))
M_times_N_q2_intgrls(f, ux, iiq2xs, dSilenceX, px, q2_exp_px_omp, M, q2ss, Nupx, nThrds)
t4 = _tm.time()

q2_exp_px_omp_r = _N.empty((M, q2ss))
M_times_N_q2_intgrls_raw(f, ux, iiq2xs, dSilenceX, px, q2_exp_px_omp_r, M, q2ss, Nupx, nThrds)
t5 = _tm.time()

"""
f_exp_px_omp_r_nf = _N.empty((M, fss))
M_times_N_q2_intgrls_raw_no_func(fxs, ux, iiq2, dSilenceX, px, f_exp_px_omp_r_nf, M, fss, Nupx, nThrds)
t6 = _tm.time()

# f_exp_px_pure = _N.empty((M, fss))
# M_times_N_q2_intgrls_pure(fxs, ux, iiq2, dSilenceX, px, f_exp_px_omp, M, fss, Nupx, 8)
#t5 = _tm.time()
"""


t6 = _tm.time()

q2_intgrd = _N.exp(-0.5*(frr - uxrr)*(frr-uxrr) * iq2xrr)
q2_exp_px = _N.sum(q2_intgrd*pxrr, axis=2) * dSilenceX


t7 = _tm.time()


"""
print "noOMP"
print (t2-t1)
print "noOMP raw"
print (t3-t2)
"""
print "OMP"
print (t4-t3)

print "OMP raw"
print (t5-t4)
"""
print "OMP raw no func"
print (t6-t5)
"""
print "vectorized numpy"
print (t7-t6)

