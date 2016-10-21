import time as _tm
import os

#from par_intgrls_f  import M_times_N_f_intgrls, M_times_N_f_intgrls_noOMP, M_times_N_f_intgrls_noOMP_raw, M_times_N_f_intgrls_pure, M_times_N_f_intgrls_raw, M_times_N_f_intgrls_raw_no_func
from par_intgrls_f  import M_times_N_f_intgrls, M_times_N_f_intgrls_raw
M = 40

os.system("taskset -p 0xffffffff %d" % os.getpid())

#  Fill the U

Nupx  = 300
fss   = 300
U     = _N.random.randn(M)
ux    = _N.linspace(-6, 6, Nupx)
uxrr= ux.reshape((1, 1, Nupx))
FQ2   = 0.1+_N.random.rand(M)*0.3
q2    = 0.1+_N.random.rand(M)*0.3
iiq2  = 1./q2
iiq2rr= iiq2.reshape((M, 1, 1))

px    = _N.sin(2*_N.pi*_N.linspace(0, 1, Nupx)) + 0.3
pxrr     = px.reshape((1, 1, Nupx))
FQ    = _N.sqrt(FQ2)
Ur    = U.reshape((M, 1))
FQr   = FQ.reshape((M, 1))
FQ2r  = FQ2.reshape((M, 1))
dSilenceX  = 0.1

_fxs0 = _N.tile(_N.linspace(0, 1, fss), M).reshape(M, fss)
fxs  = _N.copy(_fxs0)
fxs *= (FQr*30)
fxs -= (FQr*15)
fxs += Ur

###
nThrds = 4
# t1 = _tm.time()
# f_exp_px_noomp = _N.empty((M, fss))
# M_times_N_f_intgrls_noOMP(fxs, ux, iiq2, dSilenceX, px, f_exp_px_noomp, M, fss, Nupx, nThrds)
# t2 = _tm.time()

# f_exp_px_noomp_r = _N.empty((M, fss))
# M_times_N_f_intgrls_noOMP_raw(fxs, ux, iiq2, dSilenceX, px, f_exp_px_noomp_r, M, fss, Nupx, nThrds)
t3 = _tm.time()

f_exp_px_omp = _N.empty((M, fss))
M_times_N_f_intgrls(fxs, ux, iiq2, dSilenceX, px, f_exp_px_omp, M, fss, Nupx, nThrds)
t4 = _tm.time()

f_exp_px_omp_r = _N.empty((M, fss))
M_times_N_f_intgrls_raw(fxs, ux, iiq2, dSilenceX, px, f_exp_px_omp_r, M, fss, Nupx, nThrds)
t5 = _tm.time()


# f_exp_px_omp_r_nf = _N.empty((M, fss))
# M_times_N_f_intgrls_raw_no_func(fxs, ux, iiq2, dSilenceX, px, f_exp_px_omp_r_nf, M, fss, Nupx, nThrds)
t6 = _tm.time()

# f_exp_px_pure = _N.empty((M, fss))
# M_times_N_f_intgrls_pure(fxs, ux, iiq2, dSilenceX, px, f_exp_px_omp, M, fss, Nupx, 8)
#t5 = _tm.time()

fxsr     = fxs.reshape((M, fss, 1))
fxrux = -0.5*(fxsr-uxrr)*(fxsr-uxrr)
#  f_intgrd    is M x fss x Nupx
f_intgrd  = _N.exp(fxrux*iiq2rr)   #  integrand
f_exp_px = _N.sum(f_intgrd*pxrr, axis=2) * dSilenceX
#  f_exp_px   is M x fss

t7 = _tm.time()


# print "noOMP"
# print (t2-t1)
# print "noOMP raw"
# print (t3-t2)
print "OMP"
print (t4-t3)
print "OMP raw"
print (t5-t4)
# print "OMP raw no func"
# print (t6-t5)
print "vectorized numpy"
print (t7-t6)

