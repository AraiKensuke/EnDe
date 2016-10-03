import time as _tm

from par_intgrls  import M_times_N_intrgrls
M = 40

#  Fill the U

Nupx  = 300
fss   = 60
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

t1 = _tm.time()
f_exp_px_omp = _N.empty((M, fss))
M_times_N_intrgrls(fxs, ux, iiq2, dSilenceX, px, f_exp_px_omp, M, fss, Nupx, 2)

t2 = _tm.time()

fxsr     = fxs.reshape((M, fss, 1))
fxrux = -0.5*(fxsr-uxrr)*(fxsr-uxrr)
#  f_intgrd    is M x fss x Nupx
f_intgrd  = _N.exp(fxrux*iiq2rr)   #  integrand
f_exp_px = _N.sum(f_intgrd*pxrr, axis=2) * dSilenceX
#  f_exp_px   is M x fss

t3 = _tm.time()

print (t2-t1)
print (t3-t2)

