import time as _tm
from hc_bcast import hc_bcast1, hc_bcast1_par, hc_qdr_sum
#  should we replace broadcasting in numpy?  

M  = 50
N  = 20000

fr   = _N.random.randn(M, 1)
xASr = _N.random.randn(1, N)
iq2r = _N.random.randn(M, 1)

qdrSpc2= _N.empty((M, N))
qdrSpc3= _N.empty((M, N))
t1   = _tm.time()   
qdrSpc1 = (fr - xASr)*(fr - xASr)*iq2r
t2   = _tm.time()   
hc_bcast1(fr, xASr, iq2r, qdrSpc2, M, N)
t3   = _tm.time()   
hc_bcast1_par(fr, xASr, iq2r, qdrSpc3, M, N)
t4   = _tm.time()   

print (t2-t1)
print (t3-t2)
print (t4-t3)


pkFRr  = _N.random.randn(M, 1)
mkNrms = _N.random.randn(M, 1)
qdrMKS = _N.random.randn(M, N)

cont2  = _N.empty((M, N))
    #   (Mx1) + (Mx1) - (MxN + MxN)
t1     = _tm.time()
cont1       = pkFRr + mkNrms - 0.5*(qdrSpc1 + qdrMKS)
t2     = _tm.time()
hc_qdr_sum(pkFRr, mkNrms, qdrSpc1, qdrMKS, cont2, M, N)
t3     = _tm.time()
    
print (t2-t1)
print (t3-t2)
