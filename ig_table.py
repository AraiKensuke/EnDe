import scipy.stats as _ss
#  build a table of

#  a, B   B/(a-1) is mean.  B/(a+1) is mode

#  every 5
#  an array
#for a in _N.linspace(1.01, 40, 100)**2:
#    for B in _N.linspace(1.01, 40, 100)**2:
    
a0 = 100
Bl = 0.
Bh = 100000.
#  
#  If I request for IG(a, B), for a in [a1, a2], pull up 2 cdf for [a1, B] 
#  and [a2, B].  Then 

B = 51.

#  
mode_l = Bl/(a0+1)
mode_h = Bh/(a0+1)

mode   = B/(a0+1)

sg2_l = _N.linspace(1e-8, mode_l*20, 10000)
cdfl = _ss.invgamma.cdf(sg2_l, a0, scale=Bl)
sg2_h = _N.linspace(1e-8, mode_h*20, 10000)
cdfh = _ss.invgamma.cdf(sg2_h, a0, scale=Bh)

##  GT for intrapolated cdf.  for comparison.
sg2 = _N.linspace(1e-8, mode*20, 10000)
cdf = _ss.invgamma.cdf(sg2, a0, scale=B)

#  i want sg2 values for which cdf is 1e-5, 0.02, 0.04... for IG(a1, B1)

sg2_t = _N.linspace(0, 1, 51)    #  points at which CDF is sampled
xit[0] = 1e-5
xit[50] = 0.99999
sg2_l = _N.interp(xit, cdfl, sg2_l)  #  points at which CDF == values
sg2_h = _N.interp(xit, cdfh, sg2_h)  #  

_plt.plot(sg2_l, cdfl)
_plt.plot(sg2_l, xit, ls="", marker=".", ms=12)
_plt.plot(sg2_h, cdfh)
_plt.plot(sg2_h, xit, ls="", marker=".", ms=12)

#  GT cdf
_plt.plot(sg2, cdf)   
#  intrapolated cdf
#_plt.plot(sg2_t1*p+sg2_t2*(1-p), xit, ls="", marker=".", ms=12)


#  interface    ig(a, B)

