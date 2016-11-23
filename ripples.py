import scipy.io as _sio

#  for bond day4, ex = 2, 4, 6
#  for day3, ex 
day    = 4
sdy    = ("0%d" % day) if (day < 10) else "%d" % day
ep=4-1;
ex=4-1;

anim1 = "bon"
anim2 = "bond"
anim3 = "Bon"

frip = "/home/karai/Dropbox (EastWestSideHippos)/BostonData/%(s3)s/%(s1)sripplescons%(sdy)s.mat" % {"s1" : anim1, "sdy" : sdy, "s3" : anim3}
flnp = "/home/karai/Dropbox (EastWestSideHippos)/BostonData/%(s3)s/%(s1)slinpos%(sdy)s.mat" % {"s1" : anim1, "sdy" : sdy, "s3" : anim3}

rip = _sio.loadmat(frip)    #  load matlab .mat files
mLp = _sio.loadmat(flnp)

#  these are in seconds
ex   = rip["ripplescons"].shape[1] - 1
strt = rip["ripplescons"][0, ex][0, ep][0, 0]["starttime"][0,0]
endt = rip["ripplescons"][0, ex][0, ep][0, 0]["endtime"][0,0]

r1s = _N.where((strt > 5170) & (strt < 5200))[0]
r2s = _N.where((strt > 4627) & (strt < 4645))[0]

strt[r1s]
strt[r2s]

pts=mLp["linpos"][0,ex][0,ep]["statematrix"][0][0]["time"][0,0].T[0]


##  our time starts at whatever create_mdec says is time 0
