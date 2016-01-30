import scipy.io as _scio

ex = 4-1;
ep = 4-1;

A = _scio.loadmat("bonripplescons04.mat")

#  these are in seconds
strt = A["ripplescons"][0, ex][0, ep][0, 0]["starttime"][0,0]
endt = A["ripplescons"][0, ex][0, ep][0, 0]["endtime"][0,0]
