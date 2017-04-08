#!/usr/bin/env python
import numpy as _N
pairs = 30

firstEnc = 80000
minEnc = 40000
maxEnc = 60000

minDec = 20000
maxDec = 30000

tC     = 0
str = "_N.array([\n"
for i in xrange(pairs):
    t1 = tC

    if i == 0:
        t2 = firstEnc
    else:
        t2 = int(tC + minEnc + (maxEnc - minEnc)*_N.random.rand())

    t3 = int(t2 + minDec + (maxDec - minDec)*_N.random.rand())
    tC = t3

    str += "[%(1)7d, %(2)7d], [%(2)7d, %(3)7d],\n" % {"1" : t1, "2" : t2, "3" : t3}

str += "])"

print str
