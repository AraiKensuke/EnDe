# 
def piecesfigs(ep, lstrans, tstrans, r, scxMin, scxMax, scyMin, scyMax):
    #  r is raw position
    ipg   = -1
    iPPG  = 8
    fig = _plt.figure(figsize=(13, 8))
    pg    = 1
    for ts in xrange(len(tsrans)-1):
        ipg += 1
        if (ipg % iPPG == 0) and (ipg != 0):
            if ts > 0:
                _plt.savefig("pieces_%(ep)d_%(pg)d" % {"ep" : (ep+1), "pg" : pg})
                _plt.close()
                pg += 1
            fig = _plt.figure(figsize=(13, 8))
            ipg = 0
        fig.add_subplot(4, 4, 2*ipg + 1)
        t0 = tsrans[ts];     t1 = tsrans[ts+1]
        _plt.plot(lindist[t0:t1])
        fig.add_subplot(4, 4, 2*ipg + 2)
        _plt.scatter(r[::40, 1], r[::40, 2], s=5, color="grey")
        _plt.scatter(r[t0:t1, 1], r[tsrans[ts]:tsrans[ts+1], 2], s=9, color="black")
        _plt.plot(r[t0, 1], r[t0, 2], ms=20, color="blue", marker=".")
        _plt.plot(r[t1, 1], r[t1, 2], ms=20, color="red", marker=".")
        _plt.xlim(scxMin, scxMax)
        _plt.ylim(scyMin, scyMax)

    _plt.savefig("pieces_%(ep)d_%(pg)d" % {"ep" : (ep+1), "pg" : pg})
    _plt.close()


def btwnfigs(ep, t0, t1, seg_ts, r, x, y):
    fig = _plt.figure(figsize=(9, 4))
    fig.add_subplot(2, 1, 1)
    _plt.plot(range(t0-20, t1+20), seg_ts[t0-20:t1+20], color="black", lw=4)
    _plt.ylim(0.5, 5.5)
    fig.add_subplot(2, 1, 2)
    _plt.scatter(r[:, x], r[:, y], s=5, color="grey")
    _plt.scatter(r[t0:t1, x], r[t0:t1, y], s=9, color="black")
    _plt.plot(r[t0, x], r[t0, y], ms=20, color="blue", marker=".")
    _plt.plot(r[t1, x], r[t1, y], ms=20, color="red", marker=".")
    _plt.xlim(10, 150)
    _plt.ylim(50, 180)
    _plt.savefig("btwn_%(ep)d_%(i)d" % {"ep" : (ep+1), "i" : i})
    _plt.close()
