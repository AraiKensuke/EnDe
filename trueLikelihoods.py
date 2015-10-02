class trueLikelihoods:
    #  From 
    #  Cov
    #  stdP
    #  um
    #  uP
    
    twpi = _N.sqrt(2*_N.pi)

    def lklhd(fxdMks, t):
        Lam = LAM(t)
        iSgs= _N.linalg.inv(mND.Cov)

        cmps= _N.empty((mND.M, Nx))
        rhs = _N.empty(mND.k)

        #  prob at each value of xA
        for m in xrange(mND.M):
            lnp = (xA - mND.uP[m, 0, t])**2 / (2*mND.stdP[m, 0, t]**2)
            _N.dot(iSgs[m], (fxdMks - mND.um[m, t]).T, out=rhs)
            _N.dot(fxdMks-mND.um[m, t], rhs)
            cmps[m] = mND.alp[m, 0, t] * _N.exp(-0.5*_N.dot(fxdMks-mND.um[m, t], rhs) - 0.5 * lnp)

        zs = _N.sum(cmps, axis=0)

        return zs*0.001*LAM

    def LAM(t):
        #  For our case, 
        cmps= _N.empty((mND.M, 51))
        iCov = _N.linalg.inv(mND.Cov)


        for m in xrange(mND.M):
            lnp = (xA - mND.uP[m, 0, t])**2 / (2*mND.stdP[m, 0, t]**2)
            cmps[m] = mND.alp[m, 0, t]*_N.sqrt(twpi*_N.linalg.det(mND.Cov[m])) * _N.exp(-0.5*lnp)

        zs = _N.sum(cmps, axis=0)
        return zs

    def theseMarks(mND, trainT, marks):
        nons    = _N.equal(mND.marks, None)
        mInds  = _N.where(nons == False)[0]

        mND.marks[mInds]
        mND.pos[mInds]

        marks = []
        pos   = []
        ts    = []

        iii  = -1

        #  mND.marks      --  array
        #  mND.marks[0]   --  list
        #  mND.marks]

        for mkl in mND.marks[mInds]:
            iii += 1

            for i in xrange(len(mkl[0])):
                marks.append(mkl[0][i])
                pos.append(mND.pos[mInds[iii]])
                ts.append(mInds[iii])

        marks = _N.array(marks)
        pos   = _N.array(pos)
        ts    = _N.array(ts)



        Nx     = 51
        xA     = _N.linspace(-6, 6, Nx)
        #xA     = xA.reshape(51, 1)

        N      = len(mInds)

        #  mND.alp   (M x Npf x T)
        #  mND.um    (M x T x k)
        #  mND.uP    (M x Npf x T)
        #  mND.Cov   (M x k x k)
        #  mND.stdP  (M x Npf x T)

        for n in xrange(10):
            L = evalAtFxdMks(marks[n], ts[n])
            fig = _plt.figure()
            _plt.plot(L)



