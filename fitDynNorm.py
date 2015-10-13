import numpy as _N
MVN = _N.random.multivariate_normal
import scipy.stats as _ss

class fitDynNorm:
    ITERS  = 200
    posterior_mu = None
    posterior_Cv = None

    def fit(self, dat, neps, eps, forget=1):
        oo = self

        #  setup hyperparameters for priors.  Uncertainty about mean
        hy_pr_mu_mu  = 0     
        hy_pr_mu_vr  = 10

        hy_pr_vr_a = 1
        hy_pr_vr_B = 10

        #  parameters needed to sample from posterior.  Uncertain about variance
        hy_po_mu_mu  = 0
        hy_po_mu_vr  = 0

        hy_po_vr_a  = 0
        hy_po_vr_B  = 0.5

        #  Gibbs samples from the posterior
        gsMu = _N.empty(oo.ITERS)
        gsVr = _N.empty(oo.ITERS)
        hITERS = oo.ITERS/2

        oo.posteriors_mu = _N.empty(neps)   #  mean of data distribution
        oo.posteriors_Vr = _N.empty(neps)   #  variance of data distribution

        #  we take the means of the posterior hyperparameters
        hy_po_mu_mus = _N.empty(oo.ITERS-1)
        hy_po_mu_vrs = _N.empty(oo.ITERS-1)
        hy_po_vr_as  = _N.empty(oo.ITERS-1)
        hy_po_vr_Bs  = _N.empty(oo.ITERS-1)
        ######  filtering step
        for n in xrange(neps):
            print "n %d" % n
            clstx    = dat[n*eps:(n+1)*eps]
            smpl_mn  = _N.sum(clstx)

            print smpl_mn
            gsMu[0] = smpl_mn
            gsVr[0] = _N.std(clstx)**2

            ##  Gibbs sampling.  Perform with fixed hyperparameters
            for itr in xrange(oo.ITERS-1):
                ####################  
                #  hy_pr_mu_vr prior hyperparam, 
                #  knVr is variance of data distribution
                knVr = gsVr[itr]
                hy_po_mu_vr = 1 / ((1./hy_pr_mu_vr) + (eps/knVr))
                hy_po_mu_mu = (hy_pr_mu_mu / hy_pr_mu_vr + smpl_mn / knVr) * hy_po_mu_vr
                gsMu[itr+1] = hy_po_mu_mu + _N.sqrt(hy_po_mu_vr)*_N.random.rand()
                ####################
                knMu        = gsMu[itr+1]
                hy_po_vr_a  = hy_pr_vr_a + eps*0.5    # neps is pr_sg_dof
                hy_po_vr_B  = hy_pr_vr_B + 0.5*_N.sum((clstx - knMu)**2)
                gsVr[itr+1] =  _ss.invgamma.rvs(hy_po_vr_a, scale=hy_po_vr_B)

                hy_po_mu_mus[itr] = hy_po_mu_mu
                hy_po_mu_vrs[itr] = hy_po_mu_vr
                hy_po_vr_as[itr]  = hy_po_vr_a
                hy_po_vr_Bs[itr]  = hy_po_vr_B

            #  have new "posterior hyperparameters" every time I Gibbs sample
            #  which one to use?

            hy_pr_mu_mu = _N.mean(hy_po_mu_mus[hITERS:])
            hy_pr_mu_vr = _N.mean(hy_po_mu_vrs[hITERS:])*forget
            hy_pr_vr_a  = _N.mean(hy_po_vr_as[hITERS:])
            hy_pr_vr_B  = _N.mean(hy_po_vr_Bs[hITERS:])
            

            oo.posteriors_mu[n] =  hy_pr_mu_mu + _N.sqrt(hy_pr_mu_vr)*_N.random.rand()  #  mean of data distribution
            oo.posteriors_Vr[n] =  _ss.invgamma.rvs(hy_pr_vr_a, scale=hy_pr_vr_B)
