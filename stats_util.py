#  http://www.mit.edu/~mattjj/released-code/hsmm/stats_util.py

from __future__ import division
import numpy as np
from numpy.random import random
from numpy import newaxis as na
import scipy.stats as stats
import scipy.linalg

### Sampling functions

def sample_discrete(dist,size=[]):
    assert (dist >=0).all()
    cumvals = np.cumsum(dist)
    return np.sum(random(size)[...,na] * cumvals[-1] > cumvals, axis=-1)

def sample_niw(mu_0,lmbda_0,kappa_0,nu_0):
    '''
    Returns a sample from the normal/inverse-wishart distribution, conjugate
    prior for (simultaneously) unknown mean and unknown covariance in a
    Gaussian likelihood model. Returns covariance.  '''
    # this is completely copied from Matlab's implementation, ignoring
    # the copyright. I'm sorry.
    # reference: p. 87 in Gelman's Bayesian Data Analysis

    # first sample Sigma ~ IW(lmbda_0^-1,nu_0)
    lmbda = sample_invwishart(lmbda_0,nu_0) # lmbda = np.linalg.inv(sample_wishart(np.linalg.inv(lmbda_0),nu_0))
    # then sample mu | Lambda ~ N(mu_0, Lambda/kappa_0)
    mu = np.random.multivariate_normal(mu_0,lmbda / kappa_0)

    return mu, lmbda

def sample_invwishart(lmbda,dof):
    # TODO make a version that returns the cholesky
    # TODO allow passing in chol/cholinv of matrix parameter lmbda
    n = lmbda.shape[0]
    chol = np.linalg.cholesky(lmbda)

    if (dof <= 81+n) and (dof == np.round(dof)):
        x = np.random.randn(int(dof),int(n))    # [dof x k]
    else:
        x = np.diag(np.sqrt(stats.chi2.rvs(dof-(np.arange(n)))))  #  k 

    x[np.triu_indices_from(x,1)] = np.random.randn(int(n*(n-1)/2))


    R = np.linalg.qr(x,'r')   #  x shape not fixed for given k.  
    try:
        T = scipy.linalg.solve_triangular(R.T,chol.T).T
    #  if dof too small, we get problems
    except ValueError:  # R.T is 5 x 3, not 5 x 5
        print R.T
        print chol.T
        print lmbda
        print dof
        raise
    return np.dot(T,T.T)

#  uptriinds = np.triu_indices_from(x,1)
def m_sample_invwishart(PSIs, dofs, K, uptriinds, iws):   # k = PSIs.shape[0]
    # TODO make a version that returns the cholesky
    # TODO allow passing in chol/cholinv of matrix parameter lmbda

    M    = PSIs.shape[0]
    chol = np.linalg.cholesky(PSIs)

    for m in xrange(M):
        dof = dofs[m]
        if (dof <= 81+K) and (dof == np.round(dof)):
            x = np.random.randn(int(dof), K)    # [dof x k]
        else:
            x = np.diag(np.sqrt(stats.chi2.rvs(dof-(np.arange(K)))))  #  k 

            x[uptriinds] = np.random.randn(K*(K-1)/2)

        R = np.linalg.qr(x,'r')   #  x shape not fixed for given k.  

        T = scipy.linalg.solve_triangular(R.T,chol[m].T).T

        iws[m] = np.dot(T,T.T)#, out=iws[m])
        #np.dot(T,T.T, out=iws[m])#, out=iws[m])

def sample_wishart(sigma, dof):
    '''
    Returns a sample from the Wishart distn, conjugate prior for precision matrices.
    '''

    n = sigma.shape[0]
    chol = np.linalg.cholesky(sigma)

    # use matlab's heuristic for choosing between the two different sampling schemes
    if (dof <= 81+n) and (dof == round(dof)):
        # direct
        X = np.dot(chol,np.random.normal(size=(n,dof)))
    else:
        A = np.diag(np.sqrt(np.random.chisquare(dof - np.arange(0,n),size=n)))
        A[np.tri(n,k=-1,dtype=bool)] = np.random.normal(size=(n*(n-1)/2.))
        X = np.dot(chol,A)

    return np.dot(X,X.T)

