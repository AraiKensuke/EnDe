#  http://www.mit.edu/~mattjj/released-code/hsmm/stats_util.py

from __future__ import division
import numpy as np
from numpy.random import random
from numpy import newaxis as na
import scipy.stats as stats
import scipy.linalg

#  uptriinds = np.triu_indices_from(x,1)
m_sample_invwishart(PSIs, dofs, K, uptriinds, out=iws):   # k = PSIs.shape[0]
    # TODO make a version that returns the cholesky
    # TODO allow passing in chol/cholinv of matrix parameter lmbda

    M    = PSIs.shape[0]
    chol = np.linalg.cholesky(PSIs)

    iws  = np.zeros((M, K, K))

    for m in xrange(M):
        dof = dofs[m]
        if (dof <= 81+K) and (dof == np.round(dof)):
            x = np.random.randn(int(dof), K)    # [dof x k]
        else:
            x = np.diag(np.sqrt(stats.chi2.rvs(dof-(np.arange(K)))))  #  k 

            x[uptriinds] = np.random.randn(K*(K-1)/2)

        R = np.linalg.qr(x,'r')   #  x shape not fixed for given k.  

        T = scipy.linalg.solve_triangular(R.T,chol[m].T).T
        np.dot(T,T.T, out=iws[m])
