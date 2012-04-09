import numpy as np
from scipy import linalg as slinalg
from numpy import linalg as nlinalg

from utilities import logsumexp

# for mean initialization
from sklearn import cluster

class GMM:
    '''
    This class represents a Gaussian Mixture Model. This is a mixture of (uni)multivariate gaussian distributions.

    For chord recognition M = 3, with full covariance matrix is optimal (Mauch, 2010)
    '''

    def __init__(self, M, D, covType = 'full', **kwargs):
        '''
        Initializes the mixture of multivariate gaussian distribution.

        PARAMETERS
        ----------
        M: number of mixture components
           For a single gaussian (M=1), there is no mixing matrix
        D: dimensionality of the observations
           univariate gaussian => D=1, bivariate gaussian => D=2, ..., multivariate gaussian => D=D

        kwargs:
            mu     {MxD}: matrix of means for each mixture
            Sigma  {MxDxD}: covariance matrix for each mixture
            w      {1xM}: mixing vector of gaussian weights
            covType: default='diag', 'full', 'spherical'
            zeroCorr: small float to offset zero elements where divide by zeros are possible, default = numpy epsilon (2.22e-16)
                      To not use any zero correction, set to 0.0 (WARNING: leads to numerical instability)
        '''
        
        self.M = M
        self.D = D
        if covType not in ['full', 'diag']:
            raise ValueError('GMM: invalid covariance type - ' + covType)
        else:
            self.covType = covType

        if "zeroCorr" in kwargs:
            if kwargs["zeroCorr"] < 0.0:
                raise ValueError('GMM: invalid zero correction')
            self._zeroCorr = kwargs["zeroCorr"]
        else:
            self._zeroCorr = np.finfo(float).eps
    
        # manually set distribution parameters if known
        if "mu" in kwargs:
            self._setMu(kwargs["mu"])
        else:
            self._setMu(np.random.rand(self.M, self.D))

        if "Sigma" in kwargs:
            self._setSigma(kwargs["Sigma"])
        else:
            self._setSigma(np.tile(np.eye(self.D), (self.M, 1, 1)))
        
        if "w" in kwargs:
            self._setW(kwargs["w"])
        else:
            #self._setW(np.tile(1.0 / self.M, self.M))
            wRand = np.random.rand(self.M)
            self._setW(wRand / np.sum(wRand))

    ''' 
    CLASS PROPERTIES
    ----------------
    '''
    def _getMu(self):
        ''' 
        Getter function for mu 
        
        RETURNS
        -------
        mu {MxD}
        '''
        return self._mu

    def _setMu(self, theMu):
        '''
        Setter function for mu.
        '''
        if theMu.shape != (self.M, self.D):
            raise ValueError('GMM: invalid mean vector')

        self._mu = theMu.copy()
            
    mu = property(_getMu, _setMu)

    def _getSigma(self):
        '''
        Getter function for Sigma

        RETURNS
        -------
        Sigma {MxDxD}
        '''
        return self._Sigma

    def _setSigma(self, theSigma):
        '''
        Setter function for Sigma
        '''
        if theSigma.shape != (self.M, self.D, self.D):
            raise ValueError('GMM: invalid Sigma matrix dimensions')

        self._Sigma = self._processCov(theSigma).copy()

    Sigma = property(_getSigma, _setSigma)

    def _getW(self):
        '''
        Getter function for the component distribution weights

        RETURNS
        -------
        w {1xM}
        '''
        return np.exp(self._lnw)

    def _setW(self, theWeights):
        '''
        Setter function for the component distribution weights. Stores the ln of the weights.
        '''
        if theWeights.shape != (self.M,):
            raise ValueError('GMM: invalid weight vector')
        if not np.allclose(np.sum(theWeights), 1.0):
            raise ValueError('GMM: weights should sum to 1.0')

        self._lnw = np.log(theWeights).copy()

    w = property(_getW, _setW)

    '''
    CLASS METHODS
    -------------
    '''
    def expectMax(self, X, init = 'mc', update = 'mcw', maxIter = 10, convEps = 0.01, verbose = False):
        ''' 
        Performs maximum likelihood to estimate the distribution parameters
        mu, Sigma, and w.

        PARAMETERS
        ----------
        X {N,D}: matrix of training data

        RETURNS
        -------
        lnP_history: learning curve
        '''

        # debug: save training data
        self.X = X

        N, dim = X.shape
        if dim != self.D:
            raise ValueError('GMM: training data dimensions not compatible with GMM')

        if 'm' in init or 'c' in init:
            if N >= self.M:
                # k-means requires more observations than means to run
                clusters = cluster.KMeans(k = self.M).fit(X)

        # initialize distribution parameters
        if 'm' in init:
            if N >= self.M:
                self._setMu(clusters.cluster_centers_)
            else:
                # set means randomly from data
                iRandObs = np.random.randint(N, size=(self.M, self.D))
                iCol = np.tile(np.arange(self.D), (self.M,1))
                self._setMu(X[iRandObs, iCol])

        if 'c' in init:
            # if more than one observation and not enough for kmeans, reinitialize with covariance of data
            if N > 1 and N < self.M:
                # each row represents a variable, each column an observation
                cov = np.cov(X.T)
                
                # corner case: for univariate gaussian, turn into array
                if self.D == 1:
                    cov = np.asarray([[cov]])

                # add constant along diagonal to rank-deficient covariance matrices
                # taken from GMM library netlab3.3 (matlab code)
                # GMM_WIDTH = 1.0 is arbitrary
                if nlinalg.matrix_rank(cov) < self.D:
                    cov += 1.0 * np.eye(self.D)

                self._setSigma(np.tile(cov, (self.M, 1, 1)))

            elif N >= self.M:
                # get cluster labels for training data
                labels = np.asarray(clusters.labels_, dtype = np.int)

                cov = np.zeros([self.M, self.D, self.D])

                # for each cluster
                for l in range(0,self.M):
                    # Pick out data points belonging to this centre
                    c = X[labels == l]

                    if len(c) > 0:
                        diffs = c - self._mu[l,:]
                        cov[l,:,:] = np.dot(diffs.T, diffs) / len(c)
                    else:
                        # at this point self.M number of mixtures is probably too complex a model for the data
                        # continue anyways
                        # each row represents a variable, each column an observation
                        cov[l,:,:] = np.cov(X.T)
                        
                        # corner case: for univariate gaussian, turn into array
                        if self.D == 1:
                            cov = np.asarray([[cov]])

                    # add constant along diagonal to rank-deficient covariance matrices
                    # taken from GMM library netlab3.3 (matlab code)
                    # GMM_WIDTH = 1.0 is arbitrary
                    if nlinalg.matrix_rank(cov[l,:,:]) < self.D:
                        cov[l,:,:] += 1.0 * np.eye(self.D)

                self._setSigma(cov)
                            
        # Expectation Maximization
        lnP_history = []
        for i in range(maxIter):
            # Expectation step
            lnP, posteriors = self._expect(X, verbose)
            lnP_history.append(lnP.sum())

            if verbose:
                print "EM iteration %d, lnP = %f" % (i, lnP_history[-1])

            if i > 0 and abs(lnP_history[-1] - lnP_history[-2]) < convEps:
                # if little improvement, stop training
                break

            # Maximization Step
            self._maximize(X, posteriors, update)

        # only keep covariance diagonals
        if self.covType == 'diag':
            self._Sigma *= np.eye(self.D)

        if verbose:
            if i < maxIter-1:
                print "EM converged in %d steps" % len(lnP_history)
            else:
                print "EM did not converge (maxIter reached)"

        return lnP_history

    def _expect(self, X, verbose = False):
        ''' 
        Expectation step of the expectation maximization algorithm. 
        
        PARAMETERS
        ----------
        X {NxD}: training data

        RETURNS
        -------
        lnP (N,): ln[sum_M p(l)*p(Xi | l)]
            ln probabilities of each observation in the training data,
            marginalizing over mixture components to get ln[p(Xi)]
        posteriors {NxM}: p(l | Xi)
            Posterior probabilities of each mixture component for each observation.
        '''

        N, _ = X.shape
        lnP_Xi_l = np.zeros([N, self.M])

        # zero correction
        self._Sigma[self._Sigma == 0.0] += self._zeroCorr

        # for each mixture component
        for l in range(0,self.M):
            X_mu = X - self._mu[l,:]

            if self.covType == 'diag':
                sig_l = np.diag(self._Sigma[l,:,:])
                lnP_Xi_l[:,l] = -0.5 * (self.D * np.log(2.0*np.pi) + np.sum((X_mu ** 2) / sig_l, axis=1) + np.sum(np.log(sig_l)))

            elif self.covType == 'full':
                try:
                    # cholesky decomposition => U*U.T = _Sigma[l,:,:]
                    U = slinalg.cholesky(self._Sigma[l,:,:], lower=True)
                except slinalg.LinAlgError:
                    # reinitialization trick is from scikit learn GMM
                    if verbose:
                        print "Sigma is not positive definite. Reinitializing ..."
                    self._Sigma[l,:,:] = 1e-6 * np.eye(self.D)
                    U = 1000.0 * self._Sigma[l,:,:]
                    
                Q = slinalg.solve_triangular(U, X_mu.T, lower=True)
                lnP_Xi_l[:,l] = -0.5 * (self.D * np.log(2.0 * np.pi) + 2.0 * np.sum(np.log(np.diag(U))) + np.sum(Q ** 2, axis=0))

        lnP_Xi_l += self._lnw
        
        # calculate sum of probabilities (marginalizing over mixtures)
        lnP = logsumexp(lnP_Xi_l, axis=1)

        posteriors = np.exp(lnP_Xi_l - lnP[:,np.newaxis])
        
        return lnP, posteriors
            
    def _maximize(self, X, posteriors, update):
        '''
        Maximization step of the expectation maximization algorithm. 
        
        PARAMETERS
        ----------
        X {NxD}: training data
        posteriors {NxM}: p(l | Xi)
            Posterior probabilities of each mixture component for each observation.
        update: which model parameters to update subset of 'mcw'
        '''
        
        N, _ = X.shape
        w = posteriors.sum(axis=0)

        # zero correction, avoid divide by zero
        w[w == 0.0] += self._zeroCorr

        if 'w' in update:
            self._lnw = np.log(w / N)

        if 'm' in update:
            self._mu = np.dot(posteriors.T, X) / w[:, np.newaxis]

        if 'c' in update:
            # for each mixture
            for l in range(0,self.M):
                X_mu = X - self._mu[l,:]
                self._Sigma[l,:,:] = np.dot(X_mu.T, posteriors[:,[l]] * X_mu) / w[l]
                # add a prior for numerical stability, used in many matlab EM libraries
                self._Sigma[l,:,:] += np.eye(self.D)*(1e-2)
            
    def calcLnP(self, X):
        '''
        Calculate the ln probability of the given observations under the model

        PARAMETERS
        ----------
        X {NxD}: observations

        RETURNS
        -------
        lnP (N,): ln[sum_M p(l)*p(Xi | l)]
            ln probabilities of each observation in the training data,
            marginalizing over mixture components to get ln[p(Xi)]
        '''
        
        lnP, _ = self._expect(X)
        return lnP

    def calcDerivLnP(self, X):
        '''
        Calculate the partial derivative of the ln probability of the given observations under the model
        with respect to the observations.

        PARAMETERS
        ----------
        X {NxD}: observations

        RETURNS
        -------
        lnP_deriv (N,D)
        '''

        N, _ = X.shape
        lnP_Xi_l = np.zeros([N, self.D, self.M])

        # zero correction
        self._Sigma[self._Sigma == 0.0] += self._zeroCorr

        # for each mixture component
        for l in range(0,self.M):
            X_mu = X - self._mu[l,:]
            mu_X = self._mu[l,:] - X

            if self.covType == 'diag':
                sig_l = np.diag(self._Sigma[l,:,:]) # (D,)
                lnP_Xi_l[:,:,l] = (-0.5 * (self.D * np.log(2.0*np.pi) + np.sum((X_mu ** 2) / sig_l, axis=1) + np.sum(np.log(sig_l))) +
                                  np.log(mu_X/sig_l))
            elif self.covType == 'full':
                try:
                    # cholesky decomposition => U*U.T = _Sigma[l,:,:]
                    L = slinalg.cholesky(self._Sigma[l,:,:], lower=True)
                except slinalg.LinAlgError:
                    # reinitialization trick is from scikit learn GMM
                    if verbose:
                        print "Sigma is not positive definite. Reinitializing ..."
                    self._Sigma[l,:,:] = 1e-6 * np.eye(self.D)
                    L = 1000.0 * self._Sigma[l,:,:]
                    
                # solve LQ=X_mu
                Q = slinalg.solve_triangular(L, X_mu.T, lower=True)

                # solve Lx=I for L^-1
                invL = slinalg.solve_triangular(L, np.eye(self.D), lower=True)
                invSig = np.dot(invL.T, invL)
                
                lnP_Xi_l[:,:,l] = (-0.5 * (self.D * np.log(2.0 * np.pi) + 2.0 * np.sum(np.log(np.diag(L))) + np.sum(Q ** 2, axis=0))[:,np.newaxis] +
                                  np.log(np.dot(mu_X, invSig)))

        lnP_Xi_l += self._lnw
        
        # calculate sum of probabilities (marginalizing over mixtures)
        lnP_deriv = logsumexp(lnP_Xi_l, axis=2)

        return lnP_deriv

    def _processCov(self, Cov):
        ''' 
        Helper function.
        Manipulate the given covariance matrix to conform to the covariance matrix type of the GMM.

        So far only supports full and diagonal covariance matrices.

        PARAMETERS
        ----------
        Cov {MxDxD}

        RETURNS
        -------
        Cov' {MxDxD} Covariance matrix of type self.covType
        '''

        if self.covType == 'full':
            Cprime = Cov
        elif self.covType == 'diag':
            Cprime = Cov * np.eye(self.D)

        return Cprime
