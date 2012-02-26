import numpy as np
from emission import *

class GHMM:
    ''' 
    Creates a hidden Markov model with a mixture of multivariate gaussians emission distribution
    '''

    def __init__(self, N, **args):
        '''
        GHMM constructor for model lambda = (pi, A, B)

        PARAMETERS
        ----------
        N: number of hidden states
        
        args:
            labels {1xN} hidden state labels
            pi {1xN}: initial state distribution
            A  {NxN}: transition matrix
            B  {1xN}: list of GMM emission distributions for each hidden state
        '''
        
        self.N = N

        # initialize hidden state labels
        if 'labels' in args:
            self._setLabels(args['labels'])
        else:
            self._setLabels(range(self.N))

        # initialize initial state distribution
        if 'pi' in args:
            self._setPi(args['pi'])
        else:
            self._setPi(np.ones((1, self.N)) / self.N)

        # initialize transition matrix
        if 'A' in args:
            self._setA(args['A'])
        else:
            aRand = np.random.rand(self.N, self.N)
            aRand / aRand.sum(axis=1)[:,np.newaxis]
            self._setA(aRand)

        # initialize emission distributions
        if 'B' in args:
            self._setB(args['B'])
        else:
            pass

    ''' 
    CLASS PROPERTIES
    ----------------
    '''
    def _getLabels(self):
        return self._labels

    def _setLabels(self, theLabels):
        pass

    labels = property(_getLabels, _setLabels)

    def _getPi(self):
        return self._A

    def _setPi(self, thePi):
        if thePi.shape != (1,self.N):
            raise ValueError('GHMM: invalid pi dimensions')
        if not np.allclose(thePi.sum(), 1.0) or np.any(thePi < 0.0):
            raise ValueError('GHMM: invalid pi values')

        self._pi = thePi.copy()

    pi = property(_getPi, _setPi)

    def _getA(self):
        return self._A

    def _setA(self, theA):
        if theA.shape != (self.N, self.N):
            raise ValueError('GHMM: invalid A dimensions')
        if not np.allclose(theA.sum(axis=1), 1.0) or np.any(theA < 0.0):
            raise ValueError('GHMM: invalid A values')

        self._A = theA.copy()

    A = property(_getA, _setA)

    def _getB(self):
        return self._B

    def _setB(self, theB):
        if len(theB) != self.N:
            raise ValueError('GHMM: invalid B dimensions')
        if any(not isinstance(emis, GMM) for emis in theB): 
            raise ValueError('GHMM: B elements must be of class emission.GMM')
        
        self._B = theB.copy()

    B = property(_getB, _setB)

    ''' 
    CLASS METHODS
    -------------
    '''
    def viterbi(self, O, labels = True):
        '''
        Calculates the q*, the most probable state sequence corresponding from the observations O.
        As the function name suggests, the viterbi algorithm is used.

        PARAMETERS
        ----------
        O {NxD}: observation matrix with a sequence of N observations, each having dimension D
        labels: whether to return the state labels, or the state indices

        RETURNS
        -------
        lnP_seq: ln probability of q*
        qstar {Nx1}: labels/indices of states in q*
        '''

        Nobs, D = O.shape

        # check dimensions of provided observations agree with the trained emission distributions
        dim = self._B[0].mu.shape[1]
        if D != dim:
            raise ValueError('GHMM: observation dimension does not agree with the trained emission distributions for the model')

        # calculate lnP for each observation for each state's emission distribution
        # lnP_obs {Nobs, N}
        for i in range(0,N):
            
