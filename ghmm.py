import numpy as np
from emission import *
from utilities import logsumexp

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
        if len(theLabels) != self.N:
            raise ValueError('GHMM: invalid number of state labels')

        self._labels = theLabels

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
        
        self._B = theB

    B = property(_getB, _setB)

    ''' 
    CLASS METHODS
    -------------
    '''
    def baumWelch(self, O, init = 'pa', update = 'pab', maxIter = 10, convEps = 0.01, verbose = False):
        '''
        Estimates the model parameters using the classic Baum-Welch expectation maximization algorithm

        PARAMETERS
        ----------
        [O {TxD}]: list of observation matrices with a sequence of T observations, each having dimension D
        init: p - init pi, a - init transitions, b - init emissions
        update: p - update pi, a - update transitions, b - update emissions
        maxIter: maximum number of iterations to run the EM algorithm
        convEps: convergence threshold
        verbose: print progress
        '''

        # init params
        if 'p' in init:
            self._setPi(np.ones((1, self.N)) / self.N)
    
        if 'a' in init:
            aRand = np.random.rand(self.N, self.N)
            aRand / aRand.sum(axis=1)[:,np.newaxis]
            self._setA(aRand)

        lnP_history = []
        for i in range(maxIter):
            lnP_curr = 0
            for o in O:
                lnP, lnAlpha = self._forward(o)
                lnBeta = self._backward(o)
                lnGamma = lnAlpha + lnBeta - lnP

                lnP_curr += lnP

                
            lnP_history.append(lnP_curr)

            # check convergence criterion
            if i > 0 and abs(lnP_history[-1] - lnP_history[-2]) < convEps:
                break
        
    # TODO: SCALING
    def _forward(self, O):
        '''
        Calculates the forward variable, alpha: the probability of the partial observation
        sequence O1 O2 ... Ot (until time t) and state Si at time t.

        PARAMETERS
        ----------
        O {TxD}: observation matrix with a sequence of T observations, each having dimension D
        
        RETURNS
        -------
        lnP {Float}: log probability of the observation sequence O
        lnAlpha {T,N}: log of the forward variable: the probability of the partial observation
                       sequence O1 O2 ... Ot (until time t) and state Si at time t.
        '''

        T, D = O.shape

        # check dimensions of provided observations agree with the trained emission distributions
        dim = self._B[0].mu.shape[1]
        if D != dim:
            raise ValueError('GHMM: observation dimension does not agree with the trained emission distributions for the model')

        # calculate lnP for each observation for each state's emission distribution
        # lnP_obs {T, N}
        lnP_obs = np.zeros([T,self.N])
        for i in range(0,self.N):
            lnP_obs[:,i] = self._B[i].calcLnP(O)

        # forward variable, alpha {T,N}
        lnAlpha = np.zeros([T,self.N])

        # Step 1: Initialization
        lnAlpha[0,:] = np.log(self._pi) + lnP_obs[0,:]

        # Step 2: Induction
        for t in range(1,T-1):
            ln_alpha[t,:] = logsumexp(lnAlpha[[t-1],:].T + np.log(self._A), axis=0) + lnP_obs[t,:]

        # Step 3: Termination
        lnP = logsumexp(lnAlpha[T,:])

        return lnP, lnAlpha

    # TODO: SCALING by same coefficients as alpha
    def _backward(self, O):
        '''
        Calculates the backward variable, beta: the probability of the partial observation 
        sequence 0T OT-1 ... Ot+1 (backwards to time t+1) and State Si at time t+1

        PARAMETERS
        ----------
        O {TxD}: observation matrix with a sequence of T observations, each having dimension D
        
        RETURNS
        -------
        lnBeta {T,N}: log of the backward variable: the probability of the partial observation 
                      sequence 0T OT-1 ... Ot+1 (backwards to time t+1) and State Si at time t+1
        '''
        
        T, D = O.shape

        # check dimensions of provided observations agree with the trained emission distributions
        dim = self._B[0].mu.shape[1]
        if D != dim:
            raise ValueError('GHMM: observation dimension does not agree with the trained emission distributions for the model')

        # calculate lnP for each observation for each state's emission distribution
        # lnP_obs {T, N}
        lnP_obs = np.zeros([T,self.N])
        for i in range(0,self.N):
            lnP_obs[:,i] = self._B[i].calcLnP(O)

        # backward variable, beta {T,N}
        # Step 1: Initialization
        # since ln(1) = 0
        lnBeta = np.zeros([T,self.N])

        # Step 2: Induction
        for t in reversed(range(0,T-1)):
            lnBeta[t,:] = logsumexp(np.log(self._A) + lnP_obs[t+1,:] + lnBeta[t+1,:], axis=1)

        return lnBeta

    def viterbi(self, O, labels = True):
        '''
        Calculates the q*, the most probable state sequence corresponding from the observations O.
        As the function name suggests, the viterbi algorithm is used.

        PARAMETERS
        ----------
        O {TxD}: observation matrix with a sequence of T observations, each having dimension D
        labels: whether to return the state labels, or the state indices

        RETURNS
        -------
        pstar: ln probability of q*
        qstar {Tx1}: labels/indices of states in q* (normal python array of len T)
        '''

        T, D = O.shape

        # check dimensions of provided observations agree with the trained emission distributions
        dim = self._B[0].mu.shape[1]
        if D != dim:
            raise ValueError('GHMM: observation dimension does not agree with the trained emission distributions for the model')

        # calculate lnP for each observation for each state's emission distribution
        # lnP_obs {T, N}
        lnP_obs = np.zeros([T,self.N])
        for i in range(0,self.N):
            lnP_obs[:,i] = self._B[i].calcLnP(O)

        # lnDelta {TxN}: best score along a single path, at time t, accounting for the first t observations and ending in state Si
        lnDelta = np.zeros([T,self.N])
        # lnPsi {TxN}: arg max of best scores for each t and j state
        lnPsi = np.zeros([T,self.N], dtype=np.int)

        # Step 1: initialization
        lnDelta[0,:] = np.log(self._pi) + lnP_obs[0,:]

        # Step 2: recursion
        for t in range(1,T):
            pTrans = lnDelta[[t-1],:].T + np.log(self._A)
            lnDelta[t,:] = np.max(pTrans, axis=0) + lnP_obs[t,:]
            lnPsi[t,:] = np.argmax(pTrans, axis=0)

        lnDelta[lnDelta <= -1e200] = -np.Inf

        # Step 3: termination
        qstar_t = np.argmax(lnDelta[T-1,:])
        pstar = lnDelta[T-1,qstar_t]

        qstar = []
        for t in reversed(range(0,T)):
            qstar.append(qstar_t)
            qstar_t = lnPsi[t,qstar_t]

        qstar.reverse()

        # return labels
        if (labels):
            qstar = [self._labels[q] for q in qstar]
        
        return pstar, qstar

    def derivOptCrit(self, O):
        '''
        PARAMETERS
        ----------
        O {TxD}: observation matrix with a sequence of T observations, each having dimension D

        RETURNS
        -------
        dC / dy {TxD}: derivative of the optimization criterion for each observation
        '''

        T, D = O.shape

        lnAlpha = self._forward(O)[1]
        lnBeta = self._backward(O)

        # calculate lnP for each observation for each state's emission distribution
        # lnP_obs {T, N}
        lnP_obs = np.zeros([T,self.N])
        for i in range(0,self.N):
            lnP_obs[:,i] = self._B[i].calcLnP(O)

        # calculate derivative of the optimization criterion for each observation for each state's emission distribution
        dlnP = np.zeros([T,self.N,D])
        for i in range(0,self.N):
            dlnP[:,i,:] = self._B[i].calcDerivLnP(O)

        return np.exp(logsumexp((lnBeta + lnAlpha - lnP_obs)[:,:,np.newaxis] + dlnP, axis=1))
