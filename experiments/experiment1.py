from learnHMM import *
from ghmm import *

'''
MUMT 609
February 21, 2012
Gregory Burlet

Chord recognition experiment

PARAMETERS
----------
M {int}: Number of mixtures for the GMM emission distributions
covType {string}: Type of covariance matrix for the GMM emission distributions
    'diag' -> diagonal
    'full' -> full
addOne {boolean}: flag to add one to pi and A before normalization to avoid 0 states 
'''
M = 3
covType = 'full'
addOne = True

# learn HMM model lambda = (pi, A, B) from ground truth
pi, A, B, labels, Xtest, ytest = learnHMM(M, addOne, covType)

N = A.shape[0]

# fill the HMM with the learned parameters
hmm = GHMM(N, labels = labels, pi = pi, A = A, B = B)

# find optimal state sequence
pstar, qstar = hmm.viterbi(Xtest)
