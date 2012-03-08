from learnHMM import *
from ghmm import *

'''
MUMT 609
February 21, 2012
Gregory Burlet

Chord recognition experiment

PARAMETERS
----------
'''
M = 3               # number of gaussian components in the emission distribution mixture
covType = 'full'    # covariance structure for emission distributions (diag or full)
features = 'tb'     # features to use: t (treble) or b (bass) or both
featureNorm = 'L1'  # L_1, L_2, or L_inf feature normalization
leaveOutSong = 3    # leave one out validation (choose song id to leave out)
obsThresh = 0       # chords with number of observations below obsThresh are discluded
addOne = True       # add one to pi and A before normalization

# learn HMM model lambda = (pi, A, B) from ground truth
pi, A, B, labels, Xtest, ytest = learnHMM(3, covType = covType, features = features, featureNorm = featureNorm, leaveOneOut = leaveOutSong, obsThresh=obsThresh)

# number of chords in ground truth
N = A.shape[0]

# fill the HMM with the learned parameters
hmm = GHMM(N, labels = labels, pi = pi, A = A, B = B)

# find optimal state sequence
pstar, qstar = hmm.viterbi(Xtest)

# report error
numCorr = 0
for qInd in range(0,len(ytest)):
    if qstar[qInd] == ytest[qInd]:
        numCorr += 1

acc = float(numCorr) / len(ytest)
print "recognition accuracy: ", acc
