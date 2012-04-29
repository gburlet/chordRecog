# experiment 4 
# average leave one out

import sys
sys.path.insert(0, "..")

import numpy as np
from learnHMMaligned import *
import ghmm

# PARAMETERS
# ----------
K = 5               # number of songs to leave out
M = 7               # number of gaussian components in the emission distribution mixture
covType = 'diag'    # covariance structure for emission distributions (diag or full)
quality = 'full'    # chord quality: full or simple
rotate = False      # rotate chromas and use quality as chord label
key = True          # include key information in chord labels
features = 'tb'     # features to use: t (treble) or b (bass) or both
obsThresh = 0       # chords with number of observations below obsThresh are discluded
addOne = True       # add one to pi and A before normalization
tieStates = 11      # number of tied states (chord duration modeling)

numSongs = 649

# holdOut songnumber range (low,high) inclusive (not song ID's but in order of appearance in data files)
# choose random songs to holdout
holdOut = [(s,s) for s in list(np.random.randint(1,numSongs+1,K))]

holdOutAcc =  []
for holdOut in holdOuts:
    pi, A, B, labels, Xtest, Ytest, AIC = learnHMM(M=M, addOne=True, features=features, chordQuality=chordQuality, rotateChroma=rotate, key=key, featureNorm='L1', covType=covType, holdOut=holdOut, obsThresh=0)

    if tieStates is not None:
        pi, A, B, labels = tieStates(pi, A, B, labels, D = 11)

    # number of chords in ground truth
    N = len(labels)

    # fill the HMM with the learned parameters
    hmm = ghmm.GHMM(N, labels = labels, pi = pi, A = A, B = B)

    accs = {}
    # find optimal state sequence for each holdout test song
    for sid in Xtest:
        pstar, qstar = hmm.viterbi(Xtest[sid])

        # report error
        numCorr = 0
        for qInd in range(len(Ytest[sid])):
            if tieStates is not None:
                result = Ytest[sid][qInd].split("_")[0]
            else:
                result = Ytest[sid][qInd]

            if qstar[qInd] == result:
                numCorr += 1

        acc = float(numCorr) / len(Ytest[sid])
        accs[sid] = acc

    holdOutAcc.append(accs)

# calculate average recognition accuracy over holdout songs
acc = 0.0
for hold in holdOutAcc:
    for song in hold:
        acc += hold[song]
acc /= len(holdOut)

print "average accuracy: ", acc
