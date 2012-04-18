# experiment 2
# K-fold cross-validation

import sys
sys.path.insert(0, "..")

import numpy as np
from learnHMMaligned import *
import ghmm

# PARAMETERS
# ----------
K = 5                   # number fold cross validation
M = 5                   # number mixtures
features = 'tb'         # 't' for treble, 'b' for bass
chordQuality = 'full'   # chord quality 'full' or 'simple'
rotate = False          # rotate chromas
key = False             # include key information in labels
covType = 'full'        # emission distribution covariance structure


numSongs = 649
hop = np.floor(numSongs/K)

holdOuts = []
for i in range(K-1):
    holdOuts.append(((i*hop)+1, (i+1)*hop))

holdOuts.append((holdOuts[-1][1]+1, numSongs))

holdOutAcc =  []
for holdOut in holdOuts:
    pi, A, B, labels, Xtest, Ytest, AIC = learnHMM(M=M, addOne=True, features=features, chordQuality=chordQuality, rotateChroma=rotate, key=key, featureNorm='L1', covType=covType, holdOut=holdOut, obsThresh=0)

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
            if qstar[qInd] == Ytest[sid][qInd]:
                numCorr += 1

        acc = float(numCorr) / len(Ytest)
        accs[sid] = acc

    holdOutAcc.append(accs)

# calculate average recognition accuracy over blocks
acc = 0.0
for block in holdOutAcc:
    for song in block:
        acc += block[song]
acc /= numSongs

print acc
