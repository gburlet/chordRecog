import sys
sys.path.insert(0, "..")

from learnHMMaligned import *
import ghmm
import pickle

'''
Chord recognition experiment
Using chordino chroma vector features aligned framewise with ground-truth annotations

Use "fake transition matrix" => don't worry about transition information, just worry about the probability it is that chord

PARAMETERS
----------
USE minimal AIC value: hmm_M=7_sig=diag_quality=full_rotate=0_key=1, AIC:  -136151623.928
'''
M = 7               # number of gaussian components in the emission distribution mixture
covType = 'diag'    # covariance structure for emission distributions (diag or full)
quality = 'full'    # chord quality: full or simple
rotate = False      # rotate chromas and use quality as chord label
key = True          # include key information in chord labels
features = 'tb'     # features to use: t (treble) or b (bass) or both
holdOut = (1,1)     # holdOut songnumber range (low,high) inclusive (not song ID's but in order of appearance in data files)
obsThresh = 0       # chords with number of observations below obsThresh are discluded
addOne = True       # add one to pi and A before normalization
tieStates = 11      # number of tied states (chord duration modeling)

# learn HMM model lambda = (pi, A, B) from ground truth
pi, A, B, labels, Xtest, Ytest, AIC = learnHMM(M=M, addOne=addOne, features=features, chordQuality=quality, rotateChroma=rotate, key=key, featureNorm='L1', covType=covType, holdOut=holdOut, obsThresh=obsThresh)

if tieStates is not None:
    pi, A, B, labels = tieStates(pi, A, B, labels, D = 11)

# number of chords in ground truth
N = A.shape[0]

# fill the HMM with the learned parameters
hmm = ghmm.GHMM(N, labels = labels, pi = pi, A = A, B = B)

# pickle hmm model for future
rot = '1' if rotate else '0'
key = '1' if key else '0'
tie = str(tieStates) if tieStates is not None else 'NA'
fName = '../trainedhmms/exp1_M=' + str(M) + '_sig=' + covType + '_quality=' + quality + '_rotate=' + rot + '_key=' + key + '_tied=' + tie + 'holdOut=[' + str(holdOut[0]) + '_' = str(holdOut[1]) + ']' 
outP = open(fName, 'w')
pickle.dump(hmm, outP)
outP.close()

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

# calculate average recognition accuracy over holdout songs
acc = 0.0
for hold in accs:
    acc += accs[hold]
acc /= len(Ytest)

print "average accuracy: ", acc
