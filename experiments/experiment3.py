# experiment 3
# AIC calculations

import pickle

import sys
sys.path.insert(0, "..")
from learnHMMaligned import *
import ghmm

expInd = 6

# parameter combinations
exp = {
    0: {'M': 3, 'sig': 'diag', 'rotate': True, 'quality': 'simple', 'key': True},
    1: {'M': 6, 'sig': 'diag', 'rotate': True, 'quality': 'simple', 'key': True},
    2: {'M': 8, 'sig': 'diag', 'rotate': True, 'quality': 'simple', 'key': True},

    3: {'M': 3, 'sig': 'full', 'rotate': True, 'quality': 'simple', 'key': True},
    4: {'M': 6, 'sig': 'full', 'rotate': True, 'quality': 'simple', 'key': True},
    5: {'M': 8, 'sig': 'full', 'rotate': True, 'quality': 'simple', 'key': True},

    6: {'M': 3, 'sig': 'diag', 'rotate': False, 'quality': 'simple', 'key': True},
    7: {'M': 6, 'sig': 'diag', 'rotate': False, 'quality': 'simple', 'key': True},
    8: {'M': 8, 'sig': 'diag', 'rotate': False, 'quality': 'simple', 'key': True},

    9: {'M': 3, 'sig': 'full', 'rotate': False, 'quality': 'simple', 'key': True},
    10: {'M': 6, 'sig': 'full', 'rotate': False, 'quality': 'simple', 'key': True},
    11: {'M': 8, 'sig': 'full', 'rotate': False, 'quality': 'simple', 'key': True},

    12: {'M': 3, 'sig': 'diag', 'rotate': True, 'quality': 'full', 'key': True},
    13: {'M': 6, 'sig': 'diag', 'rotate': True, 'quality': 'full', 'key': True},
    14: {'M': 8, 'sig': 'diag', 'rotate': True, 'quality': 'full', 'key': True},

    15: {'M': 3, 'sig': 'full', 'rotate': True, 'quality': 'full', 'key': True},
    16: {'M': 6, 'sig': 'full', 'rotate': True, 'quality': 'full', 'key': True},
    17: {'M': 8, 'sig': 'full', 'rotate': True, 'quality': 'full', 'key': True},

    18: {'M': 3, 'sig': 'diag', 'rotate': False, 'quality': 'full', 'key': True},
    19: {'M': 6, 'sig': 'diag', 'rotate': False, 'quality': 'full', 'key': True},
    20: {'M': 8, 'sig': 'diag', 'rotate': False, 'quality': 'full', 'key': True},

    21: {'M': 3, 'sig': 'full', 'rotate': False, 'quality': 'full', 'key': True},
    22: {'M': 6, 'sig': 'full', 'rotate': False, 'quality': 'full', 'key': True},
    23: {'M': 8, 'sig': 'full', 'rotate': False, 'quality': 'full', 'key': True},

    24: {'M': 3, 'sig': 'diag', 'rotate': True, 'quality': 'simple', 'key': False},
    25: {'M': 6, 'sig': 'diag', 'rotate': True, 'quality': 'simple', 'key': False},
    26: {'M': 8, 'sig': 'diag', 'rotate': True, 'quality': 'simple', 'key': False},

    27: {'M': 3, 'sig': 'full', 'rotate': True, 'quality': 'simple', 'key': False},
    28: {'M': 6, 'sig': 'full', 'rotate': True, 'quality': 'simple', 'key': False},
    29: {'M': 8, 'sig': 'full', 'rotate': True, 'quality': 'simple', 'key': False},

    30: {'M': 3, 'sig': 'diag', 'rotate': False, 'quality': 'simple', 'key': False},
    31: {'M': 6, 'sig': 'diag', 'rotate': False, 'quality': 'simple', 'key': False},
    32: {'M': 8, 'sig': 'diag', 'rotate': False, 'quality': 'simple', 'key': False},

    33: {'M': 3, 'sig': 'full', 'rotate': False, 'quality': 'simple', 'key': False},
    34: {'M': 6, 'sig': 'full', 'rotate': False, 'quality': 'simple', 'key': False},
    35: {'M': 8, 'sig': 'full', 'rotate': False, 'quality': 'simple', 'key': False},

    36: {'M': 3, 'sig': 'diag', 'rotate': True, 'quality': 'full', 'key': False},
    37: {'M': 6, 'sig': 'diag', 'rotate': True, 'quality': 'full', 'key': False},
    38: {'M': 8, 'sig': 'diag', 'rotate': True, 'quality': 'full', 'key': False},

    39: {'M': 3, 'sig': 'full', 'rotate': True, 'quality': 'full', 'key': False},
    40: {'M': 6, 'sig': 'full', 'rotate': True, 'quality': 'full', 'key': False},
    41: {'M': 8, 'sig': 'full', 'rotate': True, 'quality': 'full', 'key': False},

    42: {'M': 3, 'sig': 'diag', 'rotate': False, 'quality': 'full', 'key': False},
    43: {'M': 6, 'sig': 'diag', 'rotate': False, 'quality': 'full', 'key': False},
    44: {'M': 8, 'sig': 'diag', 'rotate': False, 'quality': 'full', 'key': False},

    45: {'M': 3, 'sig': 'full', 'rotate': False, 'quality': 'full', 'key': False},
    46: {'M': 6, 'sig': 'full', 'rotate': False, 'quality': 'full', 'key': False},
    47: {'M': 8, 'sig': 'full', 'rotate': False, 'quality': 'full', 'key': False},
}

pi, A, B, labels, Xtest, Ytest, AIC = learnHMM(M=exp[expInd]['M'], addOne=True, features='tb', chordQuality=exp[expInd]['quality'], rotateChroma=exp[expInd]['rotate'], key=exp[expInd]['key'], featureNorm='L1', covType=exp[expInd]['sig'], holdOut=(1,1), obsThresh=0)

print "AIC: ", AIC

# number of chords in ground truth
N = A.shape[0]

# fill the HMM with the learned parameters
hmm = ghmm.GHMM(N, labels = labels, pi = pi, A = A, B = B)

# pickle ghmm for future reference
rot = '1' if exp[expInd]['rotate'] else '0'
key = '1' if exp[expInd]['key'] else '0'
logPath = '../trainedhmms/hmm_M=' + str(exp[expInd]['M']) + '_sig=' + exp[expInd]['sig'] + '_quality=' + exp[expInd]['quality'] + '_rotate=' + rot + '_key=' + key
outP = open(logPath, 'w')

pickle.dump(hmm, outP)

outP.close()
