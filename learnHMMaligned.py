import os
import numpy as np
from emission import *

def learnHMM(M, addOne = True, features = 'tb', covType = 'full', leaveOneOut = 3, obsThresh = 0):
    '''
    PARAMETERS
    ----------
    M: number of gaussians to mix for emission probabilities
    addOne: whether one should be added to the transition matrices before normalization
    features: treble chroma 't' or bass chroma 'b' or both
    featureNorm: L1, L2, Linf, or none normalization
    covType: type of the covariance matrix for the GMM emission distribution
             'diag', 'full'
    leaveOneOut: songID to leave out of the training phase and save for model validation
    

    RETURNS
    -------
    pi {1xN}: initial state probabilities
    A {NxN}: state transition probabilities
    B {1xN}: emission distribution for each state
    '''

    groundTruth = open('data/gtruth_constqnetout.csv', 'r')
    # sid, timestamp, chordName, chroma vector

    piDict = {}
    aDict = {}
    bDict = {}
    QLabels = set()

    if features == 'tb':
        D = 24
    else:
        D = 12

    pSid = -1
    numSongs = 0;
    pChordName = ""
    
    # initialize hold out data storage
    Xtest = []
    ytest = []

    for obsNum, obs in enumerate(groundTruth):        
        obs = obs.split(",")
        sid = int(obs[0])
        chordName = obs[2]
        # first timestamps in songs usually have no annotation
        if chordName.strip() == "NANA":
            continue

        if features == 't':
            chroma = np.asfarray(obs[3:15])
        elif features == 'b':
            chroma = np.asfarray(obs[15:27])
        elif features == 'tb':
            chroma = np.asfarray(obs[3:27])

        # skip silence (really there are no chords in either the treble or bass)            
        if np.sum(chroma) == 0:
            continue

        # features are already normalized by neural net softmax output!

        if sid != leaveOneOut:
            if sid != pSid:
                # new training sequence
                # update pi
                if chordName in piDict:
                    piDict[chordName] += 1
                else:
                    piDict[chordName] = 1
                pSid = sid
                print "processing song: ", sid
                pChordName = chordName
                numSongs += 1
            else:
                #update A
                if pChordName in aDict:
                    if chordName in aDict[pChordName]:
                        aDict[pChordName][chordName] += 1.0
                    else:
                        aDict[pChordName][chordName] = 1.0
                else:
                    aDict[pChordName] = {chordName: 1.0}
            
            # update B
            if chordName in bDict:
                bDict[chordName].append(chroma)
            else:
                bDict[chordName] = [chroma]

            # update state labels
            QLabels.add(chordName)
        else:
            # this is part of the validation set
            # add observation to Xtest
            Xtest.append(chroma)

            # add the observations chord label
            ytest.append(chordName)
    
    # close file pointer    
    groundTruth.close()

    print "done parsing, learning state transitions and emissions ..."
    QLabels = sorted(QLabels)
    
    # prune states with insufficient observations
    QLabels = [q for q in QLabels if len(bDict[q]) >= obsThresh]

    N = len(QLabels)

    # initialize arrays
    pi = np.zeros((1,N))
    A = np.zeros((N,N))
    B = []

    # for each state
    for i, q in enumerate(QLabels):
        # fill pi
        if q in piDict:
            pi[0,i] = piDict[q]

        # fill A
        if q in aDict:
            # traverse edges
            for j, k in enumerate(QLabels):
                if k in aDict[q]:
                    A[i,j] = aDict[q][k]
                j += 1
        
        # fill B
        # convert to numpy array
        Xtrain = np.asarray(bDict[q])
        del bDict[q]
        print "learning emissions for chord: %s, index: %d, #obs: %d" % (q, i, Xtrain.shape[0])
        bGMM = GMM(M, D, covType, zeroCorr=1e-12)
        bGMM.expectMax(Xtrain, maxIter=50, convEps=1e-6, verbose=True)
        B.append(bGMM)
            
        i += 1

    # add one to state transition probabilities before normalizing
    if addOne:
        pi += 1.0
        A += 1.0

    # transform transition histograms to probabilities
    pi /= np.sum(pi)
    A /= A.sum(axis=1)[:,np.newaxis]

    Xtest = np.asarray(Xtest).squeeze()

    return pi, A, B, QLabels, Xtest, ytest

#pi, A, B, labels, Xtest, ytest = learnHMM(3, covType='full', features = 'tb', leaveOneOut = 3, obsThresh=0)
