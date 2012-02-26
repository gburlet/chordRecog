import os
import numpy as np
from emission import *

def learnHMM(M, addOne = True, features = 't', covType = 'full', leaveOneOut = 3):
    '''
    PARAMETERS
    ----------
    M: number of gaussians to mix for emission probabilities
    addOne: whether one should be added to the transition matrices before normalization
    features: treble chroma 't' or bass chroma 'b' or both
    covType: type of the covariance matrix for the GMM emission distribution
             'diag', 'full'
    leaveOneOut: songID to leave out of the training phase and save for model validation
    

    RETURNS
    -------
    pi {1xN}: initial state probabilities
    A {NxN}: state transition probabilities
    B {1xN}: emission distribution for each state
    '''

    groundTruth = open('data/chordino.csv', 'r')

    piDict = {}
    aDict = {}
    bDict = {}
    QLabels = set()

    pSid = -1
    numSongs = 0;
    pChordName = ""
    obsNum = 0
    
    # initialize hold out data storage
    Xtest = []
    ytest = []

    for obs in groundTruth:
        # skip header
        if obsNum == 0:
            obsNum += 1
            continue
        
        obs = obs.split(",")
        sid = int(obs[0])

        if features == 't':
            chroma = np.asarray(obs[2:14], dtype=np.float)[np.newaxis,:]
        elif features == 'b':
            chroma = np.asarray(obs[14:26], dtype=np.float)[np.newaxis,:]
        elif features == 'tb':
            chroma = np.asarray(obs[2:26], dtype=np.float)[np.newaxis,:]
            
        simpleQuality = obs[35].strip()
        if simpleQuality != "NA":
            # use simple quality if available
            quality = simpleQuality
        else:
            quality = obs[34].strip()
        chordName = obs[30] + quality

        # first timestamps in songs usually have no annotation
        if chordName.strip() == "NANA":
            continue

        if sid != leaveOneOut:
            if sid != pSid:
                # new training sequence
                # update pi
                if chordName in piDict:
                    piDict[chordName] += 1
                else:
                    piDict[chordName] = 1
                pSid = sid
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
                bDict[chordName] = np.vstack((bDict[chordName], chroma))
            else:
                bDict[chordName] = chroma.copy()

            # update state labels
            QLabels.add(chordName)

            obsNum += 1
        else:
            # this is part of the validation set
            # add observation to Xtest
            Xtest.append(chroma)

            # add the observations chord label
            ytest.append(chordName)
    
    QLabels = sorted(QLabels)
    N = len(QLabels)

    # initialize arrays
    pi = np.zeros((1,N))
    A = np.zeros((N,N))
    B = []

    # for each state
    i = 0
    for q in QLabels:
        # fill pi
        if q in piDict:
            pi[0,i] = piDict[q]

        # fill A
        if q in aDict:
            # traverse edges
            j = 0
            for k in QLabels:
                if k in aDict[q]:
                    A[i,j] = aDict[q][k]
                j += 1
        
        # fill B
        if q in bDict:
            print "learning emissions for chord: %s, index: %d, #obs: %d" % (q, i, bDict[q].shape[0])
            bGMM = GMM(M, 12, covType, zeroCorr=0.01, convEps=1e-6)
            bGMM.expectMax(bDict[q], maxIter=100)
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

#pi, A, B, labels, Xtest, ytest = learnHMM(3, covType='diag')
