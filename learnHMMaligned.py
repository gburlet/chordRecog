import os
import numpy as np
from emission import *
import ghmm
import pickle

def learnHMM(M, addOne = True, features = 'tb', chordQuality = 'simple', rotateChroma = False, key = False, featureNorm = 'L1', covType = 'full', leaveOneOut = 3, obsThresh = 0):
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

    groundTruth = open('data/gtruth_chroma.csv', 'r')
    # 0, 1, 7, 10, 11, 12, 13
    # sid, timestamp, local.tonic.name, root.name, root.pc, quality, simple.quality, obs
    #  0,      1,            2,             3,        4,       5,          6,         7-

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

        # form chord name
        if key:
            if obs[2] == "NA":
                continue
            chordName = obs[2] + '.'
        else:
            chordName = ''

        if chordQuality == 'full':
            if obs[5] == "NA":
                continue
            chordName += obs[3] + obs[5]
        else:
            if obs[6] == "NA":
                continue
            chordName += obs[3] + obs[6]

        # first timestamps in songs usually have no annotation
        if chordName.strip() == "NANA":
            continue

        if features == 't':
            chroma = np.asfarray(obs[7:19])
        elif features == 'b':
            chroma = np.asfarray(obs[19:31])
        elif features == 'tb':
            chroma = np.asfarray(obs[7:31])

        # skip silence (really there are no chords in either the treble or bass)            
        if np.sum(chroma) == 0:
            continue

        if rotateChroma:
            # override chord name with quality
            chordName = obs[6] if chordQuality == 'simple' else obs[5]

            # rotate chroma
            pc = int(obs[4])
            if features == 'tb':
                # rotate treble & bass
                chroma[0:12] = np.roll(chroma[0:12], -pc)
                chroma[12:24] = np.roll(chroma[12:24], -pc)
            else:
                chroma[0:12] = np.roll(chroma[0:12], -pc)

        # perform feature normalization
        if featureNorm == 'L1':
            if features == 'tb':
                if np.sum(chroma[0:12]) != 0:
                    chroma[0:12] /= np.sum(np.abs(chroma[0:12]))
                if np.sum(chroma[12:]) != 0:
                    chroma[12:24] /= np.sum(np.abs(chroma[12:24]))
            else:
                chroma /= np.sum(np.abs(chroma))
        elif featureNorm == 'L2':
            if features == 'tb':
                if np.sum(chroma[0:12]) != 0:
                    chroma[0:12] /= np.sum(chroma[0:12] ** 2)
                if np.sum(chroma[12:]) != 0:
                    chroma[12:24] /= np.sum(chroma[12:24] ** 2)
            else:
                chroma /= np.sum(chroma ** 2)
        elif featureNorm == 'Linf':
            if features == 'tb':
                if np.sum(chroma[0:12]) != 0:
                    chroma[0:12] /= np.max(np.abs(chroma[0:12]))
                if np.sum(chroma[12:]) != 0:
                    chroma[12:24] /= np.max(np.abs(chroma[12:24]))
            else:
                chroma /= np.max(np.abs(chroma))


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

    # initialize lnP for AIC calculation
    lnP = 0.0

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
        lnP_history = bGMM.expectMax(Xtrain, maxIter=50, convEps=1e-6, verbose=True)
        B.append(bGMM)

        # update running total of lnP for AIC
        lnP += lnP_history[-1]
            
        i += 1

    # add one to state transition probabilities before normalizing
    if addOne:
        pi += 1.0
        A += 1.0

    # transform transition histograms to probabilities
    pi /= np.sum(pi)
    A /= A.sum(axis=1)[:,np.newaxis]

    Xtest = np.asarray(Xtest).squeeze()

    # AIC calculation
    if covType == 'full':
        numCovar = D ** 2
    else:
        numCovar = D

    k = (M*(D + numCovar) + M-1) * len(QLabels)
    AIC = 2*k - 2*lnP

    return pi, A, B, QLabels, Xtest, ytest, AIC

pi, A, B, labels, Xtest, ytest, AIC = learnHMM(M=2, addOne=True, features='tb', chordQuality='simple', rotateChroma=False, key=False, featureNorm='L1', covType='full', leaveOneOut=3, obsThresh=0)

print "AIC: ", AIC

# number of chords in ground truth
N = A.shape[0]

# fill the HMM with the learned parameters
hmm = ghmm.GHMM(N, labels = labels, pi = pi, A = A, B = B)

# pickle ghmm for future reference
outP = open('trainedhmms/hmm_M=2_sig=full_quality=simple_rotate=0_key=0', 'w')

pickle.dump(hmm, outP)

outP.close()
