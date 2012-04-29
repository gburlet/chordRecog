import os
import numpy as np
from emission import *

def tieStates(pi, A, B, labels, D = 11):
    '''
    Perform state tying on the chord transitions to form a negative binomial distribution duration model.
    append chord state to state labels

    PARAMETERS
    ----------
    pi: (1,N) initial state distribution
    A: (N,N) state transition matrix
    D: number of tied states
    '''

    N = A.shape[0]
    ND = N*D

    piTied = np.zeros([1,ND])
    ATied = np.zeros([ND,ND])
    BTied = []

    for i in range(N):
        # add to piTied
        piTied[0,i*D] = pi[0,i]

        # copy emission distribution pointers from B to BTied
        BTied.extend([B[i]]*D)

        # P(a_ii) + P(a_ij)
        ATied[i*D:(i*D)+D,i*D:(i*D)+D] = (A[i,i] ** (1.0/D))*np.eye(D) + np.diag([1-(A[i,i]**(1.0/D))]*(D-1), 1)
        
        # transitions to first states of other chords
        scale = (1 - (A[i,i] ** (1.0/D))) / (1 - A[i,i])
        for j in range(N):
            if j == i:
                continue
            ATied[(i+1)*D-1,j*D] = A[i,j] * scale

    # update labels by appending state index
    labelsTied = []
    for q in labels:
        for i in range(D):
            labelsTied.append(q + '_' + str(i+1))

    return piTied, ATied, BTied, labelsTied

def learnHMM(M, addOne = True, features = 'tb', chordQuality = 'simple', rotateChroma = False, key = False, featureNorm = 'L1', covType = 'full', holdOut = (-1,-1), obsThresh = 0):
    '''
    PARAMETERS
    ----------
    M: number of gaussians to mix for emission probabilities
    addOne: whether one should be added to the transition matrices before normalization
    features: treble chroma 't' or bass chroma 'b' or both
    featureNorm: L1, L2, Linf, or none normalization
    covType: type of the covariance matrix for the GMM emission distribution
             'diag', 'full'
    holdOut: song range (inclusive bounds) to leave out of the training phase and save for model validation (not song id, but in order of reading)

    RETURNS
    -------
    pi {1xN}: initial state probabilities
    A {NxN}: state transition probabilities
    B {1xN}: emission distribution for each state
    '''

    #groundTruth = open('../data/gtruth_chroma.csv', 'r')
    # 0, 1, 7, 10, 11, 12, 13
    # sid, timestamp, local.tonic.name, root.name, root.pc, quality, simple.quality, obs
    #  0,      1,            2,             3,        4,       5,          6,         7-

    groundTruth = open('data/gtruth_chroma_enharmonic.csv', 'r')
    # 0, 1, 7, 10, 11, 12
    # sid, timestamp, local.tonic.pc, root.pc, quality, simple.quality, obs  
    #  0,      1,            2,          3,       4,         5,          6-

    piDict = {}
    aDict = {}
    bDict = {}
    QLabels = set()

    if features == 'tb':
        D = 24
    else:
        D = 12

    pSid = -1
    songNum = 0;
    pChordName = ""
    
    # initialize hold out data storage
    Xtest = {}
    Ytest = {}

    for obsNum, obs in enumerate(groundTruth):        
        obs = obs.split(",")
        sid = int(obs[0])
        
        '''
        # for non-enharmonic dataset
        tonic = obs[2]
        root = obs[3]
        rootPc = obs[4]
        fQuality = obs[5]
        sQuality = obs[6]

        if features == 't':
            chroma = np.asfarray(obs[7:19])
        elif features == 'b':
            chroma = np.asfarray(obs[19:31])
        elif features == 'tb':
            chroma = np.asfarray(obs[7:31])
        '''

        # for enharmonic dataset
        tonic = obs[2]
        root = obs[3]
        rootPc = root
        fQuality = obs[4]
        sQuality = obs[5]

        if features == 't':
            chroma = np.asfarray(obs[6:18])
        elif features == 'b':
            chroma = np.asfarray(obs[18:30])
        elif features == 'tb':
            chroma = np.asfarray(obs[6:30])

        # skip silence (really there are no chords in either the treble or bass)            
        if np.sum(chroma) == 0:
            continue

        # form chord name
        if key:
            if tonic == "NA":
                continue
            chordName = tonic + '.'
        else:
            chordName = ''

        if chordQuality == 'full':
            chordName += root + fQuality
        else:
            if sQuality == "NA":
                chordName = "NA"
            else:
                chordName += root + sQuality

        # first timestamps in songs usually have no annotation
        #if chordName.strip() == "NANA":
        #    continue
        
        if rotateChroma:
            # override chord name with quality
            chordName = sQuality if chordQuality == 'simple' else fQuality

            # rotate chroma
            try:
                pc = int(rootPc)
            except ValueError:
                # root.pc is NA
                # nothing we can do here
                continue
            
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

        if sid != pSid:
            # new song
            pSid = sid
            songNum += 1

            if songNum >= holdOut[0] and songNum <= holdOut[1]:
                # this observation is part of the validation set
                print "holdout song id: ", sid, "number: ", songNum
                Xtest[sid] = [chroma]
                Ytest[sid] = [chordName]
            else:
                # this observation is part of the training set
                print "training song id: ", sid, "number: ", songNum

                # update pi
                if chordName in piDict:
                    piDict[chordName] += 1
                else:
                    piDict[chordName] = 1

                pChordName = chordName
        else:
            # same song
            if songNum >= holdOut[0] and songNum <= holdOut[1]:
                # this observation is part of the validation set
                Xtest[sid].append(chroma)
                Ytest[sid].append(chordName)
            else:
                #update A
                if pChordName in aDict:
                    if chordName in aDict[pChordName]:
                        aDict[pChordName][chordName] += 1.0
                    else:
                        aDict[pChordName][chordName] = 1.0
                else:
                    aDict[pChordName] = {chordName: 1.0}

                pChordName = chordName
            
                # update B
                if chordName in bDict:
                    bDict[chordName].append(chroma)
                else:
                    bDict[chordName] = [chroma]

                # update state labels (this is a set, only unique chord names are added)
                QLabels.add(chordName)

    # close file pointer    
    groundTruth.close()

    print "done parsing, learning state transitions and emissions ..."
    QLabels = sorted(QLabels) # turns set into ordered list
    
    # prune states with insufficient observations
    QLabels = [q for q in QLabels if len(bDict[q]) >= obsThresh]

    N = len(QLabels)

    # initialize arrays
    pi = np.zeros([1,N])
    A = np.zeros([N,N])
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
        
        # fill B
        # convert to numpy array
        Xtrain = np.asarray(bDict[q])
        del bDict[q]
        print "learning emissions for chord: %s, index: %d, #obs: %d" % (q, i, Xtrain.shape[0])
        bGMM = GMM(M, D, covType, zeroCorr=1e-12)
        lnP_history = bGMM.expectMax(Xtrain, maxIter=150, convEps=1e-3, verbose=True)
        B.append(bGMM)

        # update running total of lnP for AIC
        lnP += lnP_history[-1]

    # add one to state transition probabilities before normalizing
    if addOne:
        pi += 1.0
        A += 1.0

    # transform transition histograms to probabilities
    pi /= np.sum(pi)
    A /= A.sum(axis=1)[:,np.newaxis]

    # convert list of holdout test observations to numpy array
    for i in Xtest:
        Xtest[i] = np.asarray(Xtest[i])

    # AIC calculation
    if covType == 'full':
        numCovar = D ** 2
    else:
        numCovar = D

    k = (M*(D + numCovar) + M-1) * len(QLabels)
    AIC = 2*k - 2*lnP

    return pi, A, B, QLabels, Xtest, Ytest, AIC
