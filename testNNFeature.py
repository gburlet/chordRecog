import numpy as np
import neuralnet as nn
import activation as act
from itertools import izip

def testNNFeature(getSong = 1, chromaNorm = 'L1', constantQNorm = 'L1'):
    # initialize feature storage
    Xtrain = []     # Constant-Q transform
    Xtarget = []    # Bass and treble chromagram

    nnStruct = [256,24]

    # Set up neural network
    # uses sigmoid activation function by default at each layer
    # output activation depends on the type of chromaNorm specified
    activations = [act.Sigmoid()] * (len(nnStruct)-2)
    if chromaNorm == 'L1':
        # partitioned SoftMax output (L1 normalization for each chromagram)
        activations.append(act.SoftMax([12]))
    elif chromaNorm == 'L2':
        activations.append(act.Identity())
    elif chromaNorm == 'Linf':
        activations.append(act.Sigmoid())
    else:
        activations.append(act.Identity())

    # Instantiate neural network
    # assumes full connectivity between layer neurons.
    net = nn.NeuralNet(nnStruct, actFunc=activations)
    
    # load in trained weights
    wstar = np.load("trainedweights/wstar_grad_SSE_[0]_0.95_iter10.npy")
    net.setWeights(wstar)

    # read constant-q transform preliminary features
    qFile = open('data/logfreqspec.csv', 'r')
    # read bass and treble chromagram features
    cFile = open('data/bothchroma.csv', 'r')

    songNum = 0
    songPath = ''
    for cObs, qObs in izip(cFile, qFile):
        cObs = cObs.split(",")
        qObs = qObs.split(",")
        
        # check if we have moved to a new song
        if cObs[0]:
            # check features are in sync by audio file path
            if not qObs[0] or cObs[0] != qObs[0]:
                raise ValueError("Feature files out of sync")

            # run gathered features through the neural net and return the values
            if songNum > 0 and songNum == getSong:
                print "Processing song #%d, %s" % (songNum, songPath)

                train = np.asarray(Xtrain)
                target = np.asarray(Xtarget)
                output = net.calcOutput(train)
                return train, output, target

            songNum += 1
            songPath = cObs[0]

        if songNum != getSong:
            continue 

        # double check features are in sync by timestamp
        if float(cObs[1]) != float(qObs[1]):
            raise ValueError("Feature files out of sync")
        
        # get chromagrams
        chroma = np.asfarray(cObs[2:])

        # avoid divide by zero (this is silence in audio)
        #if np.sum(chroma) == 0:
        #    continue

        # perform feature normalization
        if chromaNorm == 'L1':
            if np.sum(chroma[0:12]) != 0:
                chroma[0:12] /= np.sum(np.abs(chroma[0:12]))
            if np.sum(chroma[12:24]) != 0:
                chroma[12:24] /= np.sum(np.abs(chroma[12:24]))
        elif chromaNorm == 'L2':
            if np.sum(chroma[0:12]) != 0:
                chroma[0:12] /= np.sum(chroma[0:12] ** 2)
            if np.sum(chroma[12:24]) != 0:
                chroma[12:24] /= np.sum(chroma[12:24] ** 2)
        elif chromaNorm == 'Linf':
            if np.sum(chroma[0:12]) != 0:
                chroma[0:12] /= np.max(np.abs(chroma[0:12]))
            if np.sum(chroma[12:24]) != 0:
                chroma[12:24] /= np.max(np.abs(chroma[12:24]))

        Xtarget.append(chroma)

        # get Constant-Q transform
        constantQ = np.asfarray(qObs[2:])
        
        # perform feature normalization
        if constantQNorm is not None and np.sum(constantQ) != 0:
            if constantQNorm == 'L1':
                constantQ /= np.sum(np.abs(constantQ))
            elif constantQNorm == 'L2':
                constantQ /= np.sum(constantQ ** 2)
            elif constantQNorm == 'Linf':
                constantQ /= np.max(np.abs(constantQ))

        Xtrain.append(constantQ)
