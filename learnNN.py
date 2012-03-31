import numpy as np
from itertools import izip
import neuralnet as nn
import activation as act

def learnNNbuff(chromaNorm = 'L1', constantQNorm = None, deltaTrain = 2, nnStruct = [256, 150, 24], errorFunc = 'SSE', verbose = False):
    '''
    Learns neural network weights with buffered feature input (batch training in segments).
    Use this function when thrashing to disk is a possibility

    PARAMETERS
    ----------
    chromaNorm: L1, L2, Linf, or None, normalization of chroma
    constantQNorm: L1, L2, Linf, or None, normalization of constant q transform
    deltaTrain {int}: how many songs to train after. Buffering features prevents thrashing to disk.
    nnStruct {List}: neural network layer list (each element is number of neurons at the layer
    errorFunc {String}: 'KLDiv' or 'SSE'
    verbose {Boolean}: report progress to standard out

    RETURN
    ------
    net: trained neural network
    '''
    
    # initialize feature storage
    Xtrain = []     # Constant-Q transform
    Xtarget = []    # Bass and treble chromagram

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

    # hire a trainer for the network
    trainer = nn.Trainer(net, errorFunc, 1)

    # read constant-q transform preliminary features
    qFile = open('data/logfreqspec.csv', 'r')
    # read bass and treble chromagram features
    cFile = open('data/bothchroma.csv', 'r')

    songNum = 0
    eligibleTrain = False
    for cObs, qObs in izip(cFile, qFile):
        cObs = cObs.split(",")
        qObs = qObs.split(",")
        
        # check if we have moved to a new song
        if cObs[0]:
            # check features are in sync by audio file path
            if not qObs[0] or cObs[0] != qObs[0]:
                raise ValueError("Feature files out of sync")
                                    
            # train the neural net with buffered features from the previous songs
            if songNum > 0 and songNum % deltaTrain == 0:
                trainer.setData(np.asarray(Xtrain), np.asarray(Xtarget))
                trainNet(trainer, verbose)
                # clear feature buffers
                del Xtrain[:]
                del Xtarget[:]

            songNum += 1

            if verbose:
                print "Processing song: ", cObs[0]
       
        # double check features are in sync by timestamp
        if float(cObs[1]) != float(qObs[1]):
            raise ValueError("Feature files out of sync")
        
        # get chromagrams
        chroma = np.asfarray(cObs[2:])

        # avoid divide by zero (this is silence in audio)
        if np.sum(chroma) == 0:
            continue

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

    # train leftovers (< deltaTrain songs)
    if len(Xtrain) > 0:
        trainer.setData(np.asarray(Xtrain), np.asarray(Xtarget))
        trainNet(trainer, verbose)

    if verbose:
        print "Done training neural network."

    qFile.close()
    cFile.close()

    return net

def learnNN(chromaNorm = 'L1', constantQNorm = 'Linf', deltaTrain = 2, nnStruct = [256, 50, 24], errorFunc = 'SSE', verbose = False):
    '''
    Learns neural network weights with unbuffered feature input: store all features in main memory and do one giant batch train.

    PARAMETERS
    ----------
    chromaNorm: L1, L2, Linf, or None, normalization of chroma
    constantQNorm: L1, L2, Linf, or None, normalization of constant q transform
    nnStruct {List}: neural network layer list (each element is number of neurons at the layer
    errorFunc {String}: 'KLDiv' or 'SSE'
    verbose {Boolean}: report progress to standard out

    RETURN
    ------
    net: trained neural network
    '''

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

    if verbose:
        print "Retrieving Features."

    # read constant-q transform preliminary features
    Xtrain = np.loadtxt('data/logfreqspec.csv', dtype=np.float, delimiter=',', usecols=range(1,257))
    # read bass and treble chromagram features
    Xtarget = np.loadtxt('data/bothchroma.csv', dtype=np.float, delimiter=',', usecols=range(1,25))

    if verbose:
        print "Normalizing Features."

    divInd = np.sum(Xtrain, axis=1) != 0
    # perform feature normalization
    if constantQNorm == 'L1':
        Xtrain[divInd,:] /= np.sum(np.abs(Xtrain[divInd,:]), axis=1)[:,np.newaxis]
    elif constantQNorm == 'L2':
        Xtrain[divInd,:] /= np.sum(Xtrain[divInd,:] ** 2, axis=1)[:,np.newaxis]
    elif constantQNorm == 'Linf':
        Xtrain[divInd,:] /= np.max(np.abs(Xtrain[divInd,:]), axis=1)[:,np.newaxis]
    del divInd

    divIndTreble = np.sum(Xtarget[:,0:12], axis=1) != 0
    divIndBass = np.sum(Xtarget[:,12:24], axis=1) != 0
    # perform feature normalization
    if chromaNorm == 'L1':
        Xtarget[divIndTreble,0:12] /= np.sum(np.abs(Xtarget[divIndTreble,0:12]), axis=1)[:,np.newaxis]
        Xtarget[divIndBass,12:24] /= np.sum(np.abs(Xtarget[divIndBass,12:24]), axis=1)[:,np.newaxis]
    elif chromaNorm == 'L2':
        Xtarget[divIndTreble,0:12] /= np.sum(Xtarget[divIndTreble,0:12] ** 2, axis=1)[:,np.newaxis]
        Xtarget[divIndBass,12:24] /= np.sum(Xtarget[divIndBass,12:24] ** 2, axis=1)[:,np.newaxis]
    elif chromaNorm == 'Linf':
        Xtarget[divIndTreble,0:12] /= np.max(np.abs(Xtarget[divIndTreble,0:12]), axis=1)[:,np.newaxis]
        Xtarget[divIndBass,12:24] /= np.max(np.abs(Xtarget[divIndBass,12:24]), axis=1)[:,np.newaxis]
    del divIndTreble
    del divIndBass

    # batch train Neural Network
    trainNet(Xtrain, Xtarget, net, errorFunc, verbose)

    if verbose:
        print "All done!"

    return net

def trainNet(trainer, verbose = False):
    '''
    Train the neural net given the set of features.

    PARAMETERS
    ----------
    Xtrain {T,D1}: training data
    Xtarget {T,D2}: target data
    '''
    
    if verbose:
        print "Training ..."

    # l-bfgs-b
    iprint = 1 if verbose else 0
    optArgs = {'bounds': (-10,10), 'm': 100, 'factr': 1e7, 'pgtol': 1e-05, 'iprint': iprint, 'maxfun': 15000}
    trainer.trainL_BFGS_B(**optArgs)
    
    # adaptive gradient descent
    # optArgs = {'etaInit': 1e-2, 'etaInc': 1.1, 'etaDec': 0.5, 'sequential': True, 'maxiter': 1, 'convEps': 1e-2}
    # trainer.trainAdaptGradDesc(**optArgs)

    #optArgs = {'eta': 1e-2, 'sequential': True, 'maxiter': 1, 'convEps': 1e-2}
    #trainer.trainGradDesc(**optArgs)
 
    if verbose:
        print "Done Training."

net = learnNNbuff(verbose = True, nnStruct = [256, 300, 200, 50, 24], deltaTrain = 1, errorFunc = 'KLDiv', chromaNorm = 'L1', constantQNorm = None)
wstar = net.flattenWeights()

# save optimal weights
np.save('wstar_adapt.npy', wstar)
print "all done!"
