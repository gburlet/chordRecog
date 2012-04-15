import numpy as np
from itertools import izip
import neuralnet as nn
import activation as act
import os

def mixlearnNNbuff(chromaNorm = 'L1', constantQNorm = None, deltaTrain = 2, nnStruct = [256, 150, 24], errorFunc = 'SSE', verbose = False, numDataPass = 1):
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

    # get feature file pointers
    qtransFiles, chromaFiles = process_dir("data/burgoyne2011chords")

    for iDataset in range(numDataPass):
        print "DATASET PASS %d" % (iDataset+1)
        # until all the feature files have been exhausted
        passInd = 0
        while len(qtransFiles) > 0 and len(chromaFiles) > 0:
            # for each pair of feature files that remain
            i = 0
            for qFile, cFile in izip(qtransFiles[:], chromaFiles[:]):
                # open Constant-Q file and restore reading offset
                qFilePtx = open(qFile["path"], 'r')
                qFilePtx.seek(qFile["offset"])
                # read an observation
                qObs = qFilePtx.readline().strip()

                # we've exhausted the observations in this song
                if not qObs:
                    print "DONE WITH SONG: %s" % qFile["path"]
                    # remove from the processing queue
                    del qtransFiles[i]
                    del chromaFiles[i]
                    continue

                # update offset
                qtransFiles[i]["offset"] = qFilePtx.tell()
                # close the file pointer
                qFilePtx.close()

                # open chroma file and restore reading offset
                cFilePtx = open(cFile["path"], 'r')
                cFilePtx.seek(cFile["offset"])
                # read an observation
                cObs = cFilePtx.readline().strip()
                # update offset
                chromaFiles[i]["offset"] = cFilePtx.tell()
                # close the file pointer
                cFilePtx.close()
                i += 1

                if passInd < 50:
                    continue

                qObs = qObs.split(",")
                cObs = cObs.split(",")

                # check features are in sync by timestamp
                if float(cObs[0]) != float(qObs[0]):
                    raise ValueError("Feature files out of sync")

                # get chromagrams
                chroma = np.asfarray(cObs[1:])

                # avoid divide by zero (this is silence in audio)
                if np.sum(chroma) < 3.0:
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
                constantQ = np.asfarray(qObs[1:])
                
                # perform feature normalization
                if constantQNorm is not None and np.sum(constantQ) != 0:
                    if constantQNorm == 'L1':
                        constantQ /= np.sum(np.abs(constantQ))
                    elif constantQNorm == 'L2':
                        constantQ /= np.sum(constantQ ** 2)
                    elif constantQNorm == 'Linf':
                        constantQ /= np.max(np.abs(constantQ))

                Xtrain.append(constantQ)

            # train on this pass
            if len(Xtrain) > 0:
                print "Xtrain: ", len(Xtrain), ", Xtarget: ", len(Xtarget)
                train = np.asarray(Xtrain)
                target = np.asarray(Xtarget)

                trainer.setData(train, target)
                trainNet(trainer, verbose)

                # clear feature buffers
                del Xtrain[:]
                del Xtarget[:]

            passInd += 1
            print "pass: ", passInd

            #if passInd > 2000:
            #    break
               
    if verbose:
        print "Done training neural network."

    return net

def trainNet(trainer, verbose = False):
    '''
    Train the neural net given the set of features.

    PARAMETERS
    ----------
    Xtrain {T,D1}: training data
    Xtarget {T,D2}: target data
    '''
    
    #if verbose:
    #    print "Training ..."
        
    # l-bfgs-b
    #iprint = 1 if verbose else 0
    #optArgs = {'bounds': None, 'm': 100, 'factr': 1e10, 'pgtol': 1e-02, 'iprint': iprint, 'maxfun': 2000}
    #trainer.trainL_BFGS_B(**optArgs)
    
    # adaptive gradient descent
    #optArgs = {'etaInit': 0.95, 'etaInc': 1.1, 'etaDec': 0.65, 'sequential': True, 'maxiter': 1, 'convEps': 1e-2}
    #trainer.trainAdaptGradDesc(**optArgs)

    optArgs = {'eta': 0.75, 'sequential': True, 'maxiter': 1, 'convEps': 1e-5}
    trainer.trainGradDesc(**optArgs)

    #optArgs = {'etaInit': 0.9, 'sequential': True, 'maxiter': 1, 'convEps': 1e-2}
    #trainer.trainDampedGradDesc(**optArgs)

    #optArgs = {'etaInit': 1e-5, 'etaInc': 1.1, 'etaDec': 0.5, 'sequential': True, 'maxiter': 1, 'convEps': 1e-2}
    #trainer.trainIndivAdaptGradDesc(**optArgs)
 
    #if verbose:
    #    print "Done Training."

def process_dir(dir):
    chromaFiles = []
    qtransFiles = []
    cName = "audio_vamp_nnls-chroma_nnls-chroma_bothchroma.csv"
    qName = "audio_vamp_nnls-chroma_nnls-chroma_logfreqspec.csv"

    if not os.path.isdir(dir):
        print "%s is not a directory" % dir
        return
    print "in dir %s" % dir
    for root, dirs, files in os.walk(dir):
        if len(files) > 0:
            # we have files, add them.
            cPresent = False
            if os.path.exists(os.path.join(root, cName)):
                # add chroma file pointer
                chromaFiles.append({"path": os.path.join(root, cName), "offset": 0})
                cPresent = True
            if os.path.exists(os.path.join(root, qName)):
                # add constant q file pointer
                qtransFiles.append({"path": os.path.join(root, qName), "offset": 0})
            elif cPresent:
                raise ValueError("Feature file missing!")

    assert len(qtransFiles) == len(chromaFiles)
    numfiles = len(qtransFiles)
    print "Got %d songs to process" % numfiles

    return qtransFiles, chromaFiles

net = mixlearnNNbuff(verbose = True, nnStruct = [256, 24], deltaTrain = 1, errorFunc = 'KLDiv', chromaNorm = 'L1', constantQNorm = 'L1', numDataPass = 4)
wstar = net.flattenWeights()

# save optimal weights
np.save('trainedweights/wstar_grad_KLDiv_[0]_0.75.npy', wstar)
print "all done!"
