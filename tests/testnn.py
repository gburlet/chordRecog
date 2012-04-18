import sys
sys.path.insert(0, "..")

import neuralnet as nn
import activation as act
import numpy as np
import matplotlib.pyplot as pl

'''
This script tests the implemented neural network as a simple universal function approximator.
Tries to learn y(x) = 0.5*sin(x)
'''
np.seterr(all='raise')

numTrain = 250

def f(x):
    return (0.5 * np.sin(x)) + 0.5, (0.3 * np.cos(x)) + 0.3

# create train samples
xtrain = np.linspace(-7,7, numTrain).reshape(numTrain,1)
ytrain = np.zeros([numTrain,2])
ytrain[:,0] = f(xtrain)[0].squeeze()
ytrain[:,1] = f(xtrain)[1].squeeze()

# Instantiate neural network with 2 layers.
# The hidden layer has 5 neurons, with 1 output.
# assumes full connectivity between layer neurons.
# uses sigmoid activation function by default at each layer.
net = nn.NeuralNet([1, 100, 50, 2], actFunc=[act.Sigmoid(), act.Sigmoid(), act.ArcTan()])

# hire a trainer for the network
trainer = nn.Trainer(net, 'SSE', 25, xtrain, ytrain)

# train using BFGS and sum of squares error
#optArgs = {'gtol': 1e-6, 'maxiter': 250}
#trainer.trainBFGS(**optArgs)

optArgs = {'bounds': None, 'm': 1000, 'factr': 1e7, 'pgtol': 1e-02, 'iprint': 1, 'maxfun': 1500}
trainer.trainL_BFGS_B(**optArgs)

# run the input through the network
ytest = net.calcOutput(xtrain)

# plot result
pl.subplot(211)
pl.plot(trainer.errHistory)
pl.xlabel('Optimization Iteration')
pl.ylabel('Error (SSE)')

pl.subplot(212)
pl.plot(xtrain, ytrain[:,0], '-')
pl.plot(xtrain, ytrain[:,1], '-')
pl.plot(xtrain, ytest[:,0], '.')
pl.plot(xtrain, ytest[:,1], '.')
pl.legend(['train target', 'network output'])

pl.show()
