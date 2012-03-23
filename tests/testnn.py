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
    return (0.5 * np.sin(x)) + 0.5

# create train samples
xtrain = np.linspace(-7,7, numTrain).reshape(numTrain,1)
ytrain = f(xtrain)

# Instantiate neural network with 2 layers.
# The hidden layer has 5 neurons, with 1 output.
# assumes full connectivity between layer neurons.
# uses sigmoid activation function by default at each layer.
net = nn.NeuralNet([1, 5, 1], actFunc=[act.Sigmoid(), act.Identity()])

# train using BFGS and sum of squares error
optArgs = {'gtol': 1e-6, 'maxiter': 250}
err = net.train(xtrain, ytrain, method='bfgs', error = 'SSE', show = 25, **optArgs)

# run the input through the network
ytest = net.calcOutput(xtrain)

# plot result
pl.subplot(211)
pl.plot(err)
pl.xlabel('Optimization Iteration')
pl.ylabel('Error (SSE)')

pl.subplot(212)
pl.plot(xtrain, ytrain, '-', xtrain, ytest, '.')
pl.legend(['train target', 'network output'])

pl.show()
