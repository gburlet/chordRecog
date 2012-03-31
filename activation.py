import numpy as np

'''
Activation functions
'''

np.seterr(all='raise')

class Sigmoid:
    ''' 
    Sigmoid activation function.
    '''   
    outputMinMax = [0.0, 1.0]

    # from my tests: np.finfo(np.float64).max = 1.7976931348623157e+308
    # and exp(a) < max, thus a < ln(max) => clamp on a should be < 709 to avoid overflow
    clamp = 150

    def __call__(self, a):
        # clamp values to avoid numerical overflow/underflow
        a[a <= -self.clamp] = -self.clamp
        a[a >= self.clamp] = self.clamp

        # apply activation function to clipped network outputs 
        return 1.0 / (1.0 + np.exp(-a))

    def derivative(self, z):
        return z * (1.0 - z)

class ArcTan:
    '''
    Hyperbolic Tangent activation function (commonly referred to as atan or tan^-1)
    '''
    outputMinMax = [-1.0, 1.0]

    def __call__(self, a):
        return np.tanh(a)

    def derivative(self, z):
        return 1.0 - (z ** 2)

class SoftMax:
    '''
    Softmax activation function. Ensures output nodes can be interpreted as
    a probability distribution. That is, the output activations sum to 1 and are all 
    0 <= yk <= 1
    '''
    outputMinMax = [0.0, 1.0]

    def __init__(self, partition = None):
        '''
        Initializes the activation function with the ability to partition output nodes
        to perform activations on each partition seperately.

        PARAMETERS
        ----------
        partition {List}: size of list is the number of partitions, entries are the partition
                          boundaries. Defaults to one partition (all of the outputs).
                          e.g., in list of 12, partition = [6], calculates softmax from 0:5 then 6:12 
        '''
        self._partition = partition

    def __call__(self, a):
        if self._partition is None:
            aMax = np.max(a)
            return np.exp(a - aMax) / np.sum(np.exp(a - aMax))
        else:
            activation = np.zeros_like(a)
            pPart = 0
            for part in self._partition:
                aPartMax = np.max(a[pPart:part])
                activation[pPart:part] = np.exp(a[pPart:part] - aPartMax) / np.sum(np.exp(a[pPart:part] - aPartMax))
                pPart = part
            # now calculate softmax over last partition boundary to the end
            aPartMax = np.max(a[pPart:])
            activation[pPart:] = np.exp(a[pPart:] - aPartMax) / np.sum(np.exp(a[pPart:] - aPartMax))

            return activation

    def derivative(self, z):
        '''
        Derivative calculation does not need to be partitioned since the function takes the yk output neuron
        values as an argument, which have already been calculated with partitions.
        '''
        return z * (1.0 - z)

class Identity:
    '''
    Identity activation function
    '''
    outputMinMax = [-np.Inf, np.Inf]

    def __call__(self, a):
        return a

    def derivative(self, z):
        return np.ones_like(z)
