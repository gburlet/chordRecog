import numpy as np

'''
Activation functions
'''

class Sigmoid:
    ''' 
    Sigmoid activation function.
    '''   
    outputMinMax = [0.0, 1.0]

    def __call__(self, a):
        return 1.0 / (1.0 + np.exp(-a))

    def derivative(self, a):
        return self(a) * (1.0 - self(a))

class ArcTan:
    '''
    Hyperbolic Tangent activation function (commonly referred to as atan or tan^-1)
    '''
    outputMinMax = [-1.0, 1.0]

    def __call__(self, a):
        return np.tanh(a)

    def derivative(self, a):
        return 1.0 - (self(a) ** 2)

class SoftMax:
    '''
    Softmax activation function. Ensures output nodes can be interpreted as
    a probability distribution. That is, the output nodes sum to 1 and are all 
    0 <= yk <= 1
    '''
    outputMinMax = [0.0, 1.0]

    def __call__(self, a):
        return np.exp(a) / np.sum(np.exp(a))

    def derivative(self, a):
        return self(a) * (1.0 - self(a))

class Identity:
    '''
    Identity activation function
    '''
    outputMinMax = [-np.Inf, np.Inf]

    def __call__(self, a):
        return a

    def derivative(self, a):
        return np.ones_like(a)
