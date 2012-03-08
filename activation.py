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

    def derivative(self, x):
        exp_x = np.exp(x)
        return exp_x / ((1.0 + exp_x) ** 2)

class ArcTan:
    '''
    Hyperbolic Tangent activation function (commonly referred to as atan or tan^-1)
    '''
    outputMinMax = [-1.0, 1.0]

    def __call__(self, a):
        return np.tanh(a)

    def derivative(self, x):
        return 1.0 - (np.tanh(x) ** 2)

class Identity:
    '''
    Identity activation function
    '''
    outputMinMax = [-np.Inf, np.Inf]

    def __call__(self, a):
        return a

    def derivative(self, x):
        np.ones_like(x)
