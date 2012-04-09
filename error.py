import numpy as np

'''
Error functions
'''

class SSE:
    '''
    En = Sum of squared error function (Euclidean loss)
    '''

    def __call__(self, output, target):
        return 0.5 * np.sum((output - target) ** 2)

    def derivative(self, output, target):
        return output - target

class KLDiv:
    '''
    Kullback-Leibler divergence. It is the average of the logarithmic difference between the
    probabilities P and Q, where the average is taken using the probabilities P. The KL divergence
    is only defined if P and Q both sum to 1, and both are >= 0 and <= 1.
    '''

    def __call__(self, output, target):
        # avoid divide by zero
        target[target == 0] = 1e-5
        output[output == 0] = 1e-5
        return -np.sum(target * (np.log(output / target)))

    def derivative(self, output, target):
        # avoid divide by zero
        output[output == 0] = 1e-5
        return -(target / output)
