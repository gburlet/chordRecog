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
