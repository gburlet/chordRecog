'''
Error functions
'''

class SSE:
    '''
    En = Sum of squared error function (Euclidean loss)
    '''

    def __call__(self, output, target):
        return 0.5 * np.sum((target - output) ** 2)

    def derivative(self, err):
        return err
