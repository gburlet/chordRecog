import activation as act

class NeuralNet():
    '''
    Creates a model of a feedforward neural network
    '''
    
    def __init__(self, neuronStruct, **args):
        '''
        Creates a model of a feedforward neural network.
        Assumes complete connectivity between each layer's nodes.
        
        PARAMETERS
        ----------
        neuronStruct {1 x N+1}: number of neurons in each of N+1 layers (including input)
        args:
            actFunc {1 x N}: list of activation functions to use in each of N layers (discluding input)
            trainFunc: training function
            errorFunc: error function
        '''

        # number of layers (including hidden and output layers but discluding input layer)
        self.N = N

        # prepare activation functions
        if "actFunc" in args:
            if len(actFn) != N:
                raise ValueError('NeuralNet: invalid number of activation functions')
            actFunc = args["actFunc"]
        else:
            actFunc = [act.Sigmoid()] * N

        if "trainFunc" in args:
            trainFunc = args["trainFunc"]
        else:
            # TODO: training functions
            pass

        if "errorFunc" in args:
            errorFunc = args["errorFunc"]
        else:
            errorFunc = error.SSE()

        # instantiate layers
        self._layers = []
        for i in range(1,N+1):
            self._layers.append(Layer(neuronStruct[i-1], neuronStruct[i], actFn[i-1])

class Layer():
    '''
    Creates a single perceptron layer of a feedforward neural network
    '''

    def __init__(self, D, M, actFn = act.Sigmoid()):
        '''
        Creates a layer object

        PARAMETERS
        ----------
        D: number of input neurons (neurons in the previous layer)
        M: number of neurons in the layer
        actFn {Activation}: activation function for the layer
        '''
        self.D = D
        self.M = M

        # absorb bias parameters into the weight parameters by defining an 
        # additional input variable Xo = 1
        self._w = np.empty([M,D+1])

        self._input = np.zeros(D+1)
        self._input[0] = 1.0

        self.output = np.zeros(M)

        # set activation function for layer
        self._actFn = actFn

        # initialize weights
        self.initWeights()

    def initWeights(self, min = -1.0, max = 1.0):
        self._w = np.random.uniform(min, max, self._w.shape)

    def calcOutput(self, input):
        self.output = self._actFn(np.dot(self._w, self.input[:,np.newaxis]))
