import numpy as np 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

test = 5

class Layer:
    
    def __init__(self, inputs, outputs, activation = None):
        self.num_input = inputs
        self.num_output = outputs
        self.weights = np.random.rand(outputs, inputs)
        self.bias = np.random.rand(outputs)
        self.activation = activation
        
    def feedforward(self, input):
        if input.size != self.num_input:
            raise ValueError ('Input array must contain {} elements.'.format(self.num_input))
        layer_output = np.dot(self.weights, input) + self.bias
        if self.activation is not None:
            return self.activation(layer_output)
        else:
            return 1 / (1 + np.exp(-layer_output))
        


L1 = Layer(5,10,sigmoid)
L2 = Layer(10,10,sigmoid)
L3 = Layer(10,5,sigmoid)

input = np.random.rand(5)

FF1 = L1.feedforward(input)
FF2 = L2.feedforward(FF1)
FF3 = L3.feedforward(FF2)

print(FF3)