import numpy as np 

def relu(x):
    return np.maximum(0,x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Layer:
    
    def __init__(self, inputs, outputs, activation):
        self.weights = 2 * np.random.rand(outputs, inputs) - 1
        self.bias = 2 * np.random.rand(outputs) - 1
        self.activation = activation
        self.output_vals = None
    
    
class Network:
    
    def __init__(self, network_layers):
        for index, layer in enumerate(network_layers[:-1]):
            if network_layers[index + 1].weights.shape[1] != layer.weights.shape[0]:
                msg = 'Layers {} and {} have unequal output / input nodes'.format(index, index + 1)
                raise ValueError (msg)
        self.layers = network_layers
        self.input = None
        self.output = None
        
    def feedforward(self, network_input):
        self.input = network_input
        layer_output = network_input
        for layer in self.layers:
            layer_output = layer.activation(np.dot(layer.weights, layer_output) + layer.bias)
        self.output = layer_output
        print(layer_output)
      
s = sigmoid  
        
test = np.array([Layer(5,10,s),Layer(10,10,s),Layer(10,10,s),Layer(10,5,s)])
 
test_network = Network(test)

test_network.feedforward(np.array([1,2,3,4,5]))





