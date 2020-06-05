import numpy as np 

np.random.seed(12345)


class Layer:
    
    def __init__(self, num_nodes, activation):
        self.num_nodes = num_nodes
        self.activation = activation
        
        
class Network:
    
    def __init__(self, num_inputs, network_layers):
        self.network_layers = network_layers
        self.network_state = [np.zeros((num_inputs,1))]
        self.network_parameters = []
        for index, layer in enumerate(network_layers):
            self.network_state.append(np.zeros((layer.num_nodes,1)))
            if index == 0:
                self.network_parameters.append([2 * np.random.rand(layer.num_nodes,num_inputs) - 1,
                                                2 * np.random.rand(layer.num_nodes,1) - 1])
            else:
                self.network_parameters.append([2 * np.random.rand(layer.num_nodes, network_layers[index - 1].num_nodes) - 1,
                                                2 * np.random.rand(1,layer.num_nodes) - 1])
            
    
    def feed_forward(self, network_input):
        self.network_state[0] = network_input
        for index, (weights, bias) in enumerate(self.network_parameters):
            x = np.dot(weights, self.network_state[index]) + bias
            self.network_state[index + 1] = self.network_layers[index].activation(np.dot(weights, self.network_state[index]) + bias)
        
        



  
        
def sigmoid(x, differential = False):
    if differential:
        return np.power(sigmoid(x),2) * np.exp(-x)
    else:
        return 1 / (1 + np.exp(-x))
    
def mean_squared_error(prediction, actual, differential = False):
    if differential:
        return prediction - actual
    else:
        return np.sum(0.5 * np.power(prediction - actual, 2))
        
        
L1 = Layer(10, sigmoid)
L2 = Layer(10, sigmoid)
output = Layer(5, sigmoid)


network = Network(5, [L1,L2,output])


print(network.network_parameters[0][0])
print(network.network_parameters[0][1])
print()

network.feed_forward(np.array([1,1,1,1,1]))

print(network.network_state[-1])