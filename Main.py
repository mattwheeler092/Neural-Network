import numpy as np 


class Layer:
    
    def __init__(self, inputs, outputs, activation):
        self.num_input = inputs
        self.num_output = outputs
        self.activation = activation
        self.output_vals = None
    
    
class Network:
    
    def __init__(self, network_layers):
        self.network_variables = []
        self.network_state = []
        for index, layer in enumerate(network_layers):
            layer_weights = 2 * np.random.rand(layer.num_output, layer.num_input) - 1
            layer_bias = 2 * np.random.rand(layer.num_output) - 1
            self.network_variables.append([layer_weights, layer_bias])
            if index == 0: self.network_state.append(np.random.rand(layer.num_input))
            self.network_state.append(np.random.rand(layer.num_output))
        self.layers = network_layers
        
    def feed_forward(self, network_input):
        self.network_state[0] = network_input
        for index, (weights, bias) in enumerate(self.network_variables):
            self.network_state[index + 1] = self.layers[index].activation(np.dot(weights, self.network_state[index]) + bias)
            
    def compute_cost(self, actual_output, cost_function):
        return cost_function(self.network_state[-1], actual_output)
   
   




      
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
        
      
s = sigmoid  

MSE = mean_squared_error
        
test = np.array([Layer(5,10,s),Layer(10,10,s),Layer(10,10,s),Layer(10,5,s)])
 
test_network = Network(test)

test_network.feed_forward(np.array([1,2,34,4,5]))

print(test_network.compute_cost(np.array([1,2,3,4,5]), MSE))


