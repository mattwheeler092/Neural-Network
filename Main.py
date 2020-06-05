import numpy as np 

class Layer:
    
    def __init__(self, inputs, outputs, activation):
        self.num_input = inputs
        self.num_output = outputs
        self.activation = activation
        self.output_vals = None
    
class Network:
    
    def __init__(self, network_layers, cost_function):
        self.network_variables = []
        self.network_state = []
        self.cost_function = cost_function
        for index, layer in enumerate(network_layers):
            layer_weights = 2 * np.random.rand(layer.num_output, layer.num_input) - 1
            layer_bias = 2 * np.random.rand(layer.num_output) - 1
            self.network_variables.append([layer_weights, layer_bias])
            if index == 0: self.network_state.append(np.zeros(layer.num_input))
            self.network_state.append(np.zeros(layer.num_output))
        self.layers = network_layers
        
    def feed_forward(self, network_input):
        self.network_state[0] = network_input
        for index, (weights, bias) in enumerate(self.network_variables):
            self.network_state[index + 1] = self.layers[index].activation(np.dot(weights, self.network_state[index]) + bias)
            
    def compute_cost(self, actual_output):
        return self.cost_function(self.network_state[-1], actual_output)
   
    def compute_gradient(self, actual_output):
        network_gradients = []
        for index, (weights, bias) in reversed(list(enumerate(self.network_variables))):
            
            if index == len(self.network_variables) - 1:
                sigma = np.multiply(self.cost_function(self.network_state[index + 1], actual_output, True), 
                                    self.layers[index].activation(np.dot(weights, self.network_state[index]) + bias, True))
                network_gradients.append([np.dot(np.transpose([sigma]),[self.network_state[index]]), sigma])
                
            else:
                sigma = np.multiply(np.dot(np.transpose(self.network_variables[index + 1][0]), network_gradients[0][1]), 
                                    self.layers[index].activation(np.dot(weights, self.network_state[index]) + bias, True))
                network_gradients.insert(0, [np.dot(np.transpose([sigma]),[self.network_state[index]]), sigma]) 
                
        print(network_gradients)
                
   




      
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
 
test_network = Network(test, MSE)

test_network.feed_forward(np.array([1,2,3,4,5]))

#print(test_network.compute_cost(np.array([1,2,3,4,5])))

test_network.compute_gradient(np.array([1,2,3,4,5]))










