import numpy as np 

np.random.seed(12345)


  
        
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


class Layer:
    
    def __init__(self, num_nodes, activation):
        self.num_nodes = num_nodes
        self.activation = activation
        
        
class Network:
    
    def __init__(self, num_inputs, network_layers):
        self.network_layers = network_layers
        self.network_state = [np.zeros(num_inputs)]
        self.network_parameters = []
        for index, layer in enumerate(network_layers):
            self.network_state.append(np.zeros(layer.num_nodes))
            if index == 0:
                self.network_parameters.append([2 * np.random.rand(layer.num_nodes,num_inputs) - 1,
                                                2 * np.random.rand(layer.num_nodes) - 1])
            else:
                self.network_parameters.append([2 * np.random.rand(layer.num_nodes, network_layers[index - 1].num_nodes) - 1,
                                                2 * np.random.rand(layer.num_nodes) - 1])
    
    def feed_forward(self, network_input):
        self.network_state[0] = network_input
        for index, (weights, bias) in enumerate(self.network_parameters):
            activation_function = self.network_layers[index].activation
            self.network_state[index + 1] = activation_function(np.dot(weights, self.network_state[index]) + bias)
        
    def compute_gradient(self, batch_input, batch_output, cost_function):
        
        network_gradients = []
        
        #for batch_index, output in enumerate(batch_output):
            
        self.feed_forward(batch_input)
    
        for index, (weights, bias) in reversed(list(enumerate(self.network_parameters))):
            
            #print(weights)
            #print(bias)
            
            print()
            #print(self.network_state)
            print()
            
            
            activation_func = self.network_layers[index].activation
            
            if index == len(self.network_parameters) - 1:
                sigma = np.multiply(cost_function(self.network_state[index + 1], batch_output, True),
                                    activation_func(np.dot(weights, self.network_state[index]) + bias, True))
                network_gradients.append([np.dot(np.transpose([sigma]), [self.network_state[index]]), sigma])
            
            else:   
                sigma = np.multiply(np.dot(np.transpose(self.network_parameters[index + 1][0]), network_gradients[0][1]),
                                    self.network_layers[index].activation(np.dot(weights, self.network_state[index]) + bias, True))
                network_gradients.insert(0, [np.dot(np.transpose([sigma]), [self.network_state[index]]), sigma])
        
        return 0


L1 = Layer(5, sigmoid)

output = Layer(3, sigmoid)

network = Network(3, [L1, output])

input = np.array([1,2,3])

output = np.array([1,1,1])

network.compute_gradient(input, output, mean_squared_error)


