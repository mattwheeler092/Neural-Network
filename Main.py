import numpy as np 

np.random.seed(12345)

class Layer:
    
    def __init__(self, inputs, outputs, activation):
        self.num_input = inputs
        self.num_output = outputs
        self.activation = activation
    
class Network:
    
    def __init__(self, network_layers):
        self.network_parameters = []
        self.network_state = []
        for index, layer in enumerate(network_layers):
            layer_weights = 2 * np.random.rand(layer.num_output, layer.num_input) - 1
            layer_bias = 2 * np.random.rand(layer.num_output) - 1
            self.network_parameters.append([layer_weights, layer_bias])
            if index == 0: self.network_state.append(np.zeros(layer.num_input))
            self.network_state.append(np.zeros(layer.num_output))
        self.layers = network_layers
        
    def feed_forward(self, network_input):
        self.network_state[0] = network_input
        for index, (weights, bias) in enumerate(self.network_parameters):
            self.network_state[index + 1] = self.layers[index].activation(np.dot(weights, self.network_state[index]) + bias)
            
    def compute_cost(self, batch_input, batch_output, cost_function):
        cost_value = 0
        for index, output in enumerate(batch_output):
            self.feed_forward(batch_input[index])
            cost_value += cost_function(self.network_state[-1], output)
        return cost_value / batch_input.shape[0]
   
    def compute_gradient(self, batch_input, batch_output, cost_function):
        network_gradients = []
        for batch_index, output in enumerate(batch_output):
            self.feed_forward(batch_input[batch_index])
            for index, (weights, bias) in reversed(list(enumerate(self.network_parameters))):
                if index == len(self.network_parameters) - 1:
                    sigma = np.multiply(cost_function(self.network_state[index + 1], output, True), 
                                        self.layers[index].activation(np.dot(weights, self.network_state[index]) + bias, True))
                    if batch_index == 0:
                        network_gradients.append([np.dot(np.transpose([sigma]),[self.network_state[index]]), sigma])
                    elif batch_index == batch_input.shape[0] - 1:
                        network_gradients[index][0] = (network_gradients[index][0] + np.dot(np.transpose([sigma]),[self.network_state[index]])) / batch_input.shape[0]
                        network_gradients[index][1] = (network_gradients[index][1] + sigma) / batch_input.shape[0]
                    else:
                        network_gradients[index][0] = network_gradients[index][0] + np.dot(np.transpose([sigma]),[self.network_state[index]])
                        network_gradients[index][1] = network_gradients[index][1] + sigma
                else:
                    if batch_index == 0:
                        sigma = np.multiply(np.dot(np.transpose(self.network_parameters[index + 1][0]), network_gradients[0][1]), 
                                            self.layers[index].activation(np.dot(weights, self.network_state[index]) + bias, True))
                        network_gradients.insert(0, [np.dot(np.transpose([sigma]),[self.network_state[index]]), sigma]) 
                    elif batch_index == batch_input.shape[0] - 1:
                        sigma = np.multiply(np.dot(np.transpose(self.network_parameters[index + 1][0]), network_gradients[index + 1][1]), 
                                            self.layers[index].activation(np.dot(weights, self.network_state[index]) + bias, True))
                        network_gradients[index][0] = (network_gradients[index][0] + np.dot(np.transpose([sigma]),[self.network_state[index]])) / batch_input.shape[0]
                        network_gradients[index][1] = (network_gradients[index][1] + sigma) / batch_input.shape[0]
                    else:
                        sigma = np.multiply(np.dot(np.transpose(self.network_parameters[index + 1][0]), network_gradients[index + 1][1]), 
                                            self.layers[index].activation(np.dot(weights, self.network_state[index]) + bias, True))
                        network_gradients[index][0] = network_gradients[index][0] + np.dot(np.transpose([sigma]),[self.network_state[index]])
                        network_gradients[index][1] = network_gradients[index][1] + sigma      
        return network_gradients
    
    def update_parameters(self, gradient_values, learning_rate):
        for index, (grad_weights, grad_bias) in enumerate(gradient_values):
            self.network_parameters[index][0] -= learning_rate * grad_weights
            self.network_parameters[index][1] -= learning_rate * grad_bias
            
                


      
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

x = np.array([np.array([1,2,3,4,5]),np.array([5,4,3,2,1])])

#print(test_network.compute_cost(np.array([1,2,3,4,5])))

y = test_network.compute_gradient(x, x, MSE)

print(y)







