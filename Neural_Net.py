import numpy as np 

np.random.seed(12345)

#======================================================================================================================================# 
        
def sigmoid(x, differential = False):
    if differential:
        return np.power(sigmoid(x),2) * np.exp(-x)
    else:
        return 1 / (1 + np.exp(-x))
    
#----------------------------------------------------------------------------------------------------------------------------------#

def relu(x, differential = False):
    if differential:
        def map_func(element):
            if element > 0: return 1
            else: return 0
        return np.array(list(map(map_func, x)))
    else:
        return np.maximum(0, x)

#----------------------------------------------------------------------------------------------------------------------------------#
    
def mean_squared_error(prediction, actual, differential = False):
    if differential:
        return prediction - actual
    else:
        return np.sum(0.5 * np.power(prediction - actual, 2))
    
#----------------------------------------------------------------------------------------------------------------------------------#

def step_decay(initial_rate, drop_frac, drop_epoch, epoch):
    return initial_rate * np.power(drop_frac, np.floor(epoch / drop_epoch))

#======================================================================================================================================# 

class Layer:
    
    def __init__(self, num_nodes, activation):
        self.num_nodes = num_nodes
        self.activation = activation
        
#======================================================================================================================================# 
        
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
                
    #----------------------------------------------------------------------------------------------------------------------------------#
    
    def feed_forward(self, network_input):
        self.network_state[0] = network_input
        for index, (weights, bias) in enumerate(self.network_parameters):
            activation_function = self.network_layers[index].activation
            self.network_state[index + 1] = activation_function(np.dot(weights, self.network_state[index]) + bias)
            
    #----------------------------------------------------------------------------------------------------------------------------------#

    def compute_cost(self, batch_input, batch_output, cost_function):
        total_cost = 0
        for index, input in enumerate(batch_input):
            self.feed_forward(input)
            total_cost += cost_function(self.network_state[-1], batch_output[index])
        return total_cost / batch_input.shape[0]
            
    #----------------------------------------------------------------------------------------------------------------------------------#
        
    def compute_gradient(self, batch_input, batch_output, cost_function):
        network_gradients = []
        for batch_index, output in enumerate(batch_output):
            self.feed_forward(batch_input[batch_index])
            for index, (weights, bias) in reversed(list(enumerate(self.network_parameters))):          
                activation_func = self.network_layers[index].activation
                if batch_index == 0:
                    if index == len(self.network_parameters) - 1:
                        sigma = np.multiply(cost_function(self.network_state[index + 1], output, True),
                                            activation_func(np.dot(weights, self.network_state[index]) + bias, True))
                        network_gradients.append([np.dot(np.transpose([sigma]), [self.network_state[index]]), sigma])
                    else:   
                        sigma = np.multiply(np.dot(np.transpose(self.network_parameters[index + 1][0]), network_gradients[0][1]),
                                            self.network_layers[index].activation(np.dot(weights, self.network_state[index]) + bias, True))
                        network_gradients.insert(0, [np.dot(np.transpose([sigma]), [self.network_state[index]]), sigma])
                else:
                    if index == len(self.network_parameters) - 1:
                        sigma = np.multiply(cost_function(self.network_state[index + 1], output, True), 
                                            self.network_layers[index].activation(np.dot(weights, self.network_state[index]) + bias, True))
                    else:
                        sigma = np.multiply(np.dot(np.transpose(self.network_parameters[index + 1][0]), network_gradients[index + 1][1]), 
                                            activation_func(np.dot(weights, self.network_state[index]) + bias, True))   
                    network_gradients[index][0] = network_gradients[index][0] + np.dot(np.transpose([sigma]),[self.network_state[index]])
                    network_gradients[index][1] = network_gradients[index][1] + sigma
                    if batch_index == batch_input.shape[0] - 1:
                        network_gradients[index][0] = network_gradients[index][0] / batch_input.shape[0]
                        network_gradients[index][1] = network_gradients[index][1] / batch_input.shape[0]
        return network_gradients
    
    #----------------------------------------------------------------------------------------------------------------------------------#

    def update_parameters(self, network_gradients, learning_rate):
        for index, (grad_weights, grad_bias) in enumerate(network_gradients):
            self.network_parameters[index][0] -= learning_rate * grad_weights
            self.network_parameters[index][1] -= learning_rate * grad_bias
            
    #----------------------------------------------------------------------------------------------------------------------------------#

    def predict(self, batch_input):
        batch_output = np.zeros((batch_input.shape[0],1))
        for index, input in enumerate(batch_input):
            self.feed_forward(input)
            batch_output[index] = self.network_state[-1]
        return batch_output
            
#======================================================================================================================================# 

L1 = Layer(10, sigmoid)

L2 = Layer(10, sigmoid)

output = Layer(1, sigmoid)

network = Network(1, [output])

input = np.array([[0],[1],[0],[1],[0],[1],[0],[1]])

output = np.array([[0],[1],[0],[1],[0],[1],[0],[1]])

cost = []
x_val = []


"""
for index in range(40000):

    grad = network.compute_gradient(input, output, mean_squared_error)
    network.update_parameters(grad, 0.1)
    cost_val = network.compute_cost(input, output, mean_squared_error)
    cost.append(cost_val)
    x_val.append(index)

import matplotlib.pyplot as plt 

plt.plot(x_val, cost)
plt.show()


for x in input:
    network.feed_forward(x)
    print('\nInput = {}\nOutput = {}\n'.format(x, network.network_state[-1]))

"""

initial = 0.1
drop = 0.5
drop_epoch = 100

x = []
y = []

for epoch in range(1000):
    x.append(epoch)
    y.append(step_decay(initial, drop, drop_epoch, epoch))
    
import matplotlib.pyplot as plt 

plt.plot(x,y)
plt.show()


