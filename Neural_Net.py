import numpy as np 


class Layer:
    
    def __init__(self, num_nodes, activation):
        self.num_nodes = num_nodes
        self.activation = activation
        
        
class Network:
    
    def __init__(self, num_inputs, num_outputs, network_layers = []):
        self.network_layers = network_layers
        self.network_state = [np.zeros(num_inputs),np.zeros(num_outputs)]
        if not network_layers:
            self.network_parameters = [2 * np.random.rand(num_inputs,num_outputs) - 1,
                                       2 * np.random.rand(1,num_outputs) - 1]
        else:
            self.network_parameters = []
            for index, layer in enumerate(network_layers):
                self.network_state.insert(-1, np.zeros(layer.num_nodes))
                if index == 0:
                    self.network_parameters.append([2 * np.random.rand(num_inputs,layer.num_nodes) - 1,
                                                    2 * np.random.rand(1,layer.num_nodes) - 1])
                else:
                    self.network_parameters.append([2 * np.random.rand(network_layers[index - 1].num_nodes,layer.num_nodes) - 1,
                                                    2 * np.random.rand(1,layer.num_nodes) - 1])
            self.network_parameters.append([2 * np.random.rand(network_layers[-1].num_nodes,num_outputs) - 1,
                                            2 * np.random.rand(1,num_outputs) - 1])
            


L1 = Layer(5,1)

network = Network(5,5)

print(network.network_parameters)




