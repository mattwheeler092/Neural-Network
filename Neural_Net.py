import numpy as np 


class Layer:
    
    def __init__(self, num_nodes, activation):
        self.num_nodes = num_nodes
        self.activation = activation
        
        
class Network:
    
    def __init__(self, num_inputs, num_outputs, layers):
        
        