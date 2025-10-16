import numpy as np
import matplotlib.pyplot as plt

# Research Dropouts
# Find how to implement regularization
class Backpropagation:
    pass

class ReluActivation:
    def __init__(self, inputs):
        self.inputs = inputs
        self.output = None

    def forward(self):
        self.output = np.maximum(0, self.inputs)

class SigmoidActivation:
    def __init__(self, inputs):
        self.inputs = inputs
        self.output = None

    def forward(self):
        self.output = 1 / (1 + np.exp(-self.inputs))

class Node:
    def __init__(self):
        self.inbound_nodes = []
        self.outbound_nodes = []
        self.value = None 

class Edge:
    def __init__(self, inbound_node, outbound_node):
        self.inbound_node = inbound_node
        self.outbound_node = outbound_node
        inbound_node.outbound_nodes.append(outbound_node)
        outbound_node.inbound_nodes.append(inbound_node)

        self.weight = np.random.randn() * np.sqrt(2 / self.inbound_node.value.shape[1]) # He initialization
        self.bias = 0.0

        self.activation = ReluActivation(self.inbound_node.value * self.weight + self.bias)

    def forward(self):
        self.activation.forward()
        self.outbound_node.value += self.activation.output 

class InputNode:
    def __init__(self, value):
        self.value = value
        self.outbound_nodes = []

class OutputNode:
    def __init__(self):
        self.value = None
        self.activation = SigmoidActivation(self.value)

    def forward(self):
        self.activation.inputs = self.value
        self.activation.forward()
        self.value += self.activation.output

class NeuralNetwork:
    pass