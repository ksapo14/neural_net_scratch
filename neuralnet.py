import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from typing import List, Tuple, Callable, Any

iris = load_iris()
X = np.array(iris.data)
y = np.array(iris.target)

# Research Dropouts
# Find how to implement regularization

class Layer:
    def __init__(self, input_size: int, output_size: int):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeroes((1, output_size))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs @ self.weights + self.biases

        return self.output
    
    def backward(self, grad_output):
        self.grad_weights = self.inputs.T @ grad_output
        self.grad_biases = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.weights.T)

        return grad_input

class OutputLayer(Layer):
    def __init__(self, input_size: int, output_size: int):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
        self.biases = np.zeroes((1, output_size))
