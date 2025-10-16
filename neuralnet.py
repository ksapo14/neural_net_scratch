import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy.typing as npt
from typing import List, Tuple, Callable, Any

iris = load_iris()
X = np.array(iris.data)
y = np.array(iris.target)

# Research Dropouts
# Find how to implement regularization

class SigmoidActivation:
    def forward(self, inputs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

class ReluActivation:
    def forward(self, inputs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.output = np.maximum(0, inputs)
        return self.output
    
class BCELoss:
    def calculate_fwd(self, predictions, actuals):
        self.predictions = predictions
        self.actuals = actuals

        loss = -np.mean(actuals * np.log(predictions) + (1 - actuals) * np.log(1 - predictions))
        return loss
    
    def calculate_back(self, predictions, actuals):
        self.predictions = predictions
        self.actuals = actuals
        n = len(self.predictions)

        gradient = -(actuals / predictions - (1 - actuals) / (1 - predictions)) / n
        return gradient

class Layer:
    def __init__(self, input_size: int, output_size: int):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeroes((1, output_size))

    def forward(self, inputs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.inputs = inputs
        self.output = inputs @ self.weights + self.biases

        return self.output
    
    def backward(self, grad_output: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.grad_weights = self.inputs.T @ grad_output
        self.grad_biases = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.weights.T)

        return grad_input

class OutputLayer(Layer):
    def __init__(self, input_size: int, output_size: int):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
        self.biases = np.zeroes((1, output_size))
