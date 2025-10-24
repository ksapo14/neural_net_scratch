#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy.typing as npt
from typing import List, Tuple, Callable, Any

iris = load_iris()
# X = np.array(iris.data)

# X = (X-X.mean(axis=0))/X.std(axis=0)

# y = np.array(iris.target)
# y = (y == 1).astype(int)
# Research Dropouts
# Find how to implement regularization

#%%
class SoftmaxActivation:
    def forward(self, inputs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        exp_shifted = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        self.output = exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)
        return self.output

    def backward(self, grad_output: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        s = self.output.reshape(-1, 1)
        j = np.diagflat(s) - np.dot(s, s.T)
        return np.dot(j, grad_output)

class SigmoidActivation:
    def forward(self, inputs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output
    
    def backward(self, grad_output: float) -> float:
        # Derivative: sigmoid(x) * (1 - sigmoid(x))
        sigmoid_derivative = self.output * (1 - self.output)
        return grad_output * sigmoid_derivative


class ReluActivation:
    def forward(self, inputs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output
    
    def backward(self, grad_output: float):
        return grad_output * (self.inputs > 0)
    
class BCELoss:
    def calculate_fwd(self, predictions: npt.NDArray[np.float64], actuals: npt.NDArray[np.float64]) -> float:
        self.predictions = predictions
        self.actuals = actuals

        loss = -np.mean(actuals * np.log(predictions) + (1 - actuals) * np.log(1 - predictions))
        return loss
    
    def calculate_back(self) -> float:
        n = len(self.predictions)

        gradient = (self.predictions - self.actuals) / n
        return gradient

class SDGOptimizer:
    def __init__(self, *, lr):
        self.lr = lr

    def update(self, layer):
        layer.weights -= self.lr * layer.grad_weights
        layer.biases -= self.lr * layer.grad_biases

class Layer:
    def __init__(self, *, input_size: int, output_size: int):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))

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
        self.biases = np.zeros((1, output_size))

class NeuralNetwork():
    def __init__(self, layers:List[Tuple[Layer, Any]]=[], epochs=1000, lr=0.01, batch_size=32):
        self.layers = layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
    
    def train(self, X, y):
        z = 0
        a = 0
        loss_fn = BCELoss()
        optimizer = SDGOptimizer(lr=self.lr)
        
        for epoch in range(self.epochs):
            for layer, activation in self.layers:
                z = layer.forward(X)
                a = activation.forward(z)

            loss = loss_fn.calculate_fwd(a3, y)
            preds = (a3 > 0.5).astype(int)
            accuracy = np.mean(preds == y)
            print(f'Loss: {loss}')
            print(f'Accuracy {accuracy}')

            grad = loss_fn.calculate_back()
            for layer, activation in reversed(self.layers):
                grad = activation.backward(grad)
                grad = layer.backward(grad)
                optimizer.update(layer)

        
# %%
X = np.array([
    [0,0],
    [1,0],
    [0,1],
    [1,1]
])

y = np.array([
    [1],
    [0],
    [0],
    [1]
])

l1 = Layer(input_size=2, output_size=2)
l2 = Layer(input_size=2, output_size=3)
l3 = Layer(input_size=3, output_size=1)

relu1 = ReluActivation()
relu2 = ReluActivation()
sigmoid = SigmoidActivation()

loss_fn = BCELoss()
optimizer = SDGOptimizer(lr=0.1)

for epoch in range(1000):

    z1 = l1.forward(X)
    a1 = relu1.forward(z1)
    z2 = l2.forward(a1)
    a2 = relu2.forward(z2)
    z3 = l3.forward(a2)
    a3 = sigmoid.forward(z3)


    loss = loss_fn.calculate_fwd(a3, y)
    preds = (a3 > 0.5).astype(int)
    accuracy = np.mean(preds == y)
    print(f'Loss: {loss}')
    print(f'Accuracy {accuracy}')

    # Backprop
    grad = loss_fn.calculate_back()
    grad = sigmoid.backward(grad)
    grad = l3.backward(grad)
    grad = relu2.backward(grad)
    grad = l2.backward(grad)
    grad = relu1.backward(grad)
    grad = l1.backward(grad)

    optimizer.update(l3)
    optimizer.update(l2)
    optimizer.update(l1)
# %%