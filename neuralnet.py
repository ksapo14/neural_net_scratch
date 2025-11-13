#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy.typing as npt
from typing import List, Tuple, Callable, Any

# iris = load_iris()
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
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
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

        gradient = (self.predictions - self.actuals) / (self.predictions * (1 - self.predictions) * n)
        return gradient

class SGDOptimizer:
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

class NeuralNetwork:
    def __init__(self, layers:List[Tuple[Layer, Any]]=[], epochs=1000, lr=0.01, batch_size=32):
        self.layers = layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
    
    def train(self, X, y):
        loss_fn = BCELoss()
        optimizer = SGDOptimizer(lr=self.lr)
        
        for epoch in range(self.epochs):
            # Forward pass
            a = X
            for layer, activation in self.layers:
                z = layer.forward(a)
                a = activation.forward(z)

            # Compute loss and accuracy
            loss = loss_fn.calculate_fwd(a, y)
            preds = (a > 0.5).astype(int)
            accuracy = np.mean(preds == y)
            print(f'Epoch {epoch+1}: Loss = {loss:.4f}, Accuracy = {accuracy:.2f}')
            
            # Backward pass
            grad = loss_fn.calculate_back()
            for layer, activation in reversed(self.layers):
                grad = activation.backward(grad)
                grad = layer.backward(grad)
                optimizer.update(layer)


        
# %%
if __name__ == "__main__":
    X = np.array([
        [0,0],
        [1,0],
        [0,1],
        [1,1]
    ])

    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    #nn = NeuralNetwork(layers=[
    #    (Layer(input_size=2, output_size=2), ReluActivation()),
    #    (Layer(input_size=2, output_size=1), SigmoidActivation())
    #])
    layer1 = Layer(input_size=2, output_size=2)
    activation1 = ReluActivation()
    layer2 = Layer(input_size=2, output_size=1)
    activation2 = SigmoidActivation()
    loss_fn = BCELoss()
    optimizer = SGDOptimizer(lr=0.1)

    for i in range(1000):
        z1 = layer1.forward(X)
        a1 = activation1.forward(z1)
        z2 = layer2.forward(a1)
        a2 = activation2.forward(z2)

        loss = loss_fn.calculate_fwd(a2, y)
        preds = (a2 > 0.5).astype(int)
        accuracy = np.mean(preds == y)
        print(f'Loss: {loss}')
        print(f'Accuracy {accuracy}')

        grad = loss_fn.calculate_back()
        grad = activation2.backward(grad)
        grad = layer2.backward(grad)
        grad = activation1.backward(grad)
        grad = layer1.backward(grad)

        optimizer.update(layer2)
        optimizer.update(layer1)
# %%