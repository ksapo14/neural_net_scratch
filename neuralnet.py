#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits
import numpy.typing as npt
from typing import List, Tuple, Callable, Any

iris = load_iris()
X = np.array(iris.data)

X = (X-X.mean(axis=0))/X.std(axis=0)

y = np.array(iris.target)
y_one_hot = np.zeros((y.size, y.max()+1))
y_one_hot[np.arange(y.size), y] = 1


#%%
class SoftmaxActivation:
    def forward(self, inputs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        exp_shifted = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        self.output = exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)
        return self.output

    def backward(self, grad_output: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return grad_output

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
        grad_output[self.inputs <= 0] = 0
        return grad_output
    
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
    
class CrossEntropyLoss:
    def calculate_fwd(self, predictions, actuals):
        self.predictions = np.clip(predictions, 1e-9, 1 - 1e-9)
        self.actuals = actuals
        return -np.mean(np.sum(actuals * np.log(self.predictions), axis=1))

    def calculate_back(self):
        n = len(self.predictions)
        return (self.predictions - self.actuals) / n

class SGDOptimizer:
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def update(self, layer):
        # Handle attribute error that keeps popping up for layers without weights/biases
        try:
            # Try convolutional layer first
            if hasattr(layer, 'kernels'):
                if hasattr(layer, 'kernels_grad') and layer.kernels_grad is not None:
                    layer.kernels -= self.lr * layer.kernels_grad
                if hasattr(layer, 'bias') and hasattr(layer, 'bias_grad') and layer.bias_grad is not None:
                    layer.bias -= self.lr * layer.bias_grad
            
            # Try fully connected layer
            elif hasattr(layer, 'weights'):
                if hasattr(layer, 'weights_grad') and layer.weights_grad is not None:
                    layer.weights -= self.lr * layer.weights_grad
                if hasattr(layer, 'bias') and hasattr(layer, 'bias_grad') and layer.bias_grad is not None:
                    layer.bias -= self.lr * layer.bias_grad
        except AttributeError:
            pass

class Layer:
    def __init__(self, *, input_size: int, output_size: int):
        self.output_size = output_size
        self.input_size = input_size
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))

    def forward(self, inputs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)

        self.inputs = inputs
        self.output = inputs @ self.weights + self.bias
        return self.output
    
    def backward(self, grad_output: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.weights_grad = self.inputs.T @ grad_output  # Changed from grad_weights
        self.bias_grad = np.sum(grad_output, axis=0, keepdims=True)  # Changed from grad_biases
        grad_input = grad_output @ self.weights.T
        return grad_input
    
class Dropout:
    def __init__(self, *, prob):
        self.prob = prob
        self.training = True
    
    def forward(self, inputs):
        if not self.training:
            return inputs
        self.screen = np.random.binomial(1, 1-self.prob, size=inputs.shape) / (1 - self.prob)
        return inputs * self.screen
    
    def backward(self, grad_output):
        return grad_output * self.screen

class NeuralNetwork:
    def __init__(self, layers:List[Tuple[Layer, Any]], loss_fn=BCELoss(), epochs=1000, lr=0.01):
        self.layers = layers
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.lr = lr
    def train(self, X, y):
        optimizer = SGDOptimizer(lr=self.lr)
        
        for epoch in range(self.epochs):
            # Forward pass
            a = X
            for layer, activation in self.layers:
                z = layer.forward(a)
                a = activation.forward(z) if activation else z

            loss = self.loss_fn.calculate_fwd(a, y)
            preds = np.argmax(a, axis=1)
            labels = np.argmax(y, axis=1)
            acc = np.mean(preds == labels)

            print(f"Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.2f}")

            # Backprop
            grad = self.loss_fn.calculate_back()
            for layer, activation in reversed(self.layers):
                grad = activation.backward(grad) if activation else grad
                grad = layer.backward(grad)
                optimizer.update(layer)
    def predict(self, features):
        a = features
        for layer, activation in self.layers:
                z = layer.forward(a)
                a = activation.forward(z) if activation else z

        return np.argmax(a, axis=1)


        
# %%
if __name__ == "__main__":
    # X = np.array([
    #     [0,0],
    #     [1,0],
    #     [0,1],
    #     [1,1]
    # ])

    # y = np.array([
    #     [0],
    #     [1],
    #     [1],
    #     [0]
    # ])

    #nn = NeuralNetwork(layers=[
    #    (Layer(input_size=2, output_size=2), ReluActivation()),
    #    (Layer(input_size=2, output_size=1), SigmoidActivation())
    #])
    layer1 = Layer(input_size=4, output_size=8)
    activation1 = ReluActivation()
    dropout1 = Dropout(prob=0)
    layer2 = Layer(input_size=8, output_size=3)
    activation2 = SoftmaxActivation()
    loss_fn = CrossEntropyLoss()

    nn = NeuralNetwork(
        layers=[
            (layer1, activation1),
            # (Layer(input_size=layer1.output_size, output_size=layer1.output_size), dropout1),
            (layer2, activation2)
        ],
        loss_fn=loss_fn,
        epochs=1000,
        lr=0.1
    )

    nn.train(X, y_one_hot)
    print(nn.predict(X))
# %%