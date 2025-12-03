#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from skimage.util import view_as_windows
from neuralnet import BCELoss, ReluActivation, NeuralNetwork, SGDOptimizer, CrossEntropyLoss, SoftmaxActivation, Layer

#%%
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist.data[:1000].reshape(-1, 28, 28, 1) / 255.0
y = mnist.target.astype(np.int64)[:1000]
y_one_hot = np.zeros((y.size, y.max()+1))
y_one_hot[np.arange(y.size), y] = 1

# %%

class ConvolutionalLayer:
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = np.zeros(out_channel)
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernels = np.random.normal(loc=0.0, scale=0.1, size=(kernel_size, kernel_size, in_channel, out_channel))
        self.kernels_grad = None
        self.bias_grad = None

    def augment(self, x):
        pass

    def pad(self, x):
        pad = self.padding
        return np.pad(x, ((pad,pad), (pad,pad), (0,0)), mode="constant") if pad > 0 else x

    def forward(self, inputs):
        if inputs.ndim == 2:
            inputs = inputs[..., None]
            
        self.inp_back = inputs
        inputs = self.pad(inputs)

        k = self.kernel_size

        input_window = view_as_windows(
            arr_in=inputs, 
            window_shape=(k, k, self.in_channel), 
            step=(self.stride, self.stride, 1)
        )[:, :, 0, :, :, :]

        self.inp_window_back = input_window

        conv = np.sum(input_window[..., None] * self.kernels, axis=(2, 3, 4)) + self.bias

        return conv 
    
    def backward(self, output_grad):
        inputs = self.inp_back
        if inputs.ndim == 2:
            inputs = inputs[..., None]

        inputs_pad = self.pad(inputs)
        k = self.kernel_size
        s = self.stride

        self.bias_grad = np.sum(output_grad, axis=(0, 1))  # (out_channel,)

        inp_windows = self.inp_window_back  # already cached in forward
        H_out, W_out = output_grad.shape[0], output_grad.shape[1]

        inp_cols = inp_windows.reshape(H_out * W_out, -1)  # each row is a flattened patch

        out_flat = output_grad.reshape(H_out * W_out, -1)

        kernels_grad_flat = inp_cols.T.dot(out_flat)
        self.kernels_grad = kernels_grad_flat.reshape(k, k, self.in_channel, self.out_channel)

        kernels_flat = self.kernels.reshape(-1, self.out_channel)  # (k*k*in_channel, out_channel)
        cols_grad = out_flat.dot(kernels_flat.T)                    # (H_out*W_out, k*k*in_channel)

        grad_input_pad = np.zeros_like(inputs_pad, dtype=np.float64)
        idx = 0
        for i in range(0, H_out):
            for j in range(0, W_out):
                patch = cols_grad[idx].reshape(k, k, self.in_channel)
                h_start = i * s
                w_start = j * s
                grad_input_pad[h_start:h_start + k, w_start:w_start + k, :] += patch
                idx += 1

        if self.padding > 0:
            p = self.padding
            grad_input = grad_input_pad[p:-p, p:-p, :]
        else:
            grad_input = grad_input_pad

        return grad_input
        
    
class MaxPoolLayer:
    def __init__(self, pooling_size=2):
        self.pooling_size = pooling_size

    def forward(self, inputs):
        if inputs.ndim == 2:
            inputs = inputs[..., None]  

        self.input_shape = inputs.shape
        
        H, W, C = inputs.shape
        pool_size = self.pooling_size

        input_window = view_as_windows(
            inputs,
            window_shape=(pool_size, pool_size, C),
            step=(pool_size, pool_size, 1)
        )[:, :, 0, :, :, :]

        H_out, W_out = input_window.shape[0], input_window.shape[1]
        reshaped = input_window.transpose(0, 1, 4, 2, 3).reshape(H_out, W_out, C, -1)
        
        pooled = np.max(reshaped, axis=-1)
        self.max_indices = np.argmax(reshaped, axis=-1)  # Cache for backward

        return pooled
    
    def backward(self, output_grad):
        C = self.input_shape[2]
        pool_size = self.pooling_size
        
        input_grad = np.zeros(self.input_shape)
        
        for i in range(output_grad.shape[0]):
            for j in range(output_grad.shape[1]):
                h_start = i * pool_size
                w_start = j * pool_size
                
                for c in range(C):
                    # Get the flattened index of max
                    flat_idx = self.max_indices[i, j, c]
                    
                    # Convert to 2D position in the pool window
                    max_h = flat_idx // pool_size
                    max_w = flat_idx % pool_size
                    
                    # Route gradient to that position
                    input_grad[h_start + max_h, w_start + max_w, c] += output_grad[i, j, c]
        
        return input_grad
    

class FlattenLayer:
    def __init__(self):
        self.input_shape = None
    
    def forward(self, inputs):
        self.input_shape = inputs.shape
        
        return inputs.flatten().reshape(1, -1)
    
    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)

class ConvNeuralNetwork(NeuralNetwork):
    def __init__(self, layers, loss_fn=BCELoss(), epochs=1000, lr=0.01):
        super().__init__(layers, loss_fn, epochs, lr)
    
    def train(self, X, y, batch_size=32):
        optimizer = SGDOptimizer(lr=self.lr)
        n_samples = len(X)
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            correct = 0
            
            # Process one sample at a time
            for i in range(n_samples):
                # Get single image and label
                sample = X[i]  # Shape: (28, 28, 1) for images
                label = y[i]   # Shape: (10,) for one-hot encoded
                
                # Forward pass
                a = sample
                for layer, activation in self.layers:
                    z = layer.forward(a)
                    a = activation.forward(z) if activation else z
                
                # Calculate loss
                loss = self.loss_fn.calculate_fwd(a, label)
                epoch_loss += loss
                
                # Calculate accuracy
                pred = np.argmax(a)
                true_label = np.argmax(label)
                if pred == true_label:
                    correct += 1
                
                # Backprop
                grad = self.loss_fn.calculate_back()
                for layer, activation in reversed(self.layers):
                    grad = activation.backward(grad) if activation else grad
                    grad = layer.backward(grad)
                    optimizer.update(layer)
            
            # Print epoch statistics
            avg_loss = epoch_loss / n_samples
            acc = correct / n_samples
            print(f"Epoch {epoch+1}/{self.epochs}: Loss={avg_loss:.4f}, Acc={acc:.4f}")
    
    def predict(self, features):
        # Handle single image or batch
        if features.ndim == 3:  # Single image (H, W, C)
            a = features
            for layer, activation in self.layers:
                z = layer.forward(a)
                a = activation.forward(z) if activation else z
            return np.argmax(a)
        else:  # Batch of images
            predictions = []
            for i in range(len(features)):
                a = features[i]
                for layer, activation in self.layers:
                    z = layer.forward(a)
                    a = activation.forward(z) if activation else z
                predictions.append(np.argmax(a))
            return np.array(predictions)


#%%        
layer1 = ConvolutionalLayer(1, 6, kernel_size=3, stride=1, padding=1)
activation1 = ReluActivation()
pool1 = MaxPoolLayer(2)
layer2 = ConvolutionalLayer(6, 16, kernel_size=3, stride=1, padding=1)
activation2 = ReluActivation()
pool2 = MaxPoolLayer(2)
fcl1 = Layer(input_size=16*7*7, output_size=120)
activation3 = ReluActivation()
fcl2 = Layer(input_size=120, output_size=10)
activation4 = SoftmaxActivation()
loss_fn = CrossEntropyLoss()

nn = ConvNeuralNetwork(
    layers=[
        (layer1, activation1),
        (pool1, None),
        (layer2, activation2),
        (pool2, None),
        (FlattenLayer(), None),
        (fcl1, activation3),
        (fcl2, activation4)
    ],
    loss_fn=loss_fn,
    epochs=100,
    lr=0.001
)

nn.train(X, y_one_hot)

plt.imshow(X[0].reshape(28, 28), cmap='gray')
prediction = nn.predict(X[0])

# %%
