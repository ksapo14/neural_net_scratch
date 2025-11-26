#%%
import numpy as np
import cv2
from scipy.signal import convolve2d
from skimage import data
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from skimage.util import view_as_windows
from neuralnet import ReluActivation, NeuralNetwork, SGDOptimizer, CrossEntropyLoss, SoftmaxActivation, Layer
# Research adversarial patching

#%%
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

img = mnist.data[0].reshape(28, 28).astype(np.uint8)
img = img[:,:, None]

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

    def augment(self):
        pass

    def pad(self, x):
        pad = self.padding
        return np.pad(x, ((pad,pad), (pad,pad), (0,0)), mode="constant") if pad > 0 else x

    def forward(self, inputs):
        self.inp_back = inputs
        inputs = self.pad(inputs)

        k = self.kernel_size

        input_window = view_as_windows(
            arr_in=inputs, 
            window_shape=(k, k, self.in_channel), 
            step=(self.stride, self.stride, 1)
        ) [:, :, 0, :, :, :]

        self.inp_window_back = input_window

        conv = np.sum(input_window[..., None] * self.kernels, axis=(2, 3, 4)) + self.bias
        # Solution for shape issue from ChatGPT
        # if conv.shape[-1] == 1:
        #     conv = conv[..., 0]   # squeeze last axis if only one output channel

        return conv 
    
    def backward(self, output_grad):
        inputs = self.inp_back
        inputs_pad = self.pad(inputs)
        k_size = self.kernel_size

        bias_grad = np.sum(output_grad, axis=(0,1))

        inputs_reshaped = self.inp_window_back[..., None]
        output_grad_reshaped = output_grad[:, :, None, None, None, :]

        kernels_grad = np.sum(inputs_reshaped * output_grad_reshaped, axis=(0,1))

        grad_input_pad = np.zeros_like(inputs_pad)

        for i in range(output_grad.shape[0]):
            for j in range(output_grad.shape[1]):
                h_start = i * self.stride
                w_start = j * self.stride

                grad_window = np.sum(self.kernels * output_grad[i, j], axis=-1)

                grad_input_pad[h_start:h_start+k_size, w_start:w_start+k_size, :] += grad_window

        if self.padding > 0:
            pad = self.padding
            grad_input = grad_input_pad[pad:-pad, pad:-pad, :]
        else:
            grad_input = grad_input_pad

        self.kernels_grad = kernels_grad
        self.bias_grad = bias_grad

        return grad_input

        
    
class MaxPoolLayer:
    def __init__(self, pooling_size=2):
        self.pooling_size = pooling_size

    def forward(self, inputs):
        # ensure 3D
        if inputs.ndim == 2:
            inputs = inputs[..., None]  # (H,W,1)

        C = inputs.shape[2]

        # only pool over H and W
        input_window = view_as_windows(
            inputs,
            window_shape=(self.pooling_size, self.pooling_size, C),
            step=(self.pooling_size, self.pooling_size, C)
        ) [:, :, 0, :, :, :]

        pooled = np.max(input_window, axis=(2,3))

        # # Wacky output extra dimension solution from Claude
        # pooled = pooled.squeeze(axis=-1)

        return pooled
    
    def backward(self, output_grad):
        C = self.input_shape[2]
        pool_size = self.pooling_size
        
        input_grad = np.zeros(self.input_shape)
        
        # For each output position, route gradient to max position
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

conv = layer1.forward(img)
activated = activation1.forward(conv)
pooled = pool1.forward(activated)
conv2 = layer2.forward(pooled)
activated2 = activation2.forward(conv2)
pooled2 = pool2.forward(activated2)

nn = NeuralNetwork(
    layers=[
        (layer1, activation1),
        (pool1, None),
        (layer2, activation2),
        (pool2, None),
        (fcl1, activation3),
        (fcl2, activation4)
    ],
    loss_fn=loss_fn,
    epochs=10,
    lr=0.01
)
# %%
