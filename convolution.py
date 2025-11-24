#%%
import numpy as np
import cv2
from scipy.signal import convolve2d
from skimage import data
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from skimage.util import view_as_windows

# Research adversarial patching

#%%
# Load an example image (cat-like)
img = data.camera()  # shape (512, 512)
img = cv2.resize(img, (64, 64)) / 255 # smaller
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
        return np.pad(x, ((0,0), (pad,pad), (pad,pad), (0,0)), mode="constant") if pad > 0 else x

    def forward(self, inputs):
        inputs = self.pad(inputs)

        k = self.kernel_size

        input_window = view_as_windows(
            arr_in=inputs, 
            window_shape=(k, k, self.in_channel), 
            step=(self.stride, self.stride, 1)
        )

        conv = np.sum(input_window[..., None] * self.kernels, axis=(2, 3, 4)) + self.bias
        # Solution for shape issue from ChatGPT
        if conv.shape[-1] == 1:
            conv = conv[..., 0]   # squeeze last axis if only one output channel

        return conv 
    
class MaxPoolLayer:
    def __init__(self, pooling_size=2):
        self.pooling_size = pooling_size

    def forward(self, inputs):
        # ensure 3D
        if inputs.ndim == 2:
            inputs = inputs[..., None]  # (H,W,1)

        H, W, C = inputs.shape

        # only pool over H and W
        input_window = view_as_windows(
            inputs,
            window_shape=(self.pooling_size, self.pooling_size, 1),
            step=(self.pooling_size, self.pooling_size, 1)
        )

        pooled = np.max(input_window, axis=(2,3))
        
        # Wacky output extra dimension solution from claude
        pooled = pooled.squeeze(axis=-1)

        return pooled
#%%        
layer = ConvolutionalLayer(1, 1)
pool = MaxPoolLayer(2)

conv = layer.forward(img)
pooled = pool.forward(conv)

out_img = pooled[..., 0]

# %%
