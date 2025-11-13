#%%
import numpy as np
import cv2
from scipy.signal import convolve2d
from skimage import data
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from neuralnet import *

# Research adversarial patching

#%%
# Load an example image (cat-like)
img = data.camera()  # shape (512, 512)
img = cv2.resize(img, (64, 64)) / 255 # smaller

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

    def pad(self, x):
        pass

    def forward(self, inputs):
        self.inputs = inputs
        in_h, in_w = inputs.shape[1], inputs.shape[2]
        self.out_h = (in_h - self.kernel_size + 2 * self.padding) // self.stride + 1
        self.out_w = (in_w - self.kernel_size + 2 * self.padding) // self.stride + 1
        output =
