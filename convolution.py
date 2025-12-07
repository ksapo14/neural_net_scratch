#%% Loading Nescessary Libraries
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from skimage.util import view_as_windows
from neuralnet import BCELoss, ReluActivation, NeuralNetwork, SGDOptimizer, CrossEntropyLoss, SoftmaxActivation, Layer, Dropout

#%%
# Import a random image for testing
ranom_image = Image.open("handwritten_digit.png")
random_image = np.array(ranom_image.resize((28, 28)).convert("L")) / 255.0

# Import mnist and one hot encode
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist.data[:1000].reshape(-1, 28, 28, 1) / 255.0
y = mnist.target.astype(np.int64)[:1000]
y_one_hot = np.zeros((y.size, y.max()+1))
y_one_hot[np.arange(y.size), y] = 1

X_validation = mnist.data[1000:1200].reshape(-1, 28, 28, 1) / 255.0
y_validation = mnist.target.astype(np.int64)[1000:1200]
y_validation_one_hot = np.zeros((y_validation.size, y_validation.max()+1))
y_validation_one_hot[np.arange(y_validation.size), y_validation] = 1

# %%
class ConvolutionalLayer:
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = np.zeros(out_channel)
        self.in_channel = in_channel
        self.out_channel = out_channel
        # Gaussian initialization of kernels
        self.kernels = np.random.normal(loc=0.0, scale=0.1, size=(kernel_size, kernel_size, in_channel, out_channel))
        self.kernels_grad = None
        self.bias_grad = None

    def pad(self, x):
        # Pad the input with zeros around the border
        pad = self.padding
        return np.pad(x, ((pad,pad), (pad,pad), (0,0)), mode="constant") if pad > 0 else x

    def forward(self, inputs):
        # Handle grayscale images with single channel
        if inputs.ndim == 2:
            inputs = inputs[..., None]

        # Store input for backpropagation    
        self.inp_back = inputs

        inputs = self.pad(inputs) 

        k = self.kernel_size

        # Create sliding windows
        input_window = view_as_windows(
            arr_in=inputs, 
            window_shape=(k, k, self.in_channel), 
            step=(self.stride, self.stride, 1)
        )[:, :, 0, :, :, :] # Remove extra dimension

        # Store sliding windows for backpropagation
        self.inp_window_back = input_window

        # Perform convolution operation
        conv = np.sum(input_window[..., None] * self.kernels, axis=(2, 3, 4)) + self.bias

        return conv 
    
    def backward(self, output_grad): # Output grad is upstream gradient
        inputs = self.inp_back
        # Handle grayscale images with single channel
        if inputs.ndim == 2:
            inputs = inputs[..., None]

        k = self.kernel_size
        s = self.stride
        C_in = self.in_channel
        C_out = self.out_channel

        inputs_pad = self.pad(inputs)

        H_out, W_out, _ = output_grad.shape

        # Error througought entire output feature map
        self.bias_grad = np.sum(output_grad, axis=(0, 1))

        inp_windows = self.inp_window_back
        inp_cols = inp_windows.reshape(H_out * W_out, -1)
        out_flat = output_grad.reshape(H_out * W_out, C_out)

        # Calculate gradient for kernels
        kernels_grad_flat = inp_cols.T @ out_flat
        self.kernels_grad = kernels_grad_flat.reshape(k, k, C_in, C_out)

        # Calculate gradient for input
        kernels_flat = self.kernels.reshape(-1, C_out)
        cols_grad = out_flat @ kernels_flat.T            
        cols_grad = cols_grad.reshape(H_out, W_out, k, k, C_in)

        grad_input_pad = np.zeros_like(inputs_pad)

        # Create sliding windows for input gradient
        grad_windows = view_as_windows(
            grad_input_pad,
            window_shape=(k, k, C_in),
            step=s
        ) [:, :, 0, :, :, :]

        grad_windows[...] += cols_grad

        # Remove padding from input gradient if applied
        if self.padding > 0:
            p = self.padding
            grad_input = grad_input_pad[p:-p, p:-p, :]
        else:
            grad_input = grad_input_pad

        return grad_input
        
    
class MaxPoolLayer:
    def __init__(self, pooling_size=2):
        # Detemines windoww size for pooling
        self.pooling_size = pooling_size

    def forward(self, inputs):
        # Handle grayscale images with single channel
        if inputs.ndim == 2:
            inputs = inputs[..., None]  

        self.input_shape = inputs.shape
        
        H, W, C = inputs.shape
        pool_size = self.pooling_size

        # Create sliding windows for pooling
        input_window = view_as_windows(
            inputs,
            window_shape=(pool_size, pool_size, C),
            step=(pool_size, pool_size, 1)
        )[:, :, 0, :, :, :]

        H_out, W_out = input_window.shape[0], input_window.shape[1]
        # Reshape windows to prepare for max pooling
        reshaped = input_window.transpose(0, 1, 4, 2, 3).reshape(H_out, W_out, C, -1)
        
        # Perform max pooling
        pooled = np.max(reshaped, axis=-1)
        self.max_indices = np.argmax(reshaped, axis=-1)

        return pooled
    
    def backward(self, output_grad):
        H_out, W_out, C = output_grad.shape
        pool = self.pooling_size
        H_in, W_in, _ = self.input_shape

        # Initialize input gradient
        input_grad = np.zeros((H_in, W_in, C), dtype=np.float64)

        # Create sliding windows for input gradient
        grad_windows = view_as_windows(
            input_grad,
            window_shape=(pool, pool, C),
            step=pool
        ) [:, :, 0, :, :, :]

        # Distribute gradients to the positions of max values
        max_h = self.max_indices // pool    
        max_w = self.max_indices % pool     

        # Create indices for gradient assignment
        i_idx = np.arange(H_out)[:, None, None]          
        j_idx = np.arange(W_out)[None, :, None]          
        c_idx = np.arange(C)[None, None, :]          

        # Assign gradients to positions in input gradient windows
        grad_windows[
            i_idx,
            j_idx,
            max_h,
            max_w,
            c_idx
        ] += output_grad

        return input_grad
    
# Flattens to 1D array, for fully connected layers
class FlattenLayer:
    def __init__(self):
        self.input_shape = None
    
    def forward(self, inputs):
        self.input_shape = inputs.shape
        
        return inputs.flatten().reshape(1, -1)
    
    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)
    
class Augment:
    def __init__(self):
        pass

    def flip_col(self, x, y, size=200):
        # Randomly select indices to augment
        idx = np.random.choice(len(x), size=size, replace=False)

        augment_x = x[idx]
        augment_x = 1 - augment_x # Flip colors 

        # Insert into dataset with suffled indices
        X_aug = np.concatenate((x, augment_x), axis=0)
        y_aug = np.concatenate((y, y[idx]), axis=0)

        perm = np.random.permutation(len(X_aug))
        X_aug = X_aug[perm]
        y_aug = y_aug[perm]

        return X_aug, y_aug

    def rotate(self, x, y, max_angle=15, size=200):
        idx = np.random.choice(len(x), size=size, replace=False)
        selected = x[idx]

        H, W = 28, 28
        cx, cy = H // 2, W // 2

        rotated_imgs = []

        # Math for imgage rotation
        for img in selected:
            angle_deg = np.random.uniform(-max_angle, max_angle)
            angle = np.radians(angle_deg)

            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            new_img = np.zeros_like(img)

            for i in range(H):
                for j in range(W):
                    x0 = j - cx
                    y0 = i - cy

                    xr = int(cos_a * x0 - sin_a * y0 + cx)
                    yr = int(sin_a * x0 + cos_a * y0 + cy)

                    if 0 <= xr < W and 0 <= yr < H:
                        new_img[i, j, 0] = img[yr, xr, 0]

            rotated_imgs.append(new_img)

        # Inserts images into dataset and shuffles
        rotated_imgs = np.array(rotated_imgs)
        X_aug = np.concatenate((x, np.array(rotated_imgs)), axis=0)
        y_aug = np.concatenate((y, y[idx]), axis=0)

        perm = np.random.permutation(len(X_aug))

        X_aug = X_aug[perm]
        y_aug = y_aug[perm]

        return X_aug, y_aug
    def gaussian_noise(self, x, y, mean=0.0, std=0.1, size=200):
        idx = np.random.choice(len(x), size=size, replace=False)
        selected = x[idx]

        # Adding gaussian noise to selected images
        noise = np.random.normal(loc=mean, scale=std, size=selected.shape)
        noisy_imgs = selected + noise
        noisy_imgs = np.clip(noisy_imgs, 0.0, 1.0)

        # Inserts images into dataset and shuffles
        X_aug = np.concatenate((x, noisy_imgs), axis=0)
        y_aug = np.concatenate((y, y[idx]), axis=0)

        perm = np.random.permutation(len(X_aug))

        X_aug = X_aug[perm]
        y_aug = y_aug[perm]

        return X_aug, y_aug

# For creating network specific to convolutional layers
class ConvNeuralNetwork(NeuralNetwork):
    def __init__(self, layers, loss_fn=BCELoss(), epochs=1000, lr=0.01):
        super().__init__(layers, loss_fn, epochs, lr)
    
    # Training neural network with dataset
    def train(self, X, y):
        # Initialize optimizer
        optimizer = SGDOptimizer(lr=self.lr)
        n_samples = len(X)
        
        # Training every image per epoch, updating weights after each image
        for epoch in range(self.epochs):
            epoch_loss = 0
            correct = 0
            
            # Loop through each sample
            for i in range(n_samples):
                sample = X[i]  
                label = y[i]   

                # Forward pass
                a = sample
                for layer, activation in self.layers:
                    z = layer.forward(a)
                    a = activation.forward(z) if activation else z
                
                # Calculate loss and accuracy
                loss = self.loss_fn.calculate_fwd(a, label)
                epoch_loss += loss
                
                # Calculate prediction and accuracy
                true_label = np.argmax(label)
                if pred == true_label:
                    correct += 1
                
                # Backward pass
                grad = self.loss_fn.calculate_back()
                for layer, activation in reversed(self.layers):
                    grad = activation.backward(grad) if activation else grad
                    grad = layer.backward(grad)
                    optimizer.update(layer)
            
            # Display loss and accuracy across epoch
            avg_loss = epoch_loss / n_samples
            acc = correct / n_samples
            print(f"Epoch {epoch+1}/{self.epochs}: Loss={avg_loss:.4f}, Acc={acc:.4f}")
    
    # Predicting labels for given features
    def predict(self, X):
        # Forward pass through the network to get predictions
        predictions = [] # Works with many samples
        for i in range(len(X)):
            a = X[i]
            for layer, activation in self.layers:
                z = layer.forward(a)
                a = activation.forward(z) if activation else z
            predictions.append(np.argmax(a))
        return np.array(predictions)
    
    # Validating the model on validation dataset
    def validate(self, X_val, y_val):
        y_pred = self.predict(X_val)
        y_true = np.argmax(y_val, axis=1) # Convert one-hot encoded labels to class indices

        return y_pred, y_true
        


#%% Creating Convolutional Neural Network       
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
        (FlattenLayer(), None), # Flattening layer to convert 2D to 1D
        (fcl1, activation3),
        (Dropout(prob = 0.3), None), # Dropout layer for regularization
        (fcl2, activation4)
    ],
    loss_fn=loss_fn,
    epochs=20,
    lr=0.01
)

#%% Augmenting Data for Better Training
X, y_one_hot = Augment().flip_col(X, y_one_hot, size=1000)
X, y_one_hot = Augment().rotate(X, y_one_hot, max_angle=10, size=100)
X, y_one_hot = Augment().gaussian_noise(X, y_one_hot, mean=0.0, std=0.3, size=100)

#%%
nn.train(X, y_one_hot) # Training model

# %%
pred, true = nn.validate(X_validation, y_validation_one_hot) # Validation metrics for confusion matrix

# %% Plotting Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(true, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])

disp.plot(cmap=plt.cm.Blues) # You can choose a different colormap
plt.title("Confusion Matrix")
plt.show()
# %%
accuracy = np.mean(pred == true) # Getting accuracy on validation set
# %%
