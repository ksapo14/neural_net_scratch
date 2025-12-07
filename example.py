#%%
import numpy as np
from skimage.util import view_as_windows

#%%
random_matrix = np.random.rand(4, 4)
print(random_matrix)
windows = view_as_windows(random_matrix, (3, 3))
print(windows)
# %%
