import numpy as np


x = np.random.randn(4,32,32,16)
x1 = np.stack([np.rot90(p,1) for p in x],axis=0)
print(x1.shape)