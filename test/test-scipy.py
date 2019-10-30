import numpy as np
from scipy.fftpack import dctn,idctn

y = np.random.randn(4, 4)
print(y)
print('-----------------------------------------------')
print(dctn(y,norm='ortho'))