## Building the basic acitvation functions for RNNs that can hold non-linearity as required by rnns

## Functions implemented : tanh, dtanh (Need for gradient backprop),swish, dswish ( swish derivative )

import numpy as np

def tanh(x):
  return np.tanh(x)

def dtanh(x):
  return 1- (np.tanh(x)**2)

def swish(x):
  sigma = 1/(1 + np.exp(-x))

  return x * sigma

def dswish(x):
  sigma  = 1/(1+ np.exp(-x))
  return sigma + x * sigma * (1 - sigma)

