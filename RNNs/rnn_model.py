### This is the file containing code for handling multiple sequenes of time steps across training
from rnn_cell import forward_step , backward_step
from activations import tanh
import numpy as np

def unfold_sequence(x, params , h0,):
  ## Stores the list of hidden states and outputs
  T = x.shape[0]
  h_prev = h0
  h_states = []
  outputs = []
  cache = []

  for t in range(T):
    x_t = x[t].reshape(-1,1)
    h_t , y_t , cache_t = forward_step(
      x_t,
      h_prev,
      params,
      activation = tanh 
    )

    h_states.append(h_t)
    outputs.append(y_t)
    cache.append(cache_t)

    h_prev = h_t
  return h_states , outputs , cache

def compute_loss_mse(outputs,targets):
  #Assuming the shape of outputs is same as targets list
  for i in range(outputs,targets):
    loss = np.mean((outputs[i] - targets[i])**2)
  return loss 

## Take one-hot encoded targets
def compute_loss_cross_entropy(outputs,targets):
    # Softmax -> probabilities
    exp_scores = np.exp(outputs - np.max(outputs))
    p = exp_scores / np.sum(exp_scores)

    # cross entropy
    loss = -np.sum(targets * np.log(p + 1e-9))
    return loss

''' 
Complete backprop function and update parameters function
'''
def backprop(X, )

  