from rnn_cell import forward_step , backward_step , initialize_parameters
from rnn_model import compute_loss_mse, compute_loss_cross_entropy

def train(model, data, epochs, lr):
  