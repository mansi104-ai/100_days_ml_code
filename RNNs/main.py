import numpy as np
from activations import tanh, dtanh
from rnn_cell import initialize_parameters
from train_utils import train_rnn, evaluate_rnn

def generate_dummy_sequence_data(num_samples=200, T=5, input_dim=3, num_classes=2):
    X = []
    Y = []
    for _ in range(num_samples):
        seq = np.random.randn(T, input_dim)
        label = int(seq.sum() > 0)
        target = []
        for t in range(T):
            onehot = np.zeros((num_classes, 1))
            if t == T - 1:
                onehot[label] = 1
            target.append(onehot)
        X.append(seq)
        Y.append(target)
    return X, Y

input_dim = 3
hidden_dim = 16
output_dim = 2

params = initialize_parameters(input_dim, hidden_dim, output_dim)

X_train, Y_train = generate_dummy_sequence_data(num_samples=500)
X_test, Y_test = generate_dummy_sequence_data(num_samples=100)

params, losses = train_rnn(
    X_train,
    Y_train,
    params,
    tanh,
    dtanh,
    epochs=20,
    lr=0.001
)

acc = evaluate_rnn(X_test, Y_test, params, tanh)
print("Test Accuracy:", acc)
