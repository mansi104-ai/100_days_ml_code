import numpy as np
from rnn_cell import forward_step, backward_step

def compute_loss_cross_entropy(outputs, targets):
    total_loss = 0
    T = len(outputs)
    for y_t, t_t in zip(outputs, targets):
        exp_scores = np.exp(y_t - np.max(y_t))
        p = exp_scores / np.sum(exp_scores)
        loss_t = -np.sum(t_t * np.log(p + 1e-9))
        total_loss += loss_t
    return total_loss / T

def update_parameters(params, grads, learning_rate):
    params["Wxh"] -= learning_rate * grads["dWxh"]
    params["Whh"] -= learning_rate * grads["dWhh"]
    params["Why"] -= learning_rate * grads["dWhy"]
    params["bh"]  -= learning_rate * grads["dbh"]
    params["by"]  -= learning_rate * grads["dby"]
    return params

def bptt(caches, outputs, targets, params, activation_derivative):
    dWxh = np.zeros_like(params["Wxh"])
    dWhh = np.zeros_like(params["Whh"])
    dWhy = np.zeros_like(params["Why"])
    dbh  = np.zeros_like(params["bh"])
    dby  = np.zeros_like(params["by"])

    dh_next = np.zeros((params["Whh"].shape[0], 1))
    T = len(outputs)

    for t in reversed(range(T)):
        y_t = outputs[t]
        target_t = targets[t]
        cache_t = caches[t]

        exp_scores = np.exp(y_t - np.max(y_t))
        p = exp_scores / np.sum(exp_scores)
        dy = p - target_t

        grads_t, dh_next = backward_step(
            dy,
            dh_next,
            cache_t,
            params,
            activation_derivative
        )

        dWxh += grads_t["dWxh"]
        dWhh += grads_t["dWhh"]
        dWhy += grads_t["dWhy"]
        dbh  += grads_t["dbh"]
        dby  += grads_t["dby"]

    for g in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(g, -5, 5, out=g)

    return {
        "dWxh": dWxh,
        "dWhh": dWhh,
        "dWhy": dWhy,
        "dbh": dbh,
        "dby": dby
    }

def train_rnn(X, Y, params, activation, activation_derivative, epochs=50, lr=0.001):
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for x_seq, y_seq in zip(X, Y):
            h0 = np.zeros((params["Whh"].shape[0], 1))
            from rnn_model import unfold_sequence
            h_states, outputs, caches = unfold_sequence(x_seq, params, h0, activation)
            loss = compute_loss_cross_entropy(outputs, y_seq)
            total_loss += loss
            grads = bptt(caches, outputs, y_seq, params, activation_derivative)
            params = update_parameters(params, grads, lr)
        avg_loss = total_loss / len(X)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    return params, losses

def evaluate_rnn(X, Y, params, activation):
    correct = 0
    total = 0
    for x_seq, y_seq in zip(X, Y):
        h0 = np.zeros((params["Whh"].shape[0], 1))
        from rnn_model import unfold_sequence
        _, outputs, _ = unfold_sequence(x_seq, params, h0, activation)
        y_pred = outputs[-1]
        y_true = y_seq[-1]
        if np.argmax(y_pred) == np.argmax(y_true):
            correct += 1
        total += 1
    return correct / total
