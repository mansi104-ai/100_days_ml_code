import numpy as np
from rnn_cell import forward_step, backward_step

def unfold_sequence(x, params, h0, activation):
    T = x.shape[0]
    h_prev = h0
    h_states = []
    outputs = []
    caches = []

    for t in range(T):
        x_t = x[t].reshape(-1, 1)
        h_t, y_t, cache_t = forward_step(
            x_t,
            h_prev,
            params,
            activation
        )
        h_states.append(h_t)
        outputs.append(y_t)
        caches.append(cache_t)
        h_prev = h_t

    return h_states, outputs, caches


def compute_loss_cross_entropy(outputs, targets):
    total_loss = 0
    T = len(outputs)
    for y_t, t_t in zip(outputs, targets):
        exp_scores = np.exp(y_t - np.max(y_t))
        p = exp_scores / np.sum(exp_scores)
        loss_t = -np.sum(t_t * np.log(p + 1e-9))
        total_loss += loss_t
    return total_loss / T


def bptt(outputs, targets, caches, params, activation_derivative):
    dWxh = np.zeros_like(params["Wxh"])
    dWhh = np.zeros_like(params["Whh"])
    dWhy = np.zeros_like(params["Why"])
    dbh = np.zeros_like(params["bh"])
    dby = np.zeros_like(params["by"])

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


def update_parameters(params, grads, lr):
    params["Wxh"] -= lr * grads["dWxh"]
    params["Whh"] -= lr * grads["dWhh"]
    params["Why"] -= lr * grads["dWhy"]
    params["bh"]  -= lr * grads["dbh"]
    params["by"]  -= lr * grads["dby"]
    return params
