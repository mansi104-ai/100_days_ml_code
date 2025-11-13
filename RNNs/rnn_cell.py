import numpy as np

def initialize_parameters(input_dim, hidden_dim, output_dim):
    np.random.seed(42)

    Wxh = np.random.randn(hidden_dim, input_dim) * 0.01
    Whh = np.random.randn(hidden_dim, hidden_dim) * 0.01
    Why = np.random.randn(output_dim, hidden_dim) * 0.01

    bh = np.zeros((hidden_dim, 1))
    by = np.zeros((output_dim, 1))

    parameters = {
        "Wxh": Wxh,
        "Whh": Whh,
        "Why": Why,
        "bh": bh,
        "by": by
    }

    return parameters


def forward_step(x_t, h_prev, params, activation):
    
    Wxh = params["Wxh"]
    Whh = params["Whh"]
    Why = params["Why"]
    bh  = params["bh"]
    by  = params["by"]

    a_t = Wxh @ x_t + Whh @ h_prev + bh

    h_t = activation(a_t)


    y_t = Why @ h_t + by

    cache = {
        "x_t": x_t,
        "h_prev": h_prev,
        "h_t": h_t,
        "a_t": a_t
    }

    return h_t, y_t, cache


def backward_step(dy, dh_next, cache, params, activation_derivative):
    
    Wxh = params["Wxh"]
    Whh = params["Whh"]
    Why = params["Why"]

    x_t   = cache["x_t"]
    h_prev = cache["h_prev"]
    h_t    = cache["h_t"]
    a_t    = cache["a_t"]

    dWhy = dy @ h_t.T
    dby  = dy

    dh = (Why.T @ dy) + dh_next

    da = dh * activation_derivative(a_t)

    dWxh = da @ x_t.T
    dWhh = da @ h_prev.T
    dbh  = da
    dh_prev = Whh.T @ da

    grads = {
        "dWxh": dWxh,
        "dWhh": dWhh,
        "dWhy": dWhy,
        "dbh": dbh,
        "dby": dby
    }

    return grads, dh_prev
