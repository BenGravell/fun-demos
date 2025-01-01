from autograd import grad
import autograd.numpy as np
import autograd.numpy.random as npr
import matplotlib.pyplot as plt


def activation(x):
    return np.tanh(x)


def layer(x, weight, bias):
    return activation(weight @ x + bias)


def net(x):
    for weight, bias in weights_and_biases:
        x = layer(x, weight, bias)
    return x


npr.seed(1)

feature_dim, hidden_dims, label_dim = 2, [100, 100, 100, 10], 1
in_dims = [feature_dim] + hidden_dims
out_dims = hidden_dims + [label_dim]
weights_and_biases = [[0.2*npr.randn(mi, ni), np.zeros(mi)] for mi, ni in zip(out_dims, in_dims)]
gradient = grad(net)


def visualize_net(net, ns=40):
    x1s, x2s = np.linspace(-4, 4, ns), np.linspace(-4, 4, ns)
    x1g, x2g = np.meshgrid(x1s, x2s)
    yg = np.zeros([ns, ns])
    g1g = np.zeros([ns, ns])
    g2g = np.zeros([ns, ns])
    for i in range(ns):
        for j in range(ns):
            x = np.array([x1g[i, j], x2g[i, j]])
            y = net(x)
            g = gradient(x)
            yg[i, j] = y
            g1g[i, j] = g[0]
            g2g[i, j] = g[1]

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(x1g, x2g, yg, levels=20, cmap='Spectral')
    plt.colorbar(cf)
    ax.quiver(x1g, x2g, g1g, g2g, units='width', color='k')
    return fig, ax


visualize_net(net)
plt.show()
