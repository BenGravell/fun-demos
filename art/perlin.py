# 2D Perlin noise
# Based on https://stackoverflow.com/questions/42147776/producing-2d-perlin-noise-with-numpy

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt


def perlin(x, y, n=256, seed=0):
    # permutation table
    npr.seed(seed)
    p = np.arange(n, dtype=int)
    npr.shuffle(p)
    p = np.stack([p, p]).flatten()
    # coordinates of the top-left
    xi = x.astype(int)
    yi = y.astype(int)
    # internal coordinates
    xf = x - xi
    yf = y - yi
    # fade factors
    u = fade(xf)
    v = fade(yf)
    # noise components
    n00 = gradient(p[p[xi] + yi], xf, yf)
    n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
    n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
    n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf)
    # combine noises
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)
    return lerp(x1, x2, v)


def lerp(a, b, x):
    "linear interpolation"
    return a + x * (b - a)


def poly(t, coeffs):
    pows = [1]
    for i in range(1, len(coeffs)):
        pows.append(pows[i-1]*t)
    return sum(coeff*pow for coeff, pow in zip(coeffs, pows) if coeff != 0)


def fade(t):
    return poly(t, coeffs=[0, 0, 0, 10, -15, 6])


def gradient(h, x, y):
    "grad converts h to the right gradient vector and return the dot product with (x,y)"
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    g = vectors[h % 4]
    return g[:, :, 0] * x + g[:, :, 1] * y


def fractal(n=256, num_layers=8, p0=1.0, p=2.0, r=0.5, noise_fun=None, seed=0):
    if noise_fun is None:
        noise_fun = perlin
    out = np.zeros([n, n])
    for i in range(num_layers):
        linmax = p**i
        if linmax > int(n/2):
            continue
        lin = np.linspace(0, p0*p**i, n)
        x, y = np.meshgrid(lin, lin)
        z = noise_fun(x, y, n=2*n, seed=seed+i)
        mag = r**i
        out += mag*z
    return out


if __name__ == "__main__":
    z = fractal()
    # z = fractal(1024, num_layers=16)
    # z = fractal(n=128, num_layers=5, p=1.5, r=0.8)

    cmap = 'inferno'
    plt.imshow(z, cmap=cmap)
    # plt.contourf(z, cmap=cmap)
    plt.axis('off')
