import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def r(xc, yc):
    return np.sqrt(np.square(x-xc)+np.square(y-yc))


def f(r, s, d, t):
    return np.cos(s*r - t) * np.exp(-np.square(d*r))


def g(t):
    z = np.zeros([nx, ny])
    for i in range(len(a)):
        z += a[i] * f(R[i], s[i], d[i], q[i] * t)
    return z


def squash(X):
    Z = np.tanh(X)
    return Z


def raster(t):
    return squash(g(t))


def update(t):
    im.set_data(raster(t))
    return [im]


nx, ny = 40, 40
tx = np.linspace(-40, 40, nx)
ty = np.linspace(-40, 40, ny)
x, y = np.meshgrid(tx, ty)

# Number of point sources
n = 40

# Center coordinates
Xc = 80 * (npr.rand(n)-0.5)
Yc = 80 * (npr.rand(n)-0.5)

# Amplitudes
a = (10/n) * (npr.rand(n)+0.5)

# Temporal frequencies
q = 0.1*(npr.rand(n) + 1.0)

# Spatial frequencies
s = 1.0*(npr.rand(n) + 0.5)

# Spatial decays
d = 0.05*(npr.rand(n) + 1.0)

R = []
for i in range(len(Xc)):
    R.append(r(Xc[i], Yc[i]))

# Plotting
fig = plt.figure(figsize=(4, 4))
im = plt.imshow(raster(0), cmap='seismic', vmin=-4, vmax=4, interpolation='spline16')
plt.axis('off')

ani = FuncAnimation(fig, update, interval=0, blit=True)
plt.show()
