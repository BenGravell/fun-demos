import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from matplotlib.colors import ListedColormap

tx = np.linspace(-40, 40, 100)
ty = np.linspace(-40, 40, 100)
x, y = np.meshgrid(tx, ty)


def r(xc, yc):
    return np.sqrt(np.square(x-xc)+np.square(y-yc))


# Number of point sources
n = 5

## Center coordinates
# Xc = np.linspace(0,0,n)
# Yc = np.linspace(-20,20,n)
#
## Amplitudes
# a = np.ones(n)
#
## Frequencies
# q = np.linspace(1,1.1,n)

# Center coordinates
Xc = 100 * (npr.rand(n)-0.5)
Yc = 100 * (npr.rand(n)-0.5)

# Amplitudes
a = 0.5 * (npr.rand(n)+0.5)

# Frequencies
q = npr.rand(n)+0.5

R = []
for i in range(len(Xc)):
    R.append(r(Xc[i], Yc[i]))


def f(r, t):
    return np.cos(r-0.1 * t) * np.exp(-np.square(0.05 * r))


def g(t):
    z = np.zeros_like(x)
    for i in range(len(a)):
        z += a[i] * f(R[i], q[i] * t)
    return z


def make_dvg_cmap(cmap_str):
    top = cm.get_cmap(cmap_str+'_r', 128)
    bottom = cm.get_cmap(cmap_str, 128)
    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128))))
    return ListedColormap(newcolors, name=cmap_str+'_dvg')


def imX(X):
    Z = np.tanh(X)
    #    Z = Z/np.abs(Z)
    return Z


# cmap = make_dvg_cmap('inferno')
cmap = 'seismic'

fig = plt.figure(figsize=(4, 4))
im = plt.imshow(imX(g(0)), cmap=cmap, vmin=-4, vmax=4, interpolation='bilinear')
plt.axis('off')


def update(t):
    im.set_data(imX(g(t)))
    return im,


ani = FuncAnimation(fig, update, interval=0, blit=True)
plt.show()
