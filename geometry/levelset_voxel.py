import numpy as np
import numpy.random as npr
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from functools import reduce


def sympart(X):
    return (X+X.T) / 2


def mdot(*args):
    return reduce(np.dot, args)


npr.seed(3)

n = 3
Ah = npr.randn(n, n)
A = np.dot(Ah, Ah.T)
b = npr.randn(n)
c = npr.randn()


def quad(x, A, b, c):
    return np.einsum('i...,...i', x, np.einsum('ij,j...', A, x))+np.dot(b, x)+c


def quadmin(A, b, c):
    return -la.solve(A, b) / 2


nx = 20
xs = np.repeat(np.linspace(-10, 10, nx)[:, np.newaxis], n, axis=1).T
xm = np.meshgrid(*xs)
xv = []
for i in range(len(xm)):
    xv.append(xm[i].flatten())
x = np.stack(xv)
z = quad(x, A, b, c)
zm = np.reshape(z, (nx, nx, nx))

thresh_min = 20
thresh_max = 40

voxels = np.logical_and(zm > thresh_min, zm < thresh_max)

zc = np.clip(z, thresh_min, thresh_max)
zcmin = zc.min()
zcmax = zc.max()
zc = (zc-zcmin) / (zcmax-zcmin)

viridis = cm.get_cmap('viridis')
colorsv = viridis(zc)
colors = np.reshape(colorsv, (nx, nx, nx, 4))
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(voxels, facecolors=colors, edgecolors='k')
plt.show()
