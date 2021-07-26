import numpy as np
import numpy.random as npr
import numpy.linalg as la
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

npr.seed(1)
s, r, b = 10, 28, 8 / 3  # Lorenz attractor parameters
h = 1e-2  # Step size
nt = 200  # Number of rk4 timesteps
nu = 4  # Number of linear interpolation points
ns = 20  # Number of sample initial conditions/trajectories
# nx,ny = 768,1366 # Image dimensions
nx, ny = 1080, 1920  # Image dimensions
xmin, xmax = -20, 20
ymin, ymax = -3, 48
xdel, ydel = xmax-xmin, ymax-ymin


# Lorenz attractor nonlinear dynamics
def f(x):
    return np.array([s * (x[:, 1]-x[:, 0]),
                     x[:, 0] * (r-x[:, 2])-x[:, 1],
                     x[:, 0] * x[:, 1]-b * x[:, 2]]).T


# 4th-order Runge-Kutta ODE solver
def rk4(x):
    k1 = h * f(x)
    k2 = h * f(x+k1 / 2)
    k3 = h * f(x+k2 / 2)
    k4 = h * f(x+k3)
    return x+(k1+2 * k2+2 * k3+k4) / 6


# Simulate trajectories
def sim(x0, nt):
    xhist = np.zeros([nt, ns, 3])
    xhist[0] = x0
    for t in range(nt-1):
        xhist[t+1] = rk4(xhist[t])
    return xhist


# Create many sample initial conditions randomly spread out
x0 = np.array([(xmin+xmax) / 2, 1, (ymin+ymax) / 2])+(npr.rand(ns, 3)-0.5) * np.array([xdel, 0.5, ydel]) * 0.5
xhist = sim(x0, 500)
x0 = xhist[-1]
xhist = sim(x0, nt)

# Set color by speed
chist = np.zeros([nt, ns])
for t in range(nt):
    chist[t] = la.norm(f(xhist[t]), axis=1)

# Plotting
plt.close('all')
plt.style.use('dark_background')

plot_style = '2d'
color_by_speed = True

if plot_style == '2d':
    fig, ax = plt.subplots()
    if color_by_speed:
        for i in range(ns):
            points = np.array([xhist[:, i, 0], xhist[:, i, 2]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(chist[:, i].min(), chist[:, i].max())
            lc = LineCollection(segments, cmap='plasma', norm=norm, alpha=0.8, linewidth=1)
            lc.set_array(chist[:, i])
            line = ax.add_collection(lc)
    else:
        ax.plot(xhist[:, :, 0], xhist[:, :, 2], alpha=0.8, color='w', linewidth=1)
    plt.axis('auto')

elif plot_style == '3d':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if color_by_speed:
        for i in range(ns):
            points = np.array([xhist[:, i, 0], xhist[:, i, 1], xhist[:, i, 2]]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(chist[:, i].min(), chist[:, i].max())
            lc = Line3DCollection(segments, cmap='plasma', norm=norm, alpha=0.8, linewidth=1)
            # Set the values used for colormapping
            lc.set_array(chist[:, i])
            line = ax.add_collection(lc)
        ax.set_xlim3d(xmin, xmax)
        ax.set_ylim3d(-20, 20)
        ax.set_zlim3d(ymin, ymax)
    else:
        for i in range(ns):
            ax.plot(xhist[:, i, 0], xhist[:, i, 1], xhist[:, i, 2],
                    linewidth=2, color='w', alpha=0.8)

plt.axis('off')
plt.show()
