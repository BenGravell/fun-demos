import numpy as np
import numpy.random as npr
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from scipy.spatial.distance import pdist, squareform

from mpl_toolkits.mplot3d import Axes3D

npr.seed(1)

n = 100
x = npr.randn(n, 3)

# Cylinder definition
axis = np.array([1, 2, 3])
axis = axis / la.norm(axis)

# Cylinder radius
c = 0.5
c2 = c**2

# Cylinder heights
h_lwr = -1
h_upr = 2
delta_parallel = (axis[:, None] * np.dot(x, axis)).T
delta_perpendicular = x-delta_parallel

# Radial coordinates squared
r2 = (delta_perpendicular**2).sum(axis=1)

# Axial coordinates
a = np.dot(x, axis)  # = la.norm(delta_parallel,axis=1)

# Mask points contained in cylinder
mask = np.logical_and(r2 < c2, np.logical_and(a < h_upr, a > h_lwr))

# Figure setup
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Colors
scatter_color = np.array([0.2, 0.8, 0.4, 1]) * (mask[:, None] * 0.8+0.2)

# Scatter plot sample points
ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=scatter_color)

# Cylinder
nx_cyl = 100
nz_cyl = 100
x_cyl = np.linspace(-c, c, nx_cyl)
z_cyl = np.linspace(h_lwr, h_upr, nz_cyl)
Xc, Zc = np.meshgrid(x_cyl, z_cyl)
Yc = np.sqrt(c2-Xc**2)
Ycn = -Yc

# Rotate
u = np.array([0, 0, 1])
rotax = np.cross(u, axis)
rotax = rotax / la.norm(rotax)
alpha = np.arccos(np.dot(u, axis))
sinalpha = np.sin(alpha)
cosalpha = np.cos(alpha)
mcosalpha = 1-cosalpha
rotmat = np.array([[rotax[0]**2 * (mcosalpha)+cosalpha, rotax[0] * rotax[1] * mcosalpha-sinalpha * rotax[2],
                    rotax[0] * rotax[2] * mcosalpha+sinalpha * rotax[1]],
                   [rotax[0] * rotax[1] * mcosalpha+sinalpha * rotax[2], rotax[1]**2 * (mcosalpha)+cosalpha,
                    rotax[1] * rotax[2] * mcosalpha-sinalpha * rotax[0]],
                   [rotax[0] * rotax[2] * mcosalpha-sinalpha * rotax[1],
                    rotax[1] * rotax[2] * mcosalpha+sinalpha * rotax[0], rotax[2]**2 * (mcosalpha)+cosalpha]])

Xcf = Xc.flatten()
Ycf = Yc.flatten()
Ycnf = Ycn.flatten()
Zcf = Zc.flatten()

XX = np.vstack([Xcf, Ycf, Zcf]).T
XXn = np.vstack([Xcf, Ycnf, Zcf]).T
XXrot = np.zeros_like(XX)
XXnrot = np.zeros_like(XXn)
for i in range(XX.shape[0]):
    XXrot[i] = np.dot(rotmat, XX[i])
    XXnrot[i] = np.dot(rotmat, XXn[i])

Xcf = XXrot[:, 0]
Ycf = XXrot[:, 1]
Zcf = XXrot[:, 2]
Xcnf = XXnrot[:, 0]
Ycnf = XXnrot[:, 1]
Zcnf = XXnrot[:, 2]

Xc = np.reshape(Xcf, [nx_cyl, nz_cyl])
Yc = np.reshape(Ycf, [nx_cyl, nz_cyl])
Zc = np.reshape(Zcf, [nx_cyl, nz_cyl])

Xcn = np.reshape(Xcnf, [nx_cyl, nz_cyl])
Ycn = np.reshape(Ycnf, [nx_cyl, nz_cyl])
Zcn = np.reshape(Zcnf, [nx_cyl, nz_cyl])

# Draw parameters
rstride = 20
cstride = 10
cyl_color = "tab:blue"
ax.plot_surface(Xc, Yc, Zc, alpha=0.5, rstride=rstride, cstride=cstride, color=cyl_color)
ax.plot_surface(Xcn, Ycn, Zcn, alpha=0.5, rstride=rstride, cstride=cstride, color=cyl_color)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
