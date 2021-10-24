import numpy as np
import numpy.random as npr
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from scipy.spatial.distance import pdist, squareform

npr.seed(2)

# Lennard-Jones potential parameters
epsilon = 1.0
sigma = 1.25
a = 24*epsilon*sigma**6
b = 2*sigma**6

# Delta time (timestep)
dt = 1e-2

# Spatial dimension, pick 2 or 3
sd = 2

# Target number of particles
n_target = 256

# Actual number of particles
n_root = round(n_target**(1/sd))
n = n_root**sd

# Mass of particles
m = 1.0*np.ones(n)

# Single pair equilibrium distance
deq = (2**(1/6))*sigma

friction = False

if not friction:
    # Force from Lennard-Jones potential
    def forcemag(x):
        x6 = x**-6
        return a*(x6 - b*x6**2)/x


    def force(x):
        px = squareform(pdist(x))
        frc = np.zeros([n, sd])
        for j in range(n):
            allbutj = [i for i in range(n) if i != j]
            frcmag = forcemag(px[allbutj, j])
            frcdir = x[allbutj, :] - x[j, :]
            frcdiru = frcdir/la.norm(frcdir, axis=1)[:, None]
            frc[j, :] = np.dot(frcmag, frcdiru)
        return frc


    # ODE solver - Semi-Implicit Euler
    # Simple  --> computes fast
    # Symplectic --> good long-range accuracy due to energy conservation
    def semi_euler(x, v):
        vnew = v + force(x)*dt/m[:, None]
        xnew = x + vnew*dt
        return xnew, vnew
else:
    # NON-PHYSICAL
    # Force from Lennard-Jones potential + friction
    def forcemag(x):
        x6 = x**-6
        return a*(x6 - b*x6**2)/x


    # Friction coefficient
    c = 1.0


    def force(x, v):
        px = squareform(pdist(x))
        frc = np.zeros([n, sd])
        for j in range(n):
            allbutj = [i for i in range(n) if i != j]
            frcmag = forcemag(px[allbutj, j])
            frcdir = x[allbutj, :] - x[j, :]
            frcdiru = frcdir/la.norm(frcdir, axis=1)[:, None]
            frc[j, :] = np.dot(frcmag, frcdiru) - c*v[j]
        return frc


    # ODE solver - Semi-Implicit Euler
    # Simple  --> computes fast
    # Symplectic --> good long-range accuracy due to energy conservation
    def semi_euler(x, v):
        vnew = v + force(x, v)*dt/m[:, None]
        xnew = x + vnew*dt
        return xnew, vnew

# Inital conditions
x, v = np.zeros([n, sd]), np.zeros([n, sd])

# Grid
q = 0
if sd == 2:
    for i in range(round(n**0.5)):
        for j in range(round(n**0.5)):
            posarray = np.array([i, j])
            x[q] = 1.0*deq*posarray
            q = q + 1
elif sd == 3:
    for i in range(round(n**(1/3))):
        for j in range(round(n**(1/3))):
            for k in range(round(n**(1/3))):
                posarray = np.array([i, j, k])
                x[q] = 1.0*deq*posarray
                q = q + 1

# Add randomness
# x += 0.01*npr.randn(n,sd)
v += 0.1*npr.randn(n, sd)

scat_size = 50

if sd == 2:
    fig, ax = plt.subplots()
    scat = ax.scatter(x[:, 0], x[:, 1], s=scat_size)
elif sd == 3:
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=scat_size)

xmin = x[:, 0].min()
xmax = x[:, 0].max()
ymin = x[:, 1].min()
ymax = x[:, 1].max()
xdel = xmax - xmin
ydel = ymax - ymin

view_margin = 0.50

ax.set_xlim(xmin - view_margin*xdel, xmax + view_margin*xdel)
ax.set_ylim(ymin - view_margin*ydel, ymax + view_margin*ydel)
if sd == 3:
    zmin = x[:, 2].min()
    zmax = x[:, 2].max()
    zdel = zmax - zmin
    ax.set_zlim(zmin - view_margin*zdel, zmax + view_margin*zdel)
# ax.axis('equal')
ax.axis('off')
ax.set_position([0, 0, 1, 1])

stride = 1

if sd == 2:
    def update(t):
        global x;
        global v
        for i in range(stride):
            x, v = semi_euler(x, v)
        scat.set_offsets(x)
        return scat,


    ao = ani.FuncAnimation(fig, update, interval=0, blit=True)
elif sd == 3:
    def update(t):
        global x;
        global v
        for i in range(stride):
            x, v = semi_euler(x, v)
        global scat
        scat.remove()
        scat = ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=scat_size, c='tab:blue')
        return scat,


    ao = ani.FuncAnimation(fig, update, interval=0, blit=False)
plt.show()
