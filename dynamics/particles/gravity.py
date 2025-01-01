import numpy as np
import numpy.random as npr
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from scipy.spatial.distance import pdist, squareform


def pol2cart(r, p):
    return r * np.cos(p), r * np.sin(p)


n = 20  # number of particles
m = 1.0*np.ones(n)  # mass of particles
m[0] = 20  # one large mass
g = 9.81  # gravity acceleration

r = 1.0 + 0.1 * npr.rand(n)  # initial position radii
p = 2 * np.pi * npr.rand(n)  # initial position phases
x = np.array(pol2cart(r, p)).T  # initial positions
x[0] = [0, 0]  # put large mass at origin

v = 6.0 * np.array(pol2cart(r, p+np.pi / 2)).T+0.1 * npr.randn(n, 2)  # initial velocity
v[0] = [0, 0]  # make large mass at rest initially

fig, ax = plt.subplots()
colors = np.tile([[0.2], [0.4], [0.7]], n).T
colors[0] = [0.8, 0.4, 0.2]
scat = ax.scatter(x[:, 0], x[:, 1], 30 * m**0.66, c=colors)
ax.axis('equal')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.axis('off')
ax.set_position([0, 0, 1, 1])

reg = 0.001  # regularizer amount in force denominator

h = 0.00001  # stepsize


slicebank = []
for j in range(n):
    slicebank.append([i for i in range(n) if i != j])


def acc(x):
    px = squareform(pdist(x))
    a = np.zeros([n, 2])
    for j in range(n):
        sl = slicebank[j]
        a[j] = np.dot((g * m[sl] / (reg + px[sl, j]**3)), x[sl]-x[j])
    return a


def euler(x, v, acc):
    xnew = x + h * v
    vnew = v + h * acc(x)
    return xnew, vnew


def sie1(x, v, acc):
    vnew = v + h * acc(x)
    xnew = x + h * vnew
    return xnew, vnew


def sie2(x, v, acc):
    xnew = x + h * v
    vnew = v + h * acc(xnew)
    return xnew, vnew


def rk4(x, v, acc):
    rkc, kc = np.array([1, 2, 2, 1])[:, None, None] / 6, [1 / 2, 1 / 2, 1]
    dim = [5, 4]
    dim.extend(x.shape)
    dat = np.zeros(dim)
    dat[0, 0], dat[1, 0] = x, v
    for i in range(4):
        if i > 0: dat[0, i], dat[1, i] = x+kc[i-1] * dat[3, i-1], v+kc[i-1] * dat[4, i-1]
        dat[2, i] = acc(dat[0, i])
        dat[3, i], dat[4, i] = h * dat[1, i], h * dat[2, i]
    return x+np.sum(rkc * dat[3], axis=0), v+np.sum(rkc * dat[4], axis=0)


def energy(x, v):
    kinetic_energy = 0.5*np.sum(m[:, None]*v**2)

    px = squareform(pdist(x))
    ge = np.zeros([n, 2])
    for j in range(n):
        sl = slicebank[j]
        ge[j] = -g*m[j]*np.sum(m[sl]/px[sl, j])
    potential_energy = np.sum(ge)

    total_energy = kinetic_energy + potential_energy

    spacer = '    '
    print('%.3f' % kinetic_energy, end=spacer)
    print('%.3f' % potential_energy, end=spacer)
    print('%.3f' % total_energy, end=spacer)
    print('')
    return total_energy



# TODO this code is broken - something is wrong with either the acceleration function or the energy function
#  check the dot product / sums over masses/positions

odesolver = euler
# odesolver = rk4

# odesolver = sie1
# odesolver = sie2


def update(t):
    global x
    global v
    x, v = odesolver(x, v, acc)
    energy(x, v)
    scat.set_offsets(x-(x * m[:, None]).sum(0) / m.sum())
    return scat,


ao = ani.FuncAnimation(fig, update, interval=0, blit=True)
plt.show()
