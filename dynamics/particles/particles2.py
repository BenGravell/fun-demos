import numpy as np
import numpy.random as npr
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from scipy.spatial.distance import pdist, squareform


def cart2pol(x, y):
    rho = np.sqrt(x**2+y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def force(x):
    return x**2


# def force(x):
#    return 1.0*x**2-10.0*np.log(x)

## Lennard-Jones
# def force(x):
#    x6 = x**6
#    x13 = x6*x6*x
#    return (x6-2)/x13


npr.seed(3)
n = 40
m = 1.0 * np.ones(n)
m[0] = 100
b = 0.0 * np.ones(n)
c = 1.0 * np.ones(n)

rho = 2+2.0 * npr.rand(n)
phi = 2 * np.pi * npr.rand(n)
# phi = 2*np.pi*np.linspace(0,1,n)

x = np.array(pol2cart(rho, phi)).T
x[0, :] = [0, 0]

v = 20.0 * np.array(pol2cart(rho, phi+np.pi / 2)).T
v[0, :] = [0, 0]

fig, ax = plt.subplots()
scat = ax.scatter(x[:, 0], x[:, 1], 10 * m**0.5)
al = 10
ax.set_xlim(-al, al)
ax.set_ylim(-al, al)
h = 0.001
stride = 1


def calc_va(x, v):
    px = squareform(pdist(x))
    frc = np.zeros([n, 2])
    for j in range(n):
        sl = [i for i in range(n) if i != j]
        frcmag = m[j] * m[sl] * force(px[sl, j]) * c[sl]
        frcdir = x[sl, :]-x[j, :]
        frcdiru = frcdir / la.norm(frcdir, axis=1)[:, None]
        frc[j, :] = np.dot(frcmag, frcdiru)
    a = frc / m[:, None]-v * b[:, None]
    return v, a


def euler_xv(x, v):
    v, a = calc_va(x, v)
    xnext = x+h * v
    vnext = v+h * a
    return xnext, vnext


def rk4_xv(x, v):
    x1, v1 = x, v
    v1, a1 = calc_va(x1, v1)
    kx1, kv1 = h * v1, h * a1

    x2, v2 = x+kx1 / 2, v+kv1 / 2
    v2, a2 = calc_va(x2, v2)
    kx2, kv2 = h * v2, h * a2

    x3, v3 = x+kx2 / 2, v+kv2 / 2
    v3, a3 = calc_va(x3, v3)
    kx3, kv3 = h * v3, h * a3

    x4, v4 = x+kx3, v+kv3
    v4, a4 = calc_va(x4, v4)
    kx4, kv4 = h * v4, h * a4

    xnext = x+(kx1+2 * kx2+2 * kx3+kx4) / 6
    vnext = v+(kv1+2 * kv2+2 * kv3+kv4) / 6
    return xnext, vnext


def update(t):
    global x
    global v
    for st in range(stride):
        #        x,v = euler_xv(x,v)
        x, v = rk4_xv(x, v)
    scat.set_offsets(x)
    return scat,


aniobj = ani.FuncAnimation(fig, update, interval=1, blit=True)

# nt=4000
# av=np.zeros([nt,n])
# for i in range(nt):
#    av[i,:] = la.norm(v,axis=1)
#    update(0)
# plt.figure()
# plt.plot(np.mean(av,axis=1),color=[0.2,0.4,0.9],linewidth=4)
# plt.plot(np.max(av,axis=1),color=[0.3,0.3,0.3])
# plt.plot(np.min(av,axis=1),color=[0.3,0.3,0.3])
# plt.plot(av,color=[0.2,0.5,0.7],alpha=0.02)
plt.show()