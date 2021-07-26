import numpy as np
import numpy.random as npr
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.animation as ani


class Particle():
    particles = []

    def __init__(self, m, b, c, x, v):
        self.m = m
        self.b = b
        self.c = c
        self.x = x
        self.v = v
        self.particles.append(self)

    @property
    def a(self):
        return self.f / self.m-self.b * self.v

    def force(self, x):
        return x * x * x / 3-2 * x-1 / x

    @property
    def f(self):
        a = np.zeros(2)
        for particle in self.particles:
            if particle is not self:
                dx = particle.x-self.x
                dxn = la.norm(dx)
                dxu = dx / dxn
                a += dxu * self.c * self.force(dxn)
        return a

    def step(self, h):
        if h is None:
            self.x, self.v = self.xnext, self.vnext
        else:
            self.xnext = self.x+h * self.v
            self.vnext = self.v+h * self.a


npr.seed(1)

n = 20
for i in range(n):
    Particle(m=1, b=0.5, c=1, x=2 * npr.randn(2), v=2 * npr.randn(2))

h = 0.001
fig, ax = plt.subplots()
X = np.zeros([n, 2])
for i in range(n):
    X[i, :] = Particle.particles[i].x
scat = ax.scatter(X[:, 0], X[:, 1])
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)


def update(t):
    for i in range(n):
        Particle.particles[i].step(h)
    X = np.zeros([n, 2])
    for i in range(n):
        Particle.particles[i].step(None)
        X[i, :] = Particle.particles[i].x
    scat.set_offsets(X)
    return scat,


aniobj = ani.FuncAnimation(fig, update, interval=1, blit=True)
