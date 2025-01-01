import numpy as np
import numpy.linalg as la
import numpy.random as npr
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    def __init__(self, b, v, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        xs, ys, zs = list(zip(b, v))
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


# This class uses the (scalar, vector) decomposition of quaternions
class Quaternion:
    def __init__(self, r, v):
        self.r = r
        self.v = v

    @property
    def conj(self):
        return Quaternion(self.r, -self.v)

    @property
    def normsq(self):
        return self.r**2 + np.sum(self.v**2)

    @property
    def norm(self):
        return np.sqrt(self.normsq)

    @property
    def inv(self):
        return self.conj / self.normsq

    def __repr__(self):
        return "quaternion" + str((self.r, self.v))

    def __add__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(self.r + other.r, self.v + other.v)
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(self.r - other.r, self.v - other.v)
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            r = self.r * other.r - np.dot(self.v, other.v)
            v = self.r * other.v + other.r * self.v + np.cross(self.v, other.v)
            return Quaternion(r, v)
        else:
            return Quaternion(self.r * other, self.v * other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, Quaternion):
            return self * other.inv
        else:
            return Quaternion(self.r / other, self.v / other)

    def __rtruediv__(self, other):
        return other * self.inv


def hprod(p, q):
    return q * p / q


def mix(a, b, t):
    return t * a + (1 - t) * b


def normalize(v):
    return v / la.norm(v)


npr.seed(1)

# Starting point
p = Quaternion(0, normalize(npr.randn(3)))

# Angle to sweep thru
theta_max = 2 * np.pi * (30 / 360)

plt.close("all")
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

num_directions = 10
num_samples = 8

for i in range(num_directions):
    # Randomly select a rotation axis
    a = normalize(npr.randn(3))
    color = (1 + a) / 2

    for t in np.linspace(0, 1, num_samples):
        b = np.array([0, 0, 0])
        theta = t * theta_max
        q = Quaternion(np.cos(theta / 2), np.sin(theta / 2) * a)
        pr = hprod(p, q)
        v = pr.v

        arrow = Arrow3D(b, v, mutation_scale=10, color=mix(color, np.zeros(3), t))
        ax.add_artist(arrow)

ax.set_proj_type("ortho")
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
ax.set_zlim([-1.1, 1.1])
plt.show()
