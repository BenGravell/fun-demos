import numpy as np
import matplotlib.pyplot as plt


def abs2(x):
    return np.square(x.real) + np.square(x.imag)


d, n = 256, 32  # pixel density & number of iterations
# d, n = 1024, 256  # pixel density & number of iterations
r = 2  # escape radius (must be greater than 2)
r2 = r**2

x = np.linspace(-2.5, 1.5, 4*d + 1)
y = np.linspace(-1.5, 1.5, 3*d + 1)

X, Y = np.meshgrid(x, y)
C = X + Y*1j

Z = np.zeros_like(C)
T = np.zeros(C.shape, dtype=int)

for k in range(n):
    mask = abs2(Z) < r2
    Z[mask] = Z[mask]**2 + C[mask]
    T[mask] = k + 1

plt.imshow(T, cmap=plt.cm.inferno_r)
plt.show()
