import numpy as np
import matplotlib.pyplot as plt


def abs2(x):
    return np.square(x.real) + np.square(x.imag)


pixel_density = 512
num_iterations = 64
escape_radius = 2
assert escape_radius >= 2
escape_radius2 = escape_radius*escape_radius

xmin, xmax = -2.5, 1.5
ymin, ymax = -1.5, 1.5
x = np.linspace(xmin, xmax, int(xmax-xmin)*pixel_density + 1)
y = np.linspace(ymin, ymax, int(ymax-ymin)*pixel_density + 1)

X, Y = np.meshgrid(x, y)
C = X + Y*1j

Z = np.zeros_like(C)
T = np.zeros(C.shape, dtype=int)

for k in range(num_iterations):
    mask = abs2(Z) < escape_radius2
    Z[mask] = Z[mask]**2 + C[mask]
    T[mask] = k + 1

plt.imshow(T, cmap='coolwarm')
plt.show()
