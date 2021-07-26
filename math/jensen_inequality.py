import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

N = 40
x_lwr = 3
x_upr = -1
x = x_lwr + (x_upr-x_lwr)*npr.rand(N)


def f(x):
    return x**2


y = f(x)

x_sample_mean = np.mean(x, axis=0)
y_sample_mean = np.mean(y)
y_from_x_sample_mean = f(x_sample_mean)


x_grid = np.linspace(x_lwr, x_upr, 1000)
y_grid = f(x_grid)

plt.close('all')
plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(x, y, c='tab:blue', s=50, alpha=0.6, zorder=10, label='Samples')
ax.scatter(x_sample_mean, y_sample_mean, c='tab:red', s=200, zorder=11, label=r'$\mathbb{E}[f(x)]$')
ax.scatter(x_sample_mean, y_from_x_sample_mean, c='tab:orange', s=200, zorder=12, label=r'$f(\mathbb{E}[x])$')
ax.plot(x_grid, y_grid, linestyle='-', color='k', lw=2, alpha=0.8, zorder=1, label=r'$f(x)$')
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.legend()
plt.tight_layout()
plt.show()
