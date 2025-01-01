import numpy as np
import numpy.random as npr
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm


def rand_pd(n):
    E = np.diag(npr.rand(n))
    V = la.qr(npr.rand(n, n))[0]
    return V.dot(E.dot(V.T))


# Fix random state for reproducibility
np.random.seed(4)

# Number of modes
N = 2

# Mode probabilities
p = np.array([0.7, 0.3])

# Mode centers
means = np.zeros([N, 2])
for i in range(N):
    means[i] = 10*npr.rand(2)

# Mode covariances
covs = np.zeros([N, 2, 2])
for i in range(N):
    covs[i] = rand_pd(2)

# Number of samples
n = 100000

# Sample the number of samples from each mode
s = npr.choice(np.arange(N), p=p, size=n)
nss = np.unique(s, return_counts=True)[1]

# Sample from each mode
samples_list = []
for mean, cov, ns in zip(means, covs, nss):
    samples_list.append(npr.multivariate_normal(mean=mean, cov=cov, size=ns))
samples = np.vstack(samples_list)

C = np.hstack([np.ones(nss[0]), -np.ones(nss[1])])

xmin = np.min(samples[:, 0])
xmax = np.max(samples[:, 0])
ymin = np.min(samples[:, 1])
ymax = np.max(samples[:, 1])

# Plotting
cmap = 'RdBu'

plt.close('all')
fig, ax = plt.subplots(figsize=(6, 6))
fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
norm = SymLogNorm(linthresh=10, vmin=-1000, vmax=1000)
hb = ax.hexbin(samples[:, 0], samples[:, 1], C=C, reduce_C_function=np.sum, norm=norm, gridsize=50, cmap=cmap)
ax.axis([xmin, xmax, ymin, ymax])
ax.set_title("Hexagon binning")
cb = fig.colorbar(hb, ax=ax)
plt.show()
