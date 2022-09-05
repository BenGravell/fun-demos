import autograd.numpy as np
import autograd.numpy.linalg as la
from autograd import grad
import numpy.random as npr
from matplotlib import pyplot as plt
from matplotlib import animation


def split(a, b):
    # Split the whole number a into b near-equal whole numbers
    c = int(a/b)
    return [c if i >= a - b*c else c+1 for i in range(b)]


def rand_range(n, low, high):
    x = npr.rand(n)
    return low + x*(high-low)


def rand_psd(d):
    # Generate a random d x d positive semidefinite matrix
    E = np.diag(npr.rand(d))
    U = npr.randn(d, d)
    V = la.qr(U)[0]
    P = V.dot(E.dot(V.T))
    return P


def generate_data(n=1000, num_centers=5):
    ns = split(n, num_centers)
    x = np.zeros([n, dim])
    y = np.zeros(n, dtype=int)
    a = 0

    # Put true means in grid cells to ensure well-separated data
    grid_size = np.round(np.ceil(num_centers**0.5)).astype(int)
    grid_idxs = np.arange(grid_size**2)
    selected_idxs = npr.permutation(grid_idxs)[0:num_centers]
    selected_subs = np.array([np.unravel_index(selected_idx, shape=(grid_size, grid_size)) for selected_idx in selected_idxs])
    means_true = 10*selected_subs.astype(float)
    means_true += 5*np.array([npr.rand(dim) for i in range(num_centers)])

    # Randomly generate covariances
    covrs_true = np.array([rand_psd(dim) for i in range(num_centers)])

    for i, (true_mean, true_covr) in enumerate(zip(means_true, covrs_true)):
        b = a + ns[i]
        x[a:b] = npr.multivariate_normal(true_mean, true_covr, size=b-a)
        y[a:b] = i
        a = b

    idx = np.arange(n)
    npr.shuffle(idx)
    x = x[idx]
    y = y[idx]

    return x, y, means_true


def batch_indices(iter):
    idx = iter % num_batches
    return slice(idx * batch_size, (idx+1) * batch_size)


def loss(means, iter):
    # Loss over the examples defined by the batch_indices rule0
    idx = batch_indices(iter)
    sqdists = np.array([np.sum((x[idx] - mean)**2, axis=1) for mean in means])
    return np.mean(np.min(sqdists, axis=0))


def assign(means):
    # Clustering evaluation
    sqdists = np.array([np.sum((x - mean)**2, axis=1) for mean in means])
    y = np.argmin(sqdists, axis=0)
    return y


def init_means(x):
    return x[npr.permutation(x.shape[0])[0:k]]


def noise_schedule(iter):
    return 1.0/(1+iter)


def gd(means, noise_amt=1.0):
    # Gradient descent w/ random perturbations
    global iter
    out = means - lr*g(means, iter) + noise_amt*noise_schedule(iter)*npr.randn(k, dim)
    iter += 1
    return out


def plot_clusters(x, means, fig=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    scat1 = ax.scatter(x[:, 0], x[:, 1], c=assign(means), s=10, alpha=0.7, edgecolor='none', cmap=cmap)
    scat2 = ax.scatter(means[:, 0], means[:, 1], c=np.arange(k), s=100, marker='d', cmap=cmap, edgecolors='k', linewidths=2)
    fig.tight_layout()
    return fig, ax, scat1, scat2


if __name__ == "__main__":
    # Initialize
    seed = 1
    npr.seed(seed)

    # Problem data
    dim = 2  # dimensionality of each example, must be 2 for this script
    kt = 9  # number of true clusters
    n = 100*kt  # total number of examples
    batch_size = int(n/10)  # number of examples per batch
    num_batches = int(np.ceil(n/batch_size))

    x, y_true, means_true = generate_data(n, kt)

    k = kt  # number of assumed clusters
    g = grad(loss)  # gradient of loss
    lr = 0.1  # learning rate
    means = init_means(x)
    iter = 0

    # Visualization settings
    cmap = 'turbo'
    fps = 60
    fig, ax, scat1, scat2 = plot_clusters(x, means)
    stride = 1

    def animation_update(i):
        global means
        for j in range(stride):
            means = gd(means)
        scat2.set_offsets(means)
        scat1.set_array(assign(means))
        return [scat1, scat2]

    ani = animation.FuncAnimation(fig, animation_update, interval=1000/fps, blit=True)
    plt.show()
