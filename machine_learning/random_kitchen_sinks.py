"""Random kitchen sinks with random Fourier features.

See https://people.eecs.berkeley.edu/~brecht/kitchensinks.html
See http://www.argmin.net/2017/12/05/kitchen-sinks/
"""


from time import time
import numpy as np
from numpy import pi, cos, dot, outer, eye, zeros, ones, floor, ceil, sign, maximum, absolute
from numpy.random import randn, rand
import numpy.random as npr
import numpy.linalg as la
import scipy.linalg as sla
import scipy.sparse.linalg as ssla
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons


# Data settings
seed = 1
npr.seed(seed)
n_samples = 4000  # Number of total data examples (train + test)
feature_noise_scale = 0.3  # Intensity of the noise applied to the features
label_noise_scale = 0.5  # Intensity of the noise applied to the labels
vmin, vmax = -2, 2  # Label lower and upper bounds

# Create data
X, y = make_moons(n_samples=n_samples, noise=feature_noise_scale, random_state=seed)
X, _ = la.qr(X)
# X = StandardScaler().fit_transform(X)
y = 2*y - 1  # Convert from 0, 1 labels to -1, 1 labels
y = y + label_noise_scale*npr.randn(n_samples)
y = y + 0.5*np.sign(y)  # Move label values away from zero
y = np.clip(y, vmin, vmax)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
x_min, x_max = X[:, 0].min()*1.1, X[:, 0].max()*1.1
y_min, y_max = X[:, 1].min()*1.1, X[:, 1].max()*1.1
diam = np.max([x_max - x_min, y_max - y_min])

# Dimensions and predictor settings
N, d = X_train.shape  # Training data dimensions
D = 640  # Number of random features
reg = 1e-1  # Regularization weight
gam = 10.0/diam  # Gaussian kernel standard deviation


def train(Z, y, solver='exact', solver_options=None, show_time=True):
    # NOTE: 'approx' may be slower than 'exact'
    #       for small and medium problem instances if the convergence tolerance is small
    if show_time:
        time_start = time()

    D = Z.shape[1]
    if solver == 'exact':
        A = dot(Z.T, Z) + (reg*reg)*eye(D)
        B = dot(Z.T, y)
        alpha = sla.solve(A, B, assume_a='sym')
    elif solver == 'approx':
        if solver_options is None:
            solver_options = dict(atol=1e-10, btol=1e-10, conlim=1e10, iter_lim=200, show=True)
        alpha = ssla.lsqr(Z, y, damp=reg, **solver_options)[0]
    else:
        raise ValueError

    if show_time:
        time_now = time()
        time_elapsed = time_now - time_start
        print('Training completed in %.6f seconds\n' % time_elapsed)
    return alpha


def soft_threshold(alpha, t):
    return sign(alpha)*maximum(absolute(alpha) - t, 0)


def featurize(X, w, b):
    N = X.shape[0]
    return cos(dot(X, w) + outer(ones(N), b))


def predict(X, w, b, alpha):
    return dot(featurize(X, w, b), alpha)


def threshold(y, level=0):
    return 2*(y > level).astype(np.int64) - 1


def score(y, y_true, thresh=False):
    if thresh:
        error = np.mean(y-y_true != 0)
        print('Mean error, threshold: %.2f%%' % (100 * error))
    else:
        error = np.mean(np.abs(y-y_true))
        print('Mean error,       raw: %.2f%%' % (100 * error))
    return error


# Model 1: Train a random kitchen sink
w = gam*randn(d, D)  # feature params, frequencies
b = 2*pi*rand(D)  # feature params, phases
Z_train = featurize(X_train, w, b)  # features
alpha = train(Z_train, y_train, solver='exact')  # trained weights

# Test
y_test_predict = predict(X_test, w, b, alpha)
y_test_predict_thresh = threshold(y_test_predict)
y_test_thresh = threshold(y_test)
print("Prediction error, original parameter")
score(y_test_predict, y_test, thresh=False)
score(y_test_predict_thresh, y_test_thresh, thresh=True)
print('')

# Model 2: LASSO via soft thresholding
l1reg = 0.5
alpha_sparse = soft_threshold(alpha, t=l1reg)
print("Number of nonzero entries in original parameter = %6d" % np.count_nonzero(alpha))
print("Number of nonzero entries in    LASSO parameter = %6d" % np.count_nonzero(alpha_sparse))

# Test
y_test_predict = predict(X_test, w, b, alpha_sparse)
y_test_predict_thresh = threshold(y_test_predict)
y_test_thresh = threshold(y_test)
print("Prediction error, LASSO parameter")
score(y_test_predict, y_test, thresh=False)
score(y_test_predict_thresh, y_test_thresh, thresh=True)
print('')

# Model 3: Re-fit using selected features
mask = absolute(alpha) > l1reg
w = w[:, mask]
b = b[mask]
D = b.size
Z_train = featurize(X_train, w, b)  # features
alpha = train(Z_train, y_train, solver='exact')  # trained weights

# Test
y_test_predict = predict(X_test, w, b, alpha)
y_test_predict_thresh = threshold(y_test_predict)
y_test_thresh = threshold(y_test)
print("Prediction error, sparse re-fit parameter")
score(y_test_predict, y_test, thresh=False)
score(y_test_predict_thresh, y_test_thresh, thresh=True)
print('')

# Plot the decision boundary. For that, we will assign a color to each point in the mesh.
ng = 100
xx, yy = np.meshgrid(np.linspace(x_min, x_max, ng),
                     np.linspace(y_min, y_max, ng))
X_grid = np.c_[xx.ravel(), yy.ravel()]
y_grid_raw = predict(X_grid, w, b, alpha)
y_grid_raw = y_grid_raw.reshape(xx.shape)
y_grid = threshold(y_grid_raw)


# Plotting
plt.close('all')
fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(12, 8))
cm1 = 'bwr_r'
cm2 = 'RdBu'
cm3 = 'PiYG'
s = 20
edgecolor = 'k'
num_levels = 10

ax[0, 0].scatter(X_test[:, 0], X_test[:, 1], s=s, c=y_test, cmap=cm1, vmin=vmin, vmax=vmax, alpha=0.8, edgecolors=edgecolor)
ax[1, 0].scatter(X_test[:, 0], X_test[:, 1], s=s, c=y_test_thresh, cmap=cm1, vmin=-1, vmax=1, alpha=0.8, edgecolors=edgecolor)
ax[0, 0].set_title('True')

ax[0, 1].contourf(xx, yy, y_grid_raw, cmap=cm2, vmin=vmin, vmax=vmax, alpha=0.8, levels=np.linspace(vmin, vmax, num_levels), extend='both')
ax[0, 1].scatter(X_test[:, 0], X_test[:, 1], s=s, c=y_test_predict, cmap=cm1, vmin=vmin, vmax=vmax, alpha=0.8, edgecolors=edgecolor)

ax[1, 1].contourf(xx, yy, y_grid, cmap=cm2, vmin=-1, vmax=1, alpha=0.8, levels=np.linspace(-1, 1, num_levels), extend='both')
ax[1, 1].scatter(X_test[:, 0], X_test[:, 1], s=s, c=y_test_predict_thresh, cmap=cm1, vmin=-1, vmax=1, alpha=0.8, edgecolors=edgecolor)
ax[0, 1].set_title('Prediction')

ax[0, 2].scatter(X_test[:, 0], X_test[:, 1], s=s, c=y_test_predict-y_test, cmap=cm3, vmin=-2, vmax=2, alpha=0.5)
ax[1, 2].scatter(X_test[:, 0], X_test[:, 1], s=s, c=y_test_predict_thresh-y_test_thresh, cmap=cm3, vmin=-2, vmax=2, alpha=0.5)
ax[0, 2].set_title('Error')

fig.tight_layout()


# Show individual random features.
num_show_features_max = 8**2
num_show_features = D if D <= num_show_features_max else num_show_features_max
nrows = int(ceil(num_show_features**0.5))
ncols = int(ceil(num_show_features**0.5))
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(8, 8))
Z_grid = featurize(X_grid, w, b)
k = 0
a_max = np.max(alpha)
for i in range(nrows):
    for j in range(ncols):
        if k < D:
            zz = alpha[k]*Z_grid[:, k].reshape(xx.shape)
            ax[i, j].contourf(xx, yy, zz, cmap=cm2, vmin=-a_max, vmax=a_max,
                              levels=np.linspace(-a_max, a_max, 100), extend='both')
            ax[i, j].axis('off')
            k += 1
fig.suptitle('Predictor decomposed by individual weighted random features')
fig.tight_layout()

plt.show()
