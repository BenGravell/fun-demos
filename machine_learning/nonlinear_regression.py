import numpy as np
import matplotlib.pyplot as plt

# Data generation
np.random.seed(1)
n = 100  # number of examples
# Raw features w/ noise
x_true = np.linspace(0, 6, n)
x = x_true + 0.1*np.random.randn(n)
# Labels w/ noise
y_true = np.sin(2*x_true) + 0.1*x_true**2
y = y_true + 0.5*np.random.randn(n)


def featurize(x):
    return np.vstack([x**i for i in range(d)]).T


def predict(x, w):
    # Model is linear-in-nonlinear-features
    phi = featurize(x)
    return phi @ w


# Training via least-squares
d = 8  # Degree of the polynomial
# Nonlinear features
phi = featurize(x)
w = np.linalg.lstsq(phi, y, rcond=None)[0]

# Prediction
y_pred = predict(x_true, w)

# Plotting
plt.plot(x_true, y_true, lw=2, color='tab:green', linestyle='--', alpha=0.5, label='Ground truth')
plt.plot(x_true, y_pred, lw=2, color='tab:blue', alpha=0.7, label='Prediction')
plt.scatter(x, y, s=20, color='k', alpha=0.3, edgecolors='none', label='Training data')
plt.legend()
plt.axis('off')
plt.show()
