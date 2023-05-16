import streamlit as st

import numpy as np
import matplotlib.pyplot as plt


def featurize(x, d):
    # Nonlinear polynomial features
    return np.vstack([x**i for i in range(d+1)]).T


def predict(x, d, w):
    # Model is linear-in-nonlinear-features
    phi = featurize(x, d)
    return phi @ w

d_true = 8
w_true = np.array([-1.735e-03, 1.850e+00, 1.001e-01, -1.098e+00, 1.930e-03, 1.641e-01, -5.610e-04, -7.326e-03, 3.900e-05])

def generate_data(seed, noise_std=0.1, n=100):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(low=-np.pi, high=np.pi, size=n))
    phi_true = featurize(x, 8)
    y_true = phi_true @ w_true
    y = y_true + noise_std*rng.standard_normal(size=n)
    return x, y, y_true


def data_plot(x, y, y_true, y_pred):
    fig, ax = plt.subplots()
    ax.plot(x, y_true, lw=2, color='tab:green', linestyle='--', alpha=0.7, label='Ground Truth')
    ax.plot(x, y_pred, lw=2, color='tab:blue', alpha=0.7, label='Prediction')
    ax.scatter(x, y, s=10, color='k', alpha=0.4, edgecolors='none', label='Training Data')
    ax.legend()
    # ax.axis('off')
    return fig, ax

def weight_plot(d, w, d_true, w_true):
    fig, ax = plt.subplots()
    ax.bar(np.arange(d+1), w, alpha=0.7, label="Trained Model")
    ax.bar(np.arange(d_true+1), w_true, width=0.4, color='k', alpha=0.5, label="Ground Truth")
    ax.legend()
    return fig, ax

with st.sidebar:
    seed = st.slider("Seed", min_value=1, max_value=10, value=1, step=1)
    n = st.select_slider("Number of examples", [10, 30, 100, 300, 1000], value=10)
    noise_std = st.select_slider("Noise Standard Deviation", np.logspace(-2, 0, 101), format_func=lambda x: x.round(3), value=0.1)
    d = st.slider("Model Polynomial Degree", min_value=1, max_value=20, value=8, step=1)


# Generate dataset
x, y, y_true = generate_data(seed, noise_std, n)

# Training via least-squares
phi = featurize(x, d)
w = np.linalg.lstsq(phi, y, rcond=None)[0]

# Prediction
y_pred = predict(x, d, w)

# Metrics
rmse = np.sqrt(np.mean(np.square(y_pred - y_true)))
st.metric("RMSE", rmse.round(3))

# Plotting
data_fig, data_ax = data_plot(x, y, y_true, y_pred)
st.pyplot(data_fig)

st.header("Model Weights")
weight_fig, weight_ax = weight_plot(d, w, d_true, w_true)
st.pyplot(weight_fig)
