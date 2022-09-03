from numpy import inf, array, zeros, eye, diag, arange, linspace, sin, pi
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from proplot import Colormap

# Visualization settings
fps = 60  # frames per second
cmap = Colormap('IceFire', left=0.1, right=0.9)

# Dynamical system settings
n, nt, dt = 10, 200, 0.01  # number of state, number of timesteps, inter-sample time
ts = dt*arange(nt)


def gen_A(a=0):
    t = a / fps

    freqs_D = 0.05*linspace(1, 1.3, n)
    phases_D = linspace(0.3, 0.5, n)

    freqs_F = 0.08*linspace(1, 1.4, n-1)
    phases_F = linspace(0.6, 0.9, n-1)

    Dp = 1 + sin(freqs_D*2*pi*t + 2*pi*phases_D)
    Fp = 1 + sin(freqs_F*2*pi*t + 2*pi*phases_F)

    D = diag(linspace(2.0, 0.5, n) + Dp)
    F = diag(linspace(1.0, 4.0, n - 1) + Fp, 1)
    A = eye(n) - (D + F - F.T)*dt

    return A


def gen_x0(a=0):
    freqs = 0.2*linspace(1, 1, n)
    phases = linspace(0, 0.5, n)
    t = a / fps
    return sin(freqs*2*pi*t + 2*pi*phases)


def get_color(x):
    t = (1 + x)/2
    return cmap(t)


A = gen_A(0)
As = zeros([nt, n, n])
As[0] = eye(n)
for k in range(nt-1):
    As[k+1] = A @ As[k]

# Boundary curve, based on sub-multiplicativity of inf-norm
br = array([la.norm(M, ord=inf) for M in As])

x0 = gen_x0()
x = As @ x0

# Set up the plot
fig, ax = plt.subplots(figsize=(6, 6))
lines = [ax.plot(ts, x[:, i], color=get_color(x0[i]), lw=2, alpha=0.5)[0] for i in range(n)]
plt.plot(ts, zeros(nt), 'k', alpha=0.2)
line_bnd_upr = ax.plot(ts, br, 'k', alpha=0.2)[0]
line_bnd_lwr = ax.plot(ts, -br, 'k', alpha=0.2)[0]
plt.ylim([-2.5, 2.5])
plt.axis('off')
plt.tight_layout()


def update(a):
    A = gen_A(a)
    As = zeros([nt, n, n])
    As[0] = eye(n)
    for k in range(nt - 1):
        As[k + 1] = A@As[k]
    br = array([la.norm(M, ord=inf) for M in As])
    line_bnd_upr.set_ydata(br)
    line_bnd_lwr.set_ydata(-br)

    x0 = gen_x0(a)
    x = As @ x0
    for i in range(n):
        lines[i].set_ydata(x[:, i])
        lines[i].set_color(get_color(x0[i]))
    return ax,


ani = animation.FuncAnimation(fig, update, interval=1000/fps, blit=True)
plt.show()
