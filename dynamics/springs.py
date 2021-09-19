from copy import copy
import numpy as np
import numpy.linalg as la
import numpy.random as npr
import scipy.linalg as sla
from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.colors import ListedColormap


def specrad(A):
    return np.max(np.abs(la.eig(A)[0]))


def mix(a, b, x):
    return x*a + (1-x)*b


def edgemod(A, i, j, val):
    A[i, i] -= val
    A[i, j] += val
    A[j, i] += val
    A[j, j] -= val
    return


def cont2discrete(a, b, dt):
    # This does the same thing as the function scipy.signal.cont2discrete with method="zoh"

    # Build an exponential matrix
    em_upper = np.hstack((a, b))

    # Need to stack zeros under the a and b matrices
    em_lower = np.hstack((np.zeros((b.shape[1], a.shape[1])),
                          np.zeros((b.shape[1], b.shape[1]))))

    em = np.vstack((em_upper, em_lower))
    ms = sla.expm(dt*em)

    # Dispose of the lower rows
    ms = ms[:a.shape[0], :]

    # Split
    ad = ms[:, 0:a.shape[1]]
    bd = ms[:, a.shape[1]:]
    return ad, bd


def make_ctime_system(N, G=None, M=None, m=None):
    # Create coupled spring-mass-damper system
    # N: Number of nodes
    # G: Number of spring-to-ground connections
    # M: Number of spring-to-spring edges
    # m: Number of actuators

    # Spring rates and damping coefficients for connections to ground
    spring_base = 0.01*mix(npr.rand(N), 0.5, 0.5)
    damp_base = 0.02*mix(npr.rand(N), 0.5, 0.5)
    if G is None:
        G = int(N/10)
    idxs_no_ground = npr.randint(N, size=N-G)
    for idx in idxs_no_ground:
        spring_base[idx] = 0.0
        damp_base[idx] = 0.0

    # Create edges programmatically
    if M is None:
        # Connect all adjacent node pairs (results in a line graph)
        edges = [{i, i + 1} for i in range(N - 1)]
        M = len(edges)
    else:
        # Randomly connect M node pairs (results in a complicated graph, will look weird visually)
        M = min(M, int((N*(N - 1))/2))
        edges = []
        for k in range(M):
            edge = {npr.randint(N), npr.randint(N)}
            while len(edge) == 1 or edge in edges:
                edge = {npr.randint(N), npr.randint(N)}
            edges.append(edge)

    # Spring rates and damping coefficients for connections between nodes
    spring_edges = 0.1*5.0*mix(npr.rand(M), 0.5, 0.5)
    damp_edges = 0.2*0.20*mix(npr.rand(M), 0.5, 0.5)

    A11 = np.zeros([N, N])  # Position-to-velocity
    A12 = np.eye(N)  # Velocity-to-velocity
    A21 = -np.diag(spring_base)  # Position-to-acceleration
    A22 = -np.diag(damp_base)  # Velocity-to-acceleration

    for edge, spring, damp in zip(edges, spring_edges, damp_edges):
        i, j = edge
        edgemod(A21, i, j, spring)
        edgemod(A22, i, j, damp)

    Ac = np.block([[A11, A12],
                   [A21, A22]])

    if m is None:
        m = int(N/10)
    # Evenly space m actuators
    if m == 1:
        actuators = [0]
    else:
        s = int((N - 1)/(m - 1))
        actuators = [i*s for i in range(m)]
    m = len(actuators)

    strengths = np.ones(m)  # Keep strengths as one for proper scaling of arrows on plot
    Bc = np.zeros([2*N, m])
    for i, (idx, strength) in enumerate(zip(actuators, strengths)):
        Bc[N + idx, i] = strength
    return Ac, Bc, actuators


def make_penalty(N, m):
    # State + action penalty
    Q = sla.block_diag(100*np.eye(N)/N,  # position penalties
                       1.0*np.eye(N)/N,  # velocity penalties
                       1.0*np.eye(m)/m)  # action penalties
    return Q


def make_initial_state(N, method=None):
    x0 = np.hstack([np.zeros(N), np.zeros(N)])

    if method is None:
        method = 'zeros'

    if method == 'zeros':
        pass
    elif method == 'ones':
        # All positions @ 1, zero velocity
        x0[0:N] = np.ones(N)
    elif method == 'ends':
        # Both ends only
        x0[[0, N - 1]] = 2
    elif method == 'sine':
        # Smooth sine shape
        x0[0:N] = np.cos(np.linspace(0, 2*np.pi, N))
    elif method == 'rand':
        # Random
        x0[0:N] = 2*npr.rand(N) - 1

    return x0


def dare(A, B, Q, R, E=None, S=None):
    """
    Solve the discrete-time algebraic Riccati equation.
    Wrapper around scipy.linalg.solve_discrete_are.
    Pass a copy of input matrices to protect them from modification.
    """
    return solve_discrete_are(copy(A), copy(B), copy(Q), copy(R), copy(E), copy(S))


def gain(P, A, B, Q):
    n, m = B.shape
    AB = np.hstack([A, B])
    H = np.dot(AB.T, np.dot(P, AB)) + Q
    Hux = H[n:n+m, 0:n]
    Huu = H[n:n+m, n:n+m]
    K = -la.solve(Huu, Hux)
    return K


def riccati_direct(A, B, Q):
    n, m = B.shape
    Qxx = Q[0:n, 0:n]
    Quu = Q[n:n+m, n:n+m]
    Qxu = Q[0:n, n:n+m]
    P = dare(A, B, Qxx, Quu, E=None, S=Qxu)
    K = gain(P, A, B, Q)
    return P, K


class Disturbance:
    def __init__(self, N):
        n = 2*N
        self.N = N
        self.n = n
        self.noise = np.zeros(n)
        self.w = np.zeros(n)
        self.spike_clocks = np.zeros(N)
        self.spike_mags = np.zeros(N)
        self.disturb_mean = np.zeros(n)
        # self.disturb_covr = sla.block_diag(1e-9*np.eye(N), 1e-3*np.eye(N))
        self.disturb_std_pos = 0
        self.disturb_std_vel = 0.03

    def calc_spike(self, x, mag=1.0, time_scale=2.0):
        def f(x):
            return np.tanh(x)/(1 + 4*x)
        return mag*0.05*f(time_scale*0.01*x)

    def update(self, t, period=600):
        if t % period == 0:
            idx = npr.randint(self.N)
            self.spike_clocks[idx] = 0
            self.spike_mags[idx] = (npr.rand() + 1)*npr.choice([-1, 1])
        spike = np.zeros(self.n)
        spike[self.N:] = self.spike_mags*self.calc_spike(self.spike_clocks)
        self.spike_clocks += 1
        # self.noise = mix(npr.multivariate_normal(self.disturb_mean, self.disturb_covr), self.noise, 0.001)
        # self.w = spike + self.noise
        self.noise = np.hstack([self.disturb_std_pos*npr.randn(self.N), self.disturb_std_vel*npr.randn(self.N)])
        self.w = spike
        return self.w


class SpringScatter:
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, N=20, G=None, M=None, m=None, dt=0.1, state0=None, seed=1,
                 show_curve=True, show_scatter=True, show_action=True, show_disturbance=True):
        npr.seed(seed)

        self.N = N
        self.G = G
        self.n = 2*N
        self.Ac, self.Bc, self.actuators = make_ctime_system(N, G, M, m)
        self.m = len(self.actuators)
        n, m = self.n, self.m
        self.dt = dt
        self.A, self.B = cont2discrete(self.Ac, self.Bc, self.dt)
        self.Q = make_penalty(N, m)
        self.disturbance = Disturbance(N)

        self.show_curve = show_curve
        self.show_scatter = show_scatter
        self.show_action = show_action
        self.show_disturbance = show_disturbance

        # Control design
        print('Designing controller...', end='')
        self.P, self.K = riccati_direct(self.A, self.B, self.Q)
        self.AK = self.A + np.dot(self.B, self.K)
        print('finished.')

        # Initial state
        if state0 is None:
            state0 = make_initial_state(N)

        self.state = np.copy(state0)
        self.action = np.zeros(m)
        self.offs = np.arange(N)

        # Setup the figure and axes
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        # Then setup FuncAnimation
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=5,
                                           init_func=self.setup_plot, blit=True)

    def scat_data(self, state):
        pos, vel = state[0:self.N], state[self.N:2*self.N]
        x = self.offs
        y = pos
        s = 150*(900/(self.N**2))*np.ones(self.N)
        c = 5*np.abs(vel)
        return x, y, s, c

    def quiv_data(self, action, x, y):
        X = x[self.actuators]
        Y = y[self.actuators]
        U = np.zeros(self.m)
        V = 400*action
        C = 10*np.abs(action)
        return X, Y, U, V, C

    def quivd_data(self, w, x, y):
        w2 = w[self.N:]
        X = x
        Y = y
        U = np.zeros(self.N)
        V = 400*w2/self.dt
        C = 10*np.abs(w2)
        return X, Y, U, V, C

    def setup_plot(self):
        """Initial drawing of the plot."""
        self.artists = []

        x, y, s, c = self.scat_data(self.state)

        if self.show_curve:
            self.curve, = self.ax.plot(x, y, color='k', linewidth=4, zorder=110)
            self.artists.append(self.curve)

        # Scatter points for each node
        if self.show_scatter:
            scat_cmap = ListedColormap(cm.get_cmap('Greys', 512)(np.linspace(0.9, 0.4, 256)))
            scat_norm = Normalize(0, 1)
            self.scat = self.ax.scatter(x, y, c=c, s=s, norm=scat_norm, marker='o',
                                        cmap=scat_cmap, edgecolor='none', alpha=0.8, zorder=100)
            self.artists.append(self.scat)

        # Quiver arrows for the control actions
        if self.show_action:
            X, Y, U, V, C = self.quiv_data(self.action, x, y)
            quiv_norm = Normalize(0, 1)
            quiv_cmap = ListedColormap(cm.get_cmap('Blues', 512)(np.linspace(0.4, 0.8, 256)))
            self.quiv = self.ax.quiver(X, Y, U, V, C, pivot='tail', units='dots', scale=1.0, scale_units='dots',
                                       norm=quiv_norm, cmap=quiv_cmap, alpha=0.8, zorder=300)
            self.artists.append(self.quiv)

        # Quiver arrows for the disturbances
        if self.show_disturbance:
            Xd, Yd, Ud, Vd, Cd = self.quivd_data(self.disturbance.w, x, y)
            quivd_norm = Normalize(0, 1)
            quivd_cmap = ListedColormap(cm.get_cmap('Reds', 512)(np.linspace(0.5, 0.9, 256)))
            self.quivd = self.ax.quiver(Xd, Yd, Ud, Vd, Cd, pivot='tail', units='dots', scale=1.0, scale_units='dots',
                                       norm=quivd_norm, cmap=quivd_cmap, alpha=0.8, zorder=200)
            self.artists.append(self.quivd)

        # Reference horizontal line in the middle
        self.ax.axhline(0, color='k', alpha=0.2)

        # Plot options
        self.ax.axis([-1, self.N, -2, 2])
        self.ax.axis('off')

        return self.artists

    def update(self, t):
        # Simulation updates
        self.disturbance.update(t)
        self.action = np.dot(self.K, self.state)
        self.state = np.dot(self.A, self.state) + np.dot(self.B, self.action) + self.disturbance.w
        x, y, s, c = self.scat_data(self.state)

        # Update curve
        if self.show_curve:
            self.curve.set_ydata(y)

        # Update scatter plot
        if self.show_scatter:
            # Set scatter positions
            xy = np.vstack([x, y]).T
            self.scat.set_offsets(xy)
            # Set sizes
            self.scat.set_sizes(s)
            # Set colors
            self.scat.set_array(c)

        # Update quiver for action
        if self.show_action:
            X, Y, U, V, C = self.quiv_data(self.action, x, y)
            self.quiv.set_offsets(np.vstack([X, Y]).T)
            self.quiv.set_UVC(U, V, C)

        # Update quiver for disturbance
        if self.show_disturbance:
            Xd, Yd, Ud, Vd, Cd = self.quivd_data(self.disturbance.w, x, y)
            self.quivd.set_offsets(np.vstack([Xd, Yd]).T)
            self.quivd.set_UVC(Ud, Vd, Cd)

        return self.artists


if __name__ == '__main__':
    plt.close('all')
    SpringScatter(N=40)
    plt.show()
