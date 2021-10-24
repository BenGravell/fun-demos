import numpy as np
import numpy.random as npr
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.animation as ani

###############################################################################
# Kuramoto network
###############################################################################

###################
# System definition
###################
npr.seed(1)  # Random seed for repeatability
n = 40  # Number of oscillators
w = 1.0 + npr.rand(n)  # Intrinsic angular velocities
dt = 0.01  # Sampling time


def incidence(A):
    m = int(A.sum() / 2)
    B = np.zeros([n, m])
    k = 0
    for i in range(n):
        for j in range(i):
            if A[i, j]:
                B[[i, j], k] = 1
                k = k+1
    return B


def laplacian(B):
    return B @ B.T


## Unweighted adjacency matrix
# A = np.zeros([n,n])
# add_edge = True;
# while add_edge:
#    # Add edge randomly
#    i,j = npr.randint(n),npr.randint(n)
#    if i==j:
#        continue
#    A[[i,j],[j,i]] = 1
#
#    # Unweighted edge incidence matrix
#    B = incidence(A)
#
#    # Graph Laplacian
#    L = laplacian(B)
#
#    # Check if graph is connected
#    L_eigvals,L_eigvecs = la.eig(L)
#    L_eigvals = np.sort(np.real(L_eigvals))
#
#    if (L_eigvals<1e-12).sum() == 1:
#        add_edge = False

# Unweighted adjacency matrix
A = np.ones([n, n])-np.diag(np.ones(n))  # fully connected

# Unweighted edge incidence matrix
B = incidence(A)

# Graph Laplacian
L = laplacian(B)

W = np.diag(w)
C = B.T @ W @ B
E, V = la.eig(C)
E = np.sort(np.real(E))
Emin = E[np.where(E > 1e-12)].min()
gamma = np.pi / 2
kappa_min = abs(max(w)-min(w)) / (Emin * np.sin(gamma))

# Weighted adjacency matrix (couplings "kappa" included as weights)
kappa = 1.0 * kappa_min
A = kappa * A


# Kuramoto oscillator dynamics (zero-order hold discretization)
def f(theta):
    dtheta = np.zeros(n)
    for i in range(n):
        neighbors = [j for j in range(n) if i != j and A[i, j] > 0]
        coupling = A[i, neighbors] @ (np.sin(theta[i]-theta[neighbors]))
        dtheta[i] = dt * w[i] * (1-coupling)
    return dtheta


#########################
# Simulation and plotting
#########################

# Initial condition
theta0 = gamma * npr.rand(n)
theta = np.copy(theta0)

# Display radii
r_mid = 1
r_lwr = 0.8 * r_mid
r_upr = 1.2 * r_mid
r = np.linspace(r_lwr+0.05, r_upr-0.05, n)


def display_x(theta):
    return np.array([r * np.cos(theta), r * np.sin(theta)]).T


def initial_plot(theta):
    fig, ax = plt.subplots()

    # Draw particles/nodes
    dispx = display_x(theta)
    scat_size = 50
    scat = ax.scatter(dispx[:, 0], dispx[:, 1], s=scat_size, alpha=0.5, edgecolor='none')

    # Draw orbit curves
    ax.add_artist(plt.Circle((0, 0), r_mid, facecolor=[0, 0, 0, 0], edgecolor=[0, 0, 0, 0.8], lw=2))
    ax.add_artist(plt.Circle((0, 0), r_lwr, facecolor=[0, 0, 0, 0], edgecolor=[0, 0, 0, 0.4], lw=1))
    ax.add_artist(plt.Circle((0, 0), r_upr, facecolor=[0, 0, 0, 0], edgecolor=[0, 0, 0, 0.4], lw=1))

    # Set axis options
    xmin = -r_mid
    xmax = +r_mid
    ymin = -r_mid
    ymax = +r_mid
    xdel = xmax-xmin
    ydel = ymax-ymin
    view_margin = 0.20
    ax.set_xlim(xmin-view_margin * xdel, xmax+view_margin * xdel)
    ax.set_ylim(ymin-view_margin * ydel, ymax+view_margin * ydel)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.set_position([0, 0, 1, 1])

    return fig, ax, scat


fig, ax, scat = initial_plot(theta)


def update(t):
    global theta
    theta = theta+f(theta)
    scat.set_offsets(display_x(theta))
    return scat,


ao = ani.FuncAnimation(fig, update, interval=0, blit=True)
plt.show()
