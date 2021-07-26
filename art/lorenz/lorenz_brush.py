import numpy as np
import numpy.random as npr
import numpy.linalg as la
from scipy.ndimage import convolve
import matplotlib.pyplot as plt


###############################################################################
# Simulation functions
###############################################################################

# Lorenz attractor nonlinear dynamics
def f(x, s=10, r=28, b=8 / 3):
    return np.array([s * (x[:, 1]-x[:, 0]),
                     x[:, 0] * (r-x[:, 2])-x[:, 1],
                     x[:, 0] * x[:, 1]-b * x[:, 2]]).T


# 4th-order Runge-Kutta ODE solver
def rk4(x, h=None):
    if h is None:
        h = 1e-3
    k1 = h * f(x)
    k2 = h * f(x+k1 / 2)
    k3 = h * f(x+k2 / 2)
    k4 = h * f(x+k3)
    return x+(k1+2 * k2+2 * k3+k4) / 6


# Simulate trajectories and paint onto canvas at each step
def sim(x0, nt, nu, nx, ny, xmin, ymin, xdel, ydel, h=None):
    M = np.zeros([ny, nx])
    xn = np.copy(x0)
    for t in range(nt):
        x = np.copy(xn)
        xn = rk4(x, h)
        dx = (xn-x) / nu
        for u in range(nu):
            xu = x+u * dx
            mi = (ny * (1.0-((xu[:, 2]-ymin) / ydel))).astype(int)
            mj = (nx * ((xu[:, 0]-xmin) / xdel)).astype(int)
            mi = np.clip(mi, 0, ny-1)
            mj = np.clip(mj, 0, nx-1)
            M[mi, mj] += 1  # Unique entries incremented (no weight)
        print('Simulating timestep %d / %d' % (t+1, nt))
    return xn, M


###############################################################################
# Graphics functions
###############################################################################

# Brush falloff function
def brush_falloff(R, ftype='bump'):
    if ftype == 'bump':
        return np.exp(-1 / (1-R**2))
    elif ftype == 'inv_quad':
        return (R-1)**2


# Brush image kernel
def create_brush(br):
    brush = np.zeros([2 * br+1, 2 * br+1])
    for i in range(2 * br+1):
        for j in range(2 * br+1):
            R = la.norm([i-br, j-br]) / br
            if R < 1:
                brush[i, j] += brush_falloff(R, 'inv_quad')
    return brush


# Draw and save the image
def im(M, show_plot=True, save_plot=True, cmap='inferno'):
    if show_plot:
        plt.imshow(M, cmap=cmap, interpolation='gaussian')
        plt.axis('off')
        plt.show()
    if save_plot:
        plt.imsave('lorenz.png', M, cmap=cmap)


###############################################################################
# Main function
###############################################################################
if __name__ == "__main__":
    load = False

    if load:
        M = np.load('lorenz.npy')
        brush = create_brush(2)
        M = convolve(M, brush)
        M = np.tanh(M/M.max())
        im(M, cmap='inferno')
    else:
        npr.seed(1)

        quality = 'draft'
        # quality = 'medium'
        # quality = 'final'

        if quality == 'draft':
            nt_init, nt = 100, 500  # Number of ODE solver steps
            h = 2/nt  # ODE solver timestep
            nx, ny = 480, 270  # Image dimensions
            nu = 5  # Number of linear interpolation points
            ns = 100  # Number of sample initial conditions/trajectories
        elif quality == 'medium':
            nt_init, nt = 400, 2000  # Number of ODE solver steps
            h = 2/nt  # ODE solver timestep
            nx, ny = 1280, 720  # Image dimensions
            nu = 10  # Number of linear interpolation points
            ns = 500  # Number of sample initial conditions/trajectories
        elif quality == 'final':
            nt_init, nt = 1000, 5000  # Number of ODE solver steps
            h = 2/nt  # ODE solver timestep
            nx, ny = 3840, 2160  # Image dimensions
            nu = 20  # Number of linear interpolation points
            ns = 1000  # Number of sample initial conditions/trajectories
        else:
            raise ValueError

        xmin, xmax = -20, 20
        ymin, ymax = -3, 48

        xdel, ydel = xmax-xmin, ymax-ymin

        # Create many sample initial conditions randomly spread out
        xnom = np.array([(xmin+xmax) / 2, 1, (ymin+ymax) / 2])
        x = xnom+0.5 * (npr.rand(ns, 3)-0.5) * np.array([ydel, 0.5, xdel])

        # Simulate to let trajectories evolve onto the invariant manifold
        x, M = sim(x, nt_init, nu, nx, ny, xmin, ymin, xdel, ydel, h)

        # Simulate to paint canvas
        x, M = sim(x, nt, nu, nx, ny, xmin, ymin, xdel, ydel, h)

        # Save preprocessed data
        np.save('lorenz.npy', M)

        # Convolve with brush to smooth edges out
        brush = create_brush(2)
        M = convolve(M, brush)
        M = np.tanh(M / M.max())
        im(M, cmap='inferno')
