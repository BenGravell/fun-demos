import numpy as np
import numpy.random as npr
import numpy.linalg as la
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
import matplotlib.animation as ani
from matplotlib import cm
from matplotlib.animation import FuncAnimation


def mix(a, b, z):
    return (1-z)*a + z*b


def unit_saturate(x, strength=1.0):
    return np.tanh(strength*x)/np.tanh(strength)


def clipped_mean(x, pct_lwr=25, pct_upr=75):
    return np.mean(x[np.logical_and(np.percentile(x, pct_lwr) < x, np.percentile(x, pct_upr) > x)])


def cart2pol(x, y):
    rho = np.sqrt(x**2+y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def mod_angle(a, interval='pos'):
    if interval == 'pos':
        b = (a/(2*np.pi) % 1) * 2*np.pi
    elif interval == 'sym':
        b = (((a+np.pi)/(2*np.pi) % 1) * 2*np.pi) - np.pi
    else:
        raise ValueError
    return b


def policy(state, target):
    # Settings
    vmin, vmax = 0.0, 6.0
    wmin, wmax = -30.0, 30.0

    v_scales = np.array([1.0, 1.0, 1.0, 4.0, 4.0])
    w_scales = np.array([1.0, 80.0, 10.0, 10.0, 10.0])

    # rule_weights = np.array([0.2, 0.3, 0.2, 0.3, 0.0])
    rule_weights = np.array([0.0, 0.2, 0.2, 0.2, 0.4])
    # rule_weights = np.array([1, 0, 0, 0, 0])
    # rule_weights = np.array([0, 1, 0, 0, 0])
    # rule_weights = np.array([0, 0, 1, 0, 0])
    # rule_weights = np.array([0, 0, 0, 1, 0])
    # rule_weights = np.array([0, 0, 0, 0, 1])

    num_comps = 5

    # Extract data
    coords = state[:, 0:2]
    angles = state[:, 2]

    v_comps = np.zeros([num_comps, N])
    w_comps = np.zeros([num_comps, N])

    # Rule 0: Base rates
    v_comps[0, :] = vmax*npr.rand(N)
    w_comps[0, :] = wmax*(2*npr.rand(N) - 1)

    # Sample pairs
    pairs = []
    for i in range(M):
        pair = (npr.randint(N), npr.randint(N))
        while pair in pairs or pair[0] == pair[1]:
            pair = (npr.randint(N), npr.randint(N))
        pairs.append(pair)

    sq_dists = [np.sum(np.square(coords[i] - coords[j])) for i, j in pairs]
    dists = np.sqrt(sq_dists)
    d_angles = [angles[i] - angles[j] for i, j in pairs]

    for i in range(N):
        Mi = 0
        ks = []
        nbr_dists = []
        nbr_d_angles = []
        nbr_coords = []
        nbr_angles = []
        for j, pair in enumerate(pairs):
            match = False
            if i == pair[1]:
                k = pair[0]
                match = True
            elif i == pair[0]:
                k = pair[1]
                match = True

            if match:
                Mi += 1
                ks.append(k)
                nbr_coords.append(coords[k])
                nbr_angles.append(angles[k])

                nbr_dists.append(dists[j])
                nbr_d_angles.append(d_angles[j])

        if not ks:
            continue

        nbr_dists = np.array(nbr_dists)
        nbr_d_angles = np.array(nbr_d_angles)
        nbr_coords = np.array(nbr_coords)
        nbr_angles = np.array(nbr_angles)

        weights = 1/nbr_dists

        # Rule 1: Alignment
        delta_angles = nbr_d_angles - angles[i]
        avg_angle = np.average(delta_angles, weights=weights/np.sum(weights))
        normalized_angle = mod_angle(avg_angle, interval='sym') / (Mi*np.pi)

        w_comps[1, i] = normalized_angle
        # print(normalized_angle)

        # Rule 2: Separation
        nbr_delta_coords = coords[i] - nbr_coords
        nbr_delta_angles = mod_angle(np.arctan2(nbr_delta_coords[:, 1], nbr_delta_coords[:, 0]), interval='pos')
        nbr_relative_angles = nbr_delta_angles - nbr_angles
        nbr_relative_angles = mod_angle(nbr_relative_angles, interval='sym')
        w_comps[2, i] -= np.sum(nbr_relative_angles*weights)*(Mi**0.5)

    v_comps[1, :] = v_comps[0, :]
    v_comps[2, :] = v_comps[0, :]

    # Rule 3: Cohesion (global)
    center = np.mean(coords, axis=0)
    delta_coords = center - coords
    dist_delta_coords = la.norm(center-coords, axis=1)
    target_angles = mod_angle(np.arctan2(delta_coords[:, 1], delta_coords[:, 0]), interval='pos')
    delta_angles = target_angles - angles
    delta_angles = mod_angle(delta_angles, interval='sym')
    v_comps[3] = dist_delta_coords
    w_comps[3] = delta_angles

    # Rule 4: Target seeking
    delta_coords = target - coords
    dist_delta_coords = la.norm(target-coords, axis=1)
    target_angles = mod_angle(np.arctan2(delta_coords[:, 1], delta_coords[:, 0]), interval='pos')
    delta_angles = target_angles - angles
    delta_angles = mod_angle(delta_angles, interval='sym')
    v_comps[4] = dist_delta_coords
    w_comps[4] = delta_angles

    # Mix the component rules
    v = np.sum((rule_weights*v_scales)[:, None]*v_comps, axis=0)
    w = np.sum((rule_weights*w_scales)[:, None]*w_comps, axis=0)

    # Saturate
    v = np.clip(v, vmin, vmax)
    w = np.clip(w, wmin, wmax)
    return v, w


def f(state, target, controls=None):
    if controls is None:
        # Get control actions
        controls = policy(state, target)
    v, w = controls

    # Dynamics
    angles = state[:, 2]
    c = np.cos(angles)
    s = np.sin(angles)

    xdot = np.array([v*c, v*s, w]).T
    return xdot


def step(state, target, controls=None, h=None):
    if h is None:
        h = global_h
    state_new = state + h*f(state, target, controls)
    state_new[:, 2] = mod_angle(state_new[:, 2], interval='pos')
    return state_new


def get_target(t):
    target = np.array([target_path_x_center + target_path_radius*np.cos(target_path_xfreq*t),
                       target_path_y_center + target_path_radius*np.sin(target_path_yfreq*t)])
    return target


def physics_update(state, t, h=None):
    target = get_target(t)
    controls = policy(state, target)
    state = step(state, target, controls, h=h)
    return state, target


if __name__ == "__main__":
    # NOTE: You may need to run this script in a terminal for the animation to run properly

    # npr.seed(1)

    # Number of birds
    N = 100

    # Number of sampled pairs
    # M = 100
    M = 2*N  # must be < N (N-1) / 2

    # Number of states
    n = 3

    # Simulation time step length
    global_h = 0.05
    subsamples = 100

    # Plotting stride number
    stride = 1

    # Axis limit
    al = 10

    # Initial target coords and path settings
    target = np.array([al / 2, al / 2])
    target_path_x_center = al / 2
    target_path_y_center = al / 2
    target_path_xfreq = 0.005
    target_path_yfreq = 0.010
    target_path_radius = al / 2

    # Initial state
    coords = al*npr.rand(N, 2)
    angles = 2*np.pi*npr.rand(N, 1)
    state = np.hstack([coords, angles])
    controls = policy(state, target)

    visualization_scheme = 'raster'
    # visualization_scheme = 'quiver'
    if visualization_scheme == 'quiver':
        def init_quiver(ax, state, scale):
            X = state[:, 0]
            Y = state[:, 1]
            U = np.cos(state[:, 2])
            V = np.sin(state[:, 2])
            quiver = ax.quiver(X, Y, U, V, alpha=0.5,
                               scale=scale, scale_units='inches', headlength=8, headaxislength=6, headwidth=6)
            return quiver

        def update_quiver(quiver, state, wrap=False):
            XY = state[:, 0:2]
            if wrap:
                XY = XY % al
            U = np.cos(state[:, 2])
            V = np.sin(state[:, 2])
            quiver.set_offsets(XY)
            quiver.set_UVC(U, V)
            return quiver

        def update(gt, quiver_prey, scatter_target_prey1, scatter_target_prey2):
            global state
            global target
            global controls

            t = gt/subsamples
            if gt % subsamples == 0:
                target = get_target(t)
                controls = policy(state, target)
            state = step(state, target, controls, h=global_h/subsamples)

            quiver_prey = update_quiver(quiver_prey, state)
            scatter_target_prey1.set_offsets(target)
            scatter_target_prey2.set_offsets(target)

            return [quiver_prey, scatter_target_prey1, scatter_target_prey2]


        # Initialize plot
        plt.close('all')
        fig, ax = plt.subplots(figsize=(6, 6))

        scatter_target_prey1 = ax.scatter(target[0], target[1], s=400, c='k')
        scatter_target_prey2 = ax.scatter(target[0], target[1], s=200, c='w')

        quiver_prey = init_quiver(ax, state, scale=10)

        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

        xoff, yoff = 0.5*al, 0.5*al
        xlwr, xupr = 0-xoff, al+xoff
        ylwr, yupr = 0-yoff, al+yoff
        ax.set_xlim(xlwr, xupr)
        ax.set_ylim(ylwr, yupr)
        ax.set_aspect('equal')
        ax.grid('on')
        fig.tight_layout()

        # Animate simulation
        aniobj = ani.FuncAnimation(fig, update, fargs=[quiver_prey, scatter_target_prey1, scatter_target_prey2], interval=1, blit=True)
        plt.show()

    elif visualization_scheme == 'raster':
        def make_raster_bins(xmin=-0.5*al, xmax=al+0.5*al, ymin=-0.5*al, ymax=al+0.5*al, nx=400, ny=400):
            bins = [np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)]
            return bins

        def state2raster(state):
            x, y = state[:, 0], state[:, 1]
            raster = np.histogram2d(x, y, bins=bins)[0].astype('float')/N
            # raster[raster > 0] += vmin
            return raster

        vmin, vmax = 0, 1
        nx, ny = 500, 500
        bins = make_raster_bins(nx=nx, ny=ny)
        raster = 0*state2raster(state)

        class State:
            def __init__(self, state, raster, target):
                self.state = state
                self.raster = raster
                self.target = target
                self.controls = policy(self.state, self.target)

            def update_target(self, t):
                self.target = get_target(t)

            def update_controls(self):
                self.controls = policy(self.state, self.target)
                return self.controls

            def update(self, h):
                self.state = step(self.state, self.target, self.controls, h=h)
                self.raster = mix(state2raster(self.state), self.raster, 1-0.02)

        S = State(state, raster, target)

        def update(gt, S, subsamples=1):
            t = gt/subsamples

            if gt % subsamples == 0:
                S.update_target(gt)
                S.update_controls()
            S.update(h=global_h/subsamples)

            im.set_data(unit_saturate(S.raster, strength=2000.0).T)
            return [im]

        # Initialize the plot
        fig, ax = plt.subplots()
        cmap_big = cm.get_cmap('turbo', 512)
        cmap = ListedColormap(cmap_big(np.linspace(0.00, 0.50, 256)))
        im = plt.imshow(raster, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower', interpolation='none')
        ax.axis('off')
        fig.tight_layout()

        # Create the animation
        animation = FuncAnimation(fig, update, fargs=[S, subsamples], interval=1, blit=True)
        plt.show()
