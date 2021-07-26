import numpy as np
import numpy.random as npr
import numpy.linalg as la
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
import matplotlib.animation as ani
from matplotlib import cm


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


def f_prey(state, target, avoid):
    # Settings
    sense_radius = 0.5
    vmin, vmax = 0.5, 4.0
    wmin, wmax = -3.0, 3.0

    # Extract data
    N = state.shape[0]
    coords = state[:, 0:2]
    angles = state[:, 2]
    c = np.cos(angles)
    s = np.sin(angles)

    # Compute control inputs
    v_scales = np.array([0.1, 1.0, 0.0, 10.0, 2.0])
    w_scales = np.array([0.0, 1.0, 0.3, 5.0, 2.0])

    num_comps = 5

    v_comps = np.zeros([num_comps, N])
    w_comps = np.zeros([num_comps, N])

    dr = squareform(pdist(coords))
    # da = np.array([angles[i]-angles for i in range(N)])

    for i in range(N):
        # determine neighbors
        nbr_idxs = dr[i] < sense_radius
        nbr_coords = coords[nbr_idxs]
        nbr_angles = angles[nbr_idxs]
        N_nbr = np.sum(nbr_idxs)

        # Rule 0: Base rates
        v_comps[0, i] = 1.0
        w_comps[0, i] = 0.0

        # Rule 1: Alignment
        avg_angle = np.mean(nbr_angles)
        delta_angle = avg_angle - angles[i]
        normalized_angle = mod_angle(delta_angle, interval='sym') / (N_nbr*np.pi)
        normalized_angle_error = np.abs(delta_angle)
        v_comps[1, i] = normalized_angle_error
        w_comps[1, i] = normalized_angle

        # Rule 2: Separation
        nbr_delta_coords = coords[i] - nbr_coords
        nbr_delta_angles = mod_angle(np.arctan2(nbr_delta_coords[:, 1], nbr_delta_coords[:, 0]), interval='pos')
        nbr_relative_angles = nbr_delta_angles - nbr_angles
        nbr_relative_angles = mod_angle(nbr_relative_angles, interval='sym')
        w_comps[2, i] -= np.sum(nbr_relative_angles)*(N_nbr**0.5)

    # Rule 3: Target seeking
    delta_coords = target - coords
    dist_delta_coords = la.norm(target-coords, axis=1)
    target_angles = mod_angle(np.arctan2(delta_coords[:, 1], delta_coords[:, 0]), interval='pos')
    delta_angles = target_angles - angles
    delta_angles = mod_angle(delta_angles, interval='sym')
    v_comps[3] = np.minimum(dist_delta_coords/al, 1.0)
    w_comps[3] = delta_angles

    # Rule 4: Danger avoidance
    delta_coords = avoid - coords
    dist_delta_coords = la.norm(avoid-coords, axis=1)
    target_angles = mod_angle(np.arctan2(delta_coords[:, 1], delta_coords[:, 0]), interval='pos')
    delta_angles = target_angles - angles
    delta_angles = mod_angle(delta_angles, interval='sym')
    v_comps[4] = 1/dist_delta_coords
    w_comps[4] = -delta_angles/dist_delta_coords

    # Sum the contributions of each rule
    v = np.zeros(N)
    w = np.zeros(N)
    for i in range(num_comps):
        v += v_scales[i]*v_comps[i]
        w += w_scales[i]*w_comps[i]

    # Saturate
    v = np.clip(v, vmin, vmax)
    w = np.clip(w, wmin, wmax)

    xdot_action = np.array([v*c, v*s, w]).T
    xdot = xdot_action
    return xdot


def f_pred(state, target):
    # Settings
    sense_radius = 0.5
    vmin, vmax = 0.2, 2.0
    wmin, wmax = -2.0, 2.0

    # Extract data
    N = state.shape[0]
    coords = state[:, 0:2]
    angles = state[:, 2]
    c = np.cos(angles)
    s = np.sin(angles)

    # Compute control inputs
    v_scales = np.array([0.2, 2.0, 0.0, 20.0])
    w_scales = np.array([0.0, 0.2, 1.0, 5.0])

    num_comps = 4

    v_comps = np.zeros([num_comps, N])
    w_comps = np.zeros([num_comps, N])

    dr = squareform(pdist(coords))
    # da = np.array([angles[i]-angles for i in range(N)])

    for i in range(N):
        # determine neighbors
        nbr_idxs = dr[i] < sense_radius
        nbr_coords = coords[nbr_idxs]
        nbr_angles = angles[nbr_idxs]
        N_nbr = np.sum(nbr_idxs)

        # Rule 0: Base rates
        v_comps[0, i] = 1.0
        w_comps[0, i] = 0.0

        # Rule 1: Alignment
        avg_angle = np.mean(nbr_angles)
        delta_angle = avg_angle - angles[i]
        normalized_angle = mod_angle(delta_angle, interval='sym') / (N_nbr*np.pi)
        normalized_angle_error = np.abs(delta_angle)
        v_comps[1, i] = normalized_angle_error
        w_comps[1, i] = normalized_angle

        # Rule 2: Separation
        nbr_delta_coords = coords[i] - nbr_coords
        nbr_delta_angles = mod_angle(np.arctan2(nbr_delta_coords[:, 1], nbr_delta_coords[:, 0]), interval='pos')
        nbr_relative_angles = nbr_delta_angles - nbr_angles
        nbr_relative_angles = mod_angle(nbr_relative_angles, interval='sym')
        w_comps[2, i] -= np.sum(nbr_relative_angles)*(N_nbr**0.5)

    # Rule 3: Target seeking
    delta_coords = target - coords
    dist_delta_coords = la.norm(target-coords, axis=1)
    target_angles = mod_angle(np.arctan2(delta_coords[:, 1], delta_coords[:, 0]), interval='pos')
    delta_angles = target_angles - angles
    delta_angles = mod_angle(delta_angles, interval='sym')
    v_comps[3] = np.minimum(dist_delta_coords/al, 1.0)
    w_comps[3] = delta_angles

    # Sum the contributions of each rule
    v = np.zeros(N)
    w = np.zeros(N)
    for i in range(num_comps):
        v += v_scales[i]*v_comps[i]
        w += w_scales[i]*w_comps[i]

    # Saturate
    v = np.clip(v, vmin, vmax)
    w = np.clip(w, wmin, wmax)

    xdot_action = np.array([v*c, v*s, w]).T
    xdot = xdot_action
    return xdot


# def euler(x):
#     return x + h*f(x)


# def rk4(x):
#     weights = np.array([1, 2, 2, 1])
#     N = weights.size
#     n = x.size
#     kxi = np.zeros([N+1, n])
#     for i in range(N):
#         kxi[i+1] = h*f(x + kxi[i]*weights[i])
#     return x + kxi[1:]*weights[:, None]/np.sum(weights)


def step(state_prey, state_pred, target_prey, target_pred, avoid_prey):
    state_prey_new = state_prey + h*f_prey(state_prey, target_prey, avoid_prey)
    state_pred_new = state_pred + h*f_pred(state_pred, target_pred)
    state_prey_new[:, 2] = mod_angle(state_prey_new[:, 2], interval='pos')
    state_pred_new[:, 2] = mod_angle(state_pred_new[:, 2], interval='pos')
    return state_prey_new, state_pred_new


def init_quiver(ax, state, target, norm, cmap, scale):
    coords = state[:, 0:2]
    X = state[:, 0]
    Y = state[:, 1]
    U = np.cos(state[:, 2])
    V = np.sin(state[:, 2])
    C = la.norm(target - coords, axis=1)
    quiver = ax.quiver(X, Y, U, V, C, norm=norm, cmap=cmap, alpha=0.5,
                       scale=scale, scale_units='inches', headlength=8, headaxislength=6, headwidth=6)
    return quiver


def update_quiver(quiver, state, target, wrap=False):
    coords = state[:, 0:2]
    XY = state[:, 0:2]
    if wrap:
        XY =XY % al
    U = np.cos(state[:, 2])
    V = np.sin(state[:, 2])
    C = la.norm(target-coords, axis=1)
    quiver.set_offsets(XY)
    quiver.set_UVC(U, V, C)
    return quiver


def update(t):
    global state_prey
    global state_pred
    global target_prey
    global target_pred
    global quiver_prey
    global quiver_pred
    global scatter_target_prey1
    global scatter_target_prey2
    global scatter_target_pred1
    global scatter_target_pred2
    global scatter_avoid_prey1
    global scatter_avoid_prey2

    for st in range(stride):
        target_prey = np.array([target_path_x_center + target_path_radius*np.cos(target_path_xfreq*t),
                                target_path_y_center + target_path_radius*np.sin(target_path_yfreq*t)])
        # target_pred = np.mean(state_prey[:, 0:2], axis=0)
        # avoid_prey = np.mean(state_pred[:, 0:2], axis=0)
        target_pred = np.array([clipped_mean(state_prey[:, 0]), clipped_mean(state_prey[:, 1])])
        avoid_prey = np.array([clipped_mean(state_pred[:, 0]), clipped_mean(state_pred[:, 1])])

        state_prey, state_pred = step(state_prey, state_pred, target_prey, target_pred, avoid_prey)
    quiver_prey = update_quiver(quiver_prey, state_prey, target_prey)
    quiver_pred = update_quiver(quiver_pred, state_pred, target_pred)
    scatter_target_prey1.set_offsets(target_prey)
    scatter_target_prey2.set_offsets(target_prey)
    scatter_target_pred1.set_offsets(target_pred)
    scatter_target_pred2.set_offsets(target_pred)
    scatter_avoid_prey1.set_offsets(avoid_prey)
    scatter_avoid_prey2.set_offsets(avoid_prey)

    return [quiver_prey, quiver_pred,
            scatter_target_prey1, scatter_target_prey2,
            scatter_target_pred1, scatter_target_pred2,
            scatter_avoid_prey1, scatter_avoid_prey2]


if __name__ == "__main__":
    # npr.seed(1)

    # Number of birds
    N_prey = 200
    N_pred = 20

    # Number of states
    n = 3

    # Simulation time step length
    h = 0.02

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

    # Color normalizer
    norm_prey = Normalize(0.0, al/4)
    norm_pred = Normalize(0.0, al/4)

    # Initial state
    coords_prey = al*npr.rand(N_prey, 2)
    angles_prey = 2*np.pi*npr.rand(N_prey, 1)
    state_prey = np.hstack([coords_prey, angles_prey])

    coords_pred = al*npr.rand(N_pred, 2)
    angles_pred = 2*np.pi*npr.rand(N_pred, 1)
    state_pred = np.hstack([coords_pred, angles_pred])


    # Colormap
    cmap_big_prey = cm.get_cmap('Blues', 512)
    cmap_prey = ListedColormap(cmap_big_prey(np.linspace(0.40, 0.80, 256)))
    cmap_big_pred = cm.get_cmap('Reds', 512)
    cmap_pred = ListedColormap(cmap_big_pred(np.linspace(0.40, 0.80, 256)))

    # Initialize plot
    plt.close('all')
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize=(6, 6))

    scatter_target_prey1 = ax.scatter(target[0], target[1], s=400, c='k')
    scatter_target_prey2 = ax.scatter(target[0], target[1], s=200, c='w')
    scatter_target_pred1 = ax.scatter(target[0], target[1], s=200, c=np.array(cmap_big_prey(0.9))[None, :], alpha=0.4)
    scatter_target_pred2 = ax.scatter(target[0], target[1], s=100, c=np.array(cmap_big_prey(0.1))[None, :], alpha=0.4)
    scatter_avoid_prey1 = ax.scatter(target[0], target[1], s=200, c=np.array(cmap_big_pred(0.9))[None, :], alpha=0.4)
    scatter_avoid_prey2 = ax.scatter(target[0], target[1], s=100, c=np.array(cmap_big_pred(0.1))[None, :], alpha=0.4)

    quiver_prey = init_quiver(ax, state_prey, target, norm_prey, cmap_prey, scale=10)
    quiver_pred = init_quiver(ax, state_pred, target, norm_pred, cmap_pred, scale=6)

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
    aniobj = ani.FuncAnimation(fig, update, interval=1, blit=True)
