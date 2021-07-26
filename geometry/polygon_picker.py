import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon


class Cursor(object):
    def __init__(self, ax):
        self.ax = ax
        self.lx = ax.axhline(color='tab:grey', alpha=0.5)  # the horiz line
        self.ly = ax.axvline(color='tab:grey', alpha=0.5)  # the vert line

        # text location in axes coords
        self.txt = ax.text(0.7, 0.9, '', transform=ax.transAxes)

    def mouse_move(self, event):
        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata
        # update the line positions
        self.lx.set_ydata(y)
        self.ly.set_xdata(x)

        self.txt.set_text('x=%1.2f, y=%1.2f' % (x, y))
        self.ax.figure.canvas.draw()


coords = []

plt.close('all')
fig, ax = plt.subplots()

n_grid = 10
xmin, xmax = 0, 1
ymin, ymax = 0, 1

# Plot setup
x_major_ticks = np.linspace(xmin, xmax, n_grid+1)
y_major_ticks = np.linspace(ymin, ymax, n_grid+1)
ax.set_xticks(x_major_ticks)
ax.set_yticks(y_major_ticks)
plt.grid('on')
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
# plt.axis('target')
ax.set_xticklabels(['' for _ in ax.get_xticklabels()])
ax.set_yticklabels(['' for _ in ax.get_yticklabels()])

square_points = np.array([[0.2, 0.2],
                          [0.2, 0.3],
                          [0.3, 0.3],
                          [0.3, 0.2],
                          [0.2, 0.2]])
square = Path(square_points)
square_patch = Polygon(square_points)
ax.add_artist(square_patch)

plt.show()

num_vertices = 8
cursor = Cursor(ax)
fig.canvas.mpl_connect('motion_notify_event', cursor.mouse_move)
points = plt.ginput(num_vertices, timeout=0)
points.append(points[0])
points_arr = np.array(points)

plt.plot(points_arr[:, 0], points_arr[:, 1])

path = Path(points, closed=True)

gx, gy = np.meshgrid(np.linspace(xmin, xmax, n_grid+1),
                     np.linspace(ymin, ymax, n_grid+1))
grid_points = list(zip(gx.flatten(), gy.flatten()))

inside = path.contains_points(grid_points, radius=1e-9)
ax.scatter(gx.flatten(), gy.flatten(), c=inside.astype(float), cmap="RdYlGn")

inside_square = path.contains_path(square)
