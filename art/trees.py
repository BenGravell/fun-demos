import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import collections as mc
from matplotlib.colors import ListedColormap
import proplot as proplot

from math import sin, cos


def rand_range(val_range):
    low, high = val_range
    return low + (high-low)*npr.rand()


def sind(x):
    return sin(x*2*np.pi/360)


def cosd(x):
    return cos(x*2*np.pi/360)


def custom_colormap(cmap_str=None):
    if cmap_str is None:
        cmap_str = 'grey'
    # Single colors
    if cmap_str in ['grey', 'red', 'orange', 'green', 'blue', 'purple']:
        colors = cm.get_cmap(cmap_str.capitalize()+'s_r', 512)
        cmap = ListedColormap(colors(np.linspace(0.0, 0.9, 256)))
    # Warm colormaps
    elif cmap_str == 'fire':
        hots = cm.get_cmap('afmhot', 512)
        cmap = ListedColormap(hots(np.linspace(0.0, 0.7, 256)))
    elif cmap_str == 'inferno':
        hots = cm.get_cmap('inferno', 512)
        cmap = ListedColormap(hots(np.linspace(0.05, 0.95, 256)))
    elif cmap_str == 'sunset':
        cmap = proplot.Colormap('Sunset_r', right=0.9)
    elif cmap_str == 'passion':
        purplereds = cm.get_cmap('PuRd_r', 512)
        darks = purplereds(np.linspace(0.00, 0.35, 144))
        lights = purplereds(np.linspace(0.50, 0.80, 112))
        cmap = ListedColormap(np.vstack([darks, lights]))
    elif cmap_str == 'twilight':
        purples = cm.get_cmap('twilight', 1024)
        cmap = ListedColormap(purples(np.linspace(0.6, 0.9, 256)))
    # Cool colormaps
    elif cmap_str == 'viridis':
        cmap = proplot.Colormap('viridis', left=0.10, right=0.95)
    elif cmap_str == 'cividis':
        cmap = proplot.Colormap('cividis', left=0.05, right=0.95)
    elif cmap_str == 'glacial':
        cmap = proplot.Colormap('Glacial_r', right=0.9)
    elif cmap_str == 'mako':
        cmap = proplot.Colormap('Mako', right=0.9)
    elif cmap_str == 'marine':
        cmap = proplot.Colormap('Marine_r', right=0.9)
    elif cmap_str == 'boreal':
        cmap = proplot.Colormap('Boreal_r', right=0.9)
    # Natural colormaps
    elif cmap_str == 'natural_flat':
        terrain_r = cm.get_cmap('terrain_r', 512)
        browns = terrain_r(np.linspace(0.25, 0.45, 192))
        greens = terrain_r(np.linspace(0.55, 0.70, 64))
        cmap = ListedColormap(np.vstack([browns, greens]))
    elif cmap_str == 'natural_real':
        brown = proplot.Colormap('Browns3_r', left=0.0, right=0.6)
        green = proplot.Colormap('Greens3', left=0.2, right=0.7)
        cmap = proplot.Colormap(brown, green, ratios=(10, 8))
    else:
        raise ValueError("Invalid custom colormap string '%s'" % cmap_str)
    return cmap


class TreeOptions:
    def __init__(self, depth=6, length_scale=1.0, width_scale=5.0, angle_scale=1.0, alpha_scale=1.0,
                 length_factor=0.8, width_factor=0.75, angle_factor=1.2, alpha_factor=0.9,
                 length_range=None, width_range=None, angle_range=None,
                 num_children_choices=None, num_children_weights=None,
                 bx=0.0, by=0.0, base_angle=90.0, cmap_str=None):
        # Tree depth
        self.depth = depth
        # Scales
        self.length_scale = length_scale
        self.width_scale = width_scale
        self.angle_scale = angle_scale
        self.alpha_scale = alpha_scale
        # Factors
        self.length_factor = length_factor
        self.width_factor = width_factor
        self.angle_factor = angle_factor
        self.alpha_factor = alpha_factor
        # Ranges
        if length_range is None:
            self.length_range = [0.3, 1.0]
        else:
            self.length_range = length_range
        if width_range is None:
            self.width_range = [2, 3]
        else:
            self.width_range = width_range
        if angle_range is None:
            self.angle_range = [-15, 15]
        else:
            self.angle_range = angle_range
        # Children
        if num_children_choices is None:
            self.num_children_choices = [2, 3, 4]
        else:
            self.num_children_choices = num_children_choices
        if num_children_weights is None:
            self.num_children_weights = [0.6, 0.25, 0.15]
        else:
            self.num_children_weights = num_children_weights
        # Base
        self.bx = bx
        self.by = by
        self.base_angle = base_angle
        # Color
        self.cmap = custom_colormap(cmap_str)


class Branch:
    # Initialize id counter
    current_sid = 0

    def __init__(self, opt, parent=None, length=None, width=None, angle=None):
        self.sid = Branch.current_sid
        Branch.current_sid += 1
        self.level = 0 if parent is None else parent.level + 1
        self.children = None
        self.opt = opt
        if length is None:
            self.length = self.genval(self.opt.length_scale, self.opt.length_range, self.opt.length_factor) if length is None else length
        else:
            self.length = length
        if width is None:
            self.width = self.genval(self.opt.width_scale, self.opt.width_range, self.opt.width_factor)
        else:
            self.width = width
        if angle is None:
            self.angle = self.genval(self.opt.angle_scale, self.opt.angle_range, self.opt.angle_factor)
        else:
            self.angle = angle

    def genval(self, scale, val_range, factor):
        return scale * rand_range(val_range) * (factor**self.level)

    def grow(self):
        if self.children is None:
            num_children = npr.choice(self.opt.num_children_choices, p=self.opt.num_children_weights)
            self.children = [Branch(opt=self.opt, parent=self) for _ in range(num_children)]
        else:
            for child in self.children:
                child.grow()

    def print(self):
        if self.level > 0:
            print('|  '*(self.level-1), end='')
            print('|---', end='')
        print('%03d' % self.sid)
        if self.children is not None:
            for child in self.children:
                child.print()

    def get_draw_data(self, draw_data=None, bx=0.0, by=0.0, base_angle=90.0, length_scale=1.0):
        if draw_data is None:
            draw_data = [[], [], []]
        segs, widths, cvals = draw_data
        abs_angle = base_angle + self.angle
        dx = self.length*cosd(abs_angle)
        dy = self.length*sind(abs_angle)
        dx_draw = length_scale*dx
        dy_draw = length_scale*dy
        nx = bx+dx
        ny = by+dy
        nx_draw = bx+dx_draw
        ny_draw = by+dy_draw
        seg = [(bx, by), (nx_draw, ny_draw)]
        segs.append(seg)
        widths.append(self.width)
        color = np.array(self.opt.cmap(self.level/self.opt.depth))
        alpha = self.opt.alpha_scale * (self.opt.alpha_factor**self.level)
        color[-1] = alpha
        cvals.append(color)
        draw_data = segs, widths, cvals
        if self.children is not None:
            for child in self.children:
                child.get_draw_data(draw_data, nx, ny, abs_angle, length_scale)
        return segs, widths, cvals

    def draw(self, ax, length_scale=1.0, width_scale=1.0, alpha_scale=1.0):
        segs, widths, cvals = self.get_draw_data(None, self.opt.bx, self.opt.by, self.opt.base_angle, length_scale)
        widths = np.array(widths)
        cvals = np.array(cvals)
        widths *= width_scale
        cvals[:, -1] *= alpha_scale
        lc = mc.LineCollection(segs, colors=cvals, linewidths=widths)
        ax.add_collection(lc)
        return lc

    def draw_layered(self, ax):
        length_scales = [0.9, 1.0, 1.1]
        width_scales = [1.2, 1.0, 0.8]
        alpha_scales = [0.15, 0.25, 0.35]
        lc_list = []
        for length_scale, width_scale, alpha_scale in zip(length_scales, width_scales, alpha_scales):
            lc = self.draw(ax, length_scale, width_scale, alpha_scale)
            lc_list.append(lc)
        return lc_list


def create_tree(bx=0.0, by=0.0, base_angle=90.0, cmap_str=None):
    opt = TreeOptions(bx=bx, by=by, base_angle=base_angle, cmap_str=cmap_str)
    tree = Branch(opt, length=rand_range([1.0, 1.2]), angle=rand_range([-5, 5]))
    for _ in range(opt.depth):
        tree.grow()
    return tree


if __name__ == "__main__":
    npr.seed(4)

    # cmap_str_list = ['grey', 'red', 'orange', 'green', 'blue', 'purple']
    # cmap_str_list = ['fire', 'inferno', 'sunset', 'passion', 'twilight']
    # cmap_str_list = ['viridis', 'cividis', 'glacial', 'mako', 'marine', 'boreal']
    # cmap_str_list = ['natural_flat', 'natural_real']
    cmap_str_list = ['natural_real', 'fire', 'marine', 'boreal', 'sunset',
                     'glacial', 'inferno', 'mako', 'cividis', 'passion']

    # pattern = 'horizontal'
    pattern = 'radial'

    plt.close('all')

    if pattern == 'horizontal':
        num_trees = 6
        fig, ax = plt.subplots(figsize=(10, 6))
    elif pattern == 'radial':
        num_trees = 10
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        raise ValueError

    for i in range(num_trees):
        if pattern == 'horizontal':
            tree_spacing = 1.5
            bx = tree_spacing*i
            by = 0.0
            base_angle = 90.0
        elif pattern == 'radial':
            base_angle = -i*(360/num_trees) + 90
            r = 0.5
            bx = r*cosd(base_angle)
            by = r*sind(base_angle)
        else:
            raise ValueError
        cmap_str = cmap_str_list[i % len(cmap_str_list)]
        tree = create_tree(bx, by, base_angle, cmap_str)
        # tree.print()
        tree.draw(ax, length_scale=1.1, alpha_scale=0.8)
        # tree.draw_layered(ax)

    # Add a circle to cover up radial pattern center
    if pattern == 'radial':
        circle = plt.Circle((0, 0), 3.0*r, color=[0.1, 0.1, 0.1], zorder=100)
        ax.add_patch(circle)
        num_ellipses = num_trees*4
        for i in range(num_ellipses):
            phase = (i/num_ellipses)*2*np.pi
            # print(phase)
            t = np.linspace(0, 2*np.pi, 1000)
            xb = 2.99*r*np.cos(t)
            yb = 0.5*r*np.sin(t)
            x = np.cos(phase)*xb - np.sin(phase)*yb
            y = np.sin(phase)*xb + np.cos(phase)*yb
            xy = np.vstack([x, y]).T
            poly = plt.Polygon(xy, color='w', zorder=1100+i, alpha=0.05)
            ax.add_patch(poly)
        # plt.text(0, 0, 'Arbor', color='w',
        #          horizontalalignment='center', verticalalignment='center', fontfamily='Jost*',
        #          fontweight='light', fontsize=60,  zorder=200)

    ax.axis('equal')
    ax.axis('off')
    ax.autoscale()
    fig.tight_layout()
    plt.show()
