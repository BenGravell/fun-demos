# Generative flower art using fractal Perlin noise

import os
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import proplot as pplt

from perlin import fractal


# Global settings
plt.close('all')
seed = 1
npr.seed(seed)

bkgd_color = 'black'
# bkgd_color = 'white'

quality = 'draft'
# quality = 'medium'
# quality = 'high'
# quality = 'ultra'

if quality == 'draft':
    n = 100
elif quality == 'medium':
    n = 500
elif quality == 'high':
    n = 4000
elif quality == 'ultra':
    n = 12000
else:
    raise ValueError


cmap_str_set = 'single'
# cmap_str_set = 'triple'
# cmap_str_set = 'all'

if cmap_str_set == 'single':
    cmap_strs = ['Rocket']

elif cmap_str_set == 'triple':
    cmap_strs = ['Rocket',
                 'Mako',
                 'Boreal_r']
elif cmap_str_set == 'all':
    cmap_strs = ['Rocket',
                 'Amp_r',
                 'Fire_r',
                 'Turbid_r',
                 'Speed_r',
                 'Boreal_r',
                 'Tempo_r',
                 'Haline',
                 'Mako',
                 'Ice',
                 'Glacial_r',
                 'Dense_r',
                 'Sunset_r',
                 'inferno']
else:
    raise ValueError

save = False


# Choose the number of petals
num_petals = 3

# Create angle data
ang = np.linspace(0, 2*np.pi, n)

petal_color_overlap = 0.2


def rescale(x, ymin, ymax):
    xmin, xmax = np.min(x), np.max(x)
    s = (ymax - ymin) / (xmax - xmin)
    y = ymin + s * x - s * xmin
    return y


# Match endpoints while preserving mean
def rematch(x):
    n = x.size
    original_mean = np.mean(x)
    ramp = -np.linspace(x[0], x[-1], n)
    x += ramp
    new_mean = np.mean(x)
    x += original_mean - new_mean
    return x


for cmap_str in cmap_strs:
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},
                           figsize=(5, 5),
                           facecolor=bkgd_color)

    for j in range(num_petals-1, -1, -1):
        seed += 1

        # Create radius data
        r = fractal(n, num_layers=npr.randint(3, 7), p0=npr.uniform(2.5, 8.0), seed=seed)

        for i in range(n):
            r[i] = rematch(r[i])

        rmin = 0
        rmax = (j+1)/num_petals
        r = rescale(r, rmin, rmax)

        # Make the colormap
        left = max((num_petals-j-1-petal_color_overlap)/num_petals, 0.0)
        right = min((num_petals-j+petal_color_overlap)/num_petals, 1.0)
        cmap = pplt.Colormap(cmap_str, left=left, right=right)

        # Plot each radial line
        for i in range(n):
            ax.plot(ang, r[i], color=cmap(i/(n-1)), lw=8.0/(n**0.5), alpha=min(4.0/(n**0.5), 1))

    # Set plot options
    ax.axis('off')
    fig.tight_layout()
    if save:
        filename = 'flower'+'_'+cmap_str.split('_')[0].lower()+'.png'
        path_out = os.path.join(filename)
        fig.savefig(path_out, dpi=800, facecolor=fig.get_facecolor(), edgecolor='none')
