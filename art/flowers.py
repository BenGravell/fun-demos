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

quality = 'draft'
# quality = 'medium'
# quality = 'final'

if quality == 'draft':
    n = 100
    cmap_strs = ['Rocket']
    save = False
elif quality == 'medium':
    n = 500
    cmap_strs = ['Rocket',
                 'Mako',
                 'Boreal_r']
    save = False
elif quality == 'final':
    n = 4000
    save = True
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


for cmap_str in cmap_strs:
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},
                           figsize=(5, 5),
                           facecolor=bkgd_color)

    # Create angle data
    unit = np.linspace(0, 1, n)
    ang = 2*np.pi*(unit - 0.25)

    # Choose the number of petals
    num_petals = npr.choice([3, 4, 5])
    for j in range(num_petals-1, -1, -1):
        seed += 1

        # Create radius data
        r = fractal(n,
                    p0=npr.uniform(2.5, 8.0),
                    num_layers=npr.randint(3, 7),
                    seed=seed)

        # Choose the fraction of n lines to display
        disp_frac = npr.uniform(0.3, 0.7)

        m = int(disp_frac*n)
        r = r[0:m]

        # Match endpoints while preserving mean
        for i in range(m):
            mean = np.mean(r[i])
            shift = -np.linspace(r[i, 0], r[i, -1], n)
            r[i] = r[i] + shift
            r[i] = r[i] - np.mean(r[i]) + mean

        # Shift & scale to fit in [tmin, tmax] (normalization)
        tmin, tmax = 0, (j+1)/num_petals
        rmin, rmax = np.min(r), np.max(r)
        r = tmin + (tmax-tmin)*(r-rmin)/(rmax-rmin)

        # Make the colormap
        petal_color_overlap = 0.25
        left_raw = max((num_petals-j-1-petal_color_overlap)/num_petals, 0.0)
        right_raw = min((num_petals-j+petal_color_overlap)/num_petals, 1.0)
        left0 = 0.0
        right0 = 1.0
        left = left_raw*(right0 - left0)
        right = right_raw*(right0 - left0)
        cmap = pplt.Colormap(cmap_str, left=left, right=right)

        # Plot each radial line
        for i in range(m):
            ax.plot(ang, r[i],
                    color=cmap(i/(m-1)),
                    lw=8.0/(m**0.5),
                    alpha=min(4.0/(m**0.5), 1))

    # Set plot options
    ax.axis('off')
    fig.tight_layout()
    if save:
        folder = 'flowers'
        filename = 'flower'+'_'+cmap_str.split('_')[0].lower()+'.png'
        path_out = os.path.join(folder, filename)
        fig.savefig(path_out,
                    dpi=800,
                    facecolor=fig.get_facecolor(),
                    edgecolor='none')
