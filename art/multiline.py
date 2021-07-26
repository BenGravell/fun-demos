import sys
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from matplotlib import cm
import pyqtgraph as pg
from pyqtgraph.ptime import time
from pyqtgraph.Qt import QtGui, QtCore

from sig import ema, smooth, StateSpaceFilter

from time import sleep

# Initialize plot
app = QtGui.QApplication([])
pg.setConfigOptions(antialias=True)
plot = pg.plot()
plot.setWindowTitle('MultiLine')

# Simulation settings
nPlots = 20
nSamples = 100
data = np.zeros([nPlots, nSamples])

time_ssfilters = [StateSpaceFilter(a1=0.95, a2=0.05, n=3) for i in range(nSamples)]
space_ssfilter = StateSpaceFilter(a1=0.90, a2=0.05, n=3)

# Plot settings
curve_spacing = 0.0

# colormap_str = 'viridis'
colormap_str = 'inferno'
color_range = [0.0, 1.0]

# colormap_str = 'Blues_r'
# # colormap_str = 'Greens_r'
# color_range = [0.0, 0.7]

colormap = cm.get_cmap(colormap_str)
colors = 255*np.array([colormap(t) for t in np.linspace(color_range[1], color_range[0], nPlots)])
colors[:, 3] = 255*np.logspace(0, -0.5, nPlots)
# colors[:, 3] = 255*np.logspace(0, 0, nPlots)

curves = []
for idx in range(nPlots):
    pen = pg.mkPen(color=colors[idx], width=1)
    curve = pg.PlotCurveItem(pen=pen)
    plot.addItem(curve)
    curve.setPos(0, idx*curve_spacing)
    curves.append(curve)

plot.setYRange(-2.0, 2.0 + curve_spacing*nPlots)
plot.setXRange(0, nSamples)
plot.resize(800, 800)

lastTime = time()
fps = None
fps_tgt = 60
dt_tgt = 1/fps_tgt
fudge_factor = 0.85


def update():
    global curves, plot, data, lastTime, fps, nPlots

    # Roll the previous data
    data = np.roll(data, 1, axis=0)

    # Step data forward using smoothed curve as input to filter
    x = npr.randn(nSamples)
    y = np.array([space_ssfilter.step(val) for val in x])
    z = np.array([ssfilter.step(val) for val, ssfilter in zip(y, time_ssfilters)])
    data[0] = z

    # Set curve data
    for idx in range(nPlots):
        curves[idx].setData(data[idx])

    # FPS counter
    now = time()
    dt = now-lastTime
    if dt < dt_tgt:
        sleep(fudge_factor*(dt_tgt-dt))
    now = time()
    dt = now-lastTime
    lastTime = now
    if fps is None:
        fps = 1.0/dt
    else:
        s = np.clip(dt*3., 0, 1)
        fps = fps*(1-s) + (1.0/dt)*s
    plot.setTitle('%0.2f fps' % fps)

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)


# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
