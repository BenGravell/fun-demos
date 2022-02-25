import sys
import numpy as np
import numpy.random as npr

# import matplotlib.pyplot as plt
# from matplotlib import cm
import proplot as pplt

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
nPlots = 50
nSamples = 200
data = np.zeros([nPlots, nSamples])

time_ssfilters = [StateSpaceFilter(a1=0.98, a2=0.1, n=3) for i in range(nSamples)]
space_ssfilter = StateSpaceFilter(a1=0.98, a2=0.05, n=2)

# Plot settings
curve_spacing = 0.0

# colormap_str = 'Rocket_r'
colormap_str = 'Mako_r'
left, right = 0.0, 1.0

colormap = pplt.Colormap(colormap_str, left=left, right=right)
colors = 255*np.array([colormap(t) for t in np.linspace(0, 1, nPlots)])
# colors[:, 3] = 255*np.logspace(0, -3.0, nPlots)
colors[:, 3] = 255*np.logspace(0, -1.0, nPlots)
# colors[:, 3] = 255*np.logspace(0, 0, nPlots)

curves = []
for idx in reversed(range(nPlots)):
    pen = pg.mkPen(color=colors[idx], width=1)
    curve = pg.PlotCurveItem(pen=pen)
    plot.addItem(curve)
    curve.setPos(0, idx*curve_spacing)
    curves.insert(0, curve)

plot.setYRange(-2.0, 2.0 + curve_spacing*nPlots)
plot.setXRange(0, nSamples)
plot.resize(800, 800)

lastTime = time()
fps = None


def update():
    global curves, plot, data, lastTime, fps, nPlots

    # Roll the previous data
    data = np.roll(data, 1, axis=0)

    # Step data forward using smoothed curve as input to filter
    x = 0.001*npr.randn(nSamples)
    y = np.array([space_ssfilter.step(val) for val in x])
    z = np.array([ssfilter.step(val) for val, ssfilter in zip(y, time_ssfilters)])
    # z -= np.mean(z)
    # z *= 100/np.sum(np.abs(z))
    data[0] = z

    # Set curve data
    for idx in range(nPlots):
        curves[idx].setData(data[idx])

    # FPS counter
    now = time()
    dt = now-lastTime
    lastTime = now
    if fps is None:
        fps = 1.0/dt
    else:
        s = np.clip(dt*3.0, 0, 1)
        fps = fps*(1-s) + (1.0/dt)*s
    plot.setTitle('%0.2f fps' % fps)


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)


# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
