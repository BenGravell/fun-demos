import numpy as np
import numpy.random as npr
from scipy.stats import norm
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import cm

from sig import StateSpaceFilter


def rand_range(n, low, high):
    x = npr.rand(n)
    return low + x*(high-low)


def randn_range(n, low, high, mass_contained=0.90):
    scale = 1/norm.ppf((mass_contained+1)/2)
    x = (scale*npr.randn(n) + 1)/2
    return low + x*(high-low)


def deg2rad(x):
    return (2*np.pi/360)*x


def f(x, mag, freq, phase):
    y = mag*np.sin(deg2rad(freq*x + phase))
    return y


npr.seed(1)

n = 50

freq_lo = 0.99
freq_hi = 1.00

mags_base = np.clip(randn_range(n, 0.5, 1.5), 0.2, 2.0)
freqs_base = rand_range(n, freq_lo, freq_hi)
phases_base = randn_range(n, 0, 10)

freqs_dyna = rand_range(n, 2, 2)

x = np.linspace(0, 360*freq_hi, 1000)
y = np.array([f(x, mag, freq, phase) for mag, freq, phase in zip(mags_base, freqs_base, phases_base)]).T

phase_dyna_ssfilters = [StateSpaceFilter(a1=0.95, a2=0.1, n=2) for i in range(n)]
mag_dyna_ssfilters = [StateSpaceFilter(a1=0.98, a2=0.02, n=4) for i in range(n)]
mag_dyna_glob_ssfilter = StateSpaceFilter(a1=0.999, a2=0.001, n=6)

fig = plt.figure(figsize=(12, 4))
ax = plt.axes(xlim=(0, 360*freq_hi), ylim=(-1.2, 1.2))
plt.axis('off')

cmap = cm.get_cmap('Blues_r', 256)
lines = ax.plot(x, y, lw=2, alpha=0.1)


def animate(count):
    for line, mag_base, freq_base, phase_base, freq_dyna, phase_dyna_ssfilter, mag_dyna_ssfilter in zip(lines, mags_base, freqs_base, phases_base, freqs_dyna, phase_dyna_ssfilters, mag_dyna_ssfilters):

        mag_dyna = 0.1*mag_dyna_ssfilter.step(npr.randn()) + 0.01*mag_dyna_glob_ssfilter.step(npr.randn())
        mag = mag_base*np.tanh(mag_dyna)

        freq = freq_base + freq_dyna

        # phase_dyna = -count * freq_dyna

        phase_dyna = 0.5*phase_dyna_ssfilter.step(npr.randn()) - count*freq_dyna
        phase = phase_base + phase_dyna

        y = f(x, mag, freq, phase)
        line.set_data(x, y)

        color = cmap(np.clip(np.abs(mag), 0, 1))
        line.set_color(color)
    return lines


anim = animation.FuncAnimation(fig, animate, interval=1000/60, blit=True)
plt.show()
