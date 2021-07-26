import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt


class StateSpaceFilter:
    def __init__(self, a1, a2, n, state=None):
        # Construct state-space dynamics matrices
        A = a1*np.eye(n) + a2*np.diag(np.ones(n-1), -1)
        B = np.zeros(n)
        B[0] = 1.0
        C = np.zeros(n)
        C[-1] = 1.0
        D = np.array(0.0)

        # Initialize attributes
        self.n = n
        self.A, self.B, self.C, self.D = A, B, C, D
        self.state = state if state is not None else np.zeros(n)
        self.observation = 0.0

    def step(self, action):
        self.state = np.dot(self.A, self.state) + np.dot(self.B, action)
        self.observation = np.dot(self.C, self.state) + np.dot(self.D, action)
        return self.observation


def smooth(x, window_len=11, window='hanning', truncate='fit'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')

    if truncate is None:
        pass
    elif truncate == 'start':
        y = y[y.size-x.size:]
    elif truncate == 'end':
        y = y[0:x.size]
    elif truncate == 'fit':
        y = np.interp(np.linspace(0, 1, x.size), np.linspace(0, 1, y.size), y)

    return y


def ema(x, a, y0=None):
    n = len(x)
    y = np.zeros(n)
    if y0 is None:
        y[0] = x[0]
    else:
        y[0] = y0
    for i in range(n-1):
        y[i+1] = a*x[i+1] + (1-a)*y[i]
    return y


if __name__ == "__main__":
    T = 1000
    u = 0.1*npr.randn(T)
    # my_filter = StateSpaceFilter(a1=0.95, a2=0.1, n=4)
    my_filter = StateSpaceFilter(a1=0.99, a2=0.01, n=2)
    y = np.array([my_filter.step(u[t]) for t in range(T)])
    plt.plot(y)
