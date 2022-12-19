import numpy as np
import matplotlib.pyplot as plt
import data_analysis.read_data as data
from scipy.fft import fft
from scipy import signal

# path = '/media/ck/Samsung_T5/data/'
path = '/home/kun/Downloads/data/data/'
t, ex, ey, ez, bx, by, bz, b_total = data.read_timeseries(path, 0)

T = 0.25
t = np.array(t) * T
ex = np.array(ex)
ey = np.array(ey)
ez = np.array(ez)
bx = np.array(bx)
by = np.array(by)
bz = np.array(bz)
N = np.size(t)
y = fft(by)

x = np.linspace(0., 2 * np.pi / (2 * T), len(t) // 2)
plt.plot(x, np.log((2. / N * np.abs(y[0:len(t) // 2])) ** 2))
plt.show()
