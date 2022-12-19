import numpy as np
import matplotlib.pyplot as plt
import data_analysis.read_data as data
# import pywt as wt
from scipy.fft import fft
from scipy import signal

path = '/media/kun/Samsung_T5/data/data1d_parallel/'

t, ex, ey, ez, bx, by, bz = data.read_timeseries(path)

T = 0.02
t = np.array(t) * T
ex = np.array(ex)
ey = np.array(ey)
ez = np.array(ez)
bx = np.array(bx)
by = np.array(by)
bz = np.array(bz)
a, b = wt.dwt(by, 'db1')
y = fft(by)

# # windowed fft
# f, t, Zxx = signal.stft(by, fs=1000., nperseg=1000)
#
# plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=0.03, shading='gouraud')
# plt.colorbar()
# plt.title('STFT Magnitude')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()

# N = np.size(t)
# xf = np.linspace(0, 1. / (2.0*T), N//2)
# print(xf)
# xf = xf[:]
# print(xf)
# X, Y = np.meshgrid(t, xf)
# # plt.plot(xf, 2. / N * np.abs(b[:N//2]))
# plt.plot(np.abs(b), np.abs(a))
# # plt.ylim(0, 0.02)
# fig1 = plt.figure()
# plt.plot(t, by)
# plt.show()
# By = np.array(by)
# length = np.size(by[0])
#
# x = np.linspace(0, 1, num=length)
# plt.plot(x, By[3500])
# plt.show()

# continues wavelet transformation
coefs, frequencies = wt.cwt(by, np.arange(1,2500,25), 'mexh', sampling_period=0.02, method='fft')
# coefs = coefs[0:0:-1]
coefs = np.flip(coefs, 0)
# plt.imshow(np.abs(coefs), extent=[0, 100, 0, abs(frequencies.max())], cmap='PRGn', aspect='auto',
#            vmax=abs(coefs).max(), vmin=abs(coefs).min())
X, Y = np.meshgrid(t, frequencies)
plt.pcolor(X, Y, np.abs(coefs))
plt.colorbar()
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
