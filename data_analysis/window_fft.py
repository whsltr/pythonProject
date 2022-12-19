import numpy as np
import matplotlib.pyplot as plt
import data_analysis.read_data as data
from scipy.fft import fft
from scipy import signal

# path = '/home/ck/Documents/hybrid2D_PUI/data/'
# path = '/media/ck/15814792801/'
path = '/home/kun/Downloads/data/data/'

t, ex, ey, ez, bx, by, bz, b_total = data.read_timeseries(path)

T = 0.01
t = np.array(t) * T
ex = np.array(ex)
ey = np.array(ey)
ez = np.array(ez)
bx = np.array(bx)
by = np.array(by)
bz = np.array(bz)
N = np.size(t)
# remove linear trend along axis form data
signal.detrend(bx)
signal.detrend(by)
signal.detrend(bz)
signal.detrend(ex)
signal.detrend(ey)
signal.detrend(ez)
# windowed fft,    f is frequencies, t is segment times,
ff, tf, Zxx = signal.stft(by, fs=1. / T, nperseg=1000)
# replace 0 elements with median value
m = np.median(Zxx[Zxx > 0])
Zxx[Zxx == 0] = m
ff = ff[0:25] * 2 * np.pi
tf = tf[:-2]
Zxx = Zxx[0:25, :-2]
fig = plt.figure()
ax = plt.pcolormesh(tf, ff, np.log(np.abs(Zxx)), cmap='jet', shading='gouraud', vmin=-14)
plt.plot([0, tf[-1]], [1, 1], color='w', linestyle='-', lw=1)
cbar = fig.colorbar(ax)
cbar.set_label(r'by')
fig = plt.figure()

plt.plot(t, by)
plt.xlabel(r't')
plt.ylabel(r'$B_y$')
# plt.show()
# plot the the change of wave amplitude as the the increase

Zxx = (abs(Zxx[13, :]) + abs(Zxx[12, :]) + abs(Zxx[14, :]) + abs(Zxx[15, :]) + abs(Zxx[11, :])) / 5
fig = plt.figure()
plt.plot(tf, Zxx)


# windowed fft,    f is frequencies, t is segment times,
ff, tf, Zxx = signal.stft(bx, fs=1. / T, nperseg=500)
# replace 0 elements with median value
m = np.median(Zxx[Zxx > 0])
Zxx[Zxx == 0] = m
ff = ff[0:25]
tf = tf[:-200]
Zxx = Zxx[0:25, :-200]
fig = plt.figure()
plt.pcolormesh(tf, ff, np.log(np.abs(Zxx)), cmap='jet', shading='gouraud', vmin=-14)
cbar = plt.colorbar()
plt.plot([0, tf[-1]], [1, 1], color='w', linestyle='-', lw=1)
cbar.set_label(r'bx')
fig = plt.figure()
plt.plot(t, bx)
plt.xlabel(r't')
plt.ylabel(r'$B_x$')

# plt.show()

# windowed fft,    f is frequencies, t is segment times,
ff, tf, Zxx = signal.stft(bz, fs=1. / T, nperseg=500)
# replace 0 elements with median value
m = np.median(Zxx[Zxx > 0])
Zxx[Zxx == 0] = m
ff = ff[0:25]
tf = tf[:-200]
Zxx = Zxx[0:25, :-200]
fig = plt.figure()
plt.pcolormesh(tf, ff, np.log(np.abs(Zxx)), cmap='jet', shading='gouraud', vmin=-14)
plt.plot([0, tf[-1]], [1, 1], color='w', linestyle='-', lw=1)
cbar = plt.colorbar()
cbar.set_label(r'bz')
fig = plt.figure()
plt.plot(t, bz)
plt.xlabel(r't')
plt.ylabel(r'$B_z$')

# windowed fft,    f is frequencies, t is segment times,
ff, tf, Zxx = signal.stft(ex, fs=1. / T, nperseg=500)
# replace 0 elements with median value
m = np.median(Zxx[Zxx > 0])
Zxx[Zxx == 0] = m
ff = ff[0:25]
tf = tf[:-200]
Zxx = Zxx[0:25, :-200]
fig = plt.figure()
plt.pcolormesh(tf, ff, np.log(np.abs(Zxx)), cmap='jet', shading='gouraud', vmax=0, vmin=-14)
plt.plot([0, tf[-1]], [1, 1], color='w', linestyle='-', lw=1)
cbar = plt.colorbar()
cbar.set_label(r'ex')
fig = plt.figure()
plt.plot(t, ex)
plt.xlabel(r't')
plt.ylabel(r'$E_x$')
# plt.show()


# windowed fft,    f is frequencies, t is segment times,
ff, tf, Zxx = signal.stft(ey, fs=1. / T, nperseg=500)
# replace 0 elements with median value
m = np.median(Zxx[Zxx > 0])
Zxx[Zxx == 0] = m
ff = ff[0:25]
tf = tf[:-200]
Zxx = Zxx[0:25, :-200]
fig = plt.figure()
plt.pcolormesh(tf, ff, np.log(np.abs(Zxx)), cmap='jet', shading='gouraud', vmin=-14)
plt.plot([0, tf[-1]], [1, 1], color='w', linestyle='-', lw=1)
cbar = plt.colorbar()
cbar.set_label(r'ey')
fig = plt.figure()
plt.plot(t, ey)
plt.xlabel(r't')
plt.ylabel(r'$E_y$')

# windowed fft,    f is frequencies, t is segment times,
ff, tf, Zxx = signal.stft(ez, fs=1. / T, nperseg=500)
# replace 0 elements with median value
m = np.median(Zxx[Zxx > 0])
Zxx[Zxx == 0] = m
ff = ff[0:25]
tf = tf[:-200]
Zxx = Zxx[0:25, :-200]
fig = plt.figure()
plt.pcolormesh(tf, ff, np.log(np.abs(Zxx)), cmap='jet', shading='gouraud', vmin=-14)
plt.plot([0, tf[-1]], [1, 1], color='w', linestyle='-', lw=1)
cbar = plt.colorbar()
cbar.set_label(r'$E_z$')
fig = plt.figure()
plt.plot(t, ez)
plt.xlabel(r't')
plt.ylabel(r'$E_z$')
plt.show()

# plot the the change of wave amplitude as the the increase
