# import numpy as np
# import matplotlib.pyplot as plt
# import data_analysis.read_data as data
# from scipy.fft import fft, fftfreq, fftshift
# from scipy import signal
#
# path = '/media/ck/Samsung_T5/data/0.0024/'
#
# By = data.read2d_timeseries(path)
# T = 0.2
# dx = 0.5
# By = np.array(By, dtype=float)
# # By = By[:2000, :]
# # Z = fft2(By)
# # Z = Z / len(By[:, 0]) / len(By[0, :])
# # tf = np.linspace(0., 1.0 / (2. * T), len(By[0, :]) // 2) * 2 * np.pi
# # xf = np.linspace(0., 1. / (2. * dx), len(By[:, 0]) // 2) * 2 * np.pi
# tf = fftfreq(len(By[:, 0]), T) * 2 * np.pi
# xf = fftfreq(len(By[0, :]), dx) * 2 * np.pi
# xf = fftshift(xf)
# tf = fftshift(tf)
# By1 = By[:, 0]
# f = fft(By1)
# omega = np.linspace(0, 2 * np.pi / 2 / T, len(By1) // 2)
# t = np.arange(0,len(By1)*0.2, 0.2)
#
# plt.figure()
# plt.plot(t, By1)
# plt.ylabel('$B_y$')
# plt.xlabel('t')
#
# plt.figure()
# plt.plot(omega, (2. / len(By1) * np.abs(f[0:len(By1) // 2])) ** 2)
# plt.xlabel('$\omega\Omega^{-1}$')
# plt.ylabel('power')
#
# # continue wavelet transform
# plt.figure()
# widths = np.linspace(0.1, 1, 20)
# cwtmatr = signal.cwt(By1, signal.ricker, widths)
# plt.imshow(cwtmatr, extent=[0, len(By1)*0.2, 0, 1], camp='image.cmap', aspect='auto', vmax=abs(cwtmatr).max())
#
# plt.show()

# import pywt
# import numpy as np
# import matplotlib.pyplot as plt
# x = np.arange(512)
# y = np.sin(2*np.pi*x/32)
# coef, freqs=pywt.cwt(y,np.arange(1,129),'gaus1')
# plt.matshow(coef) # doctest: +SKIP
# plt.show() # doctest: +SKIP

# import pywt
# import numpy as np
# import matplotlib.pyplot as plt
# t = np.linspace(-1, 1, 200, endpoint=False)
# sig  = np.cos(2 * np.pi * 7 * t)
# widths = np.arange(1, 100)
# cwtmatr, freqs = pywt.cwt(sig, widths, 'morl2')
# freqs = freqs/0.01
# cwtmatr = cwtmatr ** 2
#
# plt.pcolormesh(t, freqs, cwtmatr, cmap='jet')
# plt.colorbar()
# # plt.imshow(cwtmatr, extent=[-1, 1, 1, 60], cmap='PRGn', aspect='auto',
# #            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())  # doctest: +SKIP
# plt.show() # doctest: +SKIP

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import data_analysis.read_data as data

path = '/home/kun/Downloads/data/data/'
t, ex, ey, ez, bx, by, bz, b_total = data.read_timeseries(path, 0)
Bx = np.array(bz)
dt = 0.1
fs = 1 / dt
w = 48
# Bx = Bx[400:1000]
sig = np.array(Bx)
freq = np.linspace(0, fs / 2, 200)
widths = w * fs / (2 * freq * np.pi)
t = np.linspace(0, len(Bx) * dt, len(Bx))
# t, dt = np.linspace(0, 1, 201, retstep=True)
# fs = 1/dt
# w = 6.
# sig = np.cos(2*np.pi*(50 + 10*t)*t) + np.sin(40*np.pi*t)
# freq = np.linspace(1, fs/2, 100)
# widths = w*fs / (2*freq*np.pi)
cwtm = signal.cwt(sig, signal.morlet2, widths, w=w)
for i in range(1, 8):
    path = '/home/kun/Downloads/data/data/'
    t, ex, ey, ez, bx, by, bz, b_total = data.read_timeseries(path, i)
    Bx = np.array(bz)
    # dt = 0.1
    # fs = 1/dt
    # w = 48
    # Bx = Bx[400:1000]
    sig = np.array(Bx)
    freq = np.linspace(0, fs / 2, 200)
    widths = w * fs / (2 * freq * np.pi)
    t = np.linspace(0, len(Bx) * dt, len(Bx))
    # t, dt = np.linspace(0, 1, 201, retstep=True)
    # fs = 1/dt
    # w = 6.
    # sig = np.cos(2*np.pi*(50 + 10*t)*t) + np.sin(40*np.pi*t)
    # freq = np.linspace(1, fs/2, 100)
    # widths = w*fs / (2*freq*np.pi)
    cwtm1 = signal.cwt(sig, signal.morlet2, widths, w=w)
    cwtm += cwtm1

cwtm = cwtm / 8

plt.pcolormesh(t, freq * np.pi * 2, np.log(np.abs(cwtm)), cmap='jet', shading='gouraud', vmin=-7, )
plt.xlabel('t(s)')
plt.ylabel(r'$\omega$')
plt.colorbar()
fig = plt.figure()
plt.plot(t, sig)
plt.xlabel("t(s)")
plt.ylabel('$B_x(nt)$')

ff, tf, Zxx = signal.stft(Bx, fs=1. / dt, nperseg=800)
m = np.median(Zxx[Zxx > 0])
Zxx[Zxx == 0] = m
fig = plt.figure()
ax = plt.pcolormesh(tf, ff * np.pi * 2, np.log(np.abs(Zxx)), cmap='jet', shading='gouraud', vmin=-8)
plt.xlabel('t(s)')
plt.ylabel('$\omega$')
plt.colorbar()
plt.show()
