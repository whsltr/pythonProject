import matplotlib
import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
import data_analysis.read_data as data
from scipy.fft import fft, fft2, fftfreq, fftshift
from scipy import signal

# plt.rcParams['font.sans-serif'] = ['Times New Roman']
# plt.rcParams['axes.unicode_minus'] = False
# print(matplotlib.matplotlib_fname())

path = '/home/kun/Downloads/data/data/'
# path = '/home/ck/Documents/hybrid2D_PUI/data/'
by = 'by'
bz = 'bz'
By = data.read2d_timeseries(path, by)
Bz = data.read2d_timeseries(path, bz)
T = 0.05
dx = 0.25
By = np.array(By, dtype=float)
By = By[::-1, :]
Bz = np.array(Bz, dtype=float)
Bz = Bz[::-1, :]
leng = int(len(By[:, 0]))
By = By + 1j * Bz
By1 = By[600:, :]
By2 = By[300:, :]
By3 = By
By = [By1, By2, By3]
Z = []
tf = []
xf = []
polarization = []
power = []
tf_r = []
for i in range(3):
    Zi = fft2(By[i])
    Zi = Zi / len(By[i][:, 0]) / len(By[i][0, :])
    Zi = fftshift(Zi)
    polarizationi = (np.abs(Zi[len(Zi[:, 0]) // 2::-1, ::-1]) - np.abs(Zi[len(Zi[:, 0]) // 2:, :])) / \
                    (np.abs(Zi[len(Zi[:, 0]) // 2::-1, ::-1]) + np.abs(Zi[len(Zi[:, 0]) // 2:, :]))
    print(np.amax(polarizationi))
    # polarizationi = np.log10(np.abs(Zi[len(Zi[:, 0]) // 2:, ::1]))
    poweri = np.abs(Zi[len(Zi[:, 0]) // 2::-1, ::-1]) + np.abs(Zi[len(Zi[:, 0]) // 2:, :])
    # tf = np.linspace(0., 1.0 / (2. * T), len(By[0, :]) // 2) * 2 * np.pi
    # xf = np.linspace(0., 1. / (2. * dx), len(By[:, 0]) // 2) * 2 * np.pi
    tfi = fftfreq(len(By[i][:, 0]), T) * 2 * np.pi
    xfi = fftfreq(len(By[i][0, :]), dx) * 2 * np.pi
    xfi = fftshift(xfi)
    tfi = fftshift(tfi)
    tfi_r = tfi
    tfi = tfi[len(tfi) // 2::]
    # By_r = By[:, 0]
    # f = fft(By[i])
    Z.append(Zi)
    if i == 0:
        polarizationi[poweri < 10**-3.5] = None
    elif i==1:
        polarizationi[poweri < 10**-3] = None
    else:
        polarizationi[poweri < 10**-2.5] = None
    polarization.append(polarizationi)
    power.append(poweri)
    tf.append(tfi)
    xf.append(xfi)
    tf_r.append(tfi_r)
omega = np.linspace(0, 2 * np.pi / 2 / T, len(By1) // 2)

from scipy.constants import c, mu_0, epsilon_0, k, e, m_e, proton_mass

B = 3.0e-9
n = 3.0e6
T = 4.3
v = 400
N = 2048 * 400
# charge exchange rate
gama = 3.4e-4
beta = n * T * 1.6e-19
pressureb = 1 / 2 / mu_0 * B ** 2
beta = beta / pressureb
print(beta)
omega_i = e * B / proton_mass
omega_e = e * B / m_e
omega_pi = n * e ** 2 / epsilon_0 / proton_mass
omega_pe = n * e ** 2 / epsilon_0 / m_e

func = lambda omega: np.sqrt(omega ** 2 / c ** 2 * (
        1 - omega_pi / (omega * (omega + omega_i)) - omega_pe / (omega * (omega - omega_e)))) \
                     * c / np.sqrt(omega_pi)

func1 = lambda omega: np.sqrt(omega ** 2 / c ** 2 * (
        1 - omega_pi / (omega * (omega - omega_i)) - omega_pe / (omega * (omega + omega_e)))) \
                      * c / np.sqrt(omega_pi)
omega = np.linspace(0.001, 1. * np.pi * omega_i, 10001)
omega1 = omega / omega_i

k = func(omega)
v = omega1 / k
delta_omega = ((v + 10) / v) * omega1
font1 = {'family': 'Computer Modern Roman',
         'weight': 'normal',
         'size': 16}
font2 = {'family': 'TIMES',
         'weight': 'normal',
         'size': 16}

k1 = func1(omega)
k1 = k1[k1 < 2]
omega1_shf = omega1[0:len(k1)]
v = omega1_shf / k1
delta_omega = ((v + 10) / v) * omega1_shf

norm = matplotlib.colors.Normalize(vmin=-9, vmax=-3)
text = ['(a)$0<t\Omega_p <20$', '(b)$0<t\Omega_p <50$', '(c)0<t$\Omega_p <80$']
text1 = ['(d)$0<t\Omega_p <20$', '(e)$0<\Omega_p <50$', '(f)0<$\Omega_p <80$']
fig, ax = plt.subplots(nrows=2, ncols=3, sharey=True, figsize=(12, 12))
for i in range(3):

    # ax[i].plot(-k1, -((-v + 10) / (-v)) * omega1_shf, 'w', label='shfited whister')

    # plt.plot(k1, delta_omega, 'p--', label='Alfven wave')
    # plt.plot(-k1, ((-v + 10) / -v) * omega1_shf, 'p')
    # plt.plot(k1, -omega1_shf-delta_omega, 'b')
    # plt.plot(-k1, -omega1_shf-delta_omega, 'b')

    # k = k[k<2]
    # omega = omega[0:len(k)]
    # fig = plt.figure(figsize=(8, 6))
    # plt.plot(k, omega1, 'w--', label='whister wave')
    # plt.plot(-k, omega1, 'w--')
    # ax[i].plot(k, -omega1, 'w--')
    # ax[i].plot(-k, -omega1, 'w--')
    k11 = func1(omega)
    k11 = k1[k1 < 2]
    omega11 = omega1[0:len(k1)]
    # ax[i].plot(k11, omega11, 'r--')
    # ax[i].plot(-k11, omega11, 'r--')
    # plt.plot(k1, -omega1, 'p--')
    # plt.plot(-k1, -omega1, 'p--')
    #
    # xf, tf = np.meshgrid(xf, tf)
    # Z[i] = fftshift(Z[i])
    # xlen = len(xf)
    # tlen = len(tf)
    # xf = xf[xlen / 2 - xlen / 8:xlen / 2 + xlen / 8]
    # tf = tf[tlen / 2: tlen / 2 + tlen / 32]
    # Z = Z[xlen / 2 - xlen / 8:xlen / 2 + xlen / 8, tlen / 2: tlen / 2 + tlen / 32]
    # plt.figure(figsize=(8, 6))
    h1 = ax[0, i].pcolormesh(xf[i]*np.cos(0/360*2*np.pi), tf[i], np.log10(np.abs(power[i]) ** 2), cmap='jet', norm=norm)
    ax[0, i].set_xlabel('$k_{||}\lambda_p$', font1)
    if i == 0:
        ax[0, i].set_ylabel('$\omega/ \Omega_p$', font1)
    ax[0, i].tick_params(labelsize=16)
    ax[0, i].set_xlim(-0.8, 0.8)
    ax[0, i].set_ylim(0., 2.)
    # ax[0, i].text(-0.75, 1.8, text[i], fontsize=12, color='white')
    ax[0, i].set_title(text[i], fontsize=16)
    h2 = ax[1, i].pcolormesh(xf[i] * np.cos(0 / 360 * 2 * np.pi), tf[i], polarization[i], cmap='jet', vmin=-1, vmax=1)
    ax[1, i].set_xlabel('$k_{||}\lambda_p$', font1)
    if i == 0:
        ax[1, i].set_ylabel('$\omega/ \Omega_p$', font1)
    ax[1, i].tick_params(labelsize=16)
    ax[1, i].set_xlim(-1, 1)
    ax[1, i].set_ylim(0., 2.)
    # ax[1, i].text(-0.75, 1.8, text1[i], fontsize=16, color='k')
    ax[1, i].set_title(text1[i], fontsize=16)
    # plt.legend()
    # ax[i].xlim(-1.5, 1.5)
    # ax[i].ylim(-1, 0.1)

# colorbar left blow width and hight
l = 0.92
b = 0.59
w = 0.015
h = 0.35

l1 = 0.92
b1 = 0.1
w1 = 0.015
h11 = 0.35

rect = [l, b, w, h]
rect1 = [l1, b1, w1, h11]
cbar_ax = fig.add_axes(rect)
cbar_ax1 = fig.add_axes(rect1)
cb = plt.colorbar(h1, cax=cbar_ax)
cb1 = plt.colorbar(h2, cax=cbar_ax1)
cb.ax.tick_params(labelsize=12)
cb1.ax.tick_params(labelsize=12)
# cb1 = plt.colorbar(h)
# cb1.ax.tick_params(labelsize=12)
# cb1.formatter.set_powerlimits((-2, 0))
# plt.legend()
plt.savefig(path + 'spectra.png', bbox_inches='tight')
plt.show()
