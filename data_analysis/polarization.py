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
by = 'by'
bz = 'bz'
By = data.read2d_timeseries(path, by)
Bz = data.read2d_timeseries(path, bz)
T = 0.1
dx = 0.5
By = np.array(By, dtype=float)
By = By[::-1, :]
Bz = np.array(Bz, dtype=float)
Bz = Bz[::-1, :]
leng = int(len(By[:, 0]))
By = By + 1j * Bz
By1 = By[500:, :]
By2 = By[250:, :]
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
    poweri = np.log10(poweri)
    poweri[poweri < -3] = -3
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
         'size': 14}
font2 = {'family': 'TIMES',
         'weight': 'normal',
         'size': 14}

k1 = func1(omega)
k1 = k1[k1 < 2]
omega1_shf = omega1[0:len(k1)]
v = omega1_shf / k1
delta_omega = ((v + 10) / v) * omega1_shf

# norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
# text = ['(a)', '(b)', '(c)']
# fig, ax = plt.subplots(nrows=2, ncols=3, sharey=True, figsize=(18, 6))
# for i in range(3):
#
#     k11 = func1(omega)
#     k11 = k1[k1 < 2]
#     omega11 = omega1[0:len(k1)]
#     # Z[i] = Z[i][len(Z[i][:, 0])//2:, :]
#
#     h1 = ax[0, i].pcolormesh(xf[i] * np.cos(0 / 360 * 2 * np.pi), tf[i], polarization[i], cmap='jet', )
#     h2 = ax[1, i].pcolormesh(xf[i] * np.cos(0 / 360 * 2 * np.pi), tf_r[i], np.log10(np.abs(Z[i])), cmap='jet',
#                              vmax=-1, vmin=-5)
#     ax[0, i].set_xlabel('$k_{||}d_i^{-1}$', font1)
#     if i == 0:
#         ax[0, i].set_ylabel('$\omega\Omega_i^{-1}$', font1)
#     ax[0, i].tick_params(labelsize=12)
#     ax[0, i].set_xlim(-0.8, 0.8)
#     ax[0, i].set_ylim(-2., 2.)
#     ax[0, i].text(-0.75, 1.8, text[i], fontsize=12, color='white')
#     ax[1, i].set_xlabel('$k_{||}d_i^{-1}$', font1)
#     if i == 0:
#         ax[1, i].set_ylabel('$\omega\Omega_i^{-1}$', font1)
#     ax[1, i].tick_params(labelsize=12)
#     ax[1, i].set_xlim(-0.8, 0.8)
#     ax[1, i].set_ylim(0., 2.)
#     ax[1, i].text(-0.75, 1.8, text[i], fontsize=12, color='white')
#     # plt.legend()
#     # ax[i].xlim(-1.5, 1.5)
#     # ax[i].ylim(-1, 0.1)
#
# # colorbar left blow width and hight
# l = 0.92
# b = 0.12
# w = 0.015
# h = 1 - 2 * b
#
# rect = [l, b, w, h]
# cbar_ax = fig.add_axes(rect)
# cb = plt.colorbar(h1, cax=cbar_ax)
# cb.ax.tick_params(labelsize=12)
# # cb1 = plt.colorbar(h)
# # cb1.ax.tick_params(labelsize=12)
# # cb1.formatter.set_powerlimits((-2, 0))
# # plt.legend()
# plt.savefig(path + 'polarization.png', bbox_inches='tight')
# plt.show()

COLORMAP_FILE = '/home/kun/Pictures/bremm.png'


class ColorMap2D:
    def __init__(self, filename: str, transpose=False, reverse_x=False, reverse_y=False, xclip=None, yclip=None):
        """
        Maps two 2D array to an RGB color space based on a given reference image.
        Args:
            filename (str): reference image to read the x-y colors from
            rotate (bool): if True, transpose the reference image (swap x and y axes)
            reverse_x (bool): if True, reverse the x scale on the reference
            reverse_y (bool): if True, reverse the y scale on the reference
            xclip (tuple): clip the image to this portion on the x scale; (0,1) is the whole image
            yclip  (tuple): clip the image to this portion on the y scale; (0,1) is the whole image
        """
        self._colormap_file = filename or COLORMAP_FILE
        self._img = plt.imread(self._colormap_file)
        self._img = self._img[:, :, 0:3]
        # self._img = self._img[::-1, ::-1]
        if transpose:
            self._img = self._img.transpose()
        if reverse_x:
            self._img = self._img[::-1, :, :]
        if reverse_y:
            self._img = self._img[:, ::-1, :]
        if xclip is not None:
            imin, imax = map(lambda x: int(self._img.shape[0] * x), xclip)
            self._img = self._img[imin:imax, :, :]
        if yclip is not None:
            imin, imax = map(lambda x: int(self._img.shape[1] * x), yclip)
            self._img = self._img[:, imin:imax, :]
        if issubclass(self._img.dtype.type, np.integer):
            self._img = self._img / 255.0

        self._width = len(self._img)
        self._height = len(self._img[0])

        self._range_x = (0, 1)
        self._range_y = (0, 1)

    @staticmethod
    def _scale_to_range(u: np.ndarray, u_min: float, u_max: float) -> np.ndarray:
        return (u - u_min) / (u_max - u_min)

    def _map_to_x(self, val: np.ndarray) -> np.ndarray:
        xmin, xmax = self._range_x
        val = self._scale_to_range(val, xmin, xmax)
        rescaled = (val * (self._width - 1))
        return rescaled.astype(int)

    def _map_to_y(self, val: np.ndarray) -> np.ndarray:
        ymin, ymax = self._range_y
        val = self._scale_to_range(val, ymin, ymax)
        rescaled = (val * (self._height - 1))
        return rescaled.astype(int)

    def __call__(self, val_x, val_y):
        """
        Take val_x and val_y, and associate the RGB values
        from the reference picture to each item. val_x and val_y
        must have the same shape.
        """
        if val_x.shape != val_y.shape:
            raise ValueError(f'x and y array must have the same shape, but have {val_x.shape} and {val_y.shape}.')
        self._range_x = (np.amin(val_x), np.amax(val_x))
        self._range_y = (np.amin(val_y), np.amax(val_y))
        x_indices = self._map_to_x(val_x)
        y_indices = self._map_to_y(val_y)
        i_xy = np.stack((x_indices, y_indices), axis=-1)
        rgb = np.zeros((*val_x.shape, 3))
        for indices in np.ndindex(val_x.shape):
            img_indices = tuple(i_xy[indices])
            rgb[indices] = self._img[img_indices]
        return rgb

    def generate_cbar(self, nx=100, ny=100):
        "generate an image that can be used as a 2D colorbar"
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        return self.__call__(*np.meshgrid(x, y))


# generate data
x = y = np.linspace(-2, 2, 300)
xx, yy = np.meshgrid(x, y)
ampl = np.exp(-(xx ** 2 + yy ** 2))
phase = (xx ** 2 - yy ** 2) * 6 * np.pi
data = ampl * np.exp(1j * phase)
data_x, data_y = np.abs(data), np.angle(data)
print(np.array((1, 2, 3)))

# Here is the 2D colormap part
cmap_2d = ColorMap2D('/home/kun/Pictures/color.png', reverse_x=True)  # , xclip=(0,0.9))
rgb = cmap_2d(power[2], polarization[2])
cbar_rgb = cmap_2d.generate_cbar()

# plot the data
fig, plot_ax = plt.subplots(figsize=(8, 6))
plot_extent = (-2 * np.pi, 2 * np.pi, 0, 5 * 2 * np.pi)
plot_ax.imshow(rgb, aspect='auto', extent=plot_extent, origin='lower')
plot_ax.set_xlim(-0.8, 0.8)
plot_ax.set_ylim(0, 2)
plot_ax.set_xlabel(r'$kk_i$', font1)
plot_ax.set_ylabel(r'$\omega/\Omega_i$', font1)
plot_ax.tick_params(labelsize=12)
# plot_ax.set_title('data')

#  create a 2D colorbar and make it fancy
plt.subplots_adjust(left=0.1, right=0.65)
bar_ax = fig.add_axes([0.68, 0.15, 0.15, 0.7])
cmap_extent = (data_x.min(), data_x.max(), polarization[2].min(), polarization[2].max())
bar_ax.imshow(cbar_rgb, extent=cmap_extent, aspect='auto', origin='lower', )
bar_ax.set_xlabel(r'Power', font1)
bar_ax.set_ylabel(r'$\varepsilon$', font1)
bar_ax.yaxis.tick_right()
bar_ax.yaxis.set_label_position('right')
for item in ([bar_ax.title, bar_ax.xaxis.label, bar_ax.yaxis.label] +
             bar_ax.get_xticklabels() + bar_ax.get_yticklabels()):
    item.set_fontsize(10)
plt.show()
