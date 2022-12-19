import numpy as np
import matplotlib.pyplot as plt
import math

# from sympy.solvers import solve
# from scipy.optimize import fsolve
#
# from scipy.constants import c, mu_0, epsilon_0, k, e, m_e, proton_mass
#
# B = 3.0e-9
# n = 3.0e6
# T = 4.3
# v = 400
# # N = 2048 * 400
# # # charge exchange rate
# # gama = 3.4e-4
# # beta = n * T * 1.6e-19
# # pressureb = 1 / 2 / mu_0 * B ** 2
# # beta = beta / pressureb
# omega_i = e * B / proton_mass
# omega_e = e * B / m_e
# omega_pi = n * e ** 2 / epsilon_0 / proton_mass
# omega_pe = n * e ** 2 / epsilon_0 / m_e
#
# func = lambda omega: np.sqrt(omega ** 2 / c ** 2 * (
#         1 - omega_pi / (omega * (omega + omega_i)) - omega_pe / (omega * (omega - omega_e)))) \
#                      * c / np.sqrt(omega_pi)
#
# func1 = lambda omega: np.sqrt(omega ** 2 / c ** 2 * (
#         1 - omega_pi / (omega * (omega - omega_i)) - omega_pe / (omega * (omega + omega_e)))) \
#                      * c / np.sqrt(omega_pi)
# omega = np.linspace(0.001, 0.4 * np.pi * omega_i, 10001)
# omega1 = omega / omega_i
#
# k = func(omega)
# # k = k[k<2]
# # omega = omega[0:len(k)]
# fig = plt.figure()
# plt.plot(k, omega1, 'r--')
# plt.plot(-k, omega1, 'r--')
# plt.plot(k, -omega1, 'r--')
# plt.plot(-k, -omega1, 'r--')
# k1 = func1(omega)
# k1 = k1[k1<2]
# omega1 = omega1[0:len(k1)]
# plt.plot(k1, omega1, 'r--')
# plt.plot(-k1, omega1, 'r--')
# plt.plot(k1, -omega1, 'r--')
# plt.plot(-k1, -omega1, 'r--')
#
# from scipy.fft import fft
#
# # Number of sample points
# N = 600
# # sample spacing
# T = 1 / 800
# x = np.linspace(0.0, N * T, N)
# y = np.sin(50 * 2. * np.pi * x) + 0.5 * np.sin(80 * 2 * np.pi * x)
# yf = fft(y)
# xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
# import matplotlib.pyplot as plt
#
# plt.figure()
# plt.plot(xf, (2.0 / N * np.abs(yf[0:N // 2])) ** 2)
# plt.grid()
# plt.figure()
# plt.plot(x, y)
# plt.show()

a = np.arange(32)
a = np.reshape(a, (4, 2, 4), )
# a.shape=(2,4)
print(a)
# a.shape=(2,2,2)
# a = np.reshape(a,(4,4),order='C')
b = np.transpose(a, (2, 1, 0)).reshape(2, -1)
c = np.transpose(a, (0, 1, 2)).reshape(-1, 4)
print(c)
# c = np.transpose(c, (1,0)).reshape(4,-1)
# print(a.shape)
# a.shape=(2,4)
c = np.concatenate((c[:4, :], c[4:, :]), axis=1)
print(c)
print(c[len(c[:, 0]) // 2:, :])
print(c[0:len(c[:, 0]) // 2, :])
print(c[2:0:-1, :])
print(c[len(c[:, 0]) // 2:, :] - c[0:len(c[:, 0]) // 2, :])
print(0.133 * np.cos(20 / 180 * 2 * np.pi))

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
        self._img = self._img[::-1, ::-1]
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
print(np.array((1,2,3)))

# Here is the 2D colormap part
cmap_2d = ColorMap2D('/home/kun/Pictures/color.png', reverse_x=True)  # , xclip=(0,0.9))
rgb = cmap_2d(data_x, data_y)
cbar_rgb = cmap_2d.generate_cbar()

# plot the data
fig, plot_ax = plt.subplots(figsize=(8, 6))
plot_extent = (x.min(), x.max(), y.min(), y.max())
plot_ax.imshow(rgb, aspect='auto', extent=plot_extent, origin='lower')
plot_ax.set_xlabel('x')
plot_ax.set_ylabel('y')
plot_ax.set_title('data')

#  create a 2D colorbar and make it fancy
plt.subplots_adjust(left=0.1, right=0.65)
bar_ax = fig.add_axes([0.68, 0.15, 0.15, 0.3])
cmap_extent = (data_x.min(), data_x.max(), data_y.min(), data_y.max())
bar_ax.imshow(cbar_rgb, extent=cmap_extent, aspect='auto', origin='lower', )
bar_ax.set_xlabel('amplitude')
bar_ax.set_ylabel('phase')
bar_ax.yaxis.tick_right()
bar_ax.yaxis.set_label_position('right')
for item in ([bar_ax.title, bar_ax.xaxis.label, bar_ax.yaxis.label] +
             bar_ax.get_xticklabels() + bar_ax.get_yticklabels()):
    item.set_fontsize(7)
plt.show()

# from matplotlib import colors
# import numpy as np
#
#
# class colorbar2d:
#     def __init__(self, list1, list2, minColor=None, maxColor=None, maxv=1, s=1, step=0.05):
#
#         self.list1_max, self.list1_min = max(list1), min(list1)
#         self.list2_max, self.list2_min = max(list2), min(list2)
#         self.maxv = maxv
#         self.maxs = s
#
#         # 将所选特征0-1化
#         if minColor == None and maxColor == None:
#             self.h = (list1 - self.list1_min) / (self.list1_max - self.list1_min)
#             self.limit_color = False
#         else:
#             color_dic = {'red': 0, 'orange': 18, 'yellow': 30, 'green': 56,
#                          'cyan': 88, 'blue': 112, 'purple': 140, 'r': 0,
#                          'o': 18, 'y': 30, 'g': 56, 'c': 88,
#                          'b': 112, 'm': 140}
#             self.mincolor = color_dic.get(minColor) / 180.0
#             self.maxcolor = color_dic.get(maxColor) / 180.0
#             self.h = (list1 - self.list1_min) / (self.list1_max - self.list1_min) * (
#                         self.maxcolor - self.mincolor) + self.mincolor
#             self.limit_color = True
#
#         self.v = (list2 - self.list2_min) / (self.list2_max - self.list2_min) * self.maxv + (1 - self.maxv)
#         self.s = [self.maxs for i in range(len(list1))]
#
#         self.hsv = np.zeros(shape=(len(list1), 3))
#         self.hsv[:, 0] = self.h
#         self.hsv[:, 1] = self.s
#         self.hsv[:, 2] = self.v
#
#     def rgb(self):
#         self.rgb = colors.hsv_to_rgb(self.hsv)
#         return self.rgb
#
#     def hsv(self):
#         return self.hsv
#
#     def colorbar(self):
#         h = 0.05
#         xx, yy = np.meshgrid(np.arange(self.list1_min, self.list1_max, h),
#                              np.arange(self.list2_min, self.list2_max, h))
#         xx, yy = xx.ravel(), yy.ravel()
#         if self.limit_color == False:
#             colorbar_h = (xx - self.list1_min) / (self.list1_max - self.list1_min)
#             colorbar_v = (yy - self.list2_min) / (self.list2_max - self.list2_min)
#         else:
#             colorbar_h = (xx - self.list1_min) / (self.list1_max - self.list1_min) * (
#                         self.maxcolor - self.mincolor) + self.mincolor
#
#         colorbar_v = (yy - self.list2_min) / (self.list2_max - self.list2_min) * self.maxv + (1 - self.maxv)
#         colorbar_s = [self.maxs for i in range(len(xx))]
#
#         hsv = np.zeros(shape=(len(xx), 3))
#         hsv[:, 0] = colorbar_h
#         hsv[:, 1] = colorbar_s
#         hsv[:, 2] = colorbar_v
#         rgb = colors.hsv_to_rgb(hsv)
#         return xx, yy, rgb
