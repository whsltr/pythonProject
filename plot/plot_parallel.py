from matplotlib.collections import LineCollection
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np


def readfile(path):
    f = open(path + '.txt')
    omegai1 = []
    omegar1 = []
    polarization1 = []
    k_para = []
    for line in f.readlines():
        line = line.split()
        if not line:
            break

        omegar1.append(float(line[1]))
        omegai1.append(float(line[2]))
        polarization1.append(float(line[3]))
        k_para.append(float(line[0]))

    f.close()
    return k_para, np.array(omegar1), np.array(omegai1), np.array(polarization1)


fig, ax = plt.subplots()
plt.subplots_adjust(left=0.148, bottom=0.124)

pth = '/home/kun/Documents/mathematics/beam/0.01/10va/theta_0/' + '1_1'
k_para, omega_r, omega_i, polarization = readfile(pth)
omega_r = omega_r
omega_i = omega_i
xmin = min(k_para)
xmax = max(k_para)
x = np.linspace(xmin, xmax, len(k_para))
y = 10 * np.cos(0/360 * 2*np.pi) * x + 1
ax.plot(x, y, 'k--')
segments = [np.column_stack([x[i:i+2], omega_r[i:i+2]]) for i in range(len(x)-1)]
ax.axis([-0.2, 0.2, -0.6, 0.4])
norm = colors.Normalize(vmin=-0.2, vmax=0.25)
lc = LineCollection(segments, cmap='jet', array=omega_i, linewidth=4, norm=norm)
line = ax.add_collection(lc)

pth = '/home/kun/Documents/mathematics/beam/0.01/10va/theta_0/' + '20va_1_2'
k_para, omega_r, omega_i, polarization = readfile(pth)
omega_r = omega_r
omega_i = omega_i
xmin = min(k_para)
xmax = max(k_para)
x = np.linspace(xmin, xmax, len(k_para))
segments = [np.column_stack([x[i:i+2], omega_r[i:i+2]]) for i in range(len(x)-1)]
# ax.axis([-0.2, 0.2, -0.4, 0.4])
# norm = colors.Normalize(vmin=-0.04, vmax=0.020)
lc = LineCollection(segments, cmap='jet', array=omega_i, linewidth=4, norm=norm)
line = ax.add_collection(lc)

pth = '/home/kun/Documents/mathematics/beam/0.01/10va/theta_0/' + '20va_-1_1'
k_para, omega_r, omega_i, polarization = readfile(pth)
omega_r = -omega_r
omega_i = omega_i
xmin = min(k_para)
xmax = max(k_para)
x = np.linspace(-12, 0, len(k_para))
y = 10 * np.cos(0/360 * 2*np.pi) * x + 1
ax.plot(x, y, 'k--')
segments = [np.column_stack([x[i:i+2], omega_r[i:i+2]]) for i in range(len(x)-1)]
# ax.axis([-0.2, 0.2, -0.4, 0.4])
# norm = colors.Normalize(vmin=-0.04, vmax=0.020)
lc = LineCollection(segments, cmap='jet', array=omega_i, linewidth=4, norm=norm)
line = ax.add_collection(lc)

pth = '/home/kun/Documents/mathematics/beam/0.01/10va/theta_0/' + '-1_2'
k_para, omega_r, omega_i, polarization = readfile(pth)
omega_r = -omega_r
omega_i = omega_i
xmin = min(k_para)
xmax = max(k_para)
x = np.linspace(-xmax, -xmin, len(k_para))
segments = [np.column_stack([x[i:i+2], omega_r[i:i+2]]) for i in range(len(x)-1)]
# ax.axis([-0.2, 0.2, -0.4, 0.4])
# norm = colors.Normalize(vmin=-0.04, vmax=0.020)
lc = LineCollection(segments, cmap='jet', array=omega_i, linewidth=4, norm=norm)
line = ax.add_collection(lc)

pth = '/home/kun/Documents/mathematics/beam/0.01/10va/theta_0/' + '1_3'
k_para, omega_r, omega_i, polarization = readfile(pth)
omega_r = omega_r[::-1]
omega_i = omega_i[::-1]
xmin = min(k_para)
xmax = max(k_para)
x = np.linspace(xmin, xmax, len(k_para))
segments = [np.column_stack([x[i:i+2], omega_r[i:i+2]]) for i in range(len(x)-1)]
# ax.axis([-0.2, 0.2, -0.8, 0.4])
# norm = colors.Normalize(vmin=-0.2, vmax=0.2)
lc = LineCollection(segments, cmap='jet', array=omega_i, linewidth=4, norm=norm)
line = ax.add_collection(lc)

pth = '/home/kun/Documents/mathematics/beam/0.01/10va/theta_0/' + '-1_3'
k_para, omega_r, omega_i, polarization = readfile(pth)
omega_r = -omega_r
omega_i = omega_i
xmin = min(k_para)
xmax = max(k_para)
x = np.linspace(-xmax, -xmin, len(k_para))
segments = [np.column_stack([x[i:i+2], omega_r[i:i+2]]) for i in range(len(x)-1)]
# ax.axis([-0.2, -0.2, 0.2, -0.4, 0.4])
# norm = colors.Normalize(vmin=-0.04, vmax=0.020)
lc = LineCollection(segments, cmap='jet', array=omega_i, linewidth=4, norm=norm)
line = ax.add_collection(lc)

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14}
ax.set_xlabel('$k \lambda_p$', font=font1)
ax.set_ylabel('$\omega / \Omega_p$', font=font1)
ax.set_title(r'$\alpha = 0 \degree$', fontsize=18)
# ax.set_yticks([-4, -2, 0, 2, 4])
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.axhline(0, color='grey', linestyle='-')
ax.axvline(0, color='grey', linestyle='-')
ax.text(0.12, 0.32, r'$\mathrm{L^+}$', fontdict=font1, bbox=dict(boxstyle="square",ec='red',fc='white'))
ax.text(-0.12, 0.32, r'$\mathrm{L^-}$', fontdict=font1, bbox=dict(boxstyle="square",ec='red',fc='white'), ha='right')
ax.text(-0.12, -0.55, r'$\mathrm{R^+}$', fontdict=font1, bbox=dict(boxstyle="square",ec='red',fc='white'), ha='right')
ax.text(0.12, -0.55, r'$\mathrm{R^-}$', fontdict=font1, bbox=dict(boxstyle="square",ec='red',fc='white'))
ax.text(-0.18, 0.33, 'a', fontsize=18)
cb = plt.colorbar(line, label='$\gamma$')
cb.ax.tick_params(labelsize=14)
cb.set_label(r'$\gamma/ \Omega_p$', fontdict=font1)
# plt.scatter(x, omegar, c=omegai, cmap='jet', norm=norm)
# plt.colorbar(label='$\gamma$')
plt.show()