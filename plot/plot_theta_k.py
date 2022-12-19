from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
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

omega_r = np.array([])
omega_i = np.array([])
for i in range(91):
    print(i)
    pth = '/home/kun/Documents/mathematics/beam/0.01/10va/theta_' + str(i) + '/-1_2'
    pth1 = '/home/kun/Documents/mathematics/beam/0.01/10va/theta_' + str(i) + '/1_3'
    k, omegar, omegai, polar = readfile(pth)
    k_max = k[0]
    k_min = k[-1]
    dk = k[0] - k[1]
    dn = int(0.01 / dk + 0.001)
    omegar = omegar[::dn]
    omegai = omegai[::dn]
    print(max(omegai))
    polar = polar[::dn]
    if k_max < 12:
        omegar = np.concatenate((np.zeros(int(1200-len(omegar))), omegar), axis=0)
        omegai = np.concatenate((np.zeros(int(1200-len(omegai))), omegai), axis=0)

    omegai[omegai < 0.01] = np.nan
    omega_r = np.concatenate((omega_r, omegar[::-1]), axis=0)
    omega_i = np.concatenate((omega_i, omegai[::-1]), axis=0)

omega_r = omega_r.reshape(91, -1)
omega_i = omega_i.reshape(91, -1)

omega_r1 = np.array([])
omega_i1 = np.array([])
for i in range(65):
    print(i)
    pth = '/home/kun/Documents/mathematics/beam/0.01/10va/theta/' + str(i) + '-1_2'
    k, omegar, omegai, polar = readfile(pth)
    k_max = k[0]
    k_min = k[-1]
    dk = k[0] - k[1]
    dn = int(0.01 / dk + 0.001)
    omegar = omegar[::dn]
    omegai = omegai[::dn]
    polar = polar[::dn]

    omegai[omegai < 0.01] = np.nan
    omega_r1 = np.concatenate((omega_r1, omegar[::-1]), axis=0)
    omega_i1 = np.concatenate((omega_i1, omegai[::-1]), axis=0)

omega_r1 = omega_r1.reshape(65, -1)
omega_i1 = omega_i1.reshape(65, -1)
omega_r[:65, :100] = omega_r1
omega_i[:65, :100] = omega_i1

omega_r2 = np.array([])
omega_i2 = np.array([])
for i in range(17, 91):
    print(i)
    pth = '/home/kun/Documents/mathematics/beam/0.01/10va/theta_' + str(i) + '/1_3'
    k, omegar, omegai, polar = readfile(pth)
    k_max = k[0]
    k_min = k[-1]
    dk = (k[0] - k[10]) / 10
    dn = int(0.01 / dk + 0.001)
    omegar = omegar[::dn]
    omegai = omegai[::dn]
    polar = polar[::dn]

    if k_max < 8:
        omegar = np.concatenate((np.zeros(int(800-len(omegar))), omegar), axis=0)
        omegai = np.concatenate((np.zeros(int(800-len(omegai))), omegai), axis=0)
    if len(omegai)<800:
        omegar = np.concatenate((omegar, omegar[-1]), axis=0)
        omegai = np.concatenate((omegai, omegai[-1]), axis=0)
    omegai[omegai < 0.001] = np.nan
    omega_r2 = np.concatenate((omega_r2, omegar[::-1]), axis=0)
    omega_i2 = np.concatenate((omega_i2, omegai[::-1]), axis=0)

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14}

omega_r2 = omega_r2.reshape(-1, 800)
omega_i2 = omega_i2.reshape(-1, 800)
x = np.linspace(0, 8, 800)
y = np.linspace(17, 90, 74)
X, Y = np.meshgrid(x, y)
plt.subplots_adjust(left=0.12, bottom=0.136, right=0.94, top=0.9)
plt.pcolormesh(X, Y, omega_i2, cmap='jet')
cb = plt.colorbar()
plt.ylim(0, 90)
plt.xlabel('$k \lambda_p$', font=font1)
plt.ylabel(r'$\alpha$', fontsize=18)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
plt.text(0.4, 85, 'b', fontsize=18)
cb.ax.tick_params(labelsize=14)
cb.set_label(r'$\gamma/ \Omega_p$', fontdict=font1)

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.12, bottom=0.136, right=0.94, top=0.9)
norm = LogNorm(vmin=0.01, vmax=1.8)
x = np.linspace(0, 12, 1200)
y = np.linspace(0, 90, 91)
X, Y = np.meshgrid(x, y)
contour = ax.pcolormesh(X, Y, omega_i, cmap='jet', norm=norm)
ax.contour(X, Y, omega_r, [0], linestyles='--', colors='w')
ax.set_xlabel('$k \lambda_p$', font=font1)
ax.set_xscale('log')
ax.set_xlim(0.04, 12)
ax.set_ylabel(r'$\alpha$', fontsize=18)
ax.text(0.06, 85, 'a', fontsize=18)
plt.annotate('$\omega_r = 0$', xy=(0.13, 41), xytext=(0.3, 31), arrowprops=dict(facecolor='black', shrink=10, width=1, headwidth=6), fontsize=15)

# ax.set_title(r'$\alpha = 30 \degree$', fontsize=18)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
cb = plt.colorbar(contour, label='$\gamma$')
cb.ax.tick_params(labelsize=14)
cb.set_label(r'$\gamma/ \Omega_p$', fontdict=font1)
# plt.scatter(x, omegar, c=omegai, cmap='jet', norm=norm)
# plt.colorbar(label='$\gamma$')
plt.show()
