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
    i = 0
    for line in f.readlines():
        line = line.split()
        if not line:
            break

        if line[0][-1] == ']':
            continue
        if float(line[0]) * 10 ** 7 % (2 * 10 ** 5) == 0:
            omegar1.append(float(line[1]))
            omegai1.append(float(line[2]))
            polarization1.append(float(line[3]))
            k_para.append(float(line[0]))
        # i += 1

    f.close()
    return k_para, np.array(omegar1), np.array(omegai1), np.array(polarization1)


fig, ax = plt.subplots(2, 3, constrained_layout=True, sharey=True)
ax = [ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 0], ax[1, 1], ax[1, 2]]
z = np.linspace(84, 89, 5)

for theta in range(84, 90):
    for i in range(1, 5):
        pth = '/home/kun/Documents/mathematics/shell-like/hydrogen/small_pui_temp/' + str(theta) + '_' + str(i)
        k_para, omega_r, omega_i, polarization = readfile(pth)
        omega_r = omega_r[::-1]
        omega_i = omega_i[::-1]
        xmin = min(k_para)
        xmax = max(k_para)
        x = np.linspace(xmin, xmax, len(k_para))
        y = 10 * np.cos(0 / 360 * 2 * np.pi) * x + 1
        # ax.plot(x, y, 'k--')
        segments = [np.column_stack([x[i:i + 2], omega_r[i:i + 2]]) for i in range(len(x) - 1)]
        ax[theta - 84].axis([0., 14, 0, 11])
        norm = colors.Normalize(vmin=-0.1, vmax=0.1)
        lc = LineCollection(segments, cmap='jet', array=omega_i, linewidth=4, norm=norm)
        line = ax[theta-84].add_collection(lc)

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 14}
    ax[theta - 84].set_xlabel('$k \lambda_p$', font=font1)
    ax[theta - 84].set_ylabel('$\omega / \Omega_p$', font=font1)
    ax[theta - 84].set_title(r'$\alpha = ' + str(theta) + '\degree$', fontsize=18)
    # ax.set_yticks([-4, -2, 0, 2, 4])
    ax[theta - 84].tick_params(axis='x', labelsize=14)
    ax[theta - 84].tick_params(axis='y', labelsize=14)
    ax[theta - 84].grid()
    # ax.axhline(0, color='grey', linestyle='-')
    # ax.axvline(0, color='grey', linestyle='-')
    # ax.text(0.12, 0.32, r'$\mathrm{L^+}$', fontdict=font1, bbox=dict(boxstyle="square",ec='red',fc='white'))
    # ax.text(-0.12, 0.32, r'$\mathrm{L^-}$', fontdict=font1, bbox=dict(boxstyle="square",ec='red',fc='white'), ha='right')
    # ax.text(-0.12, -0.55, r'$\mathrm{R^+}$', fontdict=font1, bbox=dict(boxstyle="square",ec='red',fc='white'), ha='right')
    # ax.text(0.12, -0.55, r'$\mathrm{R^-}$', fontdict=font1, bbox=dict(boxstyle="square",ec='red',fc='white'))
    # ax.text(-0.18, 0.33, 'a', fontsize=18)
cb = plt.colorbar(line, label='$\gamma$')
cb.ax.tick_params(labelsize=14)
cb.set_label(r'$\gamma/ \Omega_p$', fontdict=font1)
# plt.scatter(x, omegar, c=omegai, cmap='jet', norm=norm)
# plt.colorbar(label='$\gamma$')
plt.show()
