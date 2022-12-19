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


fig, ax1 = plt.subplots()

for i in range(10, 11):
    pth = '/home/kun/Documents/mathematics/shell/84' + '_' + str(i)
    k_para, omega_r, omega_i, polarization = readfile(pth)
    k_min = min(k_para)
    k_max = max(k_para)
    k = np.linspace(k_min, k_max, len(omega_r))
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 14}
    color = 'tab:blue'
    ax1.plot(k, omega_r[::-1], color=color)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.set_ylim([0, 8])
    ax1.set_xlim([0, 6])
    ax1.axhline(0, color='k', )

    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.plot(k, omega_i[::-1], color=color)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=14)
    ax2.set_ylim([-0.025, 0.1])
    plt.subplots_adjust(left=0.148, bottom=0.124, right=0.84, top=0.9)
    # plt.savefig('/home/kun/Documents/ye/111/01t' + str(n1) + '_' + str(vd1) + '.png')
    # ax2.axhline(0, 0, k_max, color='k', )
ax1.set_xlabel(r'$k\lambda_{p}$', font=font1)
ax2.set_ylabel(r'$\gamma/\Omega_{p}$', color=color, font=font1)
ax1.set_ylabel(r'$\omega/\Omega_{p}$', color=color, font=font1)
ax1.grid()
# ax2.grid()
plt.title(r'$n_1 = ' + str(0.1) + '$', font=font1)
plt.show()
