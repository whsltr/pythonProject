import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D

pth = '/media/kun/Samsung_T5/Mathematica/test/0.01/output_ptp' + '80.txt'
f = open(pth)
omegai10 = []
omegar10 = []
polarization10 = []
ny = 800
for line in f.readlines():
    line = line.split()
    if not line:
        break
    if line[0] == 'Indeterminate':
        line[0] = 0
    if line[1] == 'Indeterminate':
        line[1] = 0
    if line[2] == 'Indeterminate':
        line[2] = 0
    omegar10.append(float(line[0]))
    omegai10.append(float(line[1]))
    polarization10.append(float(line[2]))

f.close()
omegai10 = np.array(omegai10).reshape(399, -1)
omegar10 = np.array(omegar10).reshape(399, -1)
polarization10 = np.array(polarization10).reshape(399, -1)

omegai10 = omegai10[:, ::-1]
omegar10 = omegar10[:, ::-1]
polarization10 = polarization10[:, ::-1]

# omegai = np.concatenate((omegai, omegai1), axis=0)
# omegai = omegai[:, ::-12]

pth = '/media/kun/Samsung_T5/Mathematica/test/0.01/output_ptp' + '81.txt'
f = open(pth)
omegai11 = []
omegar11 = []
polarization11 = []
for line in f.readlines():
    line = line.split()
    if not line:
        break
    if line[0] == 'Indeterminate':
        line[0] = 0
    if line[1] == 'Indeterminate':
        line[1] = 0
    if line[2] == 'Indeterminate':
        line[2] = 0
    omegar11.append(float(line[0]))
    omegai11.append(float(line[1]))
    polarization11.append(float(line[2]))

f.close()
omegai11 = np.array(omegai11).reshape(399, -1)
omegar11 = np.array(omegar11).reshape(399, -1)
polarization11 = np.array(polarization11).reshape(399, -1)

omegai1 = np.concatenate((omegai10, omegai11), axis=1)
omegar1 = np.concatenate((omegar10, omegar11), axis=1)
polarization1 = np.concatenate((polarization10, polarization11), axis=1)

pth = '/media/kun/Samsung_T5/Mathematica/test/0.01/output_ptp' + '810.txt'
f = open(pth)
omegai10 = []
omegar10 = []
polarization10 = []
for line in f.readlines():
    line = line.split()
    if not line:
        break
    if line[0] == 'Indeterminate':
        line[0] = 0
    if line[1] == 'Indeterminate':
        line[1] = 0
    if line[2] == 'Indeterminate':
        line[2] = 0
    omegar10.append(float(line[0]))
    omegai10.append(float(line[1]))
    polarization10.append(float(line[2]))

f.close()
omegai10 = np.array(omegai10).reshape(100, -1)
omegar10 = np.array(omegar10).reshape(100, -1)
polarization10 = np.array(polarization10).reshape(100, -1)

omegai10 = omegai10[:, ::-1]
omegar10 = omegar10[:, ::-1]
polarization10 = polarization10[:, ::-1]

# omegai = np.concatenate((omegai, omegai1), axis=0)
# omegai = omegai[:, ::-12]

pth = '/media/kun/Samsung_T5/Mathematica/test/0.01/output_ptp' + '811.txt'
f = open(pth)
omegai11 = []
omegar11 = []
polarization11 = []
for line in f.readlines():
    line = line.split()
    if not line:
        break
    if line[0] == 'Indeterminate':
        line[0] = 0
    if line[1] == 'Indeterminate':
        line[1] = 0
    if line[2] == 'Indeterminate':
        line[2] = 0
    omegar11.append(float(line[0]))
    omegai11.append(float(line[1]))
    polarization11.append(float(line[2]))

f.close()
omegai11 = np.array(omegai11).reshape(100, -1)
omegar11 = np.array(omegar11).reshape(100, -1)
polarization11 = np.array(polarization11).reshape(100, -1)

omegai2 = np.concatenate((omegai10, omegai11), axis=1)
omegar2 = np.concatenate((omegar10, omegar11), axis=1)
polarization2 = np.concatenate((polarization10, polarization11), axis=1)

pth = '/media/kun/Samsung_T5/Mathematica/test/0.01/output_ptp' + '820.txt'
f = open(pth)
omegai10 = []
omegar10 = []
polarization10 = []
for line in f.readlines():
    line = line.split()
    if not line:
        break
    if line[0] == 'Indeterminate':
        line[0] = 0
    if line[1] == 'Indeterminate':
        line[1] = 0
    if line[2] == 'Indeterminate':
        line[2] = 0
    omegar10.append(float(line[0]))
    omegai10.append(float(line[1]))
    polarization10.append(float(line[2]))

f.close()
omegai10 = np.array(omegai10).reshape(100, -1)
omegar10 = np.array(omegar10).reshape(100, -1)
polarization10 = np.array(polarization10).reshape(100, -1)

omegai10 = omegai10[:, ::-1]
omegar10 = omegar10[:, ::-1]
polarization10 = polarization10[:, ::-1]

# omegai = np.concatenate((omegai, omegai1), axis=0)
# omegai = omegai[:, ::-12]

pth = '/media/kun/Samsung_T5/Mathematica/test/0.01/output_ptp' + '821.txt'
f = open(pth)
omegai11 = []
omegar11 = []
polarization11 = []
for line in f.readlines():
    line = line.split()
    if not line:
        break
    if line[0] == 'Indeterminate':
        line[0] = 0
    if line[1] == 'Indeterminate':
        line[1] = 0
    if line[2] == 'Indeterminate':
        line[2] = 0
    omegar11.append(float(line[0]))
    omegai11.append(float(line[1]))
    polarization11.append(float(line[2]))

f.close()
omegai11 = np.array(omegai11).reshape(100, -1)
omegar11 = np.array(omegar11).reshape(100, -1)
polarization11 = np.array(polarization11).reshape(100, -1)

omegai3 = np.concatenate((omegai10, omegai11), axis=1)
omegar3 = np.concatenate((omegar10, omegar11), axis=1)
polarization3 = np.concatenate((polarization10, polarization11), axis=1)

pth = '/media/kun/Samsung_T5/Mathematica/test/0.01/output_ptp' + '830.txt'
f = open(pth)
omegai10 = []
omegar10 = []
polarization10 = []
for line in f.readlines():
    line = line.split()
    if not line:
        break
    if line[0] == 'Indeterminate':
        line[0] = 0
    if line[1] == 'Indeterminate':
        line[1] = 0
    if line[2] == 'Indeterminate':
        line[2] = 0
    omegar10.append(float(line[0]))
    omegai10.append(float(line[1]))
    polarization10.append(float(line[2]))

f.close()
omegai10 = np.array(omegai10).reshape(201, -1)
omegar10 = np.array(omegar10).reshape(201, -1)
polarization10 = np.array(polarization10).reshape(201, -1)

omegai10 = omegai10[:, ::-1]
polarization10 = polarization10[:, ::-1]
omegar10 = omegar10[:, ::-1]

# omegai = np.concatenate((omegai, omegai1), axis=0)
# omegai = omegai[:, ::-12]

pth = '/media/kun/Samsung_T5/Mathematica/test/0.01/output_ptp' + '831.txt'
f = open(pth)
omegai11 = []
omegar11 = []
polarization11 = []
for line in f.readlines():
    line = line.split()
    if not line:
        break
    if line[0] == 'Indeterminate':
        line[0] = 0
    if line[1] == 'Indeterminate':
        line[1] = 0
    if line[2] == 'Indeterminate':
        line[2] = 0
    omegar11.append(float(line[0]))
    omegai11.append(float(line[1]))
    polarization11.append(float(line[2]))

f.close()
omegai11 = np.array(omegai11).reshape(201, -1)
omegar11 = np.array(omegar11).reshape(201, -1)
polarization11 = np.array(polarization11).reshape(201, -1)

omegai4 = np.concatenate((omegai10, omegai11), axis=1)
omegar4 = np.concatenate((omegar10, omegar11), axis=1)
polarization4 = np.concatenate((polarization10, polarization11), axis=1)
omegai1 = np.concatenate((omegai1, omegai2, omegai3, omegai4), axis=0)
omegar1 = np.concatenate((omegar1, omegar2, omegar3, omegar4), axis=0)
polarization = np.concatenate((polarization1, polarization2, polarization3, polarization4), axis=0)
for i in polarization:
    for ii in i:
        if ii > 1:
            ii = 1 / ii

polarization[omegai1 < 0.01] = None
omegai1[omegai1 < 0.01] = None
# polarization[abs(polarization) < 0.01] = 0
polarization = np.arctan(polarization)
# omegai1 = np.concatenate((omegai10, omegai11), axis=1)
# omegar1 = np.concatenate((omegar10, omegar11), axis=1)
# polarization = np.concatenate((polarization10, polarization11), axis=1)
# for i in polarization:
#     for ii in i:
#         if ii > 1:
#             ii = 1 / ii
#
# omegai1[omegai1 < 0.001] = None
# omegai1[omegai1 > 0.1] = None
# polarization[abs(polarization) < 0.01] = 0

# file = pth + 'output4' + '.txt'
#
# f = open(file)
# omegai2 = []
# omegar2 = []
# for line in f.readlines():
#     line = line.split()
#     if not line:
#         break
#     omegar2.append(float(line[0]))
#     omegai2.append(float(line[1]))
#
# f.close()
#
# omegai[8, :] = omegai2
# omegar[8, :] = omegar2
#
# file = pth + 'output5' + '.txt'
#
# f = open(file)
# omegai2 = []
# omegar2 = []
# for line in f.readlines():
#     line = line.split()
#     if not line:
#         break
#     omegar2.append(float(line[0]))
#     omegai2.append(float(line[1]))
#
# f.close()
#
# omegai[18, :] = omegai2
# omegar[18, :] = omegar2

# fig = plt.figure(figsize=(16, 8))
# ax = fig.add_subplot(121, projection='3d')
# ax.view_init(45, 60)
# x = np.arange(0.02, 0.200001, 0.001)
# y = np.arange(0, 0.60001, 0.005)
# X, Y = np.meshgrid(x, y)
# # plt.contourf(X, Y, omegar, 10, cmap='jet')
# # plt.pcolormesh(X, Y, omegar, shading='gouraud', cmap='jet')
#
# # Normalize the colors based on omegai
# scamap = plt.cm.ScalarMappable(cmap='Reds')
# fcolors = scamap.to_rgba(omegai)
# fig = ax.scatter(X, Y, omegar, c=omegai, cmap='jet', vmin=-0.01)
# # fig = ax.plot_surface(X,Y, omegar, facecolors=fcolors)
# ax.set_xlabel(r'$k_\parallel k_i$')
# ax.set_ylabel(r'$k_\perp k_i$')
# ax.set_zlabel(r'$\omega / \Omega_i$')
# ax.set_xticks([0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21])
# # cbar = plt.colorbar(scamap)
# # plt.xlabel(r'k_\parallel')
# # plt.ylabel(r'k_perp')
# # cbar = plt.colorbar()
# # cbar.set_label(r'$\gamma / \Omega_i$')
#
#
# plt.subplot(122)
# c = plt.contour(X, Y, omegai, np.linspace(0.01, 0.04, 5), cmap='jet', vmin=0)
# # get data from contour lines
# c = c.collections[4].get_paths()[0]
# v = c.vertices
# x = np.array(v[:, 0] / 0.001, dtype=int)
# y = np.array(v[:, 1] / 0.005, dtype=int)
# n = len(x)
# z = []
# for i in range(n):
#     zz = omegar[y[i], x[i]]
#     z.append(zz)
# z = np.array(z)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot3D(v[:, 0], v[:, 1], z, 'k', linewidth=2)
# # plt.pcolormesh(X, Y, omegai, shading='gouraud', cmap='jet', vmin=0.01)
# plt.xlabel(r'$k_\parallel$')
# plt.ylabel(r'$k_\perp$')
# # cbar = plt.colorbar()
# # cbar.set_label(r'$\gamma / \Omega_i$')
# plt.show()

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14}
x = np.linspace(0.02, 0.25, 117)
y = np.linspace(0.1, 8, ny)
X, Y = np.meshgrid(x, y)
eta = omegar1 - X * 10 + 1
fig = plt.figure()
plt.pcolormesh(X, Y, omegai1, shading='gouraud', cmap='jet', )
cb = plt.colorbar()
cb.ax.tick_params(labelsize=12)
cb.set_label(r'$\gamma/ \Omega_i$', fontdict=font1)
# contour = plt.contour(x, y, polarization, cmap='cool', alpha=1, Nchunk=0.,
#                       linestyles='dotted')
# plt.clabel(contour, inline=1, )
contour = plt.contour(X, Y, omegai1, [0.08], cmap='cool', alpha=1, Nchunk=0, linestyles='dotted')
con0 = contour.collections[0].get_paths()[0].vertices
con1 = contour.collections[0].get_paths()[1].vertices
# plt.clabel(contour, inline=1)
plt.xlabel(r'$k_\parallel  k_i$', font1)
plt.ylabel(r'$k_\perp k_i$', font1)
plt.tick_params(labelsize=12)
# cbar = fig.colorbar(contour)
# cbar.set_label(r'$\gamma / \Omega_i$', fontdict=font1)
# cbar.ax.tick_params(labelsize=12)
# plt.annotate('$n=1$', xy=(0.14, 0.2), xytext=(0.2, 0.05), arrowprops=dict(facecolor='black', shrink=0.05), fontsize=15)
# plt.annotate(r'$n=2$', xy=(0.25, 0.35), xytext=(0.31, 0.2), arrowprops=dict(facecolor='black', shrink=0.05),
#              fontsize=15)
# cbar.set_clim(0.015, 0.12)

fig, ax = plt.subplots(figsize=(8, 6))
x = np.hstack((con0[:, 0], con1[:, 0]))
y = np.hstack((con0[:, 1], con1[:, 1]))
ax.plot(x, y)
ax.set_ylim(0, 12)
ax.set_xlim(0, 1)
plt.show()

# one dimentional

# # omegar1 = omegar1[::-1]
# # omegai1 = omegai1[::-1]
# # omegai4 = omegai4[::-1]
# # omegar4 = omegar4[::-1]
# k = np.arange(0.02, 0.6, 0.002)
# font1 = {'family': 'Computer Modern Roman',
#          'weight': 'normal',
#          'size': 14}
# font2 = {'family': 'TIMES',
#          'weight': 'normal',
#          'size': 14}
#
# fig, ax = plt.subplots(figsize=(8, 6))
# plt.title('ion cyclotron instability', fontsize=16)
#
# ax2 = ax.twinx()
# ax.plot(k, omegar1, 'r', label=r'$\theta = 40\degree$')
# ax2.plot(k, omegai1, 'r:', )
# ax.plot(k, omegar4, 'b', label=r'$\theta = 70\degree$')
# ax2.plot(k, omegai4, 'b:',)
# ax.tick_params(labelsize=12)
# ax2.tick_params(labelsize=12)
#
# # giving labels to the axises
# ax.set_xlabel(r"$k_\parallel k_i^{-1}$", font2)
# ax.set_ylabel(r'$\omega \Omega^{-1}$', font2)
# # ax.set_ylim([0, 1])
# # ax.tick_params(axis='y', colors='r')
# # ax.spines['left'].set_color('red')
# # ax.spines['right'].set_color('blue')
# # secondary y-axis label
# ax2.set_ylabel(r'$\gamma \Omega^{-1}$', font2)
# # ax2.tick_params(axis='y', colors='b')      #settubg up y-axis tick color to blue
# # ax2.spines['right'].set_color('blue')    #settubg up y-axis tick color to blue
# # ax2.spines['left'].set_color('red')
#
# # defining display layout
# plt.tight_layout()
# ax.legend(loc='upper center', fontsize=12)
# plt.show()
