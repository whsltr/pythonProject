import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

pth = '/home/kun/Documents/mathematics/beam/0.01/15n/' + '10va_01'
f = open(pth + '.txt')
omegai1 = []
omegar1 = []
polarization1 = []
nx = 240
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
    omegar1.append(float(line[1]))
    omegai1.append(float(line[2]))
    polarization1.append(float(line[3]))

f.close()
omegai1 = np.array(omegai1).reshape(-1, nx)
omegar1 = np.array(omegar1).reshape(-1, nx)
polarization1 = np.array(polarization1).reshape(-1, nx)

pth = '/home/kun/Documents/mathematics/beam/0.01/15n/' + '10va_00'
f = open(pth + '.txt')
omegai0 = []
omegar0 = []
polarization0 = []
nx1 = 55
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
    omegar0.append(float(line[1]))
    omegai0.append(float(line[2]))
    polarization0.append(float(line[3]))

f.close()
omegai0 = np.array(omegai0).reshape(-1, nx1)
omegar0 = np.array(omegar0).reshape(-1, nx1)
polarization0 = np.array(polarization0).reshape(-1, nx1)
omegai0 = omegai0[:, ::-1]
omegar0 = omegar0[:, ::-1]
polarization0 = polarization0[:, ::-1]

omegai10 = np.concatenate((omegai0, omegai1), axis=1)
omegar10 = np.concatenate((omegar0, omegar1), axis=1)
polarization10 = np.concatenate((polarization0, polarization1), axis=1)

pth = '/home/kun/Documents/mathematics/beam/0.01/15n/' + '10va_61'
f = open(pth + '.txt')
omegai1 = []
omegar1 = []
polarization1 = []
nx = 215
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
    omegar1.append(float(line[1]))
    omegai1.append(float(line[2]))
    polarization1.append(float(line[3]))

f.close()
omegai1 = np.array(omegai1).reshape(-1, nx)
omegar1 = np.array(omegar1).reshape(-1, nx)
polarization1 = np.array(polarization1).reshape(-1, nx)

pth = '/home/kun/Documents/mathematics/beam/0.01/15n/' + '10va_60'
f = open(pth + '.txt')
omegai0 = []
omegar0 = []
polarization0 = []
nx1 = 80
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
    omegar0.append(float(line[1]))
    omegai0.append(float(line[2]))
    polarization0.append(float(line[3]))

f.close()
omegai0 = np.array(omegai0).reshape(-1, nx1)
omegar0 = np.array(omegar0).reshape(-1, nx1)
polarization0 = np.array(polarization0).reshape(-1, nx1)
omegai0 = omegai0[:, ::-1]
omegar0 = omegar0[:, ::-1]
polarization0 = polarization0[:, ::-1]

omegai1 = np.concatenate((omegai0, omegai1), axis=1)
omegar1 = np.concatenate((omegar0, omegar1), axis=1)
polarization1 = np.concatenate((polarization0, polarization1), axis=1)
omegai1 = np.concatenate((omegai10, omegai1), axis=0)
omegar1 = np.concatenate((omegar10, omegar1), axis=0)
polarization1 = np.concatenate((polarization10, polarization1), axis=0)

# pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_17/' + '10va_10_341'
# f = open(pth + '.txt')
# omegai1 = []
# omegar1 = []
# polarization1 = []
# nx = 130
# for line in f.readlines():
#     line = line.split()
#     if not line:
#         break
#     if line[0] == 'Indeterminate':
#         line[0] = 0
#     if line[1] == 'Indeterminate':
#         line[1] = 0
#     if line[2] == 'Indeterminate':
#         line[2] = 0
#     omegar1.append(float(line[1]))
#     omegai1.append(float(line[2]))
#     polarization1.append(float(line[3]))
#
# f.close()
# omegai1 = np.array(omegai1).reshape(-1, nx)
# omegar1 = np.array(omegar1).reshape(-1, nx)
# polarization1 = np.array(polarization1).reshape(-1, nx)
#
# pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_17/' + '10va_10_340'
# f = open(pth + '.txt')
# omegai0 = []
# omegar0 = []
# polarization0 = []
# nx1 = 10
# for line in f.readlines():
#     line = line.split()
#     if not line:
#         break
#     if line[0] == 'Indeterminate':
#         line[0] = 0
#     if line[1] == 'Indeterminate':
#         line[1] = 0
#     if line[2] == 'Indeterminate':
#         line[2] = 0
#     omegar0.append(float(line[1]))
#     omegai0.append(float(line[2]))
#     polarization0.append(float(line[3]))
#
# f.close()
# omegai0 = np.array(omegai0).reshape(-1, nx1)
# omegar0 = np.array(omegar0).reshape(-1, nx1)
# polarization0 = np.array(polarization0).reshape(-1, nx1)
# omegai0 = omegai0[:, ::-1]
# omegar0 = omegar0[:, ::-1]
# polarization0 = polarization0[:, ::-1]
#
# omegai1 = np.concatenate((omegai0, omegai1), axis=1)
# omegar1 = np.concatenate((omegar0, omegar1), axis=1)
# polarization1 = np.concatenate((polarization0, polarization1), axis=1)

pth = '/home/kun/Documents/mathematics/beam/0.01/15n/' + '10va_02_151'
f = open(pth + '.txt')
omegai1_t = []
omegar1_t = []
polarization1_t = []
nx_t = 25
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
    omegar1_t.append(float(line[1]))
    omegai1_t.append(float(line[2]))
    polarization1_t.append(float(line[3]))

f.close()
omegai1_t = np.array(omegai1_t).reshape(-1, nx_t)
omegar1_t = np.array(omegar1_t).reshape(-1, nx_t)
polarization1_t = np.array(polarization1_t).reshape(-1, nx_t)

pth = '/home/kun/Documents/mathematics/beam/0.01/15n/' + '10va_02_150'
f = open(pth + '.txt')
omegai0_t = []
omegar0_t = []
polarization0_t = []
nx1_t = 70
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
    omegar0_t.append(float(line[1]))
    omegai0_t.append(float(line[2]))
    polarization0_t.append(float(line[3]))

f.close()
omegai0_t = np.array(omegai0_t).reshape(-1, nx1_t)
omegar0_t = np.array(omegar0_t).reshape(-1, nx1_t)
polarization0_t = np.array(polarization0_t).reshape(-1, nx1_t)
omegai0_t = omegai0_t[:, ::-1]
omegar0_t = omegar0_t[:, ::-1]
polarization0_t = polarization0_t[:, ::-1]

omegai1_t = np.concatenate((omegai0_t, omegai1_t), axis=1)
omegar1_t = np.concatenate((omegar0_t, omegar1_t), axis=1)
polarization1_t = np.concatenate((polarization0_t, polarization1_t), axis=1)

omegai1[20:30, :95] = omegai1_t
omegar1[20:30, :95] = omegar1_t
polarization1[20:30, :95] = polarization1_t
pth = '/home/kun/Documents/mathematics/beam/0.01/15n/' + '10va_03_171'
f = open(pth + '.txt')
omegai1_t = []
omegar1_t = []
polarization1_t = []
nx_t = 35
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
    omegar1_t.append(float(line[1]))
    omegai1_t.append(float(line[2]))
    polarization1_t.append(float(line[3]))

f.close()
omegai1_t = np.array(omegai1_t).reshape(-1, nx_t)
omegar1_t = np.array(omegar1_t).reshape(-1, nx_t)
polarization1_t = np.array(polarization1_t).reshape(-1, nx_t)

pth = '/home/kun/Documents/mathematics/beam/0.01/15n/' + '10va_03_170'
f = open(pth + '.txt')
omegai0_t = []
omegar0_t = []
polarization0_t = []
nx1_t = 20
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
    omegar0_t.append(float(line[1]))
    omegai0_t.append(float(line[2]))
    polarization0_t.append(float(line[3]))

f.close()
omegai0_t = np.array(omegai0_t).reshape(-1, nx1_t)
omegar0_t = np.array(omegar0_t).reshape(-1, nx1_t)
polarization0_t = np.array(polarization0_t).reshape(-1, nx1_t)
omegai0_t = omegai0_t[:, ::-1]
omegar0_t = omegar0_t[:, ::-1]
polarization0_t = polarization0_t[:, ::-1]

omegai1_t = np.concatenate((omegai0_t, omegai1_t), axis=1)
omegar1_t = np.concatenate((omegar0_t, omegar1_t), axis=1)
polarization1_t = np.concatenate((polarization0_t, polarization1_t), axis=1)

omegai1[30:60, 60:115] = omegai1_t
omegar1[30:60, 60:115] = omegar1_t
polarization1[30:60, 60:115] = polarization1_t
#
pth = '/home/kun/Documents/mathematics/beam/0.01/15n/' + '10va_06_201'
f = open(pth + '.txt')
omegai1_t = []
omegar1_t = []
polarization1_t = []
nx_t = 25
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
    omegar1_t.append(float(line[1]))
    omegai1_t.append(float(line[2]))
    polarization1_t.append(float(line[3]))

f.close()
omegai1_t = np.array(omegai1_t).reshape(-1, nx_t)
omegar1_t = np.array(omegar1_t).reshape(-1, nx_t)
polarization1_t = np.array(polarization1_t).reshape(-1, nx_t)

pth = '/home/kun/Documents/mathematics/beam/0.01/15n/' + '10va_06_200'
f = open(pth + '.txt')
omegai0_t = []
omegar0_t = []
polarization0_t = []
nx1_t = 25
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
    omegar0_t.append(float(line[1]))
    omegai0_t.append(float(line[2]))
    polarization0_t.append(float(line[3]))

f.close()
omegai0_t = np.array(omegai0_t).reshape(-1, nx1_t)
omegar0_t = np.array(omegar0_t).reshape(-1, nx1_t)
polarization0_t = np.array(polarization0_t).reshape(-1, nx1_t)
omegai0_t = omegai0_t[:, ::-1]
omegar0_t = omegar0_t[:, ::-1]
polarization0_t = polarization0_t[:, ::-1]

omegai1_t = np.concatenate((omegai0_t, omegai1_t), axis=1)
omegar1_t = np.concatenate((omegar0_t, omegar1_t), axis=1)
polarization1_t = np.concatenate((polarization0_t, polarization1_t), axis=1)

omegai1[60:90, 75:125] = omegai1_t
omegar1[60:90, 75:125] = omegar1_t
polarization1[60:90, 75:125] = polarization1_t
#
pth = '/home/kun/Documents/mathematics/beam/0.01/15n/' + '10va_09_241'
f = open(pth + '.txt')
omegai1_t = []
omegar1_t = []
polarization1_t = []
nx_t = 20
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
    omegar1_t.append(float(line[1]))
    omegai1_t.append(float(line[2]))
    polarization1_t.append(float(line[3]))

f.close()
omegai1_t = np.array(omegai1_t).reshape(-1, nx_t)
omegar1_t = np.array(omegar1_t).reshape(-1, nx_t)
polarization1_t = np.array(polarization1_t).reshape(-1, nx_t)

pth = '/home/kun/Documents/mathematics/beam/0.01/15n/' + '10va_09_240'
f = open(pth + '.txt')
omegai0_t = []
omegar0_t = []
polarization0_t = []
nx1_t = 20
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
    omegar0_t.append(float(line[1]))
    omegai0_t.append(float(line[2]))
    polarization0_t.append(float(line[3]))

f.close()
omegai0_t = np.array(omegai0_t).reshape(-1, nx1_t)
omegar0_t = np.array(omegar0_t).reshape(-1, nx1_t)
polarization0_t = np.array(polarization0_t).reshape(-1, nx1_t)
omegai0_t = omegai0_t[:, ::-1]
omegar0_t = omegar0_t[:, ::-1]
polarization0_t = polarization0_t[:, ::-1]

omegai1_t = np.concatenate((omegai0_t, omegai1_t), axis=1)
omegar1_t = np.concatenate((omegar0_t, omegar1_t), axis=1)
polarization1_t = np.concatenate((polarization0_t, polarization1_t), axis=1)

omegai1[90:120, 95:135] = omegai1_t
omegar1[90:120, 95:135] = omegar1_t
polarization1[90:120, 95:135] = polarization1_t

pth = '/home/kun/Documents/mathematics/beam/0.01/15n/' + '10va_12_251'
f = open(pth + '.txt')
omegai1_t = []
omegar1_t = []
polarization1_t = []
nx_t = 15
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
    omegar1_t.append(float(line[1]))
    omegai1_t.append(float(line[2]))
    polarization1_t.append(float(line[3]))

f.close()
omegai1_t = np.array(omegai1_t).reshape(-1, nx_t)
omegar1_t = np.array(omegar1_t).reshape(-1, nx_t)
polarization1_t = np.array(polarization1_t).reshape(-1, nx_t)

pth = '/home/kun/Documents/mathematics/beam/0.01/15n/' + '10va_12_250'
f = open(pth + '.txt')
omegai0_t = []
omegar0_t = []
polarization0_t = []
nx1_t = 15
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
    omegar0_t.append(float(line[1]))
    omegai0_t.append(float(line[2]))
    polarization0_t.append(float(line[3]))

f.close()
omegai0_t = np.array(omegai0_t).reshape(-1, nx1_t)
omegar0_t = np.array(omegar0_t).reshape(-1, nx1_t)
polarization0_t = np.array(polarization0_t).reshape(-1, nx1_t)
omegai0_t = omegai0_t[:, ::-1]
omegar0_t = omegar0_t[:, ::-1]
polarization0_t = polarization0_t[:, ::-1]

omegai1_t = np.concatenate((omegai0_t, omegai1_t), axis=1)
omegar1_t = np.concatenate((omegar0_t, omegar1_t), axis=1)
polarization1_t = np.concatenate((polarization0_t, polarization1_t), axis=1)

omegai1[120:251, 105:135] = omegai1_t
omegar1[120:251, 105:135] = omegar1_t
polarization1[120:251, 105:135] = polarization1_t
# pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_17/' + '10va_05_201'
# f = open(pth + '.txt')
# omegai1_t = []
# omegar1_t = []
# polarization1_t = []
# nx_t = 50
# for line in f.readlines():
#     line = line.split()
#     if not line:
#         break
#     if line[0] == 'Indeterminate':
#         line[0] = 0
#     if line[1] == 'Indeterminate':
#         line[1] = 0
#     if line[2] == 'Indeterminate':
#         line[2] = 0
#     omegar1_t.append(float(line[1]))
#     omegai1_t.append(float(line[2]))
#     polarization1_t.append(float(line[3]))
#
# f.close()
# omegai1_t = np.array(omegai1_t).reshape(-1, nx_t)
# omegar1_t = np.array(omegar1_t).reshape(-1, nx_t)
# polarization1_t = np.array(polarization1_t).reshape(-1, nx_t)
#
# pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_17/' + '10va_05_200'
# f = open(pth + '.txt')
# omegai0_t = []
# omegar0_t = []
# polarization0_t = []
# nx1_t = 25
# for line in f.readlines():
#     line = line.split()
#     if not line:
#         break
#     if line[0] == 'Indeterminate':
#         line[0] = 0
#     if line[1] == 'Indeterminate':
#         line[1] = 0
#     if line[2] == 'Indeterminate':
#         line[2] = 0
#     omegar0_t.append(float(line[1]))
#     omegai0_t.append(float(line[2]))
#     polarization0_t.append(float(line[3]))
#
# f.close()
# omegai0_t = np.array(omegai0_t).reshape(-1, nx1_t)
# omegar0_t = np.array(omegar0_t).reshape(-1, nx1_t)
# polarization0_t = np.array(polarization0_t).reshape(-1, nx1_t)
# omegai0_t = omegai0_t[:, ::-1]
# omegar0_t = omegar0_t[:, ::-1]
# polarization0_t = polarization0_t[:, ::-1]
#
# omegai1_t = np.concatenate((omegai0_t, omegai1_t), axis=1)
# omegar1_t = np.concatenate((omegar0_t, omegar1_t), axis=1)
# polarization1_t = np.concatenate((polarization0_t, polarization1_t), axis=1)
#
# omegai1[50:88, 70:145] = omegai1_t
# omegar1[50:88, 70:145] = omegar1_t
# polarization1[50:88, 70:145] = polarization1_t
# omegai = omegai[:, ::-12]
# omegai1 = np.concatenate((omegai10, omegai1), axis=0)
# omegar1 = np.concatenate((omegar10, omegar1), axis=0)
# polarization1 = np.concatenate((polarization10, polarization1), axis=0)

polarization1[omegai1 < 0.002] = None
omegar1[omegai1 < 0.02] = None
omegai1[omegai1 < 0.02] = None
polarization1[abs(polarization1) < 0.001] = 0

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14}
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 16}
x = np.linspace(0.01, 0.6, nx+nx1)
y = np.linspace(0, 12, 1200)
X, Y = np.meshgrid(x, y)
eta = (omegar1 - X*10 + 2)
fig, (ax, ax1) = plt.subplots(ncols=2, figsize=[12, 6])
cb = ax.pcolormesh(X, Y, omegai1, shading='gouraud', cmap='jet', )
cb = plt.colorbar(cb, ax=ax)
cb.ax.tick_params(labelsize=14)
cb.set_label(r'$\gamma/ \Omega_p$', fontdict=font1)

# contour = plt.contour(x, y, omegar1, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], cmap='cool', alpha=1, Nchunk=0.,
#                       linestyles='dotted')
# plt.clabel(contour, inline=1, )
contour = ax.contour(X, Y, omegar1, 6, alpha=1, Nchunk=0, linestyles='dotted', linewidths=4, colors='k',)
ax.plot([0.05, 0.17], [0.5, 0.5], color='r')
ax.plot([0.05, 0.05], [0., 0.5], color='r')
ax.plot([0.17, 0.17], [0, 0.5], color='r')
ax.plot([0.05, 0.38], [0.5, 4.5], [0.05, 0.38], [0., 0.16], color='r')
# contour1 = ax.contour(X, Y, eta, [0], alpha=1, Nchunk=0, linestyles='--', linewidths=1.5, colors='k')
# con2 = contour.collections[0].get_paths()[2].vertices
plt.clabel(contour, inline=1, fontsize=14, manual=True)
# plt.clabel(contour1, inline=1)
ax.set_xlabel(r'$k_\parallel  \lambda_p$', font1)
ax.set_ylabel(r'$k_\perp \lambda_p$', font1)
ax.text(0.05, 11.3, '(a)', fontsize=18)
# ax.set_title('(a) $T_{b\perp}=T_{b\parallel}$', fontsize=16)
ax.tick_params(labelsize=14)
axins = inset_axes(ax, width="35%", height="37%", loc=4)
cb1 = axins.pcolormesh(X, Y, omegai1, shading='gouraud', cmap='jet', )
axins.set_xlim(0.05, 0.17)
axins.set_ylim(0, 0.5)
axins.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, right=False, left=True, labelright=False,
                  labelleft=True)

# axins = zoomed_inset_axes(ax, 2.5, loc=4)
# axins.pcolormesh(X, Y, omegai1, shading='gouraud', cmap='jet')
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
# plt.xticks(visible=False)
# # plt.yticks(visible=False)
#
# #sub region of the original image
# x1, x2, y1, y2 = 0.05, 0.15, 0, 0.5
# axins.set_xlim(x1, x2)
# axins.set_ylim(y1, y2)

def readfile(path, nx):
    f = open(path + '.txt')
    omegai = []
    omegar = []
    polarization = []
    for line in f.readlines():
        line = line.split()
        omegar.append(float(line[1]))
        omegai.append(float(line[2]))
        polarization.append(float(line[3]))

    f.close()
    omegar = np.array(omegar)
    omegai = np.array(omegai)
    polarization = np.array(polarization)
    return omegar.reshape(-1, nx), omegai.reshape(-1, nx), polarization.reshape(-1, nx)

pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_40/' + '10va_01'
nx = 240
omegar1, omegai1, polarization1 = readfile(pth, nx)

pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_40/' + '10va_00'
nx1 = 55
omegar0, omegai0, polarization0 = readfile(pth, nx1)
omegai0 = omegai0[:, ::-1]
omegar0 = omegar0[:, ::-1]
polarization0 = polarization0[:, ::-1]

omegai1 = np.concatenate((omegai0, omegai1), axis=1)
omegar1 = np.concatenate((omegar0, omegar1), axis=1)
polarization1 = np.concatenate((polarization0, polarization1), axis=1)

# pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_40/' + '10va_41'
# f = open(pth + '.txt')
# omegai1 = []
# omegar1 = []
# polarization1 = []
# nx = 225
# for line in f.readlines():
#     line = line.split()
#     if not line:
#         break
#     if line[0] == 'Indeterminate':
#         line[0] = 0
#     if line[1] == 'Indeterminate':
#         line[1] = 0
#     if line[2] == 'Indeterminate':
#         line[2] = 0
#     omegar1.append(float(line[1]))
#     omegai1.append(float(line[2]))
#     polarization1.append(float(line[3]))
#
# f.close()
# omegai1 = np.array(omegai1).reshape(-1, nx)
# omegar1 = np.array(omegar1).reshape(-1, nx)
# polarization1 = np.array(polarization1).reshape(-1, nx)
#
# pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_40/' + '10va_40'
# f = open(pth + '.txt')
# omegai0 = []
# omegar0 = []
# polarization0 = []
# nx1 = 70
# for line in f.readlines():
#     line = line.split()
#     if not line:
#         break
#     if line[0] == 'Indeterminate':
#         line[0] = 0
#     if line[1] == 'Indeterminate':
#         line[1] = 0
#     if line[2] == 'Indeterminate':
#         line[2] = 0
#     omegar0.append(float(line[1]))
#     omegai0.append(float(line[2]))
#     polarization0.append(float(line[3]))
#
# f.close()
# omegai0 = np.array(omegai0).reshape(-1, nx1)
# omegar0 = np.array(omegar0).reshape(-1, nx1)
# polarization0 = np.array(polarization0).reshape(-1, nx1)
# omegai0 = omegai0[:, ::-1]
# omegar0 = omegar0[:, ::-1]
# polarization0 = polarization0[:, ::-1]
#
# omegai1 = np.concatenate((omegai0, omegai1), axis=1)
# omegar1 = np.concatenate((omegar0, omegar1), axis=1)
# polarization1 = np.concatenate((polarization0, polarization1), axis=1)
# # omegai = omegai[:, ::-12]
# omegai1 = np.concatenate((omegai10, omegai1), axis=0)
# omegar1 = np.concatenate((omegar10, omegar1), axis=0)
# polarization1 = np.concatenate((polarization10, polarization1), axis=0)
#
pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_40/' + '10va_281'
nx_t = 240
omegar1_t, omegai1_t, polarization1_t = readfile(pth, nx_t)

pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_40/' + '10va_280'
nx1_t = 55
omegar0_t, omegai0_t, polarization0_t = readfile(pth, nx1_t)
omegai0_t = omegai0_t[:, ::-1]
omegar0_t = omegar0_t[:, ::-1]
polarization0_t = polarization0_t[:, ::-1]

omegai1_t = np.concatenate((omegai0_t, omegai1_t), axis=1)
omegar1_t = np.concatenate((omegar0_t, omegar1_t), axis=1)
polarization1_t = np.concatenate((polarization0_t, polarization1_t), axis=1)

omegai1[280:, :] = omegai1_t
omegar1[280:, :] = omegar1_t
polarization1[280:, :] = polarization1_t
#
pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_40/' + '10va_241'
nx_t = 180
omegar1_t, omegai1_t, polarization1_t = readfile(pth, nx_t)

pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_40/' + '10va_240'
nx1_t = 25
omegar0_t, omegai0_t, polarization0_t = readfile(pth, nx1_t)
omegai0_t = omegai0_t[:, ::-1]
omegar0_t = omegar0_t[:, ::-1]
polarization0_t = polarization0_t[:, ::-1]

omegai1_t = np.concatenate((omegai0_t, omegai1_t), axis=1)
omegar1_t = np.concatenate((omegar0_t, omegar1_t), axis=1)
polarization1_t = np.concatenate((polarization0_t, polarization1_t), axis=1)
omegai1_t[omegai1_t < 0] = 0

omegai1[:, 90:] = omegai1_t
omegar1[:, 90:] = omegar1_t
polarization1[:, 90:] = polarization1_t
#
pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_40/' + '10va_50_151'
nx_t = 25
omegar1_t, omegai1_t, polarization1_t = readfile(pth, nx_t)

pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_40/' + '10va_50_150'
nx1_t = 25
omegar0_t, omegai0_t, polarization0_t = readfile(pth, nx1_t)
omegai0_t = omegai0_t[:, ::-1]
omegar0_t = omegar0_t[:, ::-1]
polarization0_t = polarization0_t[:, ::-1]

omegai1_t = np.concatenate((omegai0_t, omegai1_t), axis=1)
omegar1_t = np.concatenate((omegar0_t, omegar1_t), axis=1)
polarization1_t = np.concatenate((polarization0_t, polarization1_t), axis=1)

omegai1[500:697, 45:95] = omegai1_t
omegar1[500:697, 45:95] = omegar1_t
polarization1[500:697, 45:95] = polarization1_t

pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_40/' + '10va_10_341'
nx_t = 130
omegar1_t, omegai1_t, polarization1_t = readfile(pth, nx_t)

pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_40/' + '10va_10_340'
nx1_t = 20
omegar0_t, omegai0_t, polarization0_t = readfile(pth, nx1_t)

omegai0_t = omegai0_t[:500, ::-1]
omegar0_t = omegar0_t[:500, ::-1]
polarization0_t = polarization0_t[:500, ::-1]

omegai1_t = np.concatenate((omegai0_t, omegai1_t), axis=1)
omegar1_t = np.concatenate((omegar0_t, omegar1_t), axis=1)
polarization1_t = np.concatenate((polarization0_t, polarization1_t), axis=1)

omegai1[100:600, 145:] = omegai1_t
omegar1[100:600, 145:] = omegar1_t
polarization1[100:600, 145:] = polarization1_t

pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_40/' + '10va_01_221'
nx_t = 40
omegar1_t, omegai1_t, polarization1_t = readfile(pth, nx_t)

pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_40/' + '10va_01_220'
nx1_t = 10
omegar0_t, omegai0_t, polarization0_t = readfile(pth, nx1_t)

omegai0_t = omegai0_t[:, ::-1]
omegar0_t = omegar0_t[:, ::-1]
polarization0_t = polarization0_t[:, ::-1]

omegai1_t = np.concatenate((omegai0_t, omegai1_t), axis=1)
omegar1_t = np.concatenate((omegar0_t, omegar1_t), axis=1)
polarization1_t = np.concatenate((polarization0_t, polarization1_t), axis=1)

omegai1[10:99, 95:145] = omegai1_t
omegar1[10:99, 95:145] = omegar1_t
polarization1[10:99, 95:145] = polarization1_t

pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_40/' + '10va_02_151'
nx_t = 25
omegar1_t, omegai1_t, polarization1_t = readfile(pth, nx_t)

pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_40/' + '10va_02_150'
nx1_t = 70
omegar0_t, omegai0_t, polarization0_t = readfile(pth, nx1_t)

omegai0_t = omegai0_t[:, ::-1]
omegar0_t = omegar0_t[:, ::-1]
polarization0_t = polarization0_t[:, ::-1]

omegai1_t = np.concatenate((omegai0_t, omegai1_t), axis=1)
omegar1_t = np.concatenate((omegar0_t, omegar1_t), axis=1)
polarization1_t = np.concatenate((polarization0_t, polarization1_t), axis=1)
omegai1_t[omegai1_t < 0] = 0

omegai1[21:35, :95] = omegai1_t
omegar1[21:35, :95] = omegar1_t
polarization1[21:35, :95] = polarization1_t

pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_40/' + '10va_03_121'
nx_t = 15
omegar1_t, omegai1_t, polarization1_t = readfile(pth, nx_t)

pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_40/' + '10va_03_120'
nx1_t = 55
omegar0_t, omegai0_t, polarization0_t = readfile(pth, nx1_t)

omegai0_t = omegai0_t[:, ::-1]
omegar0_t = omegar0_t[:, ::-1]
polarization0_t = polarization0_t[:, ::-1]

omegai1_t = np.concatenate((omegai0_t, omegai1_t), axis=1)
omegar1_t = np.concatenate((omegar0_t, omegar1_t), axis=1)
polarization1_t = np.concatenate((polarization0_t, polarization1_t), axis=1)
omegai1_t[omegai1_t < 0] = 0

omegai1[30:35, :70] += omegai1_t
omegar1[30:35, :70] = omegar1_t
polarization1[30:35, :70] = polarization1_t

pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_40/' + '10va_035_171'
nx_t = 25
omegar1_t, omegai1_t, polarization1_t = readfile(pth, nx_t)

pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_40/' + '10va_035_170'
nx1_t = 15
omegar0_t, omegai0_t, polarization0_t = readfile(pth, nx1_t)

omegai0_t = omegai0_t[:, ::-1]
omegar0_t = omegar0_t[:, ::-1]
polarization0_t = polarization0_t[:, ::-1]

omegai1_t = np.concatenate((omegai0_t, omegai1_t), axis=1)
omegar1_t = np.concatenate((omegar0_t, omegar1_t), axis=1)
polarization1_t = np.concatenate((polarization0_t, polarization1_t), axis=1)
omegai1_t[omegai1_t < 0] = 0

omegai1[35:66, 65:105] += omegai1_t
# omegar1[35:66, 65:105] = omegar1_t
polarization1[35:66, 65:105] = polarization1_t

pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_40/' + '10va_067_221'
nx_t = 20
omegar1_t, omegai1_t, polarization1_t = readfile(pth, nx_t)

pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_40/' + '10va_067_220'
nx1_t = 25
omegar0_t, omegai0_t, polarization0_t = readfile(pth, nx1_t)

omegai0_t = omegai0_t[:, ::-1]
omegar0_t = omegar0_t[:, ::-1]
polarization0_t = polarization0_t[:, ::-1]

omegai1_t = np.concatenate((omegai0_t, omegai1_t), axis=1)
omegar1_t = np.concatenate((omegar0_t, omegar1_t), axis=1)
polarization1_t = np.concatenate((polarization0_t, polarization1_t), axis=1)
omegai1_t[omegai1_t < 0] = 0

omegai1[66:99, 80:125] += omegai1_t
# omegar1[35:66, 65:105] = omegar1_t
polarization1[66:99, 80:125] = polarization1_t

pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_40/' + '10va_96_241'
nx_t = 20
omegar1_t, omegai1_t, polarization1_t = readfile(pth, nx_t)

pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_40/' + '10va_096_240'
nx1_t = 10
omegar0_t, omegai0_t, polarization0_t = readfile(pth, nx1_t)

omegai0_t = omegai0_t[:, ::-1]
omegar0_t = omegar0_t[:, ::-1]
polarization0_t = polarization0_t[:, ::-1]

omegai1_t = np.concatenate((omegai0_t, omegai1_t), axis=1)
omegar1_t = np.concatenate((omegar0_t, omegar1_t), axis=1)
polarization1_t = np.concatenate((polarization0_t, polarization1_t), axis=1)
omegai1_t[omegai1_t < 0] = 0

omegai1[95:119, 105:135] += omegai1_t
# omegar1[35:66, 65:105] = omegar1_t
polarization1[95:119, 105:135] = polarization1_t

pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_40/' + '10va_50_281'
nx_t = 40
omegar1_t, omegai1_t, polarization1_t = readfile(pth, nx_t)

pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_40/' + '10va_50_280'
nx1_t = 25
omegar0_t, omegai0_t, polarization0_t = readfile(pth, nx1_t)

omegai0_t = omegai0_t[:, ::-1]
omegar0_t = omegar0_t[:, ::-1]
polarization0_t = polarization0_t[:, ::-1]

omegai1_t = np.concatenate((omegai0_t, omegai1_t), axis=1)
omegar1_t = np.concatenate((omegar0_t, omegar1_t), axis=1)
polarization1_t = np.concatenate((polarization0_t, polarization1_t), axis=1)

omegai1[500:700, 110:175] = omegai1_t
omegar1[500:700, 110:175] = omegar1_t
polarization1[500:700, 110:175] = polarization1_t

# pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_40/' + '10va_temp'
# nx_t = 20
# omegar1_t, omegai1_t, polarization1_t = readfile(pth, nx_t)
#
# omegai1[460:480, 55:75] = omegai1_t
# omegar1[460:480, 55:75] = omegar1_t
# polarization1[460:480, 55:75] = polarization1_t

omegai1[omegai1>0.2] = None
polarization1[omegai1 < 0.02] = None
omegar1[omegai1 < 0.02] = None
omegai1[omegai1 < 0.02] = None
polarization1[abs(polarization1) < 0.001] = 0
x = np.linspace(0.01, 0.6, nx+nx1)
y = np.linspace(0.01, 8, 800)
X, Y = np.meshgrid(x, y)
cb = ax1.pcolormesh(X, Y, omegai1, shading='gouraud', cmap='jet', )
cb = plt.colorbar(cb, ax=ax1)
cb.ax.tick_params(labelsize=14)
cb.set_label(r'$\gamma/ \Omega_p$', fontdict=font1)

# contour = plt.contour(x, y, omegar1, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], cmap='cool', alpha=1, Nchunk=0.,
#                       linestyles='dotted')
# plt.clabel(contour, inline=1, )
contour = ax1.contour(X, Y, omegar1, 7, alpha=1, Nchunk=0, linestyles='dotted', linewidths=3, colors='k')
# con2 = contour.collections[0].get_paths()[2].vertices
plt.clabel(contour, inline=1, fontsize=14)
ax1.set_xlabel(r'$k_\parallel  \lambda_p$', font1)
ax1.set_ylabel(r'$k_\perp \lambda_p$', font1)
ax1.tick_params(labelsize=14)
ax1.text(0.05, 7.5, '(b)', fontsize=18)
# ax1.set_title('(b) $T_{b\perp}=40T_{b\parallel}$', fontsize=16)
plt.savefig(pth + '.png')
plt.show()

