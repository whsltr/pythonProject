import numpy as np
import matplotlib.pyplot as plt

pth = '/home/kun/Documents/mathematics/beam/0.01/3n/' + '10va_05_2511'
f = open(pth + '.txt')
omegai1 = []
omegar1 = []
polarization1 = []
nx = 175
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

pth = '/home/kun/Documents/mathematics/beam/0.01/3n/' + '10va_05_2500'
f = open(pth + '.txt')
omegai0 = []
omegar0 = []
polarization0 = []
nx1 = 25
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

pth = '/home/kun/Documents/mathematics/beam/0.01/3n/' + '10va_17_231'
f = open(pth + '.txt')
omegai1_t = []
omegar1_t = []
polarization1_t = []
nx_t = 10
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

pth = '/home/kun/Documents/mathematics/beam/0.01/3n/' + '10va_17_230'
f = open(pth + '.txt')
omegai0_t = []
omegar0_t = []
polarization0_t = []
nx1_t = 5
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

omegai1[120:179, 10:25] = omegai1_t
omegar1[120:179, 10:25] = omegar1_t
polarization1[120:179, 10:25] = polarization1_t
# omegai = omegai[:, ::-12]
# omegai1 = np.concatenate((omegai10, omegai1), axis=0)
# omegar1 = np.concatenate((omegar10, omegar1), axis=0)
# polarization1 = np.concatenate((polarization10, polarization1), axis=0)

polarization1[omegai1 < 0.002] = 0
omegar1[omegai1 < 0.002] = 0
omegai1[omegai1 < 0.002] = 0
polarization1[abs(polarization1) < 0.001] = 0

polarization1_p = polarization1
omegai1_p = omegai1
omegar1_p = omegar1

pth = '/home/kun/Documents/mathematics/beam/0.01/3n/' + '10va_01'
f = open(pth + '.txt')
omegai1 = []
omegar1 = []
polarization1 = []
nx = 245
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

pth = '/home/kun/Documents/mathematics/beam/0.01/3n/' + '10va_00'
f = open(pth + '.txt')
omegai0 = []
omegar0 = []
polarization0 = []
nx1 = 50
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

pth = '/home/kun/Documents/mathematics/beam/0.01/3n/' + '10va_81'
f = open(pth + '.txt')
omegai1 = []
omegar1 = []
polarization1 = []
nx = 220
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

pth = '/home/kun/Documents/mathematics/beam/0.01/3n/' + '10va_80'
f = open(pth + '.txt')
omegai0 = []
omegar0 = []
polarization0 = []
nx1 = 75
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
# pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_8/' + '10va_2_351'
# f = open(pth + '.txt')
# omegai1 = []
# omegar1 = []
# polarization1 = []
# nx_t = 25
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
# omegai1 = np.array(omegai1).reshape(-1, nx_t)
# omegar1 = np.array(omegar1).reshape(-1, nx_t)
# polarization1 = np.array(polarization1).reshape(-1, nx_t)
#
# pth = '/home/kun/Documents/mathematics/beam/0.01/ptp_8/' + '10va_2_350'
# f = open(pth + '.txt')
# omegai0 = []
# omegar0 = []
# polarization0 = []
# nx1_t = 65
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
# omegai0 = np.array(omegai0).reshape(-1, nx1_t)
# omegar0 = np.array(omegar0).reshape(-1, nx1_t)
# polarization0 = np.array(polarization0).reshape(-1, nx1_t)
# omegai0 = omegai0[:, ::-1]
# omegar0 = omegar0[:, ::-1]
# polarization0 = polarization0[:, ::-1]
#
# omegai1 = np.concatenate((omegai0, omegai1), axis=1)
# omegar1 = np.concatenate((omegar0, omegar1), axis=1)
# polarization1 = np.concatenate((polarization0, polarization1), axis=1)

pth = '/home/kun/Documents/mathematics/beam/0.01/3n/' + '10va_010_251'
f = open(pth + '.txt')
omegai1_t = []
omegar1_t = []
polarization1_t = []
nx_t = 45
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

pth = '/home/kun/Documents/mathematics/beam/0.01/3n/' + '10va_010_250'
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

omegai1[99:300, 100:165] = omegai1_t
omegar1[99:300, 100:165] = omegar1_t
polarization1[99:300, 100:165] = polarization1_t

pth = '/home/kun/Documents/mathematics/beam/0.01/3n/' + '10va_077_221'
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

pth = '/home/kun/Documents/mathematics/beam/0.01/3n/' + '10va_077_220'
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

omegai1[77:100, 80:125] = omegai1_t
omegar1[77:100, 80:125] = omegar1_t
polarization1[77:100, 80:125] = polarization1_t

pth = '/home/kun/Documents/mathematics/beam/0.01/3n/' + '10va_046_181'
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

pth = '/home/kun/Documents/mathematics/beam/0.01/3n/' + '10va_046_180'
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

omegai1[46:77, 60:120] = omegai1_t
omegar1[46:77, 60:120] = omegar1_t
polarization1[46:77, 60:120] = polarization1_t

pth = '/home/kun/Documents/mathematics/beam/0.01/3n/' + '10va_01_151'
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

pth = '/home/kun/Documents/mathematics/beam/0.01/3n/' + '10va_01_150'
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

omegai1[10:30, 0:95] = omegai1_t[:20, :]
omegar1[10:30, 0:95] = omegar1_t[:20, :]
polarization1[10:30, 0:95] = polarization1_t[:20, :]
omegar1[30:46, 60:95] = omegar1_t[20:, 60:]
omegai1[30:46, 60:95] = omegai1_t[20:, 60:]
polarization1[30:46, 60:95] = polarization1_t[20:, 60:]
# omegai = omegai[:, ::-12]
# omegai1 = np.concatenate((omegai10, omegai1), axis=0)
# omegar1 = np.concatenate((omegar10, omegar1), axis=0)
# polarization1 = np.concatenate((polarization10, polarization1), axis=0)

omegai1[omegai1 > 0.28] = 0
polarization1[omegai1 < 0.02] = None
omegar1[omegai1 < 0.02] = None
omegai1[omegai1 < 0.02] = 0
polarization1[abs(polarization1) < 0.001] = 0
omegai1[50:1001, 95:] += omegai1_p
omegai1[omegai1 < 0.02] = None

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14}
x = np.linspace(0.01, 0.6, nx+nx1)
y = np.linspace(0, 12, 1201)
X, Y = np.meshgrid(x, y)
eta = (omegar1 - X*10 + 1) / (X * np.sqrt(0.01))
eta2 = omegar1 - X*10 + 2
fig = plt.figure()
plt.pcolormesh(X, Y, omegai1, shading='gouraud', cmap='jet', )
plt.ylim(0, 9.9)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=12)
cb.set_label(r'$\gamma/ \Omega_p$', fontdict=font1)

# contour = plt.contour(x, y, omegar1, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], cmap='cool', alpha=1, Nchunk=0.,
#                       linestyles='dotted')
# plt.clabel(contour, inline=1, )
contour = plt.contour(X, Y, omegar1, 7, alpha=1, Nchunk=0, linestyles='dotted', colors='w')
contour1 = plt.contour(X, Y, eta, [1], alpha=1, Nchunk=0, linestyles='--', colors='w')
contour2 = plt.contour(X, Y, eta2, [0], alpha=1, Nchunk=0, linestyles='-', colors='w')
contour3 = plt.contour(X, Y, omegai1, [0.15], alpha=1, Nchunk=0, linstyles='-', colors='w')
# con2 = contour.collections[0].get_paths()[2].vertices

plt.xlabel(r'$k_\parallel  \lambda_p$', font1)
plt.ylabel(r'$k_\perp \lambda_p$', font1)
plt.tick_params(labelsize=12)
plt.clabel(contour, inline=1, manual=False)
plt.clabel(contour1, inline=1, manual=False)
plt.clabel(contour2, inline=1, manual=False)
plt.clabel(contour3, inline=1, manual=False)

plt.savefig(pth + 'p.png')
plt.show()

