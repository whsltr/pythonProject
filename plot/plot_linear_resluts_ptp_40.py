import numpy as np
import matplotlib.pyplot as plt

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

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14}
x = np.linspace(0.01, 0.6, nx+nx1)
y = np.linspace(0.01, 8, 800)
X, Y = np.meshgrid(x, y)
eta = (omegar1 - X*10 + 1)
fig = plt.figure()
plt.pcolormesh(X, Y, omegai1, shading='gouraud', cmap='jet', )
cb = plt.colorbar()
cb.ax.tick_params(labelsize=12)
cb.set_label(r'$\gamma/ \Omega_p$', fontdict=font1)

# contour = plt.contour(x, y, omegar1, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], cmap='cool', alpha=1, Nchunk=0.,
#                       linestyles='dotted')
# plt.clabel(contour, inline=1, )
contour = plt.contour(X, Y, omegar1, 7, alpha=1, Nchunk=0, linestyles='dotted', colors='w')
# con2 = contour.collections[0].get_paths()[2].vertices
plt.clabel(contour, inline=1)
plt.xlabel(r'$k_\parallel  \lambda_p$', font1)
plt.ylabel(r'$k_\perp \lambda_p$', font1)
plt.tick_params(labelsize=12)

plt.savefig(pth + '.png')
plt.show()

