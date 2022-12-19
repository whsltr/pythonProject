import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
import data_analysis.read_data as data
from scipy.fft import fft, fft2, fftfreq, fftshift

# pth = '/media/kun/Samsung_T5/Mathematica/test/0.01/output_beam' + '0.txt'
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

omegar1[omegai1 < 0.02] = None
omegai1[omegai1 < 0.02] = None
# polarization[abs(polarization) < 0.01] = 0

path = '/home/kun/Downloads/data/0.01/data2/'

Nx = 2048
Ny = 4096
Npx = 4
Npy = 256
dx = 0.5
dy = 0.125
Lx = Nx * dx
Ly = Ny * dy
bz = []
v = []
font = {'family': 'serif',
        'weight': 'normal',
        'size': 16
        }
for t in [2000, 5000, 8000]:
    Bz = data.read_bz(path, t, Nx, Ny, Npx, Npy)
    Bz = np.array(Bz)

    print(len(Bz))
    a = np.sqrt(len(Bz))
    # a = int(a)
    # Bz = Bz.transpose(2,0,1).reshape(512,-1)
    # Bz = Bz.T
    # Bz1 = Bz.transpose(0, 1, 2).reshape(-1, 64)  # 64 is the x points in x direction per core
    Bz1 = Bz.transpose(0, 1, 2).reshape(-1, int(Nx / Npx))
    # Bz = Bz1[:128, :]  # 128 is the y point in y direction in domain
    Bz = Bz1[:Ny, :]
    for i in range(1, Npx):
        # Bz = np.concatenate((Bz, Bz1[128 * i:128 * (i + 1), :]), axis=1)
        Bz = np.concatenate((Bz, Bz1[Ny * i:Ny * (i + 1), :]), axis=1)

    Z = fft2(Bz)
    Z = Z / len(Bz[:, 0]) / len(Bz[0, :])
    yf = fftfreq(len(Bz[:, 0]), dy) * 2 * np.pi
    xf = fftfreq(len(Bz[0, :]), dx) * 2 * np.pi
    xf = fftshift(xf)
    yf = fftshift(yf)
    bz.append(Z)

t = [2000, 5000, 8000]
for i1 in t:
    index = 0
    i, vx, vy, vz, x, y = data.read_phase(path, i1)
    i = np.array(i)
    vx = np.array(vx)
    vy = np.array(vy)
    vz = np.array(vz)
    x = np.array(x)
    y = np.array(y)
    for j in i:
        if j == 0:
            index += 1
        else:
            break
    # vx_pui = vx[:index - 1]
    # vy_pui = vy[:index - 1]
    # vz_pui = vz[:index - 1]
    # x_pui = x[:index - 1]

    # background proton
    # vx = vx[index + 1:]
    # vy = vy[index + 1:]
    # vz = vz[index + 1:]
    # x = x[index + 1:]

    # pick-up ions perpendicular velocity and parallel velocity
    # v_per = np.sqrt(vy_pui ** 2 + vz_pui ** 2)
    # v_para = vx_pui

    # # background ions
    v_b_per = np.sqrt(vy ** 2 + vz ** 2)
    v_b_para = vx
    h_b = plt.hist2d(v_b_para, v_b_per, bins=200, range=[[-12, 12], [0, 20]])
    value_b, xedges_b, yedges_b = h_b[0], h_b[1], h_b[2]
    value_b[value_b == 0] = np.nan
    v.append(value_b)

    # font1 = {'family': 'Times New Roman',
    #          'weight': 'normal',
    #          'size': 14}
    # font2 = {'family': 'Times New Roman',
    #          'weight': 'normal',
    #          'size': 14}
    # # x = np.linspace(0, 256, 512)
    # x = np.linspace(0, Lx, Nx)
    # # y = np.linspace(0, 64, 128)
    # y = np.linspace(0, Ly, Ny)
    # fig = plt.figure(figsize=(22, 6))
    # plt.subplot(131)
    # plt.pcolormesh(x, y, bz[0], cmap='jet')
    # cb1 = plt.colorbar()
    # cb1.ax.tick_params(labelsize=12)
    # cb1.formatter.set_powerlimits((-2, 0))
    # plt.xlabel('$x[d_i]$', font1)
    # plt.ylabel('$y[d_i]$', font2)
    # plt.tick_params(labelsize=12)
    # plt.title(r'$t\Omega_i=$' + str(t * 0.025), fontsize=16)

norm = matplotlib.colors.Normalize(vmin=-12, vmax=-4)
fig = plt.figure(figsize=(18, 6))
plt.subplot(131)
h1 = plt.pcolormesh(xf[int(len(xf) / 2):], yf[int(len(yf) / 2):],
                    np.log10(np.abs(bz[0][:int(len(bz[0][:, 0]) / 2), :int(len(bz[0][0, :]) / 2)]) ** 2), cmap='jet',
                    norm=norm)
x = np.linspace(0.01, 0.6, nx+nx1)
y = np.linspace(0., 12, 1200)
X, Y = np.meshgrid(x, y)
contour = plt.contour(X, Y, omegai1, [0.16], colors='w', alpha=1, Nchunk=0, linestyles='dotted', linewidths=2)
plt.clabel(contour, inline=1, fontsize=12)
plt.xlabel(r'$k_{\parallel}\lambda_p$', font=font)
plt.ylabel(r'$k_{\perp}\lambda_p$', font=font)
plt.tick_params(labelsize=16)
plt.xlim(0, 1)
plt.ylim(0, 12)
plt.title('(a) $t\Omega_p=$'+str(0.01*t[0]), font=font)
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
plt.subplot(132)
x = np.linspace(0.01, 0.6, nx+nx1)
y = np.linspace(0., 8, 800)
X, Y = np.meshgrid(x, y)
h2 = plt.pcolormesh(xf[int(len(xf) / 2):], yf[int(len(yf) / 2):],
                    np.log10(np.abs(bz[1][:int(len(bz[1][:, 0]) / 2), :int(len(bz[1][0, :]) / 2)]) ** 2), cmap='jet',
                    norm=norm)
plt.xlabel(r'$k_{\parallel}\lambda_p$', font=font)
contour = plt.contour(X, Y, omegai1, [0.04], colors='k', alpha=1, Nchunk=0, linestyles='dotted', linewidths=2)
plt.tick_params(labelsize=16)
plt.xlim(0, 1)
plt.ylim(0, 12)
plt.title('(b) $t\Omega_p=$'+str(0.01*t[1]), font=font)
plt.subplot(133)
h3 = plt.pcolormesh(xf[int(len(xf) / 2):], yf[int(len(yf) / 2):],
                    np.log10(np.abs(bz[2][:int(len(bz[2][:, 0]) / 2), :int(len(bz[2][0, :]) / 2)]) ** 2), cmap='jet',
                    norm=norm)
plt.xlabel(r'$k_{\parallel}\lambda_p$', font=font)
plt.tick_params(labelsize=16)
plt.xlim(0, 1)
plt.ylim(0, 12)
plt.title('(c) $t\Omega_p=$'+str(0.01*t[2]), font=font)
# colorbar left blow width and hight
l = 0.92
b = 0.12
w = 0.015
h = 1 - 2 * b

rect = [l, b, w, h]
cbar_ax = fig.add_axes(rect)
cb = plt.colorbar(h3, cax=cbar_ax)
cb.ax.tick_params(labelsize=16)
plt.savefig(path + 'bz.png', bbox_inches='tight')

# set colorbar parameters
# cb.ax.tick_params(labelsize=16)
# cb.set_label('', fontdict=font)

# cb2 = plt.colorbar()
# cb2.ax.tick_params(labelsize=12)
# # cb2.set_label('colorbar', fontdict=font1)
# # cb2.formatter.set_powerlimits((-2, 0))
# plt.xlabel('$x[d_i]$', font1)
# plt.ylabel('$y[d_i]$', font2)
# plt.tick_params(labelsize=12)
# plt.xlabel(r'$k_xd_i$')
# plt.ylabel(r'$k_yd_i$')
# plt.xlim(0, 1.0)
# plt.ylim(0, 2.0)
# plt.title(r'$t\Omega_i=$' + str(t * 0.025), fontsize=16)
# plt.savefig(path + 'bz' + str(t) + '.png')

fig = plt.figure(figsize=(18, 4))
palette = plt.cm.jet
# Bad values (i.e., masked, nan, set to grey 0.8)
palette.set_bad('w', 1.0)
# value_b = np.ma.masked_values(value_b, value_b == 0)
X_b, Y_b = np.meshgrid(xedges_b[:-1], yedges_b[:-1])
# ax.set_facecolor('orange')
norm = matplotlib.colors.Normalize(vmin=-7, vmax=0)
plt.subplot(131)
col = plt.pcolormesh(X_b, Y_b, np.log10((v[0] / (index * yedges_b[1:]))).T, shading='gouraud', cmap=palette, vmax=0,
                     norm=norm)
plt.xlabel('$v_\parallel/v_A$', font=font)
plt.ylabel('$v_\perp/v_A$', font=font)
plt.tick_params(labelsize=16)
plt.ylim(0, 12)
plt.gca().set_aspect('equal', adjustable='box')
plt.title(r'(a) $t\Omega_p=$' + str(t[0] * 0.01), font=font)

plt.subplot(132)
col1 = plt.pcolormesh(X_b, Y_b, np.log10((v[1] / (index * yedges_b[1:]))).T, shading='gouraud', cmap=palette, vmax=0,
                      norm=norm)
plt.xlabel('$v_\parallel/v_A$', font=font)
plt.tick_params(labelsize=16)
plt.ylim(0, 12)
plt.gca().set_aspect('equal', adjustable='box')
# plt.ylabel('$v_\perp$')
plt.title(r'(b) $t\Omega_p=$' + str(t[1] * 0.01), font=font)
plt.subplot(133)
col2 = plt.pcolormesh(X_b, Y_b, np.log10((v[2] / (index * yedges_b[1:]))).T, shading='gouraud', cmap=palette, vmax=0,
                      norm=norm)
plt.xlabel('$v_\parallel/v_A$', font=font)
plt.tick_params(labelsize=16)
plt.ylim(0, 12)
plt.gca().set_aspect('equal', adjustable='box')
# plt.ylabel('$v_\perp$')
plt.title(r'(c) $t\Omega_p=$' + str(t[2] * 0.01), font=font)
cbar_ax = fig.add_axes(rect)
cb = plt.colorbar(h3, cax=cbar_ax)
cb.ax.tick_params(labelsize=16)
plt.savefig(path + 'phase.png', bbox_inches='tight')
plt.show()
