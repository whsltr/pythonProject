import numpy as np
import matplotlib.pyplot as plt
import data_analysis.read_data as data
from scipy.fft import fft, fft2, fftfreq, fftshift

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

path = '/home/kun/Downloads/data/0.01/data/'

Nx = 2048
Ny = 4096
Npx = 4
Npy = 256
dx = 0.5
dy = 0.125
Lx = Nx * dx
Ly = Ny * dy
for t in range(0, 8001, 1000):
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

    # Bz = Bz.transpose(1,0,2).reshape(512,-1)
    # Bz = Bz.T
    print(len(Bz))
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16}
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16}

    # x = np.linspace(0, 256, 512)
    x = np.linspace(0, Lx, Nx)
    # y = np.linspace(0, 64, 128)
    y = np.linspace(0, Ly, Ny)
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.pcolormesh(x, y, Bz, cmap='jet')
    cb1 = plt.colorbar()
    cb1.ax.tick_params(labelsize=16)
    cb1.formatter.set_powerlimits((-2, 0))
    plt.xlabel('$x[\lambda_i]$', font1)
    plt.ylabel('$y[\lambda_i]$', font2)
    plt.tick_params(labelsize=16)
    plt.text(50, 480, 'e', color='w', fontsize=30)
    plt.title(r'$t\Omega_i=$' + str(t * 0.01), fontsize=16)

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16}
    x = np.linspace(0.01, 0.6, nx + nx1)
    y = np.linspace(0., 12, 1200)
    X, Y = np.meshgrid(x, y)
    eta = omegar1 / X / np.sqrt(0.6 * 1836)

    Z = fft2(Bz)
    Z = Z / len(Bz[:, 0]) / len(Bz[0, :])
    yf = fftfreq(len(Bz[:, 0]), dy) * 2 * np.pi
    xf = fftfreq(len(Bz[0, :]), dx) * 2 * np.pi
    xf = fftshift(xf)
    yf = fftshift(yf)
    plt.subplot(122)
    plt.pcolormesh(xf[int(len(xf) / 2):], yf[int(len(yf) / 2):],
                   np.log10(np.abs(Z[:int(len(Z[:, 0]) / 2), :int(len(Z[0, :]) / 2)]) ** 2), cmap='jet', vmin=-12)
    cb2 = plt.colorbar()
    contour = plt.contour(X, Y, omegai1, [0.15], colors='w', alpha=1, Nchunk=0, linestyles='dotted', linewidths=2.5)
    plt.clabel(contour, inline=1, fontsize=10)
    cb2.ax.tick_params(labelsize=16)
    # cb2.set_label('colorbar', fontdict=font1)
    # cb2.formatter.set_powerlimits((-2, 0))
    plt.xlabel('$x[\lambda_p]$', font1)
    plt.ylabel('$y[\lambda_p]$', font2)
    plt.tick_params(labelsize=16)
    plt.xlabel(r'$k_x\lambda_p$')
    plt.ylabel(r'$k_y\lambda_p$')
    plt.xlim(0, 1.0)
    plt.ylim(0, 12.)
    plt.text(0.1, 11, 'f', color='w', fontsize=30)
    plt.title(r'$t\Omega_p=$' + str(t * 0.01), fontsize=16)
    plt.savefig(path + 'bz' + str(t) + '.png')

plt.show()
