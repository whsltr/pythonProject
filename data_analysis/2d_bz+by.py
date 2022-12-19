import numpy as np
import matplotlib.pyplot as plt
import data_analysis.read_data as data
from scipy.fft import fft, fft2, fftfreq, fftshift

path = '/home/kun/Downloads/data/data/'

Nx = 1024
Ny = 1024
Npx = 2
Npy = 128
dx = 1
dy = 1
Lx = Nx * dx
Ly = Ny * dy
for t in range(0, 8001, 500):
    Bz = data.read_bz(path, t, Nx, Ny, Npx, Npy)
    By = data.read_by(path, t, Nx, Ny, Npx, Npy)
    Bz = np.array(Bz)
    By = np.array(By)

    print(len(Bz))
    a = np.sqrt(len(Bz))
    # a = int(a)
    # Bz = Bz.transpose(2,0,1).reshape(512,-1)
    # Bz = Bz.T
    # Bz1 = Bz.transpose(0, 1, 2).reshape(-1, 64)  # 64 is the x points in x direction per core
    Bz1 = Bz.transpose(0, 1, 2).reshape(-1, int(Nx/Npx))
    By1 = By.transpose(0, 1, 2).reshape(-1, int(Nx/Npx))
    # Bz = Bz1[:128, :]  # 128 is the y point in y direction in domain
    Bz = Bz1[:Ny, :]
    By = By1[:Ny, :]
    for i in range(1, Npx):
        # Bz = np.concatenate((Bz, Bz1[128 * i:128 * (i + 1), :]), axis=1)
        Bz = np.concatenate((Bz, Bz1[Ny * i:Ny * (i + 1), :]), axis=1)
        By = np.concatenate((By, By1[Ny * i:Ny * (i + 1), :]), axis=1)

    # Bz = Bz.transpose(1,0,2).reshape(512,-1)
    # Bz = Bz.T
    print(len(Bz))
    # x = np.linspace(0, 256, 512)
    x = np.linspace(0, Lx, Nx)
    # y = np.linspace(0, 64, 128)

    Z = fft2(Bz) + fft2(By)
    Z = Z / len(Bz[:, 0]) / len(Bz[0, :])
    yf = fftfreq(len(Bz[:, 0]), dy) * 2 * np.pi
    xf = fftfreq(len(Bz[0, :]), dx) * 2 * np.pi
    xf = fftshift(xf)
    yf = fftshift(yf)
    plt.subplot(111)
    plt.pcolormesh(xf[int(len(xf) / 2):], yf[int(len(yf) / 2):],
                   np.log10(np.abs(Z[:int(len(Z[:, 0]) / 2), :int(len(Z[0, :]) / 2)]) ** 2), cmap='jet', vmin=-12)
    plt.colorbar()
    plt.xlabel('$k_xd_i$')
    plt.ylabel('$k_yd_i$')
    plt.xlim(0, 1.0)
    plt.ylim(0, 2.0)
    plt.title(r'$t\Omega_i=$' + str(t * 0.01))
    plt.savefig(path + 'bz+by' + str(t) + '.png')

plt.show()