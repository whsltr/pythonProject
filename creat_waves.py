import numpy as np
from scipy.fft import fft, fft2, fftfreq, fftshift, rfft2, rfftfreq
import matplotlib.pyplot as plt

# k1 = np.linspace(0.5, 1.0, 5)
# k2 = np.linspace(0, 0.5, 5)
t = np.linspace(0, 500, 1000)
# t = t/(2*np.pi)
x = np.linspace(0, 500, 1000)
# x = x/(2*np.pi)
wavey = []
wavez = []
wavesy = []
wavesz = []
fy = open("/home/kun/wavesy.txt", "w")
fz = open("/home/kun/wavesz.txt", "w")

for i in t:
    for j in x:
        # Y = 0
        # for ii in k1:
        #     for jj in k2:
                # Y += np.cos(ii*i+jj*j)
        ey = np.cos(1*j+2*i)
        ez = 0.01*np.sin(1*j+2*i)
        fy.write(str(ey)+"    ")
        fz.write(str(ez)+"    ")
        wavey.append(ey)
        wavez.append(ez)
    fy.write("\n")
    fz.write("\n")
    wavesy.append(wavey)
    wavesz.append(wavez)
fy.close()
fz.close()

file_namey = '/home/kun/wavesy.txt'
file_namez = '/home/kun/wavesz.txt'
fy = open(file_namey)
fz = open(file_namez)
Bz = []
By = []
for line in fz.readlines():
    bz = []
    for i in line.split():

        bz.append(float(i))
    Bz.append(bz)
Bz = np.array(Bz)
Bz = Bz[::-1, :]
for line in fy.readlines():
    by = []
    for i in line.split():

        by.append(float(i))
    By.append(by)
By = np.array(By)
By = By[::-1, :]
Z = fft2(By+1j*Bz)
Z = Z / len(x) / len(t)
Z = fftshift(Z)
yf = fftfreq(len(Bz[:, 0]), 0.5) * 2 * np.pi
xf = fftfreq(len(Bz[0, :]), 0.5) * 2 * np.pi
xf = fftshift(xf)
yf = fftshift(yf)
print(yf)
print(xf)
# plt.subplot(122)
plt.figure()
# plt.pcolormesh(xf[int(len(xf) / 2):], yf[int(len(yf) / 2):],
#                np.log10(np.abs(Z[:int(len(Z[:, 0]) / 2), :int(len(Z[0, :]) / 2)]) ** 2), cmap='jet', )
# plt.figure()
plt.pcolormesh(xf, yf,
               np.log10(np.abs(Z) ** 2), cmap='jet', )
cb2 = plt.colorbar()
plt.xlabel(r'$k k_i$')
plt.ylabel(r'$\omega / \Omega_i$')
plt.figure()
plt.pcolormesh(x, t, By, cmap='jet')

plt.show()
print(fftfreq(10, 1))
print(fftshift(fftfreq(10, 1)))
