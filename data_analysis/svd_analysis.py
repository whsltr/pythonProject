import numpy as np
import matplotlib.pyplot as plt
import data_analysis.read_data as data
import pywt as wt
from scipy.fft import fft
from scipy import signal
path = '/home/ck/Documents/hybrid2D_PUI/data/'
# path = '/media/ck/Samsung_T5/data/data/'
t, ex, ey, ez, bx, by, bz, b_total = data.read_timeseries(path)

T = 0.02
N = np.size(t)
print(N)
t = np.array(t) * T
ex = np.array(ex)
ey = np.array(ey)
ez = np.array(ez)
bx = np.array(bx)
by = np.array(by)
bz = np.array(bz)
# fft transform
By = fft(by)
Bx = fft(bx)
Bz = fft(bz)
Ex = fft(ex)
Ey = fft(ey)
Ez = fft(ez)
ellipsis = []
poyinting = []

for i in range(1, N // 2):
    i = int(i)
    print(i)
    # select the frequency you want to analysis. frequency is uniformly arrayed 0 to 1./(2*(1/N))
    by = By[i]
    bx = 0
    bz = Bz[i]
    ex = 0
    ey = Ey[i]
    ez = Ez[i]
    poyinting.append(ey*bz - by*ez)

    # spectral matrix of the analytic signal
    s11 = bx * np.conj(bx)
    s12 = bx * np.conj(by)
    s13 = bx * np.conj(bz)
    s23 = by * np.conj(bz)
    s32 = bz + np.conj(by)
    s22 = by * np.conj(by)
    s33 = bz * np.conj(bz)
    elli = np.imag(s23) / abs(np.imag(s23))

    A = [np.real(s11), np.real(s12), np.real(s13), np.real(s12), np.real(s22), np.real(s23), np.real(s13), np.real(s23),
         np.real(s33),
         0, -np.imag(s12), -np.imag(s13), np.imag(s12), 0, -np.imag(s23), np.imag(s13), np.imag(s23), 0]
    A = np.array(A)
    A.shape = (6, 3)
    # print(A)

    # svd method,  u is the unitary array, s is the vectors with the singular values, vh is the unitary array
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    # print(s)
    # print(vh)
    # print(u)
    ellipsis.append(elli * (s[1] / s[0]))
    print(ellipsis)
    # get the rotation matrix

    # rotation = u[:3, :] * np.linalg.inv(vh)
    # print(rotation)
    # # look what rotation
    # a = [0, 1, 0]
    # a = np.array(a)
    # a.shape = (3, 1)
    # A = A[:3, :]
    # b = np.dot(A, a)
    # c = np.dot(rotation, a)
    # # b = np.dot(rotation, a)
    # print('b = ', b)
    # print(c)
x = np.linspace(0, 1 / (2 * T), N // 2 - 1)
# if ellipsis is positive wave is
plt.plot(x, ellipsis)
fig  = plt.figure()
plt.plot(x, poyinting)
plt.show()
