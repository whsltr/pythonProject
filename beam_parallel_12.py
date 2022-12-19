import matplotlib.pyplot as plt
import numpy as np
from cmath import pi, sin, cos, sqrt
from cmath import exp
from scipy import special
from matplotlib.collections import LineCollection
from matplotlib import colors
# from matplotlib import colorbar

path = '/home/kun/Documents/mathematics/beam/0.01/10va/theta_0/'
f = open(path + '1_2.txt', 'w')
# n1 is the ring-beam, n0 is the background protons, n2 is the electron
n1 = 0.01
n0 = 1. - n1
n2 = 1.

m0 = 1.
m1 = 1.
m2 = -1836.
m22 = 1836

delta1 = 10 ** (-5)
omegar = []
omegai = []
polarization = []

dx = 0.01
xmin = 0.01
xmax = 0.2

nx = int((xmax - xmin) / dx)

vtpar0 = sqrt(0.3)
# vtpar0=0.7/math.sqrt(2)
vtper0 = vtpar0
vtpar1 = sqrt(0.01)
# vtpar1=0.2/math.sqrt(2)
vtper1 = vtpar1
vtpar2 = sqrt(0.3 * 1836)
vtper2 = vtpar2

cva = 15000

theta = 0
vr0 = 0
vr1 = 10 * sin((theta / 360) * (2 * pi))
vd1 = 10 * cos((theta / 360) * (2 * pi))
print(vd1)
vd0 = -vd1 * (n1 / n0)
vr2 = 0
vd2 = 0
b1 = vr1 / (sqrt(2)*vtper1)


def Max(x, y):
    if x >= y.real:
        return x
    else:
        return y


ymin = Max(0, b1-5)
ymax = b1+5

bs0 = 0
bs1 = vr1 / (sqrt(2) * vtper1)
bs2 = 0

As0 = 1
As1 = exp(-bs1 ** 2) + sqrt(pi) * bs1 * special.erfc(-bs1)
As2 = 1

nmin = 1
nmax = 1


# ymin = max(0, bs1 - 5)
# ymax = bs1 + 5


def Z(x):
    if abs(x) < 25:
        return 1j * sqrt(pi) * special.erfc(-1j * x) * exp(-x * x)
    elif x.imag > 1 / abs(x.real):
        return -1 / x * (1 + 1 / (2 * x * x) + 3 / (4 * x ** 4) + 15 / (8 * x ** 6))
    elif abs(x.imag) < 1 / abs(x.real):
        return 1j * np.sqrt(pi) * np.exp(-x * x) - 1 / x * (1 + 1 / (2 * x * x) + 3 / (4 * x ** 4) + 15 / (8 * x ** 6))
    else:
        return 1j * 2 * np.sqrt(pi) * np.exp(-x * x) - 1 / x * (
                1 + 1 / (2 * x * x) + 3 / (4 * x ** 4) + 15 / (8 * x ** 6))


def DZ(x):
    return -2 * (1 + x * Z(x))


def Iv(nn, x):
    return special.iv(nn, x)


def DIv(nn, x):
    return (Iv(nn - 1, x) + Iv(nn + 1, x)) / 2


def delta(x, nn):
    if x-nn == 0:
        return 1
    else:
        return 0


def fun1(yy):
    return (yy / 2)**2 * exp(-(yy-vr1/(sqrt(2)*vtper1))**2) * yy


def fun2(yy):
    return (yy / 2)**2 * exp(-(yy-vr1/(sqrt(2)*vtper1))**2) * (yy-vr1/(sqrt(2)*vtper1))


omega_0r = .01
omega_0i = 0.05
omega0 = omega_0r + 0.001 + 1j * (omega_0i + 0.0011)
omega1 = omega_0r + 0.002 + 1j * (omega_0i + 0.0012)
omega2 = omega_0r + 0.003 + 1j * (omega_0i + 0.0013)
om = [omega0, omega1, omega2]
k1 = xmin
while k1 < xmax:
    err = 1
    c = 1
    it = 0

    while delta1 < err < 1000:
        h1 = om[2] - om[1]
        h0 = om[1] - om[0]
        fu = []

        for omega in om:
            kk = k1 * k1
            D11 = omega * omega / (cva * cva) - k1 * k1
            D12 = 0
            D21 = 0
            D22 = omega * omega / (cva * cva) - kk

            n = nmin
            while n < nmax + 1:
                eta0 = (omega - k1 * vd0 - n * m0) / (sqrt(2) * k1 * vtpar0)
                eta1 = (omega - k1 * vd1 - n * m1) / (sqrt(2) * k1 * vtpar1)
                eta2 = (omega - k1 * vd2 - n * m2) / (sqrt(2) * k1 * vtpar2)
                P0 = 1/8 * (delta(n, -1) + delta(n, 1))
                Q0 = 1/8 * (delta(n, -1) + delta(n, 1))
                P2 = 1/8 * (delta(n, -1) + delta(n, 1))
                Q2 = 1/8 * (delta(n, -1) + delta(n, 1))
                P1 = ((ymax-ymin)/300) * (fun1(ymin)+fun1(ymax) + 4*sum(fun1(ymin+(2*j-1)*((ymax-ymin)/100)) for j in range(1, 51))
                                        + 2*sum(fun1(ymin+(2*j)*((ymax-ymin)/100)) for j in range(1, 50))) * 8*P0
                Q1 = ((ymax-ymin)/300) * (fun2(ymin)+fun2(ymax) + 4*sum(fun2(ymin+(2*j-1)*((ymax-ymin)/100)) for j in range(1, 51))
                                        + 2*sum(fun2(ymin+(2*j)*((ymax-ymin)/100)) for j in range(1, 50))) * 8*P0

                sc11 = -2 * n * n * ((vtper0 ** 2) / (vtpar0 ** 2) * P0 * DZ(eta0)
                                                   + 2 * Q0 * (
                                                           1 - n * m0 / (sqrt(2) * k1 * vtpar0) * Z(eta0)))
                sb11 = -2 * n * n / As1 * ((vtper1 ** 2) / (vtpar1 ** 2) * P1 * DZ(eta1)
                                                   + 2 * Q1 * (
                                                           1 - n * m1 / (sqrt(2) * k1 * vtpar1) * Z(eta1)))
                se11 = -2 * n * n * ((vtper2 ** 2) / (vtpar2 ** 2) * P2 * DZ(eta2)
                                                   + 2 * Q2 * (
                                                           1 - n * m2 / (sqrt(2) * k1 * vtpar2) * Z(eta2)))
                sc12 = -2 * n * 1j * ((vtper0 ** 2) / (vtpar0 ** 2) * P0 * DZ(eta0)
                                                   + 2 * Q0 * (
                                                           1 - n * m0 / (sqrt(2) * k1 * vtpar0) * Z(eta0)))
                sb12 = -2 * n * 1j / As1 * ((vtper1 ** 2) / (vtpar1 ** 2) * P1 * DZ(eta1)
                                                   + 2 * Q1 * (
                                                           1 - n * m1 / (sqrt(2) * k1 * vtpar1) * Z(eta1)))
                se12 = -2 * n * 1j * ((vtper2 ** 2) / (vtpar2 ** 2) * P2 * DZ(eta2)
                                                   + 2 * Q2 * (
                                                           1 - n * m2 / (sqrt(2) * k1 * vtpar2) * Z(eta2)))
                sc21 = -sc12
                sb21 = -sb12
                se21 = -se12
                sc22 = -2 * ((vtper0 ** 2) / (vtpar0 ** 2) * P0 * DZ(eta0) + 2 * Q0 * (1
                            - n * m0 / (sqrt(2) * k1 * vtpar0) * Z(eta0)))
                sb22 = -2 / As1 * ((vtper1 ** 2) / (vtpar1 ** 2) * P1 * DZ(eta1) + 2 * Q1 * (1
                            - n * m1 / (sqrt(2) * k1 * vtpar1) * Z(eta1)))
                se22 = -2 * ((vtper2 ** 2) / (vtpar2 ** 2) * P2 * DZ(eta2) + 2 * Q2 * (1
                            - n * m2 / (sqrt(2) * k1 * vtpar2) * Z(eta2)))

                D11 = D11 + (n0 / m0 * sc11 + n1 / m1 * sb11 + n2 * m22 * se11)
                D12 = D12 + (n0 / m0 * sc12 + n1 / m1 * sb12 + n2 * m22 * se12)
                D21 = D21 + (n0 / m0 * sc21 + n1 / m1 * sb21 + n2 * m22 * se21)
                D22 = D22 + (n0 / m0 * sc22 + n1 / m1 * sb22 + n2 * m22 * se22)
                n = n + 1

            fun = D11*D22-D12*D21
            fu.append(fun)

        fu0 = fu[0]
        fu1 = fu[1]
        fu2 = fu[2]

        d0 = (fu1 - fu0) / (omega1 - omega0)
        d1 = (fu2 - fu1) / (omega2 - omega1)
        a = (d1 - d0) / (h1 + h0)
        b = a * h1 + d1
        c = fu2
        des = b + sqrt(b * b - 4 * a * c) if abs(b + sqrt(b * b - 4 * a * c)) > abs(
            b - sqrt(b * b - 4 * a * c)) else b - sqrt(b * b - 4 * a * c)
        omega3 = omega2 + ((-1 * 2 * c) / des)
        err = abs((omega3 - omega2) / omega3) if abs(c) < 10 ** -5 else 1
        omega0 = omega1
        omega1 = omega2
        omega2 = omega3
        om = [omega0, omega1, omega2]
        it = it + 1
        err = 0 if it > 50 else err
    pl = -1j * omega2.real / abs(omega2.real) * -D11 / D12
    print("{:.3f}".format(k1), '  ', omega2, "  ", pl.real, '   c=', c, "   eta1=",
          (omega2.real - k1 * vd1 - nmin * m1) / (sqrt(2) * k1 * vtpar1))
    print("{:.3f}".format(k1), '  ', omega2.real, '  ', omega2.imag, '  ', pl.real, '  ',
          ((omega2.real - k1 * vd1 + 1 * m1) / (sqrt(2) * k1 * vtpar1)).real, file=f)

    k1 = k1 + dx
    omegar.append(omega2.real)
    omegai.append(omega2.imag)
    polarization.append(pl.real)

f.close()

omegar = np.array(omegar)
omegai = np.array(omegai)
x = np.linspace(xmin, xmax, nx)
segments = [np.column_stack([x[i:i+2], omegar[i:i+2]]) for i in range(len(x)-1)]

fig, ax = plt.subplots()
ax.axis([xmin, xmax, -8, 8])
# norm = colors.Normalize(vmin=-0.04, vmax=0.0230)
lc = LineCollection(segments, cmap='jet', array=omegai, linewidth=4, )
line = ax.add_collection(lc)
# plt.colorbar(line, label='$\gamma$')
# plt.scatter(x, omegar, c=omegai, cmap='jet', norm=norm)
# plt.colorbar(label='$\gamma$')
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14}
ax.set_xlabel('$k \lambda_p$', font=font1)
ax.set_ylabel('$\omega / \Omega_p$', font=font1)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.set_title(r'$\alpha = 30 \degree$', fontsize=18)
cb = plt.colorbar(line, label='$\gamma$')
cb.ax.tick_params(labelsize=14)
cb.set_label(r'$\gamma/ \Omega_p$', fontdict=font1)
plt.show()
