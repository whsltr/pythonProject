import numpy as np
import matplotlib.pyplot as plt
from cmath import pi, sin, cos, sqrt
from cmath import exp
from scipy import special
import scipy

# n1 is the ring-beam, no is the background protons, n2 is the electron
n1 = 0.01
n0 = 1. - n1
n2 = 1.

m0 = 1.
m1 = 1.
m2 = -1836.
m22 = 1836

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
vd0 = -vd1 * (n1 / n0)
vr2 = 0
vd2 = 0

bs0 = 0
bs1 = vr1 / (sqrt(2) * vtper1)
bs2 = 0

As0 = 1
As1 = exp(-bs1 ** 2) + sqrt(pi) * bs1 * special.erfc(-bs1)
As2 = 1


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


def A(nn, x):
    return 1 / 2 * exp(-x * x / 2) * Iv(nn, x * x / 2)


def B(nn, x):
    return x / 4 * exp(-x * x / 2) * (DIv(nn, x * x / 2) - Iv(nn, x * x / 2))


def C(nn, x):
    return exp(-x * x / 2) * (nn * nn / (2 * x * x) * Iv(nn, x * x / 2) -
                              x * x / 4 * DIv(nn, x * x / 2) + x * x / 4 * Iv(nn, x * x / 2))


omegar = []
omegai = []
polarization = []
k2 = 0.01
dx = 0.002
xmax = 0.2
xmin = 0.1
nx = int((xmax - xmin) / dx)
kmax = 0.02
kmin = k2
dy = 0.01
ny = int((kmax - k2) / dy + 0.9)
while k2 < kmax:
    as0 = sqrt(2) * k2 * vtper0 / m0
    as1 = sqrt(2) * k2 * vtper1 / m1
    as2 = sqrt(2) * k2 * vtper2 / m2

    omega_0r = 0.1
    omega_0i = 0.001
    omega0 = omega_0r + 0.001 + 1j * (omega_0i)
    omega1 = omega_0r + 0.002 + 1j * (omega_0i)
    omega2 = omega_0r + 0.003 + 1j * (omega_0i)
    om = [omega0, omega1, omega2]
    k1 = xmin
    it = 0
    while k1 < xmax:
        err = 1
        delta = 10 ** (-4)
        c = 1


        def eta0(omega_f, nn):
            return (omega_f - k1 * vd0 - nn * m0) / (sqrt(2) * k1 * vtpar0)


        def eta1(omega_f, nn):
            return (omega_f - k1 * vd1 - nn * m1) / (sqrt(2) * k1 * vtpar1)


        def eta2(omega_f, nn):
            return (omega_f - k1 * vd2 - nn * m2) / (sqrt(2) * k1 * vtpar2)


        while delta < err < 1000:
            h1 = om[2] - om[1]
            h0 = om[1] - om[0]
            fu = []

            for omega in om:
                kk = k1 * k1 + k2 * k2

                nmin = -1
                nmax = 1
                sc11 = sum(-2 * n * n / (as0 * as0) * ((vtper0 ** 2) / (vtpar0 ** 2) * A(n, as0) * DZ(eta0(omega, n))
                                                       + 2 * A(n, as0) * (1 - n * m0 / (sqrt(2) * k1 * vtpar0) * Z(
                            eta0(omega, n)))) for n in range(nmin, nmax + 1))
                sb11 = sum(-2 * n * n / (as1 * as1) * ((vtper1 ** 2) / (vtpar1 ** 2) * A(n, as1) * DZ(eta1(omega, n))
                                                       + 2 * A(n, as1) * (1 - n * m1 / (sqrt(2) * k1 * vtpar1) * Z(
                            eta1(omega, n)))) for n in range(nmin, nmax + 1))
                se11 = sum(-2 * n * n / (as2 * as2) * ((vtper2 ** 2) / (vtpar2 ** 2) * A(n, as2) * DZ(eta2(omega, n))
                                                       + 2 * A(n, as2) * (1 - n * m2 / (sqrt(2) * k1 * vtpar2) * Z(
                            eta2(omega, n)))) for n in range(nmin, nmax + 1))
                sc12 = sum(1j * 4 * n / as0 * ((vtper0 ** 2) / (vtpar0 ** 2) * B(n, as0) * eta0(omega, n)
                                               + B(n, as0) * (
                                                       n * m0 / (sqrt(2) * k1 * vtpar0))) * Z(
                    eta0(omega, n)) for n in range(nmin, nmax + 1))
                sb12 = sum(1j * 4 * n / as1 * ((vtper1 ** 2) / (vtpar1 ** 2) * B(n, as1) * eta1(omega, n)
                                               + B(n, as1) * (
                                                       n * m1 / (sqrt(2) * k1 * vtpar1))) * Z(
                    eta1(omega, n)) for n in range(nmin, nmax + 1))
                se12 = sum(1j * 4 * n / as2 * ((vtper2 ** 2) / (vtpar2 ** 2) * B(n, as2) * eta2(omega, n)
                                               + B(n, as2) * (
                                                       n * m2 / (sqrt(2) * k1 * vtpar2))) * Z(
                    eta2(omega, n)) for n in range(nmin, nmax + 1))
                sc13 = sum(-2 * n / as0 * (vtper0 / vtpar0) * (
                        (omega - n * m0) / (sqrt(2) * k1 * vtpar0) * A(n, as0) * DZ(eta0(omega, n))
                        - 2 * (vtpar0 ** 2) / (vtper0 ** 2) * n * m0 / (sqrt(2) * k1 * vtpar0) * A(n, as0)
                        * (1 + (omega - n * m0) / (sqrt(2) * k1 * vtpar0) * Z(eta0(omega, n)))) for n in
                           range(nmin, nmax + 1))
                sb13 = sum(-2 * n / as1 * (vtper1 / vtpar1) * (
                        (omega - n * m1) / (sqrt(2) * k1 * vtpar1) * A(n, as1) * DZ(eta1(omega, n))
                        - 2 * (vtpar1 ** 2) / (vtper1 ** 2) * n * m1 / (sqrt(2) * k1 * vtpar1) * A(n, as1)
                        * (1 + (omega - n * m1) / (sqrt(2) * k1 * vtpar1) * Z(eta1(omega, n)))) for n in
                           range(nmin, nmax + 1))
                se13 = sum(-2 * n / as2 * (vtper2 / vtpar2) * (
                        (omega - n * m2) / (sqrt(2) * k1 * vtpar2) * A(n, as2) * DZ(eta2(omega, n))
                        - 2 * (vtpar2 ** 2) / (vtper2 ** 2) * n * m2 / (sqrt(2) * k1 * vtpar2) * A(n, as2)
                        * (1 + (omega - n * m2) / (sqrt(2) * k1 * vtpar2) * Z(eta2(omega, n)))) for n in
                           range(nmin, nmax + 1))
                sc21 = -sc12
                sb21 = -sb12
                se21 = -se12
                sc22 = sum(-2 * ((vtper0 ** 2) / (vtpar0 ** 2) * C(n, as0) * DZ(eta0(omega, n))
                                 + 2 * C(n, as0) * (1 - n * m0 / (sqrt(2) * k1 * vtpar0) * Z(
                            eta0(omega, n)))) for n in range(nmin, nmax + 1))
                sb22 = sum(-2 * ((vtper1 ** 2) / (vtpar1 ** 2) * C(n, as1) * DZ(eta1(omega, n))
                                 + 2 * C(n, as1) * (1 - n * m1 / (sqrt(2) * k1 * vtpar1) * Z(
                            eta1(omega, n)))) for n in range(nmin, nmax + 1))
                se22 = sum(-2 * ((vtper2 ** 2) / (vtpar2 ** 2) * C(n, as2) * DZ(eta2(omega, n))
                                 + 2 * C(n, as2) * (1 - n * m2 / (sqrt(2) * k1 * vtpar2) * Z(
                            eta2(omega, n)))) for n in range(nmin, nmax + 1))
                sc23 = sum(1j * 2 * vtper0 / vtpar0 * (
                        (omega - n * m0) / (sqrt(2) * k1 * vtpar0) * B(n, as0) * DZ(eta0(omega, n))
                        - 2 * (vtpar0 ** 2) / (vtper0 ** 2) * n * m0 / (sqrt(2) * k1 * vtpar0) * B(n, as0)
                        * (1 + (omega - n * m0) / (sqrt(2) * k1 * vtpar0) * Z(eta0(omega, n)))) for n in
                           range(nmin, nmax + 1))
                sb23 = sum(1j * 2 * vtper1 / vtpar1 * (
                        (omega - n * m1) / (sqrt(2) * k1 * vtpar1) * B(n, as1) * DZ(eta1(omega, n))
                        - 2 * (vtpar1 ** 2) / (vtper1 ** 2) * n * m1 / (sqrt(2) * k1 * vtpar1) * B(n, as1)
                        * (1 + (omega - n * m1) / (sqrt(2) * k1 * vtpar1) * Z(eta1(omega, n)))) for n in
                           range(nmin, nmax + 1))
                se23 = sum(1j * 2 * vtper2 / vtpar2 * (
                        (omega - n * m2) / (sqrt(2) * k1 * vtpar2) * B(n, as2) * DZ(eta2(omega, n))
                        - 2 * (vtpar2 ** 2) / (vtper2 ** 2) * n * m2 / (sqrt(2) * k1 * vtpar2) * B(n, as2)
                        * (1 + (omega - n * m2) / (sqrt(2) * k1 * vtpar2) * Z(eta2(omega, n)))) for n in
                           range(nmin, nmax + 1))
                sc31 = sc13
                sb31 = sb13
                se31 = se13
                sc32 = -sc23
                sb32 = -sb23
                se32 = -se23
                sc33 = sum(-2 * (
                        ((omega - n * m0) / (sqrt(2) * k1 * vtpar0)) ** 2 * A(n, as0) * DZ(eta0(omega, n))
                        - 2 * (vtpar0 ** 2) / (vtper0 ** 2) * (omega - n * m0) / (sqrt(2) * k1 * vtpar0) * n * m0 / (
                                sqrt(2) * k1 * vtpar0) * A(n, as0)
                        * (1 + (omega - n * m0) / (sqrt(2) * k1 * vtpar0) * Z(eta0(omega, n)))) for n in
                           range(nmin, nmax + 1))
                sb33 = sum(-2 * (
                        ((omega - n * m1) / (sqrt(2) * k1 * vtpar1)) ** 2 * A(n, as1) * DZ(eta1(omega, n))
                        - 2 * (vtpar1 ** 2) / (vtper1 ** 2) * (omega - n * m1) / (sqrt(2) * k1 * vtpar1) * n * m1 / (
                                sqrt(2) * k1 * vtpar1) * A(n, as1)
                        * (1 + (omega - n * m1) / (sqrt(2) * k1 * vtpar1) * Z(eta1(omega, n)))) for n in
                           range(nmin, nmax + 1))
                se33 = sum(-2 * (
                        ((omega - n * m2) / (sqrt(2) * k1 * vtpar2)) ** 2 * A(n, as2) * DZ(eta2(omega, n))
                        - 2 * (vtpar2 ** 2) / (vtper2 ** 2) * (omega - n * m2) / (sqrt(2) * k1 * vtpar2) * n * m2 / (
                                sqrt(2) * k1 * vtpar2) * A(n, as2)
                        * (1 + (omega - n * m2) / (sqrt(2) * k1 * vtpar2) * Z(eta2(omega, n)))) for n in
                           range(nmin, nmax + 1))

                D11 = omega * omega / (cva * cva) - k1 * k1 + (
                        n0 * m0 * sc11 + n1 * m1 * sb11 + n2 * m22 * se11)
                D12 = (n0 * m0 * sc12 + n1 * m1 * sb12 + n2 * m22 * se12)
                D13 = k1 * k2 + (n0 * m0 * sc13 + n1 * m1 * sb13 + n2 * m22 * se13)
                D21 = (n0 * m0 * sc21 + n1 * m1 * sb21 + n2 * m22 * se21)
                D22 = omega * omega * (cva * cva) - kk + (
                        n0 * m0 * sc22 + n1 * m1 * sb22 + n2 * m22 * se22)
                D23 = (n0 * m0 * sc23 + n1 * m1 * sb23 + n2 * m22 * se23)
                D31 = k1 * k2 + (n0 * m0 * sc31 + n1 * m1 * sb31 + n2 * m22 * se31)
                D32 = (n0 * m0 * sc32 + n1 * m1 * sb32 + n2 * m22 * se32)
                D33 = omega * omega * (cva * cva) - k2 * k2 + (
                        n0 * m0 * sc33 + n1 * m1 * sb33 + n2 * m22 * se33)
                fun = D11 * D22 * D33 + D21 * D32 * D13 + D12 * D23 * D31 - (
                        D13 * D22 * D31 + D12 * D21 * D33 + D23 * D32 * D11)
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
            err = abs((omega3 - omega2) / omega3) if abs(c) < 10 ** -4 else 1
            omega0 = omega1
            omega1 = omega2
            omega2 = omega3
            om = [omega0, omega1, omega2]
            it = it + 1
            err = 0 if it > 250 else err
        pl = -1j * omega2.real / abs(omega2.real) * (D31 * D13 - D11 * D33) / (D33 * D12 - D13 * D32)
        pl = 1j * omega2.real / abs(omega2.real) * (D11 / D13 - D21 / D23) / (D12 / D13 - D22 / D23)
        print(k1, '  ', omega2, "  ", pl.real, '   c=', c, "   eta0=", eta0(omega2, -1))
        k1 = k1 + dx
        omegar.append(omega2.real)
        omegai.append(omega2.imag)
        polarization.append(pl.real)
    print(k2)
    k2 = k2 + dy

omegar = np.array(omegar).reshape(ny, -1)
omegai = np.array(omegai).reshape(ny, -1)
polarization = np.array(polarization).reshape(ny, -1)
# omegar = omegar[::-1, :]
# omegai = omegai[::-1, :]
omegai[omegai < 0.01] = None

font1 = {'family': 'Computer Modern Roman',
         'weight': 'normal',
         'size': 14}
x = np.linspace(xmin, xmax, nx)
y = np.linspace(kmin, kmax, ny)
X, Y = np.meshgrid(x, y)
fig = plt.figure()
plt.pcolormesh(X, Y, omegai, shading='gouraud', cmap='jet')
cb = plt.colorbar()
cb.ax.tick_params(labelsize=12)
cb.set_label(r'$\gamma/ \Omega_i$', fontdict=font1)
contour = plt.contour(x, y, polarization, 8, cmap='cool', alpha=1, vmin=-0.6, vmax=0.8, Nchunk=0.5,
                      linestyles='dotted')
plt.clabel(contour, inline=1, )
# contour = plt.contourf(X, Y, omegai, 10, cmap='cool', vmin=0.0, vmax=0.12)
plt.xlabel(r'$k_\parallel / k_i$', font1)
plt.ylabel(r'$k_\perp / k_i$', font1)
plt.tick_params(labelsize=12)
plt.show()
