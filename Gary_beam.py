import numpy as np
import matplotlib.pyplot as plt
from cmath import pi, sin, cos, sqrt
from cmath import exp
from scipy import special
import scipy

# n1 is the ring-beam, no is the background protons, n2 is the electron
n1 = 0.002
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


omegar = []
omegai = []
polarization = []
k2 = 0.01
dx = 0.002
xmax = 0.15
xmin = 0.1
nx = int((xmax - xmin) / dx + 0.9)
kmax = 0.3
kmin = k2
dy = 0.01
ny = int((kmax - k2) / dy + 0.9)
while k2 < kmax:
    ac = (k2 * vtper0 / m0) ** 2
    ab = (k2 * vtper1 / m1) ** 2
    ae = (k2 * vtper2 / m2) ** 2

    omega_0r = 0.03
    omega_0i = 0.001
    omega0 = omega_0r + 0.001 + 1j * (omega_0i + 0.0001)
    omega1 = omega_0r + 0.002 + 1j * (omega_0i + 0.0002)
    omega2 = omega_0r + 0.003 + 1j * (omega_0i + 0.0003)
    om = [omega0, omega1, omega2]
    k1 = xmin
    it = 0
    while k1 < xmax:
        err = 1
        delta = 10 ** (-7)
        c = 1

        while delta < err < 1000:
            h0 = om[2] - om[1]
            h1 = om[1] - om[0]
            fu = []

            for omega in om:
                kk = k1 * k1 + k2 * k2
                xic0 = (omega - k1 * vd0) / (sqrt(2) * k1 * vtpar0)
                xib0 = (omega - k1 * vd1) / (sqrt(2) * k1 * vtpar1)
                xie0 = (omega - k1 * vd2) / (sqrt(2) * k1 * vtpar2)
                sc11 = 0
                sb11 = 0
                se11 = 0
                sc12 = 0
                sb12 = 0
                se12 = 0
                sc13 = 0
                sb13 = 0
                se13 = 0
                sc22 = 0
                sb22 = 0
                se22 = 0
                sc23 = 0
                sb23 = 0
                se23 = 0
                sc33 = 0
                sb33 = 0
                se33 = 0

                for n in range(-1, 2):
                    xic = (omega - k1 * vd0 - n * m0) / (sqrt(2) * k1 * vtpar0)
                    xib = (omega - k1 * vd1 - n * m1) / (sqrt(2) * k1 * vtpar1)
                    xie = (omega - k1 * vd2 - n * m2) / (sqrt(2) * k1 * vtpar2)

                    sc11 += ((2 * ac) * (Iv(n, ac) - DIv(n, ac)) + n * n / ac * Iv(n, ac)) * Z(xic)
                    sb11 += ((2 * ab) * (Iv(n, ab) - DIv(n, ab)) + n * n / ab * Iv(n, ab)) * Z(xib)
                    se11 += ((2 * ae) * (Iv(n, ae) - DIv(n, ae)) + n * n / ae * Iv(n, ae)) * Z(xie)
                    sc12 += n * (Iv(n, ac) - DIv(n, ac)) * Z(xic)
                    sb12 += n * (Iv(n, ab) - DIv(n, ab)) * Z(xib)
                    se12 += n * (Iv(n, ae) - DIv(n, ae)) * Z(xie)
                    sc13 += (Iv(n, ac) - DIv(n, ac)) * DZ(xic)
                    sb13 += (Iv(n, ab) - DIv(n, ab)) * DZ(xib)
                    se13 += (Iv(n, ae) - DIv(n, ae)) * DZ(xie)
                    sc22 += n * n * Iv(n, ac) * Z(xic)
                    sb22 += n * n * Iv(n, ab) * Z(xib)
                    se22 += n * n * Iv(n, ae) * Z(xie)
                    sc23 += n * Iv(n, ac) * DZ(xic)
                    sb23 += n * Iv(n, ab) * DZ(xib)
                    se23 += n * Iv(n, ae) * DZ(xie)
                    sc33 += Iv(n, ac) * xic * DZ(xic)
                    sb33 += Iv(n, ab) * xib * DZ(xib)
                    se33 += Iv(n, ae) * xie * DZ(xie)

                ele11 = n0 * sc11 * xic0 * exp(-ac) + n1 * sb11 * xib0 * exp(-ab) + n2 * se11 * xie0 * exp(-ae) * m22
                ele12 = 1j * n0 * sc12 * xic0 * exp(-ac) + 1j * n1 * sb12 * xib0 * exp(
                    -ab) + 1j * n2 * se12 * xie0 * exp(-ae) * m22
                ele13 = 1j * n0 * sc13 * xic0 * abs(k2) / (sqrt(2) * k2) * k2 * vtper0 / m0 * exp(
                    -ac) + 1j * n1 * sb13 * xib0 * abs(k2) / (sqrt(2) * k2) \
                        * k2 * vtper1 / m1 * exp(-ab) + 1j * n2 * m22 * se13 * xie0 * abs(k2) / (
                                sqrt(2) * k2) * k2 * vtper2 / m2 * exp(-ae)
                ele21 = -ele12
                ele22 = n0 * sc22 * xic0 * exp(-ac) / ac + n1 * sb22 * xib0 * exp(
                    -ab) / ab + n2 * m22 * se22 * xie0 * exp(-ae) / ae
                ele23 = n0 * sc23 * xic0 * abs(k2) / (sqrt(2) * k2) * m0 / (k2 * vtper0) * exp(
                    -ac) + n1 * sb23 * xib0 * abs(k2) / (sqrt(2) * k2) \
                        * m1 / (k2 * vtper1) * exp(-ab) + n2 * m22 * se23 * xie0 * abs(k2) / (sqrt(2) * k2) * m2 / (
                                k2 * vtper2) * exp(-ae)
                ele31 = -ele13
                ele32 = ele23
                ele33 = -1 * n0 * sc33 * xic0 * exp(-ac) + -1 * n1 * sb33 * xib0 * exp(
                    -ab) + -1 * n2 * m22 * se33 * xie0 * exp(-ae)

                D11 = omega * omega / (cva * cva) - kk + ele11
                D12 = ele12
                D13 = ele13
                D21 = ele21
                D22 = omega * omega / (cva * cva) - k1 * k1 + ele22
                D23 = k1 * k2 + ele23
                D31 = ele31
                D32 = k1 * k2 + ele32
                D33 = omega * omega / (cva * cva) - k2 * k2 + ele33
                fun = D11 * D22 * D33 + D21 * D32 * D13 + D12 * D23 * D31 - (
                        D13 * D22 * D31 + D12 * D21 * D33 + D23 * D32 * D11)
                fu.append(fun)

            fu0 = fu[0]
            fu1 = fu[1]
            fu2 = fu[2]

            d0 = (fu1 - fu0) / h0
            d1 = (fu2 - fu1) / h1
            a = (d1 - d0) / (h1 + h0)
            b = a * h1 + d1
            c = fu2
            des = b + sqrt(b * b - 4 * a * c) if abs(b + sqrt(b * b - 4 * a * c)) > abs(
                b - sqrt(b * b - 4 * a * c)) else b - sqrt(b * b - 4 * a * c)
            omega3 = omega2 + ((-1 * 2 * c) / des)
            err = abs((omega3 - omega2) / omega3) if abs(c) < 10 ** -6 else 1
            omega0 = omega1
            omega1 = omega2
            omega2 = omega3
            om = [omega0, omega1, omega2]
            it = it + 1
            err = 0 if it > 50 else err
        pl = -1j * omega2.real / abs(omega2.real) * (D31 * D13 - D11 * D33) / (D33 * D12 - D13 * D32)
        pl = 1j * omega2.real / abs(omega2.real) * (D11 / D13 - D21 / D23) / (D12 / D13 - D22 / D23)
        print(k1, '  ', omega2, "  ", pl.real, '   c=', c)
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
