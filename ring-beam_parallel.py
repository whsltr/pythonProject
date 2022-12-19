import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import special
import scipy

# n1 is the ring-beam, no is the background protons, n2 is the electron
n1 = 0.00008
n0 = 1. - n1
n2 = 1.

m0 = 1.
m1 = 1.
m2 = -1836.
m22 = 1836

vtpar0 = math.sqrt(0.01)
# vtpar0=0.7/math.sqrt(2)
vtper0 = vtpar0
vtpar1 = math.sqrt(0.6)
# vtpar1=0.2/math.sqrt(2)
vtper1 = vtpar1
vtpar2 = math.sqrt(0.1 * 1836)
vtper2 = vtpar2

cva = 15000

theta = 0
vr0 = 0
vr1 = 20 * math.sin((theta / 360) * (2 * math.pi))
vd1 = 20 * math.cos((theta / 360) * (2 * math.pi))
vd0 = -vd1 * (n1 / n0)
vr2 = 0
vd2 = 0

bs0 = 0
bs1 = vr1 / (math.sqrt(2) * vtper1)
bs2 = 0

As0 = 1
As1 = math.exp(-bs1 ** 2) + math.sqrt(math.pi) * bs1 * math.erfc(-bs1)
As2 = 1

ymin = max(0, bs1 - 5)
ymax = bs1 + 5


def delta(x):
    if x == 0:
        return 1
    else:
        return 0


def fun1(n, b, c, y):
    return (delta(n + 1) + delta(n - 1)) * (y / 2) ** 2 * math.exp(-(y - b) * (y - b)) * (y - c)


def fun2(n, b, c, y):
    return (delta(n + 1) + delta(n - 1)) * (y / 2) * 0.5 * math.exp(-(y - b) * (y - b)) * y * (y - c)


def fun3(n, b, c, y):
    return (delta(n + 1) + delta(n - 1)) * 0.5 * 0.5 * math.exp(-(y - b) * (y - b)) * y * y * (y - c)


def Z(x):
    if abs(x) < 25:
        return 1j * math.sqrt(math.pi) * special.erfc(-1j * x) * np.exp(-x * x)
    elif x.imag > 1 / abs(x.real):
        return -1 / x * (1 + 1 / (2 * x * x) + 3 / (4 * x ** 4) + 15 / (8 * x ** 6))
    elif abs(x.imag) < 1 / abs(x.real):
        return 1j * np.sqrt(math.pi) * np.exp(-x * x) - 1 / x * (
                    1 + 1 / (2 * x * x) + 3 / (4 * x ** 4) + 15 / (8 * x ** 6))
    else:
        return 1j * 2 * np.sqrt(math.pi) * np.exp(-x * x) - 1 / x * (
                    1 + 1 / (2 * x * x) + 3 / (4 * x ** 4) + 15 / (8 * x ** 6))


def DZ(x):
    return -2 * (1 + x * Z(x))


beam_k = []
beam_r = []
beam_i = []
approxi = 0.0001
omega0 = 0.024266 + approxi * 1j
omega1 = 0.025261 + approxi * 1j
omega2 = 0.026223 + approxi * 1j

om = [omega0, omega1, omega2]

for k in np.arange(0.02, 0.21, 0.002):
    kpar = k
    kper = 0
    beam_k.append(k)

    as0 = (math.sqrt(2) * kper * vtper0) / m0
    as1 = (math.sqrt(2) * kper * vtper1) / m1
    as2 = (math.sqrt(2) * kper * vtper2) / m2

    err = 1
    delta0 = 10 ** (-9)
    while delta0 < err < 1000:
        h0 = om[1] - om[0]
        h1 = om[2] - om[1]
        fu = []

        for i in range(3):
            omega = om[i]
            D11 = (-kpar * kpar) + (omega * omega) / (cva * cva)
            D12 = 0
            D13 = (kpar * kper)
            D21 = 0
            D22 = (-kpar * kpar - kper * kper) + (omega * omega) / (cva * cva)
            D23 = 0
            D31 = kpar * kper
            D32 = 0
            D33 = (-kper * kper) + (omega * omega) / (cva * cva)

            for n in range(-1, 2):  # + is the left hand - is the right hand
                eta0 = (omega - kpar * vd0 - n * m0) / (math.sqrt(2) * kpar * vtpar0)
                eta1 = (omega - kpar * vd1 - n * m1) / (math.sqrt(2) * kpar * vtpar1)
                eta2 = (omega - kpar * vd2 - n * m2) / (math.sqrt(2) * kpar * vtpar2)

                A1 = ((ymax - ymin) / 300) * (fun1(n, bs1, bs1, ymin) + fun1(n, bs1, bs1, ymax)
                                              + 4 * sum(
                            fun1(n, bs1, bs1, ymin + (2 * j - 1) * (ymax - ymin) / 100) for j in range(1, 51))
                                              + 2 * sum(
                            fun1(n, bs1, bs1, ymin + (2 * j) * (ymax - ymin) / 100) for j in range(1, 50)))
                A0 = ((ymax - ymin) / 300) * (fun1(n, bs1, 0, ymin) + fun1(n, bs1, 0, ymax)
                                              + 4 * sum(
                            fun1(n, bs1, 0, ymin + (2 * j - 1) * (ymax - ymin) / 100) for j in range(1, 51))
                                              + 2 * sum(
                            fun1(n, bs1, 0, ymin + (2 * j) * (ymax - ymin) / 100) for j in range(1, 50)))
                B1 = ((ymax - ymin) / 300) * (fun2(n, bs1, bs1, ymin) + fun2(n, bs1, bs1, ymax)
                                              + 4 * sum(
                            fun2(n, bs1, bs1, ymin + (2 * j - 1) * (ymax - ymin) / 100) for j in range(1, 51))
                                              + 2 * sum(
                            fun2(n, bs1, bs1, ymin + (2 * j) * (ymax - ymin) / 100) for j in range(1, 50)))
                B0 = ((ymax - ymin) / 300) * (fun2(n, bs1, 0, ymin) + fun2(n, bs1, 0, ymax)
                                              + 4 * sum(
                            fun2(n, bs1, 0, ymin + (2 * j - 1) * (ymax - ymin) / 100) for j in range(1, 51))
                                              + 2 * sum(
                            fun2(n, bs1, 0, ymin + (2 * j) * (ymax - ymin) / 100) for j in range(1, 50)))
                C1 = ((ymax - ymin) / 300) * (fun3(n, bs1, bs1, ymin) + fun3(n, bs1, bs1, ymax)
                                              + 4 * sum(
                            fun3(n, bs1, bs1, ymin + (2 * j - 1) * (ymax - ymin) / 100) for j in range(1, 51))
                                              + 2 * sum(
                            fun3(n, bs1, bs1, ymin + (2 * j) * (ymax - ymin) / 100) for j in range(1, 50)))
                C0 = ((ymax - ymin) / 300) * (fun3(n, bs1, 0, ymin) + fun3(n, bs1, 0, ymax)
                                              + 4 * sum(
                            fun3(n, bs1, 0, ymin + (2 * j - 1) * (ymax - ymin) / 100) for j in range(1, 51))
                                              + 2 * sum(
                            fun3(n, bs1, 0, ymin + (2 * j) * (ymax - ymin) / 100) for j in range(1, 50)))
                A = 0.125 * (delta(n + 1) + delta(n - 1))
                B = 0.125 * (delta(n + 1) + delta(n - 1))
                C = 0.125 * (delta(n + 1) + delta(n - 1))

                sp11 = (-2 * n * n / As1) * (vtper1 * vtper1 / (vtpar1 * vtpar1) * A0 * DZ(eta1) +
                                             2 * A1 * (1 - (n * m1 / (math.sqrt(2) * kpar * vtpar1) * Z(eta1))))
                sb11 = (-2 * n * n / As0) * (vtper0 * vtper0 / (vtpar0 * vtpar0) * A * DZ(eta0) +
                                             2 * A * (1 - (n * m0 / (math.sqrt(2) * kpar * vtpar0) * Z(eta0))))
                sc11 = (-2 * n * n / As2) * (vtper2 * vtper2 / (vtpar2 * vtpar2) * A * DZ(eta2) +
                                             2 * A * (1 - (n * m2 / (math.sqrt(2) * kpar * vtpar2) * Z(eta2))))
                sp12 = ((-2 * n * 1j) / As1) * (vtper1 * vtper1 / (vtpar1 * vtpar1) * B0 * DZ(eta1) +
                                                2 * B1 * (1 - (n * m1 / (math.sqrt(2) * kpar * vtpar1) * Z(eta1))))
                sb12 = ((-2 * n * 1j) / As0) * (vtper0 * vtper0 / (vtpar0 * vtpar0) * B * DZ(eta0) +
                                                2 * B * (1 - (n * m0 / (math.sqrt(2) * kpar * vtpar0) * Z(eta0))))
                sc12 = ((-2 * n * 1j) / As2) * (vtper2 * vtper2 / (vtpar2 * vtpar2) * B * DZ(eta2) +
                                                2 * B * (1 - (n * m2 / (math.sqrt(2) * kpar * vtpar2) * Z(eta2))))
                sp22 = ((-2) / As1) * (vtper1 * vtper1 / (vtpar1 * vtpar1) * C0 * DZ(eta1) +
                                       2 * C1 * (1 - (n * m1 / (math.sqrt(2) * kpar * vtpar1) * Z(eta1))))
                sb22 = ((-2) / As0) * (vtper0 * vtper0 / (vtpar0 * vtpar0) * C * DZ(eta0) +
                                       2 * C * (1 - (n * m0 / (math.sqrt(2) * kpar * vtpar0) * Z(eta0))))
                sc22 = ((-2) / As2) * (vtper2 * vtper2 / (vtpar2 * vtpar2) * C * DZ(eta2) +
                                       2 * C * (1 - (n * m2 / (math.sqrt(2) * kpar * vtpar2) * Z(eta2))))
                sp21 = -sp12
                sb21 = -sb12
                sc21 = -sc12

                ele11 = n0 * m0 * sb11 + n1 * m1 * sp11 + n2 * m22 * sc11
                ele12 = n0 * m0 * sb12 + n1 * m1 * sp12 + n2 * m22 * sc12
                ele21 = n0 * m0 * sb21 + n1 * m1 * sp21 + n2 * m22 * sc21
                ele22 = n0 * m0 * sb22 + n2 * m2 * sp22 + n2 * m22 * sc22

                D11 += ele11
                D12 += ele12
                D21 += ele21
                D22 += ele22

            fun = -D12 * D21 + D11 * D22
            fu.append(fun)

        fu0 = fu[0]
        fu1 = fu[1]
        fu2 = fu[2]
        d0 = (fu1 - fu0) / (omega1 - omega0)
        d1 = (fu2 - fu1) / (omega2 - omega1)
        a = (d1 - d0) / (h1 + h0)
        b = a * h1 + d1
        c = fu2
        des = b + np.sqrt(b * b - 4 * a * c) if abs(b + np.sqrt(b * b - 4 * a * c)) > abs(
            b - np.sqrt(b * b - 4 * a * c)) else b - np.sqrt(b * b - 4 * a * c)
        omega3 = omega2 + ((-2 * c) / des)
        err = 1 if abs(c) > 10 ** (-7) else abs((omega3 - omega2) / omega3)

        omega0 = omega1
        omega1 = omega2
        omega2 = omega3
        om = [omega0, omega1, omega2]

    beam_r.append(omega2.real)
    beam_i.append(omega2.imag)
    print(k, '  ', omega2)

fig, ax = plt.subplots()
ax.plot(beam_k, beam_r, 'r-')
ax.set_xlabel(r'$k_\parallel k_i$')
ax.set_ylabel(r'$\omega/\Omega_i$', color='r')
ax.tick_params(axis='y', labelcolor='r')
ax2 = ax.twinx()
ax2.plot(beam_k, beam_i, 'b-')
ax2.set_ylabel(r'$\gamma/\Omega_i$', color='b')
ax2.tick_params(axis='y', labelcolor='b')
plt.show()
