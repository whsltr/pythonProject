import numpy as np
import matplotlib.pyplot as plt
from cmath import pi, sin, cos, sqrt
from cmath import exp
from scipy import special
import scipy

path = '/home/kun/Documents/mathematics/shell-like/20n1/'
alpha_i = 89
f = open(path + str(alpha_i) + '_2.txt', 'w')
# n1 is the ring-beam, n0 is the background protons, n2 is the electron
n1 = np.array(
    [0.0002, 0.0005, 0.0005, 0.001, 0.0009, 0.0016, 0.0024, 0.0029, 0.0029, 0.0024, 0.0016, 0.0009, 0.001, 0.0005,
     0.0005, 0.0002]) * 10
n0 = 1. - 0.2
n2 = 1.

m0 = 1.
m1 = 1.
m2 = -1836. * 16
m22 = 1836 * 16

vtpar0 = 0.14 / sqrt(2) / sqrt(6)
# vtpar0=0.7/math.sqrt(2)
vtper0 = vtpar0
vtpar1 = np.array(
    [0.326, 0.2795, 0.2795, 0.2562, 0.2562, 0.2329, 0.2329, 0.1863, 0.1863, 0.2329, 0.2329, 0.2562, 0.2562, 0.2795,
     0.2795, 0.3260]) / sqrt(2) / sqrt(6)
# vtpar1=0.2/math.sqrt(2)
vtper1 = vtpar1
vtpar2 = 0.14 * sqrt(m22) / sqrt(6)
vtper2 = vtpar2

cva = 60 * 4

v_s = 2
theta = 0
vr0 = 0
vd0 = 0
vr2 = 0
vd2 = 0
vr1 = np.array(
    [0.2141, 0.3583, 0.5091, 0.6491, 0.7559, 0.7950, 0.8361, 0.8612, 0.8612, 0.8361, 0.7726, 0.7559, 0.6323, 0.5091,
     0.3492, 0.2087]) / sqrt(6)
vd1 = np.array(
    [0.97, 0.9261, 0.8213, 0.7126, 0.5705, 0.3958, 0.2386, 0.0794, -0.0794, -0.2383, -0.3847, -0.5705, -0.6941, -0.8213,
     -0.9027, -0.9455]) / sqrt(6)

bs0 = 0
bs2 = 0

As0 = 1
As2 = 1


# ymin = max(0, bs1 - 5)
# ymax = bs1 + 5
def Max(x, y):
    if x >= y.real:
        return x
    else:
        return y


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


def a_s(theta, v_t1, v_t2):
    v_r = v_s * np.sin(theta / 360 * 2 * np.pi)
    return (np.exp(-v_r ** 2 / (2 * v_t2 ** 2)) + np.sqrt(np.pi) * v_r / (np.sqrt(2) * v_t2) * special.erfc(
        -v_r / np.sqrt(2) / v_t2)) * np.sqrt(2 * np.pi) * v_t1 * 2 * np.pi * v_t2 ** 2


omegar = []
omegai = []
polarization = []
alpha_min = alpha_i
alpha = alpha_min
alpha_max = alpha_i
d_alpha = 0.1
ny = int((alpha_max - alpha) / d_alpha)
k_min = 2.2
k_max = 15
k = k_max
d_k = 0.02
nx = int((k_max - k_min) / d_k + 0.1 * d_k) + 1
nmin = -5
nmax = 5

as_total = sum(a_s(theta, vtpar1[i], vtper1[i]) for i in range(16))
while alpha <= alpha_max:
    omega_0r = 1.9
    omega_0i = 0.001
    omega0 = omega_0r + 0.001 + 1j * (omega_0i)
    omega1 = omega_0r + 0.002 + 1j * (omega_0i)
    omega2 = omega_0r + 0.003 + 1j * (omega_0i)
    om = [omega0, omega1, omega2]

    while k >= k_min:
        k1 = k * np.cos(alpha / 360 * 2 * np.pi)
        k2 = k * np.sin(alpha / 360 * 2 * np.pi)
        # print(k2)
        # print(k1)
        as0 = sqrt(2) * k2 * vtper0 / m0
        as2 = sqrt(2) * k2 * vtper2 / m2

        err = 1
        delta = 10 ** (-4)
        c = 1
        it = 0


        def fun_a0(nn, yy, vr, vtper):
            as1 = sqrt(2) * k2 * vtper / m1
            b = vr / np.sqrt(2) / vtper
            return special.jv(nn, as1 * yy) ** 2 * np.exp(-(yy - b) ** 2) * yy


        def fun_a1(nn, yy, vr, vtper):
            as1 = sqrt(2) * k2 * vtper / m1
            b = vr / np.sqrt(2) / vtper
            return special.jv(nn, as1 * yy) ** 2 * np.exp(-(yy - b) ** 2) * (yy - b)


        def fun_b0(nn, yy, vr, vtper):
            as1 = sqrt(2) * k2 * vtper / m1
            b = vr / np.sqrt(2) / vtper
            return special.jv(nn, as1 * yy) * (
                    (special.jv(nn - 1, as1 * yy) - special.jv(nn + 1, as1 * yy)) / 2) * np.exp(
                -(yy - b) ** 2) * yy ** 2


        def fun_b1(nn, yy, vr, vtper):
            as1 = sqrt(2) * k2 * vtper / m1
            b = vr / np.sqrt(2) / vtper
            return special.jv(nn, as1 * yy) * (
                    (special.jv(nn - 1, as1 * yy) - special.jv(nn + 1, as1 * yy)) / 2) * np.exp(
                -(yy - b) ** 2) * yy * (yy - b)


        def fun_c0(nn, yy, vr, vtper):
            as1 = sqrt(2) * k2 * vtper / m1
            b = vr / np.sqrt(2) / vtper
            return ((special.jv(nn - 1, as1 * yy) - special.jv(nn + 1, as1 * yy)) / 2) ** 2 * np.exp(
                -(yy - b) ** 2) * yy ** 3


        def fun_c1(nn, yy, vr, vtper):
            as1 = sqrt(2) * k2 * vtper / m1
            b = vr / np.sqrt(2) / vtper
            return ((special.jv(nn - 1, as1 * yy) - special.jv(nn + 1, as1 * yy)) / 2) ** 2 * np.exp(
                -(yy - b) ** 2) * yy ** 2 * (
                           yy - b)


        def A0(nn, vr, vtper, y_min, y_max):
            return ((y_max - y_min) / 300) * (fun_a0(nn, y_min, vr, vtper) + fun_a0(nn, y_max, vr, vtper) + 4 * sum(
                fun_a0(nn, y_min + (2 * j - 1) * ((y_max - y_min) / 100), vr, vtper) for j in range(1, 51))
                                              + 2 * sum(
                        fun_a0(nn, y_min + (2 * j) * ((y_max - y_min) / 100), vr, vtper) for j in range(1, 50)))


        def A1(nn, vr, vtper, y_min, y_max):
            return ((y_max - y_min) / 300) * (fun_a1(nn, y_min, vr, vtper) + fun_a1(nn, y_max, vr, vtper) + 4 * sum(
                fun_a1(nn, y_min + (2 * j - 1) * ((y_max - y_min) / 100), vr, vtper) for j in range(1, 51))
                                              + 2 * sum(
                        fun_a1(nn, y_min + (2 * j) * ((y_max - y_min) / 100), vr, vtper) for j in range(1, 50)))


        def B0(nn, vr, vtper, y_min, y_max):
            return ((y_max - y_min) / 300) * (fun_b0(nn, y_min, vr, vtper) + fun_b0(nn, y_max, vr, vtper) + 4 * sum(
                fun_b0(nn, y_min + (2 * j - 1) * ((y_max - y_min) / 100), vr, vtper) for j in range(1, 51))
                                              + 2 * sum(
                        fun_b0(nn, y_min + (2 * j) * ((y_max - y_min) / 100), vr, vtper) for j in range(1, 50)))


        def B1(nn, vr, vtper, y_min, y_max):
            return ((y_max - y_min) / 300) * (fun_b1(nn, y_min, vr, vtper) + fun_b1(nn, y_max, vr, vtper) + 4 * sum(
                fun_b1(nn, y_min + (2 * j - 1) * ((y_max - y_min) / 100), vr, vtper) for j in range(1, 51))
                                              + 2 * sum(
                        fun_b1(nn, y_min + (2 * j) * ((y_max - y_min) / 100), vr, vtper) for j in range(1, 50)))


        def C0(nn, vr, vtper, y_min, y_max):
            return ((y_max - y_min) / 300) * (fun_c0(nn, y_min, vr, vtper) + fun_c0(nn, y_max, vr, vtper) + 4 * sum(
                fun_c0(nn, y_min + (2 * j - 1) * ((y_max - y_min) / 100), vr, vtper) for j in range(1, 51))
                                              + 2 * sum(
                        fun_c0(nn, y_min + (2 * j) * ((y_max - y_min) / 100), vr, vtper) for j in range(1, 50)))


        def C1(nn, vr, vtper, y_min, y_max):
            return ((y_max - y_min) / 300) * (fun_c1(nn, y_min, vr, vtper) + fun_c1(nn, y_max, vr, vtper) + 4 * sum(
                fun_c1(nn, y_min + (2 * j - 1) * ((y_max - y_min) / 100), vr, vtper) for j in range(1, 51))
                                              + 2 * sum(
                        fun_c1(nn, y_min + (2 * j) * ((y_max - y_min) / 100), vr, vtper) for j in range(1, 50)))


        def eta0(omega_f, nn):
            return (omega_f - k1 * vd0 - nn * m0) / (sqrt(2) * k1 * vtpar0)


        def eta2(omega_f, nn):
            return (omega_f - k1 * vd2 - nn * m2) / (sqrt(2) * k1 * vtpar2)


        a0 = []
        a1 = []
        b0 = []
        b1 = []
        c0 = []
        c1 = []
        for i in range(16):
            # vr1 = v_s * np.sin(theta / 360 * 2 * np.pi)
            bs1 = vr1[i] / (np.sqrt(2) * vtper1[i])

            ymin = max(0, bs1 - 5)
            ymax = bs1 + 5
            for n in range(nmin, nmax + 1):
                a0.append(A0(n, vr1[i], vtper1[i], ymin, ymax))
                a1.append(A1(n, vr1[i], vtper1[i], ymin, ymax))
                b0.append(B0(n, vr1[i], vtper1[i], ymin, ymax))
                b1.append(B1(n, vr1[i], vtper1[i], ymin, ymax))
                c0.append(C0(n, vr1[i], vtper1[i], ymin, ymax))
                c1.append(C1(n, vr1[i], vtper1[i], ymin, ymax))

        a0 = np.array(a0).reshape(16, nmax - nmin + 1)
        a1 = np.array(a1).reshape(16, nmax - nmin + 1)
        b0 = np.array(b0).reshape(16, nmax - nmin + 1)
        b1 = np.array(b1).reshape(16, nmax - nmin + 1)
        c0 = np.array(c0).reshape(16, nmax - nmin + 1)
        c1 = np.array(c1).reshape(16, nmax - nmin + 1)

        while delta < err < 1000:
            h1 = om[2] - om[1]
            h0 = om[1] - om[0]
            fu = []

            for omega in om:
                kk = k1 * k1 + k2 * k2

                sc11 = sum(-2 * n * n / (as0 * as0) * ((vtper0 ** 2) / (vtpar0 ** 2) * A(n, as0) * DZ(eta0(omega, n))
                                                       + 2 * A(n, as0) * (1 - n * m0 / (sqrt(2) * k1 * vtpar0) * Z(
                            eta0(omega, n)))) for n in range(nmin, nmax + 1))
                se11 = sum(-2 * n * n / (as2 * as2) * ((vtper2 ** 2) / (vtpar2 ** 2) * A(n, as2) * DZ(eta2(omega, n))
                                                       + 2 * A(n, as2) * (1 - n * m2 / (sqrt(2) * k1 * vtpar2) * Z(
                            eta2(omega, n)))) for n in range(nmin, nmax + 1))
                sc12 = sum(1j * 4 * n / as0 * ((vtper0 ** 2) / (vtpar0 ** 2) * B(n, as0) * eta0(omega, n)
                                               + B(n, as0) * (
                                                       n * m0 / (sqrt(2) * k1 * vtpar0))) * Z(
                    eta0(omega, n)) for n in range(nmin, nmax + 1))
                se12 = sum(1j * 4 * n / as2 * ((vtper2 ** 2) / (vtpar2 ** 2) * B(n, as2) * eta2(omega, n)
                                               + B(n, as2) * (
                                                       n * m2 / (sqrt(2) * k1 * vtpar2))) * Z(
                    eta2(omega, n)) for n in range(nmin, nmax + 1))
                sc13 = sum(-2 * n / as0 * (vtper0 / vtpar0) * (
                        (omega - n * m0) / (sqrt(2) * k1 * vtpar0) * A(n, as0) * DZ(eta0(omega, n))
                        - 2 * (vtpar0 ** 2) / (vtper0 ** 2) * n * m0 / (sqrt(2) * k1 * vtpar0) * A(n, as0)
                        * (1 + (omega - n * m0) / (sqrt(2) * k1 * vtpar0) * Z(eta0(omega, n)))) for n in
                           range(nmin, nmax + 1))
                se13 = sum(-2 * n / as2 * (vtper2 / vtpar2) * (
                        (omega - n * m2) / (sqrt(2) * k1 * vtpar2) * A(n, as2) * DZ(eta2(omega, n))
                        - 2 * (vtpar2 ** 2) / (vtper2 ** 2) * n * m2 / (sqrt(2) * k1 * vtpar2) * A(n, as2)
                        * (1 + (omega - n * m2) / (sqrt(2) * k1 * vtpar2) * Z(eta2(omega, n)))) for n in
                           range(nmin, nmax + 1))
                sc21 = -sc12
                se21 = -se12
                sc22 = sum(-2 * ((vtper0 ** 2) / (vtpar0 ** 2) * C(n, as0) * DZ(eta0(omega, n))
                                 + 2 * C(n, as0) * (1 - n * m0 / (sqrt(2) * k1 * vtpar0) * Z(
                            eta0(omega, n)))) for n in range(nmin, nmax + 1))
                se22 = sum(-2 * ((vtper2 ** 2) / (vtpar2 ** 2) * C(n, as2) * DZ(eta2(omega, n))
                                 + 2 * C(n, as2) * (1 - n * m2 / (sqrt(2) * k1 * vtpar2) * Z(
                            eta2(omega, n)))) for n in range(nmin, nmax + 1))
                sc23 = sum(1j * 2 * vtper0 / vtpar0 * (
                        (omega - n * m0) / (sqrt(2) * k1 * vtpar0) * B(n, as0) * DZ(eta0(omega, n))
                        - 2 * (vtpar0 ** 2) / (vtper0 ** 2) * n * m0 / (sqrt(2) * k1 * vtpar0) * B(n, as0)
                        * (1 + (omega - n * m0) / (sqrt(2) * k1 * vtpar0) * Z(eta0(omega, n)))) for n in
                           range(nmin, nmax + 1))
                se23 = sum(1j * 2 * vtper2 / vtpar2 * (
                        (omega - n * m2) / (sqrt(2) * k1 * vtpar2) * B(n, as2) * DZ(eta2(omega, n))
                        - 2 * (vtpar2 ** 2) / (vtper2 ** 2) * n * m2 / (sqrt(2) * k1 * vtpar2) * B(n, as2)
                        * (1 + (omega - n * m2) / (sqrt(2) * k1 * vtpar2) * Z(eta2(omega, n)))) for n in
                           range(nmin, nmax + 1))
                sc31 = sc13
                se31 = se13
                sc32 = -sc23
                se32 = -se23
                sc33 = sum(-2 * (
                        ((omega - n * m0) / (sqrt(2) * k1 * vtpar0)) ** 2 * A(n, as0) * DZ(eta0(omega, n))
                        - 2 * (vtpar0 ** 2) / (vtper0 ** 2) * (omega - n * m0) / (sqrt(2) * k1 * vtpar0) * n * m0 / (
                                sqrt(2) * k1 * vtpar0) * A(n, as0)
                        * (1 + (omega - n * m0) / (sqrt(2) * k1 * vtpar0) * Z(eta0(omega, n)))) for n in
                           range(nmin, nmax + 1))
                se33 = sum(-2 * (
                        ((omega - n * m2) / (sqrt(2) * k1 * vtpar2)) ** 2 * A(n, as2) * DZ(eta2(omega, n))
                        - 2 * (vtpar2 ** 2) / (vtper2 ** 2) * (omega - n * m2) / (sqrt(2) * k1 * vtpar2) * n * m2 / (
                                sqrt(2) * k1 * vtpar2) * A(n, as2)
                        * (1 + (omega - n * m2) / (sqrt(2) * k1 * vtpar2) * Z(eta2(omega, n)))) for n in
                           range(nmin, nmax + 1))

                D11 = omega * omega / (cva * cva) - k1 * k1 + (
                        n0 * m0 * sc11 + n2 * m22 * se11)
                D12 = (n0 * m0 * sc12 + n2 * m22 * se12)
                D13 = k1 * k2 + (n0 * m0 * sc13 + n2 * m22 * se13)
                D21 = (n0 * m0 * sc21 + n2 * m22 * se21)
                D22 = omega * omega / (cva * cva) - kk + (
                        n0 * m0 * sc22 + n2 * m22 * se22)
                D23 = (n0 * m0 * sc23 + n2 * m22 * se23)
                D31 = k1 * k2 + (n0 * m0 * sc31 + n2 * m22 * se31)
                D32 = (n0 * m0 * sc32 + n2 * m22 * se32)
                D33 = omega * omega / (cva * cva) - k2 * k2 + (
                        n0 * m0 * sc33 + n2 * m22 * se33)

                n_pick = 0
                # i = 0
                for i in range(16):
                    # vr1[i] = v_s * sin((theta / 360) * (2 * pi))
                    # vd1 = v_s * cos((theta / 360) * (2 * pi))
                    as1 = sqrt(2) * k2 * vtper1[i] / m1
                    bs1 = vr1[i] / (sqrt(2) * vtper1[i])
                    As1 = exp(-bs1 ** 2) + sqrt(pi) * bs1 * special.erfc(-bs1)
                    # eta = a_s(theta, vtpar1, vtper1) / as_total
                    # eta = as_total
                    # n_pick = n_pick + eta
                    k_v = sqrt(2) * k1 * vtpar1[i]


                    def eta1(omega_f, nn):
                        return (omega_f - k1 * vd1[i] - nn * m1) / k_v


                    sb11 = sum(
                        -2 * n * n / (as1 * as1 * As1) * (
                                (vtper1[i] ** 2) / (vtpar1[i] ** 2) * a0[i, n + nmax] * DZ(eta1(omega, n))
                                + 2 * a1[i, n + nmax] * (1 - n * m1 / k_v * Z(
                            eta1(omega, n)))) for n in range(nmin, nmax + 1))
                    sb12 = sum(
                        1j * 4 * n / (As1 * as1) * ((vtper1[i] ** 2) / (vtpar1[i] ** 2) * b0[i, n + nmax] * eta1(omega, n)
                                                    + b1[i, n + nmax] * (
                                                            n * m1 / k_v)) * Z(
                            eta1(omega, n)) for n in range(nmin, nmax + 1))
                    sb13 = sum(-2 * n / as1 / As1 * (vtper1[i] / vtpar1[i]) * (
                            (omega - n * m1) / k_v * a0[i, n + nmax] * DZ(eta1(omega, n))
                            - 2 * (vtpar1[i] ** 2) / (vtper1[i] ** 2) * n * m1 / k_v * a1[i, n + nmax]
                            * (1 + (omega - n * m1) / k_v * Z(eta1(omega, n)))) for n in
                               range(nmin, nmax + 1))
                    sb21 = -sb12
                    sb22 = sum(-2 / As1 * ((vtper1[i] ** 2) / (vtpar1[i] ** 2) * c0[i, n + nmax] * DZ(eta1(omega, n))
                                           + 2 * c1[i, n + nmax] * (1 - n * m1 / k_v * Z(
                                eta1(omega, n)))) for n in range(nmin, nmax + 1))
                    sb23 = sum(1j * 2 / As1 * vtper1[i] / vtpar1[i] * (
                            (omega - n * m1) / k_v * b0[i, n + nmax] * DZ(eta1(omega, n))
                            - 2 * (vtpar1[i] ** 2) / (vtper1[i] ** 2) * n * m1 / k_v * b1[i, n + nmax]
                            * (1 + (omega - n * m1) / k_v * Z(eta1(omega, n)))) for n in
                               range(nmin, nmax + 1))
                    sb31 = sb13
                    sb32 = -sb23
                    sb33 = sum(-2 / As1 * (
                            ((omega - n * m1) / k_v) ** 2 * a0[i, n + nmax] * DZ(
                        eta1(omega, n))
                            - 2 * (vtpar1[i] ** 2) / (vtper1[i] ** 2) * (omega - n * m1) / k_v * n * m1 / k_v * a1[
                                i, n + nmax]
                            * (1 + (omega - n * m1) / k_v * Z(eta1(omega, n)))) for n in
                               range(nmin, nmax + 1))
                    # i = i + 1

                    D11 += n1[i] * sb11
                    D12 += n1[i] * sb12
                    D13 += n1[i] * sb13
                    D21 += n1[i] * sb21
                    D22 += n1[i] * sb22
                    D23 += n1[i] * sb23
                    D31 += n1[i] * sb31
                    D32 += n1[i] * sb32
                    D33 += n1[i] * sb33

                fun = D11 * D22 * D33 + D21 * D32 * D13 + D12 * D23 * D31 - (
                        D13 * D22 * D31 + D12 * D21 * D33 + D23 * D32 * D11)
                fu.append(fun)

            # print(n_pick)
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
            err = abs((omega3 - omega2) / omega3) if abs(c) < 10 ** -3 else 1
            omega0 = omega1
            omega1 = omega2
            omega2 = omega3
            om = [omega0, omega1, omega2]
            it = it + 1
            err = 0 if it > 250 else err
        pl = -1j * omega2.real / abs(omega2.real) * (D31 * D13 - D11 * D33) / (D33 * D12 - D13 * D32)
        pl = 1j * omega2.real / abs(omega2.real) * (D11 / D13 - D21 / D23) / (D12 / D13 - D22 / D23)
        print("{:.3f}".format(k), '  ', omega2, "  ", pl.real, '   c=', c, "   eta0=", eta0(omega2, -1))
        print("{:.3f}".format(k), '  ', omega2.real, '  ', omega2.imag, '  ', pl.real, file=f)
        k = k - d_k
        omegar.append(omega2.real)
        omegai.append(omega2.imag)
        polarization.append(pl.real)
    print(alpha)
    alpha = alpha + d_alpha

f.close()
# omegar = np.array(omegar).reshape(ny, -1)
# omegai = np.array(omegai).reshape(ny, -1)
# polarization = np.array(polarization).reshape(ny, -1)
# omegar = omegar[::-1, :]
# omegai = omegai[::-1, :]
omegai = np.array(omegai)
omegar = np.array(omegar)
# omegai[omegai < 0.01] = None

# font1 = {'family': 'Computer Modern Roman',
#          'weight': 'normal',
#          'size': 14}
# x = np.linspace(k_min, k_max, int(len(omegar)))
# y = np.linspace(alpha, alpha_max, ny)
# plt.figure()
# plt.plot(x, omegar[::-1], label='$\omega_r$')
# plt.plot(x, omegai[::-1], label='$\gamma$')
# plt.legend()
# X, Y = np.meshgrid(x, y)
# fig = plt.figure()
# plt.pcolormesh(X, Y, omegai, shading='gouraud', cmap='jet')
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=12)
# cb.set_label(r'$\gamma/ \Omega_i$', fontdict=font1)
# contour = plt.contour(x, y, polarization, 8, cmap='cool', alpha=1, vmin=-0.6, vmax=0.8, Nchunk=0.5,
#                       linestyles='dotted')
# plt.clabel(contour, inline=1, )
# # contour = plt.contourf(X, Y, omegai, 10, cmap='cool', vmin=0.0, vmax=0.12)
# plt.xlabel(r'$k_\parallel / k_i$', font1)
# plt.ylabel(r'$k_\perp / k_i$', font1)
# plt.tick_params(labelsize=12)
k = np.linspace(k_min, k_max, len(omegar))
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14}
fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel(r'$k\lambda_{De}$', font=font1)
ax1.set_ylabel(r'$\omega/\omega_{pe}$', color=color, font=font1)
ax1.plot(k, omegar[::-1], color=color)
ax1.tick_params(axis='y', labelcolor=color, labelsize=14)
ax1.tick_params(axis='x', labelsize=14)
ax1.set_ylim([1, 7])
ax1.set_xlim([0, k_max])
ax1.axhline(0, color='k', )

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel(r'$\gamma/\omega_{pe}$', color=color, font=font1)
ax2.plot(k, omegai[::-1], color=color)
ax2.tick_params(axis='y', labelcolor=color, labelsize=14)
ax2.set_ylim([-0.05, 0.2])
plt.title(r'$n_1 = ' + str(n1) + '$', font=font1)
plt.subplots_adjust(left=0.148, bottom=0.124, right=0.84, top=0.9)
# plt.savefig('/home/kun/Documents/ye/111/01t' + str(n1) + '_' + str(vd1) + '.png')
# ax2.axhline(0, 0, k_max, color='k', )

plt.show()

plt.show()
