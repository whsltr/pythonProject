import numpy as np
import matplotlib.pyplot as plt
from cmath import pi, sin, cos, sqrt
from cmath import exp
from scipy import special


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


Te = 1
me = 9.1e-31
K = 1.38e-23
epsilon = 8.85e-12
omega_p = 1
v_th = sqrt(2 * K * Te / me)
ld = v_th / sqrt(1 / 2) * omega_p


def B(k, omegaf):
    return exp(-omegaf * omegaf / (k * k * v_th * v_th)) / (sqrt(pi) * k * v_th)


def sigma(k, omegaf):
    return 1 + 1 / (k * k * ld * ld) * (1 + omegaf / (k * v_th) * Z(omegaf / (k * v_th)))


def F(x):
    return sqrt(pi) / (4 * x)


def fun1(k, omegaf):
    return B(k, omegaf) * F(k) / (k * k * abs(sigma(k, omegaf)) ** 2)


def V(kmax, omegaf):
    return kmax / 30000 * (
            fun1(0.001, omegaf) + fun1(kmax, omegaf) + 4 * sum(
        fun1((2 * j - 1) * kmax / 10000, omegaf) for j in range(1, 5000))
            + 2 * sum(fun1((2 * j) * kmax / 1000, omegaf) for j in range(1, 4999)))


omega = np.linspace(0.1, 1, 100)
k = np.linspace(0.001, 10, 100)
v = []
for omega_i in omega:
    v_i = V(0.4, omega_i)
    v.append(v_i)

v = np.array(v)

plt.plot(omega, v)
plt.xscale("log")
plt.yscale("log")
plt.show()
