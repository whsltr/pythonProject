import numpy as np
import matplotlib.pyplot as plt
from cmath import pi, sin, cos, sqrt
from cmath import exp
from scipy import special
import scipy

# n1 is the beam electron, n0 is the background proton, n2 is the electron
n1 = 0.1
n0 = 1.
n2 = 1. - n1

qm0 = -1. / 1836
qm1 = 1.
qm2 = 1.
m22 = 1 / 1836

vtpar0 = 1/sqrt(1836)
# vtpar0=0.7/math.sqrt(2)
vtper0 = 1/sqrt(1836)
vtpar1 = 1
# vtpar1=0.2/math.sqrt(2)
vtper1 = vtpar1
vtpar2 = 1
vtper2 = vtpar2

vd0 = 0
vd1 = 4
vd2 = -n1/n0 * vd1
vd2 = 0


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


omegar = []
omegai = []

approxi = 0.001
k_min = 0.001
dk = 0.001
k_max = 0.6
k0 = k_max
omega0 = sqrt(1 + 3 / 2 * k0 ** 2) + 0.001 + 1j * approxi
omega1 = sqrt(1 + 3 / 2 * k0 ** 2) + 0.002 + 1j * approxi
omega2 = sqrt(1 + 3 / 2 * k0 ** 2) + 0.003 + 1j * approxi
# omega0 = k0 * vd1 + 0.00001 + 1j * approxi
# omega1 = k0 * vd1 + 0.00002 + 1j * approxi
# omega2 = k0 * vd1 + 0.00003 + 1j * approxi
while k0 > k_min:
    # omega0 = sqrt(1 + 3 / 2 * k0 ** 2) + 0.001 + 1j * approxi
    # omega1 = sqrt(1 + 3 / 2 * k0 ** 2) + 0.002 + 1j * approxi
    # omega2 = sqrt(1 + 3 / 2 * k0 ** 2) + 0.003 + 1j * approxi
    # omega0 = k0*vd1 + 0.001
    # omega1 = k0*vd1 + 0.002
    # omega2 = k0*vd1 + 0.003

    om = [omega0, omega1, omega2]

    err = 1
    delta = 10 ** -6
    it = 0

    while err > delta:
        h1 = om[2] - om[1]
        h0 = om[1] - om[0]
        fu = []

        for ii in range(3):
            omega = om[ii]
            xip = (omega - k0 * vd0) / (sqrt(2) * k0 * vtpar0)
            xib = (omega - k0 * vd1) / (sqrt(2) * k0 * vtpar1)
            xic = (omega - k0 * vd2) / (sqrt(2) * k0 * vtpar2)

            K = 1 / (2 * k0 * k0) * (n0 * DZ(xip) + n1 * DZ(xib) + n2 * DZ(xic))
            fun = 1 - K
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
        err = abs((omega3 - omega2) / omega3) if abs(c) < 10 ** -4 else 1
        omega0 = omega1
        omega1 = omega2
        omega2 = omega3
        om = [omega0, omega1, omega2]
        it = it + 1
        err = 0 if it > 100 else err
    print(k0, '  ', omega2, '   xi=', abs(xib), '   c=', abs(c),)
    k0 = k0 - dk
    omegar.append(omega2.real)
    omegai.append(omega2.imag)

omegar = np.array(omegar)
omegai = np.array(omegai)
omegar = omegar[::-1]
omegai = omegai[::-1]

k = np.linspace(0, k_max, len(omegar))
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14}
fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel(r'$k\lambda_{De}$', font=font1)
ax1.set_ylabel(r'$\omega/\omega_{pe}$', color=color, font=font1)
ax1.plot(k, omegar, color=color)
ax1.tick_params(axis='y', labelcolor=color, labelsize=14)
ax1.tick_params(axis='x', labelsize=14)
ax1.set_ylim([-1, 3])
ax1.set_xlim([0, k_max])
ax1.axhline(0, color='k',)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel(r'$\gamma/\omega_{pe}$', color=color, font=font1)
ax2.plot(k, omegai, color=color)
ax2.tick_params(axis='y', labelcolor=color, labelsize=14)
ax2.set_ylim([-0.1, 0.3])
plt.title(r'$n_1 = ' + str(n1) + ',  v_d = ' + str(vd1) + ',  T_b=T_0' + '$', font=font1)
plt.subplots_adjust(left=0.148, bottom=0.124, right=0.84, top=0.9)
plt.savefig('/home/kun/Documents/ye/111/01t' + str(n1) + '_' + str(vd1) + '.png')
# ax2.axhline(0, 0, k_max, color='k', )

plt.show()

# x = np.linspace(-20, 20, 1000)
# x0 = 5
# fx0 = np.array([0.01*exp(-(a-x0)**2) for a in x])
# fx1 = np.array([exp(-(a)**2) for a in x])
# fx = fx0 + fx1
# plt.plot(x, fx)
# plt.show()