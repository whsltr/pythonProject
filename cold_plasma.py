import numpy as np
import matplotlib.pyplot as plt
from cmath import pi, sin, cos, sqrt
from cmath import exp
from scipy import special
import scipy

# n1 is the beam electron, n0 is the background proton, n2 is the electron
n1 = 0.001
n0 = 1. - n1

qm0 = 1
qm1 = 1.

vd0 = 0
vd1 = 2
vd2 = -n1 / n0 * vd1
theta = 0 / 360 * 2 * np.pi

omegar = []
omegai = []

approxi = 0.001
k0 = 1
k_max = k0
dk = 0.02
k_min = 0.
omega0 = -0.7 + 0.001
omega1 = -0.7 + 0.002
omega2 = -0.7 + 0.003
om = [omega0, omega1, omega2]

while k0 > 0.02:

    err = 1
    delta = 10 ** -4
    it = 0

    while err > delta:
        h1 = om[2] - om[1]
        h0 = om[1] - om[0]
        fu = []

        # d_fun = fun()
        for ii in range(3):
            omega = om[ii]
            R_up = n0 * omega ** 2 * (1 + omega - k0 * vd1) + n1 * (1 + omega) * (omega - k0 * vd1) ** 2
            R_down = (1 + omega) * omega ** 2 * (1 + omega - k0 * vd1)
            L_up = n0 * omega ** 2 * (1 - omega + k0 * vd1) + n1 * (1 - omega) * (omega - k0 * vd1) ** 2
            L_down = (1 - omega) * omega ** 2 * (1 - omega + k0 * vd1)
            # R = n0 / (1 + omega) + n1 * (omega - k0 * vd1) ** 2 / (omega ** 2 * (1 + omega - k0 * vd1))
            # L = n0 / (1 - omega) + n1 * (omega - k0 * vd1) ** 2 / (omega ** 2 * (1 - omega + k0 * vd1))

            fun = (k0 / omega) ** 4 * np.cos(theta) ** 2 * R_down * L_down - (k0 / omega) ** 2 * (
                    R_up * L_down + L_up * R_down) * (1 + np.cos(theta) ** 2) / 2 + R_up * L_up
            d_omega = omega + omega * 10 ** -4
            fun = (k0 / omega) ** 4 * np.cos(theta) ** 2 * R_down * L_down - (k0 / omega) ** 2 * (
                    R_up * L_down + L_up * R_down) * (1 + np.cos(theta) ** 2) / 2 + R_up * L_up

            R = n0 / (1 + omega) + n1 * (omega - k0 * vd1) ** 2 / (omega ** 2 * (1 + omega - k0 * vd1))
            L = n0 / (1 - omega) + n1 * (omega - k0 * vd1) ** 2 / (omega ** 2 * (1 - omega + k0 * vd1))
            # if abs(1-omega) < 10**-3 or abs(1-omega+k0*vd1) < 10**-3:
            #     fun = 10**-5
            # if abs(1+omega) < 10**-3 or abs(1+omega-k0*vd1) < 10**-3:
            #     fun = 10**-5
            # else:
            # fun = (k0/omega)**4 * np.cos(theta)**2 - (k0/omega)**2 * (R+L)*(1+np.cos(theta)**2)/2 + R*L

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
        err = abs((omega3 - omega2) / omega3) if abs(c) < 10 ** -11 else 1
        omega0 = omega1
        omega1 = omega2
        omega2 = omega3
        om = [omega0, omega1, omega2]
        it = it + 1
        err = 0 if it > 50 else err
    print(k0, '  ', omega2, '    c=', abs(c), )
    k0 = k0 - dk
    omegar.append(omega2.real)
    omegai.append(omega2.imag)

omegar = np.array(omegar)
omegai = np.array(omegai)
omegar = omegar[::-1]

k = np.linspace(k_max, 4, len(omegar))

fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel(r'$k\lambda_{De}$')
ax1.set_ylabel(r'$\omega/\omega_i$', color=color)
ax1.plot(k, omegar, color=color)
ax1.tick_params(axis='y', labelcolor=color)
# ax1.set_ylim([-40, 100])
# ax1.set_xlim([0, k_min])
ax1.axhline(0, color='k')

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel(r'$\gamma$', color=color)
ax2.plot(k, omegai, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim([-4, 10])
ax2.axhline(0, color='k')

plt.show()
