import numpy as np
import matplotlib.pyplot as plt
from cmath import pi, sin, cos, sqrt
# n1 is the beam electron, n0 is the background proton, n2 is the electron
n1 = 0.001
n0 = 1. - n1

qm0 = 1
qm1 = 1.

vd0 = 0
vd1 = 0
vd2 = -n1 / n0 * vd1
theta = 0 / 360 * 2 * np.pi

omegar = []
omegai = []

approxi = 0.001
k0 = 1.2
k_max = k0
dk = 0.02
k_min = 0.
omega = 1.7 + 0.001
def fun(k0, omega):
    R_up = n0 * omega ** 2 * (1 + omega - k0 * vd1) + n1 * (1 + omega) * (omega - k0 * vd1) ** 2
    R_down = (1 + omega) * omega ** 2 * (1 + omega - k0 * vd1)
    L_up = n0 * omega ** 2 * (1 - omega + k0 * vd1) + n1 * (1 - omega) * (omega - k0 * vd1) ** 2
    L_down = (1 - omega) * omega ** 2 * (1 - omega + k0 * vd1)
    return (k0 / omega) ** 4 * np.cos(theta) ** 2 * R_down * L_down - (k0 / omega) ** 2 * (
                    R_up * L_down + L_up * R_down) * (1 + np.cos(theta) ** 2) / 2 + R_up * L_up

while k0 > 0.02:

    err = 1
    delta = 10 ** -4
    it = 0
    n = 6
    x = 100
    while err > 10**-4:
        d_fun = (fun(k0, omega+10**-7) - fun(k0, omega-10**-7)) / (2*10**-7)
        dd_fun = (fun(k0, omega+10**-7) + fun(k0, omega-10**-7) - 2*fun(k0, omega)) / (10**-7)**2
        G = d_fun/fun(k0, omega)
        H = G**2 - dd_fun/fun(k0, omega)
        b = max(G+np.sqrt((n-1)*(n*H-G**2)), G-np.sqrt((n-1)*(n*H-G**2)))
        a = n/b
        omega = omega - a
        err = a if fun(k0, omega) < 10**-6 else 1
    k0 = k0 - dk
    print(k0, '    ', omega)

    omegar.append(omega.real)
    omegai.append(omega.imag)

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

