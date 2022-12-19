import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, mu_0, epsilon_0, k, e, m_e, proton_mass

B = 6.4e-9
n = 11.0e6
T = 14.8
v = 360e3
N = 2048 * 400
theta = 55
# charge exchange rate
gama = 3.4e-4
beta = n * T * 1.6e-19
v_the = np.sqrt(2 * T * 1.6e-19 / (m_e))
pressureb = 1 / 2 / mu_0 * B**2
beta = beta / pressureb
wave_parameter = n * e / epsilon_0 / B
print('beta=', beta)
print('v_the=', v_the)

v_a = B / np.sqrt(mu_0*proton_mass*n)
E_n = (proton_mass * v_a ** 2) / 1.6e-19
E = 2*T / E_n
omega_i = e*B / proton_mass
omega_e = e*B/m_e
omega_pi = n * e ** 2 / epsilon_0 / proton_mass
omega_ei = n * e ** 2 / epsilon_0/m_e
print('omega_e=', omega_e, '  omega_i=', omega_i, ' omega_pe=', np.sqrt(omega_ei), ' omega_pi=', np.sqrt(omega_pi))
cycle = 2 * np.pi / omega_i
deltaT = 0.025 * cycle
rate = int(deltaT * gama * N)
# print('cyclotron frequency is', omega_i)
print('cycle time is', cycle)
print('V_a is', v_a)
print('v_th=', v_the / v_a / np.sqrt(1836))
print('sqrt(beta)=', np.sqrt(beta))
print('v_beam=', v / v_a)
print('c/v_a=', c / v_a)
print('v_d=', v * np.cos(55/360 * 2 * np.pi) / v_a)
print('v_r=', v * np.sin(55/360 * 2 * np.pi) / v_a)
print(np.arctan(6.25/0.15) / (2*np.pi) * 360)
# print('normalization beta is', E, 'eV')
# print(E_n)
# print('1 is represent', E_n, 'eV')
# print(1 / (3.5e-7 * 3.5 * 0.025))
# print('number of particle exchanged per step is', rate)
# print('beta is ', beta)



