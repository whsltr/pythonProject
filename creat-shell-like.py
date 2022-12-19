import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc, erf


def a_s(theta, v_t1, v_t2):
    v_r = 2 * np.sin(theta / 360 * 2 * np.pi)
    return (np.exp(-v_r ** 2 / (2 * v_t2 ** 2)) + np.sqrt(np.pi) * v_r / (np.sqrt(2) * v_t2) * erfc(
        -v_r / np.sqrt(2) / v_t2)) * np.sqrt(2 * np.pi) * v_t1 * 2 * np.pi * v_t2 ** 2


v_t1 = 0.45
v_t2 = 0.45
v_s = 2
f_v = 0
v_x = np.linspace(-5, 5, 1000)
v_y = np.linspace(0, 5, 1000)
print(v_y[1])
v_x, v_y = np.meshgrid(v_x, v_y)
i = 0
as_total = sum(a_s(theta, v_t1, v_t2) for theta in range(-10, 191, 10))
vt = 0.45 * np.sqrt(2)
c_s = np.pi * vt**3 * (2*np.exp(-(v_s**2/vt**2))*v_s/vt + np.sqrt(np.pi)*(2*v_s**2/vt**2+1) * (1+erf(v_s/vt)))
f_s = 1 / c_s * np.exp(-(np.sqrt(v_x**2+v_y**2)-v_s)**2/vt**2)

for theta in range(-20, 201, 10):
    v_d = 2 * np.cos(theta/360 * 2 * np.pi)
    v_r = 2 * np.sin(theta/360 * 2 * np.pi)
    eta = a_s(theta, v_t1, v_t2) / as_total
    # a_s = np.sqrt(2*np.pi) * v_t1 * 2 * np.pi * v_t2**2
    f_v_temp = np.exp(-(v_x - v_d) ** 2 / 2 / v_t1 ** 2) * np.exp(-(v_y - v_r) ** 2 / 2 / v_t2 ** 2) / as_total
    f_v = f_v + f_v_temp

del_f = f_v - f_s
print(np.sum(f_v * v_y * 2 * np.pi * 0.01 * 0.005))
# f_v[f_v < 0.1 * np.max(f_v)] = np.nan
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14}
plt.figure()
plt.pcolormesh(v_x, v_y, f_v, cmap='jet')
plt.colorbar()
plt.figure()
plt.pcolormesh(v_x, v_y, f_s, cmap='jet')
plt.colorbar()
plt.figure()
plt.pcolormesh(v_x, v_y, del_f, cmap='jet')
plt.colorbar()
plt.show()

n1 = np.array(
    [0.0002, 0.0005, 0.0005, 0.001, 0.0009, 0.0016, 0.0024, 0.0029, 0.0029, 0.0024, 0.0016, 0.0009, 0.001, 0.0005,
     0.0005, 0.0002]) * 10
vtpar1 = np.array(
    [0.326, 0.2795, 0.2795, 0.2562, 0.2562, 0.2329, 0.2329, 0.1863, 0.1863, 0.2329, 0.2329, 0.2562, 0.2562, 0.2795,
     0.2795, 0.3260]) / np.sqrt(2)
vtper1 = vtpar1

vr1 = np.array(
    [0.2141, 0.3583, 0.5091, 0.6491, 0.7559, 0.7950, 0.8361, 0.8612, 0.8612, 0.8361, 0.7726, 0.7559, 0.6323, 0.5091,
     0.3492, 0.2087])
vd1 = np.array(
    [0.97, 0.9261, 0.8213, 0.7126, 0.5705, 0.3958, 0.2386, 0.0794, -0.0794, -0.2383, -0.3847, -0.5705, -0.6941, -0.8213,
     -0.9027, -0.9455])

fun = 0
for i in range(16):
    fun_temp = n1[i] * np.exp(-((v_x-vd1[i])**2 + (v_y-vr1[i])**2)/vtper1[i]**2)
    fun = fun + fun_temp

plt.pcolormesh(v_x, v_y, fun, cmap='jet')
plt.colorbar()
plt.show()
