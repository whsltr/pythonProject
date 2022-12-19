import numpy as np
import matplotlib.pyplot as plt
import data_analysis.read_data as data
import math

path = '/media/ck/Samsung_T5/data/data/'
v_ring = 3
time, vx, vy, vz, x, y = data.read_phase(path)
time = np.array(time, dtype=np.float)
vx = np.array(vx)
vy = np.array(vy)
vz = np.array(vz)
x = np.array(x)
y = np.array(y)
vxz = np.sqrt(vx ** 2 + vz ** 2)
erf = math.erf(v_ring)
vx.dtype
vx_max = np.amax(vx)
vy_max = np.amax(vy)
vz_max = np.amax(vz)
vxz_max = np.amax(vxz)
vx_min = np.min(vx)
vy_min = np.min(vy)
vz_min = np.min(vz)
vxz_min = np.min(vxz)
vx_array = np.linspace(vx_min, vx_max, num=200)
vy_array = np.linspace(vy_min, vy_max, num=200)
vz_array = np.linspace(vz_min, vz_max, num=200)
vxz_array = np.linspace(vxz_min, vxz_max, num=200)
fv_ring = vxz_array * np.exp(-(vxz_array - v_ring)**2) * 2 / (np.exp(-v_ring**2) + np.sqrt(np.pi) * v_ring * (1 + erf))
fv_maxwell = np.exp(-(vz_array**2))/ np.sqrt(np.pi)
fx = np.zeros(200)
fx = np.array(fx)

# for i in vx:
#     for j in range(199):
#         if vz_array[j] <= i < vz_array[j + 1]:
#             fx[j] += 1

num_bin = 100
ax = plt.subplot()
n, bins, patches = ax.hist(vxz, num_bin, density=True)
print(n, bins, patches)
plt.plot(vz_array, fv_maxwell)
plt.plot(vxz_array, fv_ring)
plt.show()









