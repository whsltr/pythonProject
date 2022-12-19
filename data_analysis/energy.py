import numpy as np
import matplotlib.pyplot as plt
import data_analysis.read_data as data

# path = '/home/ck/Documents/hybrid2D_PUI/data/'

# path = '/home/kun/Downloads/data/oxygen/data/'
#
# (t1, particle_total1,
#  ex1, ey1, ez1, e_total1, bx1, by1, bz1, b_total1, total1) = data.read_B_energy(path)

path = '/home/kun/Downloads/data/0.01/data2/'

(t1, thermal_x1, thermal_y1, thermal_z1, thermal_total1, flow_x1, flow_y1, flow_z1, flow_total1,
 back_thermal_x1, back_thermal_y1, back_thermal_z1, back_thermal_total1,
 back_flow_x1, back_flow_y1, back_flow_z1, back_flow_total1, back_v_total1, particle_total1,
 ex1, ey1, ez1, e_total1, bx1, by1, bz1, b_total1, total1) = data.read_energy(path)

path = '/home/kun/Downloads/data/0.01/data/'

(t, thermal_x, thermal_y, thermal_z, thermal_total, flow_x, flow_y, flow_z, flow_total,
 back_thermal_x, back_thermal_y, back_thermal_z, back_thermal_total,
 back_flow_x, back_flow_y, back_flow_z, back_flow_total, back_v_total, particle_total,
 ex, ey, ez, e_total, bx, by, bz, b_total, total) = data.read_energy(path)

T = 0.01
t = np.array(t) * T
thermal_z = np.array(thermal_z)
thermal_x = np.array(thermal_x)
thermal_y = np.array(thermal_y)
thermal_total = np.array(thermal_total)
ex = np.array(ex)
ey = np.array(ey)
ez = np.array(ez)
e_total = np.array(e_total)
bx = np.array(bx)
by = np.array(by)
bz = np.array(bz)
b_total = np.array(b_total)

ex1 = np.array(ex1)
ey1 = np.array(ey1)
ez1 = np.array(ez1)
e_total1 = np.array(e_total1)
bx1 = np.array(bx1)
by1 = np.array(by1)
bz1 = np.array(bz1)
b_total1 = np.array(b_total1)
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14}
fig = plt.figure(figsize=(10, 10))
plt.subplot(4, 1, 1)
plt.plot(t, thermal_x, 'r-', lw=2, label='x')
plt.plot(t, thermal_y, 'g-', lw=2, label='y')
plt.plot(t, thermal_z, 'b-', lw=2, label='z')
plt.plot(t, thermal_total, 'k-', lw=2, label='total')
# plt.legend(ncol=4, bbox_to_anchor=(0.35, 0.6))
plt.legend(ncol=4)
plt.ylabel('Thermal energy')

plt.subplot(4, 1, 2)
# plt.plot(t, np.log((thermal_z + thermal_y) / thermal_x / 2), 'k-', lw=2)
plt.plot(t, (thermal_z + thermal_y) / thermal_x / 2, 'k-', lw=2)
# plt.xlabel(r'$t\Omega_i')
plt.ylabel(r'$log(T_{\perp}/T_{||})$')

plt.subplot(4, 1, 3)
plt.plot(t, flow_x, 'r-', lw=2, label='x')
plt.plot(t, flow_y, 'g-', lw=2, label='y')
plt.plot(t, flow_z, 'b-', lw=2, label='z')
plt.plot(t, flow_total, 'k-', lw=2, label='total')
plt.legend(loc='best', ncol=4)
# plt.xlabel(r'$t\Omega$')
plt.ylabel(r'Bulk flow kinetic energy', font1)

plt.subplot(4, 1, 4)
plt.plot(t, particle_total, 'k-', lw=2)
plt.xlabel(r'$t\Omega$')
plt.ylabel(r'Total kinetic energy')
# plt.title('energy')
# plt.title('energy evaluation')
# plt.show()
# plt.savefig('./energy1.png')

# plot field energy

fig = plt.figure(figsize=(10, 10))
# plt.subplot(4, 1, 1)
# plt.plot(t, total, 'k-', lw=2, label='total')
# # plt.plot(t, e_total, 'g-', lw=2, label='electric')
# # plt.plot(t, b_total, 'r-', lw=2, label='magnetic')
# # plt.plot(t, particle_total, 'b-', lw=2, label='particle')
# plt.ylabel(r'$E_{total}$', font1)
# # plt.xscale("log")
# plt.tick_params(labelsize=13)
# plt.xticks(labels=None)
# plt.text(81, 2.9, 'a', font1)
# # plt.legend(ncol=4)

plt.subplot(3, 1, 1)
plt.plot(t, np.log((thermal_z + thermal_y) / thermal_x / 2), 'k-', lw=2)
plt.ylabel(r'$log(T_{\perp}/T_{||})$', font1)
# plt.plot(t, np.log((thermal_y1 + thermal_z1) / thermal_x1 / 2), 'k--', lw=2)
# plt.xscale("log")
plt.tick_params(labelsize=13)
plt.text(81, 3.5, 'a', font1)

plt.subplot(3, 1, 2)
plt.plot(t, bx - 1, 'g-', lw=2, label='$B_x$')
plt.plot(t, by, 'r', lw=2, label='$B_y$')
plt.plot(t, bz, 'b-', lw=2, label='$B_z$')
plt.plot(t, (b_total - 1), 'k-', lw=2, label='$B_{total}$')
# plt.plot(t, bx1 - 1, 'g--', lw=2, label='$B_x$')
# plt.plot(t, by1, 'r--', lw=2, label='$B_y$')
# plt.plot(t, bz1, 'b--', lw=2, label='$B_z$')
# plt.plot(t, (b_total1 - 1), 'k--', lw=2, label='$B_{total}$')
plt.ylabel(r'$\delta B^2/B_0^2$', font1)
plt.tick_params(labelsize=13)
plt.yscale("log")
# plt.xscale("log")
plt.ylim(10 ** -6, 10 ** 1)
plt.legend(ncol=4, loc='upper left')
plt.text(81, 1, 'b', font1)

plt.subplot(3, 1, 3)
plt.plot(t, ex, 'g-', lw=2, label='$E_x$')
plt.plot(t, ey, 'r-', lw=2, label='$E_y$')
plt.plot(t, ez, 'b-', lw=2, label='$E_z$')
plt.plot(t, e_total, 'k-', lw=2, label='$E_{total}$')
# plt.plot(t, ex1, 'g--', lw=2, label='$E_x$')
# plt.plot(t, ey1, 'r--', lw=2, label='$E_y$')
# plt.plot(t, ez1, 'b--', lw=2, label='$E_z$')
# plt.plot(t, e_total1, 'k--', lw=2, label='$E_{total}$')
plt.ylim(10 ** -14, 10 ** -7)
plt.ylabel(r'$\delta E^2/B_0^2$', font1)
plt.xlabel(r'$t\Omega_p$', font1)
plt.tick_params(labelsize=13)
plt.text(81, 10**-8, 'c', font1)
plt.legend(ncol=4, loc='upper left')
plt.yscale("log")
# plt.xscale("log")
# plt.xlim(5, 80)
plt.savefig(path + 'energy2.png')

plt.show()
