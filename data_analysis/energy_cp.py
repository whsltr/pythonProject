import numpy as np
import matplotlib.pyplot as plt
import data_analysis.read_data as data
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

path = '/home/kun/Downloads/data/maxwell/'
(t, thermal_x, thermal_y, thermal_z, thermal_total, flow_x, flow_y, flow_z, flow_total,
 back_thermal_x, back_thermal_y, back_thermal_z, back_thermal_total,
 back_flow_x, back_flow_y, back_flow_z, back_flow_total, back_v_total, particle_total,
 ex, ey, ez, e_total, bx, by, bz, b_total0, total) = data.read_energy(path)
b_total0 = np.array(b_total0)
b_total0 = b_total0 - 1

fig, ax = plt.subplots(figsize=(12, 8), ncols=1, nrows=2)

for i in [0.001, 0.002, 0.004, 0.008, 0.012, 0.016, 0.02]:
    path = '/home/kun/Downloads/data/' + str(i) + '/'
    # path = '/home/kun/Downloads/data/0.002/' + 'data' + str(i) + '/'

    (t, thermal_x, thermal_y, thermal_z, thermal_total, flow_x, flow_y, flow_z, flow_total,
     back_thermal_x, back_thermal_y, back_thermal_z, back_thermal_total,
     back_flow_x, back_flow_y, back_flow_z, back_flow_total, back_v_total, particle_total,
     ex, ey, ez, e_total, bx, by, bz, b_total, total) = data.read_energy(path)
    font1 = {'family': 'Computer Modern Roman',
             'weight': 'normal',
             'size': 14}
    T = 0.01
    t = np.array(t) * T
    b_total = np.array(b_total)
    b_total = b_total - 1
    # ax.plot(t, b_total, label='$v_b=$' + str(4 + 2 * i) + '$v_A$', color=marker[i - 1])
    ax[0].plot(t, b_total, '-', label='$n_b=$' + str(i) + '$n_0$')
    # axins.plot(t, b_total, color=marker[i - 1])
    x1, x2, y1, y2 = 80, 90, 6 * 10 ** -4, 3.2e-3
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)
    # axins.yaxis.set_ticks_position('right')
    # plt.setp(axins.get_xticklabels(), visible=False)
    # plt.setp(axins.get_yticklabels(), visible=False)
    # axins.set_yticks([])
    # axins.set_xticks([])

axins = zoomed_inset_axes(ax[1], zoom=4, loc=4)
marker = ('r', 'y', 'g', 'c', 'b', 'm')
marker = ('lightblue', 'cadetblue', 'darkturquoise', 'c', 'darkcyan', 'darkslategray',)
mark_inset(ax[1], axins, loc1=3, loc2=1, ec='0.5')
for i in [1, 2, 3, 4, 5, 6]:
    path = '/home/kun/Downloads/data/0.002/' + 'data' + str(i) + '/'

    (t, thermal_x, thermal_y, thermal_z, thermal_total, flow_x, flow_y, flow_z, flow_total,
     back_thermal_x, back_thermal_y, back_thermal_z, back_thermal_total,
     back_flow_x, back_flow_y, back_flow_z, back_flow_total, back_v_total, particle_total,
     ex, ey, ez, e_total, bx, by, bz, b_total, total) = data.read_energy(path)
    font1 = {'family': 'Computer Modern Roman',
             'weight': 'normal',
             'size': 14}
    T = 0.025
    t = np.array(t) * T
    b_total = np.array(b_total)
    b_total = b_total - 1
    ax[1].plot(t, b_total, label='$v_b=$' + str(4 + 2 * i) + '$v_A$', color=marker[i - 1])
    axins.plot(t, b_total, color=marker[i - 1])
x1, x2, y1, y2 = 80, 90, 6 * 10 ** -4, 3.2e-3
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.yaxis.set_ticks_position('right')
plt.setp(axins.get_xticklabels(), visible=False)
plt.setp(axins.get_yticklabels(), visible=False)
axins.set_yticks([])
axins.set_xticks([])
axins.set_yscale('log')
ax[0].set_yscale('log')
ax[0].set_ylim(10**-6, 1)
ax[0].set_ylabel(r'$(\delta B/B_0)^2$', font1)
ax[0].tick_params(labelsize=12)
ax[0].text(200, 0.5, '(a)', fontsize=12, color='black')
plt.setp(ax[0].get_xticklabels(), visible=False)
ax[1].set_yscale('log')
ax[1].set_ylim(10**-6, 1)
# ax[1].plot(t, b_total0, label='maxwellian', color='k')
ax[1].set_ylabel(r'$(\delta B/B_0)^2$', font1)
ax[1].set_xlabel(r"$\Omega_i t$", font1)
ax[1].text(200, 0.5, '(b)', fontsize=12, color='black')
ax[1].tick_params(labelsize=12)
h = ax[0].legend()
h1 = ax[1].legend(ncol=3, fontsize=12)

# set(h, 'FontName', 'Times New Roman', 'FontSize', 11, 'FontWeight', 'normal')
plt.savefig('/home/kun/Downloads/data/energy.png', bbox_inches='tight')
plt.show()
