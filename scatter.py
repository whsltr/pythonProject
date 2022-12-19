import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from matplotlib.patches import ConnectionPatch

x = np.linspace(0, 10, 1000)
y = np.sqrt(100 - x ** 2)
theta = 30
x_locate = 10 * np.cos(theta / 360 * 2 * np.pi)
y_locate = 10 * np.sin(theta / 360 * 2 * np.pi)
print(x_locate)
v_phase = 1.21
c = (x_locate ** 2 + y_locate ** 2) / 2 - v_phase * x_locate
y1 = np.sqrt(2 * c - x ** 2 + 2 * v_phase * x)
y1_rev = np.sqrt(2 * c - x ** 2 - 2 * v_phase * x)

v_phase1 = 8.36
c1 = (x_locate ** 2 + y_locate ** 2) / 2 - v_phase1 * x_locate
y2 = np.sqrt(2 * c1 - x ** 2 + 2 * v_phase1 * x)
y2_rev = np.sqrt(2 * c1 - x ** 2 - 2 * v_phase1 * x)

v_phase2 = 1
c1 = (x_locate ** 2 + y_locate ** 2) / 2 - v_phase2 * x_locate
y3 = np.sqrt(2 * c1 - x ** 2 + 2 * v_phase2 * x)
y3_rev = np.sqrt(2 * c1 - x ** 2 - 2 * v_phase2 * x)

v_r = 10 * np.sin(theta / 360 * 2 * np.pi)
v_d = 10 * np.cos(theta / 360 * 2 * np.pi)
v_x = np.linspace(v_d - 0.5, v_d + 0.5, 1000)
v_y = np.linspace(v_r - 0.5, v_r + 0.5, 1000)
v_x, v_y = np.meshgrid(v_x, v_y)
v_t1 = 0.1
v_t2 = 0.1
a_s = np.exp(-v_r ** 2 / (2 * v_t2 ** 2)) + np.sqrt(np.pi) * v_r / (np.sqrt(2) * v_t2) * erfc(
    -v_r / np.sqrt(2) / v_t2) * v_t2
f_v = np.exp(-(v_x - v_d) ** 2 / 2 / v_t1 ** 2) * np.exp(-(v_y - v_r) ** 2 / 2 / v_t2 ** 2) / a_s
f_v[f_v < 0.1 * np.max(f_v)] = np.nan
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 16}

fig = plt.figure(figsize=(15, 6))
ax1 = plt.subplot(1, 2, 1)
plt.contourf(v_x, v_y, f_v, 7, cmap='jet')
plt.plot(x, y, color='grey')
plt.plot(-x, y, color='grey')
plt.plot(x, y1, color='red')
plt.plot(-x, y1_rev, color='red')
plt.plot(x, y2, color='g')
plt.plot(-x, y2_rev, color='g')
# plt.plot(x, y3, color='blue')
# plt.plot(-x, y3_rev, color='blue')
# plt.arrow(10.1, 0, 0, 1, width=0.02, head_width=0.1, ec='green')
# plt.arrow(v_d, v_r, 2*v_r/np.sqrt(v_r**2+(v_d-v_phase)**2), -2*(v_d-v_phase)/np.sqrt(v_r**2+(v_d-v_phase)**2),
#           width=0.05, head_width=0.2, ec='red', fc='red')
# plt.arrow(v_d, v_r, -2*v_r/np.sqrt(v_r**2+(v_d-v_phase1)**2), 2*(v_d-v_phase1)/np.sqrt(v_r**2+(v_d-v_phase1)**2),
#           width=0.05, head_width=0.2, ec='green', fc='green')
# plt.arrow(v_d, v_r, -2*v_r/np.sqrt(v_r**2+(v_d-v_phase2)**2), 2*(v_d-v_phase2)/np.sqrt(v_r**2+(v_d-v_phase2)**2),
#           width=0.05, head_width=0.2, ec='blue', fc='blue')
plt.axvline(8.518, color='green', linestyle=':')
plt.axvline(8.627, color='r', linestyle=':')
plt.plot(np.array([0, np.sqrt(0)]), np.array([0, np.sqrt(100)]), '--', color='grey')
plt.plot(np.array([v_phase, v_d]), np.array([0, v_r]), '--', color='red')
plt.plot(np.array([v_phase1, v_d]), np.array([0, v_r]), '--', color='g')
# plt.plot(np.array([v_phase2, v_d]), np.array([0, v_r]), '--', color='blue')
plt.xlim(0, 12)
plt.ylim(0, 12)
plt.xlabel(r'$v_\parallel/v_A$', font=font1)
plt.ylabel(r'$v_\perp/v_A$', font=font1)
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
plt.plot([7.5, 9.5], [4, 4], 'k')
plt.plot([7.5, 9.5], [6, 6], 'k')
plt.plot([7.5, 7.5], [4, 6], 'k')
plt.plot([9.5, 9.5], [4, 6], 'k')
plt.text(0.5, 11, 'a', fontsize=20)


ax2 = plt.subplot(122)
plt.contourf(v_x, v_y, f_v, 4, cmap='jet')
plt.plot(x, y, color='grey')
plt.plot(-x, y, color='grey')
plt.plot(x, y1, color='red')
plt.plot(-x, y1_rev, color='red')
plt.plot(x, y2, color='g')
plt.plot(-x, y2_rev, color='g')
# plt.plot(x, y3, color='blue')
# plt.plot(-x, y3_rev, color='blue')
# plt.arrow(10.1, 0, 0, 1, width=0.02, head_width=0.1, ec='green')
plt.arrow(8.6, 5.1, -0.2*v_r/np.sqrt(v_r**2+(v_d-v_phase)**2), 0.2*(v_d-v_phase)/np.sqrt(v_r**2+(v_d-v_phase)**2),
          width=0.01, head_width=0.05, ec='red', fc='red')
plt.arrow(8.55, 5.01, -0.2*v_r/np.sqrt(v_r**2+(v_d-v_phase1)**2), 0.2*(v_d-v_phase1)/np.sqrt(v_r**2+(v_d-v_phase1)**2),
          width=0.01, head_width=0.05, ec='green', fc='green')
# plt.arrow(v_d, v_r, -2*v_r/np.sqrt(v_r**2+(v_d-v_phase2)**2), 2*(v_d-v_phase2)/np.sqrt(v_r**2+(v_d-v_phase2)**2),
#           width=0.05, head_width=0.2, ec='blue', fc='blue')
plt.axvline(8.55, color='green', linestyle=':')
plt.axvline(8.6, color='r', linestyle=':')
plt.plot(np.array([0, np.sqrt(0)]), np.array([0, np.sqrt(100)]), '--', color='grey')
plt.plot(np.array([v_phase, v_d]), np.array([0, v_r]), '--', color='red')
plt.plot(np.array([v_phase1, v_d]), np.array([0, v_r]), '--', color='g')
# plt.plot(np.array([v_phase2, v_d]), np.array([0, v_r]), '--', color='blue')
plt.xlim(7.5, 9.5)
plt.ylim(4, 6)
plt.xlabel(r'$v_\parallel/v_A$', font=font1)
plt.ylabel(r'$v_\perp/v_A$', font=font1)
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
plt.text(7.6, 5.8, 'b', fontsize=20)
con = ConnectionPatch((9.5, 6), (7.5, 6), coordsA=ax1.transData, coordsB=ax2.transData)
con1 = ConnectionPatch((9.5, 4), (7.5, 4), coordsA=ax1.transData, coordsB=ax2.transData)
fig.add_artist(con)
fig.add_artist(con1)

plt.show()
