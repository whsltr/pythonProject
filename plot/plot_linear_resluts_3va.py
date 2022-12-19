import numpy as np
import matplotlib.pyplot as plt

pth = '/home/kun/Documents/mathematics/beam/0.01/' + '9va_01'
f = open(pth + '.txt')
omegai1 = []
omegar1 = []
polarization1 = []
nx = 100
for line in f.readlines():
    line = line.split()
    if not line:
        break
    if line[0] == 'Indeterminate':
        line[0] = 0
    if line[1] == 'Indeterminate':
        line[1] = 0
    if line[2] == 'Indeterminate':
        line[2] = 0
    omegar1.append(float(line[1]))
    omegai1.append(float(line[2]))
    polarization1.append(float(line[3]))

f.close()
omegai1 = np.array(omegai1).reshape(-1, nx)
omegar1 = np.array(omegar1).reshape(-1, nx)
polarization1 = np.array(polarization1).reshape(-1, nx)

pth = '/home/kun/Documents/mathematics/beam/0.01/' + '3va_00'
f = open(pth + '.txt')
omegai0 = []
omegar0 = []
polarization0 = []
nx1 = 100
for line in f.readlines():
    line = line.split()
    if not line:
        break
    if line[0] == 'Indeterminate':
        line[0] = 0
    if line[1] == 'Indeterminate':
        line[1] = 0
    if line[2] == 'Indeterminate':
        line[2] = 0
    omegar0.append(float(line[1]))
    omegai0.append(float(line[2]))
    polarization0.append(float(line[3]))

f.close()
omegai0 = np.array(omegai0).reshape(-1, nx1)
omegar0 = np.array(omegar0).reshape(-1, nx1)
polarization0 = np.array(polarization0).reshape(-1, nx1)
omegai0 = omegai0[:, ::-1]
omegar0 = omegar0[:, ::-1]
polarization0 = polarization0[:, ::-1]

omegai10 = np.concatenate((omegai0, omegai1), axis=1)
omegar10 = np.concatenate((omegar0, omegar1), axis=1)
polarization10 = np.concatenate((polarization0, polarization1), axis=1)

pth = '/home/kun/Documents/mathematics/beam/0.01/' + '3va_21'
f = open(pth + '.txt')
omegai1 = []
omegar1 = []
polarization1 = []
nx = 100
for line in f.readlines():
    line = line.split()
    if not line:
        break
    if line[0] == 'Indeterminate':
        line[0] = 0
    if line[1] == 'Indeterminate':
        line[1] = 0
    if line[2] == 'Indeterminate':
        line[2] = 0
    omegar1.append(float(line[1]))
    omegai1.append(float(line[2]))
    polarization1.append(float(line[3]))

f.close()
omegai1 = np.array(omegai1).reshape(-1, nx)
omegar1 = np.array(omegar1).reshape(-1, nx)
polarization1 = np.array(polarization1).reshape(-1, nx)

pth = '/home/kun/Documents/mathematics/beam/0.01/' + '3va_20'
f = open(pth + '.txt')
omegai0 = []
omegar0 = []
polarization0 = []
nx1 = 100
for line in f.readlines():
    line = line.split()
    if not line:
        break
    if line[0] == 'Indeterminate':
        line[0] = 0
    if line[1] == 'Indeterminate':
        line[1] = 0
    if line[2] == 'Indeterminate':
        line[2] = 0
    omegar0.append(float(line[1]))
    omegai0.append(float(line[2]))
    polarization0.append(float(line[3]))

f.close()
omegai0 = np.array(omegai0).reshape(-1, nx1)
omegar0 = np.array(omegar0).reshape(-1, nx1)
polarization0 = np.array(polarization0).reshape(-1, nx1)
omegai0 = omegai0[:, ::-1]
omegar0 = omegar0[:, ::-1]
polarization0 = polarization0[:, ::-1]

omegai1 = np.concatenate((omegai0, omegai1), axis=1)
omegar1 = np.concatenate((omegar0, omegar1), axis=1)
polarization1 = np.concatenate((polarization0, polarization1), axis=1)
# omegai = omegai[:, ::-12]
omegai1 = np.concatenate((omegai10, omegai1), axis=0)
omegar1 = np.concatenate((omegar10, omegar1), axis=0)
polarization1 = np.concatenate((polarization10, polarization1), axis=0)

polarization1[omegai1 < 0.002] = None
omegar1[omegai1 < 0.002] = None
omegai1[omegai1 < 0.002] = None
polarization1[abs(polarization1) < 0.001] = 0

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14}
x = np.linspace(0.2, 0.6, nx+nx1)
y = np.linspace(0.01, 4, 399)
X, Y = np.meshgrid(x, y)
eta = (omegar1 - X*10 + 1)
fig = plt.figure()
plt.pcolormesh(X, Y, omegai1, shading='gouraud', cmap='jet', )
cb = plt.colorbar()
cb.ax.tick_params(labelsize=12)
cb.set_label(r'$\gamma/ \Omega_i$', fontdict=font1)

# contour = plt.contour(x, y, omegar1, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], cmap='cool', alpha=1, Nchunk=0.,
#                       linestyles='dotted')
# plt.clabel(contour, inline=1, )
contour = plt.contour(X, Y, omegar1, 7, alpha=1, Nchunk=0, linestyles='dotted', colors='w')
# con2 = contour.collections[0].get_paths()[2].vertices
plt.clabel(contour, inline=1)
plt.xlabel(r'$k_\parallel  \lambda_i$', font1)
plt.ylabel(r'$k_\perp \lambda_i$', font1)
plt.tick_params(labelsize=12)

plt.savefig(pth + '.png')
plt.show()

