import numpy as np
import matplotlib.pyplot as plt

pth = '/home/kun/Documents/mathematics/beam/0.01/' + '10va_01'
f = open(pth + '.txt')
omegai1 = []
omegar1 = []
polarization1 = []
eta1 = []
nx = 240
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
    eta1.append(float(line[4]))

f.close()
omegai1 = np.array(omegai1).reshape(-1, nx)
omegar1 = np.array(omegar1).reshape(-1, nx)
polarization1 = np.array(polarization1).reshape(-1, nx)
eta1 = np.array(eta1).reshape(-1, nx)

pth = '/home/kun/Documents/mathematics/beam/0.01/' + '10va_11_temp'
f = open(pth + '.txt')
omegai1_temp = []
omegar1_temp = []
polarization1_temp = []
eta1_temp = []
nx = 240
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
    omegar1_temp.append(float(line[1]))
    omegai1_temp.append(float(line[2]))
    polarization1_temp.append(float(line[3]))
    eta1_temp.append(float(line[4]))

f.close()
omegai1_temp = np.array(omegai1_temp).reshape(-1, nx)
omegar1_temp = np.array(omegar1_temp).reshape(-1, nx)
polarization1_temp = np.array(polarization1_temp).reshape(-1, nx)
eta1_temp = np.array(eta1_temp).reshape(-1, nx)
omegar1[:199, :] = omegar1_temp
omegai1[:199, :] = omegai1_temp
polarization1[:199, :] = polarization1_temp
eta1[:199, :] = eta1_temp

pth = '/home/kun/Documents/mathematics/beam/0.01/' + '10va_00'
f = open(pth + '.txt')
omegai0 = []
omegar0 = []
polarization0 = []
eta0 = []
nx1 = 50
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
    eta0.append(format(line[4]))

f.close()
omegai0 = np.array(omegai0).reshape(-1, nx1)
omegar0 = np.array(omegar0).reshape(-1, nx1)
polarization0 = np.array(polarization0).reshape(-1, nx1)
eta0 = np.array(eta0).reshape(-1, nx1)
omegai0 = omegai0[:, ::-1]
omegar0 = omegar0[:, ::-1]
polarization0 = polarization0[:, ::-1]
eta0 = eta0[:, ::-1]

omegai10 = np.concatenate((omegai0, omegai1), axis=1)
omegar10 = np.concatenate((omegar0, omegar1), axis=1)
polarization10 = np.concatenate((polarization0, polarization1), axis=1)
eta10 = np.concatenate((eta0, eta1), axis=1)

pth = '/home/kun/Documents/mathematics/beam/0.01/' + '10va_41'
f = open(pth + '.txt')
omegai1 = []
omegar1 = []
polarization1 = []
eta1 = []
nx = 240
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
    eta1.append(float(line[4]))

f.close()
omegai1 = np.array(omegai1).reshape(-1, nx)
omegar1 = np.array(omegar1).reshape(-1, nx)
polarization1 = np.array(polarization1).reshape(-1, nx)
eta1 = np.array(eta1).reshape(-1, nx)

pth = '/home/kun/Documents/mathematics/beam/0.01/' + '10va_40'
f = open(pth + '.txt')
omegai0 = []
omegar0 = []
polarization0 = []
eta0 = []
nx1 = 50
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
    eta0.append(float(line[4]))

f.close()
omegai0 = np.array(omegai0).reshape(-1, nx1)
omegar0 = np.array(omegar0).reshape(-1, nx1)
polarization0 = np.array(polarization0).reshape(-1, nx1)
eta0 = np.array(eta0).reshape(-1, nx1)
omegai0 = omegai0[:, ::-1]
omegar0 = omegar0[:, ::-1]
polarization0 = polarization0[:, ::-1]
eta0 = eta0[:, ::-1]

omegai1 = np.concatenate((omegai0, omegai1), axis=1)
omegar1 = np.concatenate((omegar0, omegar1), axis=1)
polarization1 = np.concatenate((polarization0, polarization1), axis=1)
eta1 = np.concatenate((eta0, eta1), axis=1)
# omegai = omegai[:, ::-12]
omegai1 = np.concatenate((omegai10, omegai1), axis=0)
omegar1 = np.concatenate((omegar10, omegar1), axis=0)
polarization1 = np.concatenate((polarization10, polarization1), axis=0)
eta1 = np.concatenate((eta10, eta1), axis=0)

pth = '/home/kun/Documents/mathematics/beam/0.01/' + '10va_661'
f = open(pth + '.txt')
omegai1_temp = []
omegar1_temp = []
polarization1_temp = []
eta1_temp = []
nx_temp = 25
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
    omegar1_temp.append(float(line[1]))
    omegai1_temp.append(float(line[2]))
    polarization1_temp.append(float(line[3]))
    eta1_temp.append(float(line[4]))

f.close()
omegai1_temp = np.array(omegai1_temp).reshape(-1, nx_temp)
omegar1_temp = np.array(omegar1_temp).reshape(-1, nx_temp)
polarization1_temp = np.array(polarization1_temp).reshape(-1, nx_temp)
eta1_temp = np.array(eta1_temp).reshape(-1, nx_temp)

pth = '/home/kun/Documents/mathematics/beam/0.01/' + '10va_660'
f = open(pth + '.txt')
omegai_temp = []
omegar_temp = []
polarization_temp = []
eta_temp = []
nx_temp1 = 25
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
    omegar_temp.append(float(line[1]))
    omegai_temp.append(float(line[2]))
    polarization_temp.append(float(line[3]))
    eta_temp.append(float(line[4]))

f.close()
omegai_temp = np.array(omegai_temp).reshape(-1, nx_temp1)
omegar_temp = np.array(omegar_temp).reshape(-1, nx_temp1)
polarization_temp = np.array(polarization_temp).reshape(-1, nx_temp1)
eta_temp = np.array(eta_temp).reshape(-1, nx_temp1)
omegai_temp = omegai_temp[:, ::-1]
omegar_temp = omegar_temp[:, ::-1]
polarization_temp = polarization_temp[:, ::-1]
eta_temp = eta_temp[:, ::-1]
omegai_temp = np.concatenate((omegai_temp, omegai1_temp), axis=1)
omegar_temp = np.concatenate((omegar_temp, omegar1_temp), axis=1)
polarization_temp = np.concatenate((polarization_temp, polarization1_temp), axis=1)
eta_temp = np.concatenate((eta_temp, eta1_temp), axis=1)

omegai1[64:99, 75:125] = omegai_temp
omegar1[64:99, 75:125] = omegar_temp
polarization1[64:99, 75:125] = polarization_temp
eta1[64:99, 75:125] = eta_temp
pth = '/home/kun/Documents/mathematics/beam/0.01/' + '10va_11'
f = open(pth + '.txt')
omegai1_temp = []
omegar1_temp = []
polarization1_temp = []
eta1_temp = []
nx_temp = 50
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
    omegar1_temp.append(float(line[1]))
    omegai1_temp.append(float(line[2]))
    polarization1_temp.append(float(line[3]))
    eta1_temp.append(float(line[4]))

f.close()
omegai1_temp = np.array(omegai1_temp).reshape(-1, nx_temp)
omegar1_temp = np.array(omegar1_temp).reshape(-1, nx_temp)
polarization1_temp = np.array(polarization1_temp).reshape(-1, nx_temp)
eta1_temp = np.array(eta1_temp).reshape(-1, nx_temp)

pth = '/home/kun/Documents/mathematics/beam/0.01/' + '10va_10'
f = open(pth + '.txt')
omegai_temp = []
omegar_temp = []
polarization_temp = []
eta_temp = []
nx_temp1 = 25
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
    omegar_temp.append(float(line[1]))
    omegai_temp.append(float(line[2]))
    polarization_temp.append(float(line[3]))
    eta_temp.append(float(line[4]))

f.close()
omegai_temp = np.array(omegai_temp).reshape(-1, nx_temp1)
omegar_temp = np.array(omegar_temp).reshape(-1, nx_temp1)
polarization_temp = np.array(polarization_temp).reshape(-1, nx_temp1)
eta_temp = np.array(eta_temp).reshape(-1, nx_temp1)
omegai_temp = omegai_temp[:, ::-1]
omegar_temp = omegar_temp[:, ::-1]
polarization_temp = polarization_temp[:, ::-1]
eta_temp = eta_temp[:, ::-1]
omegai_temp = np.concatenate((omegai_temp, omegai1_temp), axis=1)
omegar_temp = np.concatenate((omegar_temp, omegar1_temp), axis=1)
polarization_temp = np.concatenate((polarization_temp, polarization1_temp), axis=1)
eta_temp = np.concatenate((eta_temp, eta1_temp), axis=1)

omegai1[99:138, 90:165] = omegai_temp
omegar1[99:138, 90:165] = omegar_temp
polarization1[99:138, 90:165] = polarization_temp
eta1[99:138, 90:165] = eta_temp

pth = '/home/kun/Documents/mathematics/beam/0.01/' + '10va_141'
f = open(pth + '.txt')
omegai1_temp = []
omegar1_temp = []
polarization1_temp = []
eta1_temp = []
nx_temp = 15
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
    omegar1_temp.append(float(line[1]))
    omegai1_temp.append(float(line[2]))
    polarization1_temp.append(float(line[3]))
    eta1_temp.append(float(line[4]))

f.close()
omegai1_temp = np.array(omegai1_temp).reshape(-1, nx_temp)
omegar1_temp = np.array(omegar1_temp).reshape(-1, nx_temp)
polarization1_temp = np.array(polarization1_temp).reshape(-1, nx_temp)
eta1_temp = np.array(eta1_temp).reshape(-1, nx_temp)

pth = '/home/kun/Documents/mathematics/beam/0.01/' + '10va_140'
f = open(pth + '.txt')
omegai_temp = []
omegar_temp = []
polarization_temp = []
eta_temp = []
nx_temp1 = 15
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
    omegar_temp.append(float(line[1]))
    omegai_temp.append(float(line[2]))
    polarization_temp.append(float(line[3]))
    eta_temp.append(float(line[4]))

f.close()
omegai_temp = np.array(omegai_temp).reshape(-1, nx_temp1)
omegar_temp = np.array(omegar_temp).reshape(-1, nx_temp1)
polarization_temp = np.array(polarization_temp).reshape(-1, nx_temp1)
eta_temp = np.array(eta_temp).reshape(-1, nx_temp1)
omegai_temp = omegai_temp[:, ::-1]
omegar_temp = omegar_temp[:, ::-1]
polarization_temp = polarization_temp[:, ::-1]
eta_temp = eta_temp[:, ::-1]
omegai_temp = np.concatenate((omegai_temp, omegai1_temp), axis=1)
omegar_temp = np.concatenate((omegar_temp, omegar1_temp), axis=1)
polarization_temp = np.concatenate((polarization_temp, polarization1_temp), axis=1)
eta_temp = np.concatenate((eta_temp, eta1_temp), axis=1)

omegai1[138:170, 110:140] = omegai_temp
omegar1[138:170, 110:140] = omegar_temp
polarization1[138:170, 110:140] = polarization_temp
eta1[138:170, 110:140] = eta_temp
pth = '/home/kun/Documents/mathematics/beam/0.01/' + '10va_031'
f = open(pth + '.txt')
omegai1_temp = []
omegar1_temp = []
polarization1_temp = []
eta1_temp = []
nx_temp = 20
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
    omegar1_temp.append(float(line[1]))
    omegai1_temp.append(float(line[2]))
    polarization1_temp.append(float(line[3]))
    eta1_temp.append(float(line[4]))

f.close()
omegai1_temp = np.array(omegai1_temp).reshape(-1, nx_temp)
omegar1_temp = np.array(omegar1_temp).reshape(-1, nx_temp)
polarization1_temp = np.array(polarization1_temp).reshape(-1, nx_temp)
eta1_temp = np.array(eta1_temp).reshape(-1, nx_temp)

pth = '/home/kun/Documents/mathematics/beam/0.01/' + '10va_030'
f = open(pth + '.txt')
omegai_temp = []
omegar_temp = []
polarization_temp = []
eta_temp = []
nx_temp1 = 30
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
    omegar_temp.append(float(line[1]))
    omegai_temp.append(float(line[2]))
    polarization_temp.append(float(line[3]))
    eta_temp.append(float(line[4]))

f.close()
omegai_temp = np.array(omegai_temp).reshape(-1, nx_temp1)
omegar_temp = np.array(omegar_temp).reshape(-1, nx_temp1)
polarization_temp = np.array(polarization_temp).reshape(-1, nx_temp1)
eta_temp = np.array(eta_temp).reshape(-1, nx_temp1)
omegai_temp = omegai_temp[:, ::-1]
omegar_temp = omegar_temp[:, ::-1]
polarization_temp = polarization_temp[:, ::-1]
eta_temp = eta_temp[:, ::-1]
omegai_temp = np.concatenate((omegai_temp, omegai1_temp), axis=1)
omegar_temp = np.concatenate((omegar_temp, omegar1_temp), axis=1)
polarization_temp = np.concatenate((polarization_temp, polarization1_temp), axis=1)
eta_temp = np.concatenate((eta_temp, eta1_temp), axis=1)
polarization_temp[omegai_temp < 0.02] = 0
eta_temp[omegai_temp < 0.02] = 0
omegar_temp[omegai_temp < 0.02] = 0
omegai_temp[omegai_temp < 0.02] = 0
polarization_temp[abs(polarization_temp) < 0.001] = 0
polarization1[omegai1 < 0.02] = 0
eta1[omegai1 < 0.02] = 0
omegar1[omegai1 < 0.02] = 0
omegai1[omegai1 < 0.02] = 0
polarization1[abs(polarization1) < 0.001] = 0
omegai1[29:64, 45:95] += omegai_temp
omegar1[29:64, 45:95] += omegar_temp
polarization1[29:64, 45:95] += polarization_temp
eta1[29:64, 45:95] = eta_temp

polarization1[omegai1 < 0.02] = None
eta1[omegai1 < 0.02] = np.nan
omegar1[omegai1 < 0.02] = None
omegai1[omegai1 < 0.002] = None
polarization1[abs(polarization1) < 0.001] = 0


font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14}
x = np.linspace(0.02, 0.6, nx+nx1)
y = np.linspace(0.01, 8, 800)
X, Y = np.meshgrid(x, y)
# eta1 = (omegar1 - X*10 + 1)
fig = plt.figure()
plt.pcolormesh(X, Y, omegai1, shading='gouraud', cmap='jet', )
cb = plt.colorbar()
cb.ax.tick_params(labelsize=12)
cb.set_label(r'$\gamma/ \Omega_p$', fontdict=font1)

# contour = plt.contour(x, y, omegar1, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], cmap='cool', alpha=1, Nchunk=0.,
#                       linestyles='dotted')
# plt.clabel(contour, inline=1, )
contour = plt.contour(X, Y, omegar1, 8, alpha=1, Nchunk=0, linestyles='dotted', colors='w', inline=1)
contour1 = plt.contour(X, Y, eta1, [0], alpha=1, Nchunk=0, linestyles='--', colors='k', inline=1)
# con2 = contour.collections[0].get_paths()[2].vertices
manual_locations = [(0.17, 3), (0.226, 5.42), (0.28, 4.58), (3.5, 4.8), (4.07, 5.9), (4.67, 6.43), (5.26, 7)]
plt.clabel(contour, inline=1, manual=True)
plt.clabel(contour1, inline=1, manual=True)
plt.xlabel(r'$k_\parallel  \lambda_p$', font1)
plt.ylabel(r'$k_\perp \lambda_p$', font1)
plt.tick_params(labelsize=12)

plt.savefig(pth + '1.png')
plt.show()

