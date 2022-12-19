import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
import data_analysis.read_data as data

path = '/media/ck/15814792801/11.6/'


# path = '/home/ck/Documents/hybrid2D_PUI/data/'
def read_phase(path, t):
    file = str(t) + '.txt'
    file_name = path + file
    f = open(file_name, 'r')
    # j = []
    x = []
    y = []
    vx = []
    vy = []
    vz = []

    for line in f.readlines():
        line = line.split()
        if not line:
            break
        x.append(float(line[0]))
        y.append(float(line[1]))
        vx.append(float(line[2]))
        vy.append(float(line[3]))
        vz.append(float(line[4]))
        # vz.append(float(line[5]))
    f.close()
    return vx, vy, vz, x, y


for t in range(6000, 7001, 500):
    index = 0
    vx, vy, vz, x, y = read_phase(path, t)
    # i = np.array(i)
    vx = np.array(vx)
    vy = np.array(vy)
    vz = np.array(vz)
    x = np.array(x)
    y = np.array(y)
    # for j in i:
    #     if j == 0:
    #         index += 1
    #     else:
    #         break
    # vx_pui = vx[:index - 1]
    # vy_pui = vy[:index - 1]
    # vz_pui = vz[:index - 1]
    # x_pui = x[:index - 1]
    #
    # # background proton
    # vx = vx[index + 1:]
    # vy = vy[index + 1:]
    # vz = vz[index + 1:]
    # x = x[index + 1:]
    #
    # # pick-up ions perpendicular velocity and parallel velocity
    # v_per = np.sqrt(vy_pui ** 2 + vz_pui ** 2)
    # v_para = vx_pui

    # # background ions
    v_b_per = np.sqrt(vx ** 2 + vz ** 2)
    v_b_para = vy

    X, Y = np.mgrid[v_b_para.min():v_b_para.max():100j, v_b_per.min():v_b_per.max():100j]
    # X, Y = np.mgrid[-10:10:100j, 0:15:100j]

    # perform a kernel density estimate on the data
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([v_b_para, v_b_per])
    kernel = kde.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape) / 10.5

    # plot the results
    fig = plt.figure()

    plt.pcolormesh(X, Y, Z, shading='gouraud', cmap='jet')
    plt.xlim(-10, 10)
    plt.title(r'$t\Omega_i=$' + str(t * 0.02))
    plt.xlabel('$v_{||}/v_A$')
    plt.ylabel('$v_{\perp}/v_A$')
    plt.colorbar()
    plt.savefig('/media/ck/Samsung_T5/data/data/density' + str(t) + '.png')
    # plt.contour(X, Y, Z)

# # Create data: 200 points
# data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 3]], 200)
# x, y = data.T
#
# # Create a figure with 6 plot areas
# fig, axes = plt.subplots(ncols=6, nrows=1, figsize=(21, 5))
#
# # Everything sarts with a Scatterplot
# axes[0].set_title('Scatterplot')
# axes[0].plot(x, y, 'ko')
# # As you can see there is a lot of overplottin here!
#
# # Thus we can cut the plotting window in several hexbins
# nbins = 20
# axes[1].set_title('Hexbin')
# axes[1].hexbin(x, y, gridsize=nbins, cmap=plt.cm.BuGn_r)
#
# # 2D Histogram
# axes[2].set_title('2D Histogram')
# axes[2].hist2d(x, y, bins=nbins, cmap=plt.cm.BuGn_r)
#
# # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
# k = kde.gaussian_kde(data.T)
# xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
# zi = k(np.vstack([xi.flatten(), yi.flatten()]))
#
# # plot a density
# axes[3].set_title('Calculate Gaussian KDE')
# axes[3].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
#
# # add shading
# axes[4].set_title('2D Density with shading')
# axes[4].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
#
# # contour
# axes[5].set_title('Contour')
# axes[5].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
# axes[5].contour(xi, yi, zi.reshape(xi.shape))
plt.show()
