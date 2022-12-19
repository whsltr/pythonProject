# Libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import data_analysis.read_data as data

path = '/home/kun/Downloads/data/0.01/data/'
# path = '/home/ck/Documents/hybrid2D_PUI/data/'
for t in range(0, 8001, 500):
    vx = []
    vy = []
    vz = []
    x = []
    y = []
    # for rank in range(0, 25, 10):
    index = 0
    i, vx1, vy1, vz1, x1, y1 = data.read_phase(path, t)
    i = np.array(i)
    vx += vx1
    vy += vy1
    vz += vz1
    x += x1
    y += y1
    vx = np.array(vx)
    vx = vx
    vy = np.array(vy)
    vz = np.array(vz)
    x = np.array(x)
    y = np.array(y)
    for j in i:
        if j != 2:
            index += 1
        else:
            break
    vx_pui = vx[:index - 1]
    vy_pui = vy[:index - 1]
    vz_pui = vz[:index - 1]
    x_pui = x[:index - 1]
    #oxygen ions
    # vx_oxy = vx[index:]
    # vy_oxy = vy[index:]
    # vz_oxy = vz[index:]
    # x_oxy = x[index:]
    # y_oxy = y[index:]
    # v_per_oxy = np.sqrt(vy_oxy**2 + vz_oxy**2)
    # v_par_oxy = vx_oxy

    # pick-up ions perpendicular velocity and parallel velocity
    v_per = np.sqrt(vy_pui ** 2 + vz_pui ** 2)
    v_para = vx_pui

    # # background ions
    theta = 0
    vy1 = -vx * np.sin(theta / 360 * np.pi * 2) + vy * np.cos(theta / 360 * np.pi * 2)
    v_b_per = np.sqrt(vy1 ** 2 + vz ** 2)
    v_b_para = vx * np.cos(theta / 360 * np.pi * 2) + vy * np.sin(theta / 360 * np.pi * 2)

    # X, Y = np.mgrid[v_para.min():v_para.max():100j, v_per.min():v_per.max():100j]
    X, Y = np.mgrid[-10:10:64j, 0.01:15:64j]

    # perform a kernel density estimate on the data
    # positions = np.vstack([X.ravel(), Y.ravel()])
    # values = np.vstack([v_para, v_per])
    # kernel = st.gaussian_kde(values, bw_method=0.1)
    # Z = np.reshape(kernel(positions).T, X.shape)/Y
    #
    #
    # values_b = np.vstack([v_b_para, v_b_per])
    # kernel_b = st.gaussian_kde(values_b,bw_method=0.1)
    # Z_b = np.reshape(kernel_b(positions).T, X.shape)/Y

    # if t==0:
    #     Z = Z / Z.max()

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 14}

    # plot the results
    fig, ax = plt.subplots(figsize=(6, 5))

    # plt.pcolormesh(X, Y, Z, shading='gouraud', cmap='jet')
    # plt.pcolormesh(X, Y, Z_b, shading='gouraud', cmap='jet')
    # h = plt.hist2d(v_para, v_per, bins=200, range=[[-12, 12], [0, 15]])
    # value, xedges, yedges = h[0], h[1], h[2]
    # X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    # ax.pcolormesh(X, Y, np.log10((value / (index * yedges[1:]))).T, shading='gouraud', cmap='jet')
    # plt.hexbin(v_para, v_per, bins=100,  cmap=plt.cm.jet, )
    # plt.xlim(-12, 12)
    # plt.title(r'$t\Omega_i=$' + str(t * 0.01))
    # plt.xlabel('$v_{||}/v_A$')
    # plt.ylabel('$v_{\perp}/v_A$')
    # plt.colorbar()
    # plt.savefig(path+'density' + str(t) + '.png')
    # plt.contour(X, Y, Z)

    # fig = plt.figure(figsize=(8, 5))
    h_b = plt.hist2d(vz_pui, vy_pui, bins=200, range=[[-12, 12], [-12, 12]])
    value_b, xedges_b, yedges_b = h_b[0], h_b[1], h_b[2]
    value_b[value_b == 0] = np.nan
    palette = plt.cm.jet
    # Bad values (i.e., masked, nan, set to grey 0.8)
    palette.set_bad('w', 1.0)
    # value_b = np.ma.masked_values(value_b, value_b == 0)
    X_b, Y_b = np.meshgrid(xedges_b[:-1], yedges_b[:-1])
    # ax.set_facecolor('orange')
    col = ax.pcolormesh(X_b, Y_b, np.log10(value_b).T, shading='gouraud', cmap=palette)
    # col = ax.pcolorfast(X_b, Y_b, np.log10((value_b / (index * yedges_b[1:]))).T, shading='gouraud', cmap='jet', vmax=0)
    # col = ax.imshow(np.log10((value_b / (index * yedges_b[1:]))), cmap='jet', extent=[-12, 12, 0, 15], vmax=0)
    # ax.set_xlim(-12, 12)
    ax.axis('scaled')
    ax.set_title(r'$t\Omega_i=$' + str(t * 0.005), fontsize=14)
    ax.set_xlabel('$v_{z}/v_A$', font1)
    ax.set_ylabel('$v_{y}/v_A$', font1)
    ax.tick_params(labelsize=12)
    plt.colorbar(col)
    plt.savefig(path + 'density_v_perp' + str(t*0.005) + '.png')
    # plt.savefig('/media/ck/Samsung_T5/data/data/density' + str(t) + 'background.png')
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
