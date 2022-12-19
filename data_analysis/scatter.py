import matplotlib.pyplot as plt
import numpy as np

import data_analysis.read_data as data

# path = '/home/ck/Documents/hybrid2D_PUI/data/'
path = '/media/ck/Samsung_T5/data/data/'
# define a function to make a scatter plot with histograms
def scatter_hist_pui(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, s=0.1)
    ax.set_xlabel(r'$v_{\parallel}$')
    ax.set_ylabel(r'$v_{\bot}$')
    ax.set_ylim(0, 20)
    ax.set_xlim(-20, 20)
    ax.set_xticks(np.arange(-20, 20, 2))
    ax.set_yticks(np.arange(0, 20, 2))
    ax.set_title('Pick-up ions velocity distribution', y=-0.3)
    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax / binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')


def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, s=0.1)
    ax.set_ylim(0, )
    ax.set_xlabel(r'$v_{\parallel}$')
    ax.set_ylabel(r'$v_{\bot}$')
    ax.set_title('Background ions velocity distribution', y=-0.3)

    # now determine nice limits by hand:
    binwidth = 0.01
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax / binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')


for t in range(0, 50001, 10000):
    index = 0
    i, vx, vy, vz, x, y = data.read_phase(path, t)
    i = np.array(i)
    vx = np.array(vx)
    vy = np.array(vy)
    vz = np.array(vz)
    x = np.array(x)
    y = np.array(y)
    for j in i:
        if j == 0:
            index += 1
        else:
            break
    vx_pui = vx[:index - 1]
    vy_pui = vy[:index - 1]
    vz_pui = vz[:index - 1]
    x_pui = x[:index - 1]


    # background proton
    vx = vx[index + 1:]
    vy = vy[index + 1:]
    vz = vz[index + 1:]
    x = x[index + 1:]

    # pick-up ions perpendicular velocity and parallel velocity
    v_per = np.sqrt(vy_pui ** 2 + vz_pui ** 2)
    v_para = vx_pui

    # background ions
    v_b_per = np.sqrt(vx ** 2 + vz ** 2)
    v_b_para = vy

    # definitions for the axes
    left, width = 0.1, 0.7
    bottom, height = 0.2, 0.35
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.15, height]
    #
    # plot pick-up ions velocity distribution

    # start with a square figure
    fig = plt.figure(figsize=(6., 6.))
    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    # use the previously defined function
    scatter_hist_pui(v_para, v_per, ax, ax_histx, ax_histy)
    # set the proportion of y axis to x axis
    # plt.set_aspect(1.)

    # plot background ions

    # start with a square figure
    fig = plt.figure(figsize=(6., 6.))
    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    # use the previously defined function
    scatter_hist(v_b_para, v_b_per, ax, ax_histx, ax_histy)
    # set the proportion of y axis to x axis
    # plt.set_aspect(1.)

plt.show()
