from matplotlib import pyplot as plt
import numpy as np

##generating some  data
x, y = np.meshgrid(
    np.linspace(0, 1, 100),
    np.linspace(0, 1, 100),
)
g = np.concatenate((x[:, :50]*2, x[:, 49::-1]*2), axis=1)
directions = (np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) + 1) * np.pi
magnitude = np.exp(-(x * x + y * y))


##normalize data:
def normalize(M):
    return (M - np.min(M)) / (np.max(M) - np.min(M))


d_norm = normalize(directions)
m_norm = normalize(magnitude)

fig, (plot_ax, bar_ax) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

plot_ax.imshow(
    np.dstack((d_norm, np.zeros_like(directions), m_norm)),
    aspect='auto',
    extent=(0, 100, 0, 100),
)

bar_ax.imshow(
    np.dstack((x, np.zeros_like(x)+0.5, np.zeros_like(y)+1, y)),
    extent=(
        np.min(directions), np.max(directions),
        np.min(magnitude), np.max(magnitude),
    ),
    aspect='auto',
    origin='lower',
)
bar_ax.set_xlabel('direction')
bar_ax.set_ylabel('magnitude')

plt.show()
