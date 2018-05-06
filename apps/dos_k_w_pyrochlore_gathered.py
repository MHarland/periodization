#!/usr/bin/env pytriqs
import sys
from numpy import array, load
from matplotlib import pyplot as plt, cm

rows = int(sys.argv[1])
cols = int(sys.argv[2])
vmax = int(sys.argv[3])

path = [[0,0,0],[.5,0,.5],[.5,.25,.75],[3/8.,3/8.,.75],[.5,.5,.5],[0,0,0]]
path_labels = ['$\Gamma$', 'X', 'W', 'K', 'L', '$\Gamma$']
ax_pos = [0, 0]

fig = plt.figure(figsize = (8.27, 11.69))

for nr, npy_file in enumerate(sys.argv[4:]):
    ax = plt.subplot2grid((rows, cols), ax_pos)
    data = load(npy_file)
    x = data[0]
    y = data[1]
    z = data[2]
    k_ticks_indices = data[3]
    im = ax.imshow(z.T, cmap = cm.copper, interpolation = 'gaussian', extent = [0, len(x), y[0], y[-1]], vmin = 0, vmax = vmax, origin = 'lower')
    ax.set_ylabel('$\omega$')
    ax.set_ylim(-10,1)
    ax.set_xlabel('$k$')
    ax.set_xticks([k_ticks_indices[i] for i in range(len(k_ticks_indices))])
    ax.set_xticklabels(path_labels)
    ax.set_title(npy_file[4:-4])
    ext = im.get_extent()
    ax.set_aspect(abs(2*(ext[0] - ext[1])/float(ext[2] - ext[3])))
    ax_pos[1] = ax_pos[1] + 1
    if ax_pos[1] == cols:
        ax_pos[1] = 0
        ax_pos[0] = ax_pos[0] + 1

plt.tight_layout()
plt.savefig('dos_k_w.png', dpi = 600)
