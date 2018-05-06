#!/usr/bin/env pytriqs
import sys
from numpy import array
from ClusterDMFT.periodization.periodization import PeriodizationBase as Periodization, g_k_to_imshow_data
from matplotlib import pyplot as plt, cm

rows = int(sys.argv[1])
cols = int(sys.argv[2])
vmax = int(sys.argv[3])

path = [[0,0,0],[.5,0,.5],[.5,.25,.75],[3/8.,3/8.,.75],[.5,.5,.5],[0,0,0]]
path_labels = ['$\Gamma$', 'X', 'W', 'K', 'L', '$\Gamma$']
ax_pos = [0, 0]

fig = plt.figure(figsize = (8.27, 11.69))

for nr, arch in enumerate(sys.argv[4:]):
    ax = plt.subplot2grid((rows, cols), ax_pos)
    lat = Periodization(archive = arch)
    x, y, z, k_ticks = g_k_to_imshow_data(lat.get_tr_g_lat_pade(), path, lat.bz_grid)
    im = ax.imshow(z, cmap = cm.copper, interpolation = 'gaussian', extent = [0, len(x), y[0], y[-1]], vmin = 0, vmax = vmax, origin = 'lower')
    ax.set_ylabel('$\omega$')
    ax.set_xlabel('$k$')
    ax.set_xticks([k_ticks[i][0] for i in range(len(k_ticks))])
    ax.set_xticklabels(path_labels)
    ax.set_title(arch[:-3])
    ext = im.get_extent()
    ax.set_aspect(abs(.5*(ext[0] - ext[1])/float(ext[2] - ext[3])))
    ax_pos[1] = ax_pos[1] + 1
    if ax_pos[1] == cols:
        ax_pos[1] = 0
        ax_pos[0] = ax_pos[0] + 1
    del lat

plt.tight_layout()
plt.savefig('dos_k_w.png', dpi = 600)
