#!/usr/bin/env pytriqs
from ClusterDMFT.periodization import Periodization
from matplotlib import pyplot as plt, cm
import sys

n_start = int(sys.argv[1])
n_stop = int(sys.argv[2])
n_step = int(sys.argv[3])
max_w = int(sys.argv[4])
for arch in sys.argv[5:len(sys.argv)]:
    x = Periodization(archive = arch)
    for n in range(n_start, n_stop, n_step):
        x.plot_dos_loc(pade_n_omega_n = n, pade_eta = 10**-10, dos_n_points = 1200, dos_window = (-max_w, max_w), name = str(n), clip_threshold = 0, color = cm.jet((n - n_start) /float(n_stop-n_start)))
    filename = 'dos_' + arch[0:-3] + '_' + str(n_start) + str(n_stop) + str(n_step) + '.png'
    plt.gca().set_ylim(0, 2)
    plt.savefig(filename)
    plt.close()
    print filename + ' ready'
    del x
