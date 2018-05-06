#!/usr/bin/env pytriqs
from ClusterDMFT.cdmft import CDmft
from ClusterDMFT.evaluation.analytical_continuation import pade_tr as pade
from pytriqs.gf.local import BlockGf, GfImFreq, GfReFreq
from pytriqs.plot.mpl_interface import oplot
from numpy import pi
from matplotlib import pyplot as plt, cm
import sys

n_start = int(sys.argv[1])
n_stop = int(sys.argv[2])
n_step = int(sys.argv[3])
max_w = int(sys.argv[4])
max_y = int(sys.argv[5])

for arch in sys.argv[6:len(sys.argv)]:
    x = CDmft(archive = arch)
    g = x.load('g_c_iw')
    for n in range(n_start, n_stop, n_step):
        g_w = pade(g, pade_n_omega_n = n, pade_eta = 0.05, dos_n_points = 1200, dos_window = (-max_w, max_w), clip_threshold = 0)
        oplot(g_w, RI = 'S', name = str(n), color = cm.jet((n - n_start) /float(n_stop - n_start)))
    filename = 'dos_' + arch[0:-3] + '_' + str(n_start) + str(n_stop) + str(n_step) + '.pdf'
    plt.gca().set_ylim(0, max_y)
    plt.savefig(filename)
    plt.close()
    print filename + ' ready'
    del x

