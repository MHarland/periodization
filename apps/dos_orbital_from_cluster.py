#!/usr/bin/env pytriqs
from ClusterDMFT.cdmft import CDmft
from ClusterDMFT.evaluation.analytical_continuation import pade_spin as pade
from pytriqs.gf.local import BlockGf, GfImFreq, GfReFreq
from pytriqs.plot.mpl_interface import oplot
from numpy import pi
from matplotlib import pyplot as plt, cm
import sys

n_start = int(sys.argv[1])
n_stop = int(sys.argv[2])
n_step = int(sys.argv[3])
max_w = int(sys.argv[4])
max_a = int(sys.argv[5])

cmaps_pool = [cm.Reds, cm.Blues, cm.Greens, cm.Greys]
for arch in sys.argv[6:len(sys.argv)]:
    print 'loading '+arch+' ...'
    x = CDmft(archive = arch)
    g = x.load('G_sym_iw')
    cmaps = list()
    n_orbs = int(g.n_blocks * .5)
    for i in range(n_orbs):
        cmaps.append(cmaps_pool[i%(len(cmaps_pool))])
    for n in range(n_start, n_stop, n_step):
        for s, b in g:
            g_w = pade(g, s, pade_n_omega_n = n, pade_eta = 10**-2, dos_n_points = 1200, dos_window = (-max_w, max_w), clip_threshold = 0)
            oplot(g_w, RI = 'S', name = s[0]+s[2]+' '+str(n), color = cmaps[int(s[0])](((n - n_start) /float(n_stop - n_start))*.5 + .5))
    filename = 'dos_orb_' + arch[0:-3] + '_' + str(n_start) + str(n_stop) + str(n_step) + '.png'
    plt.gca().set_ylim(0, max_a)
    plt.savefig(filename)
    plt.close()
    print filename + ' ready'
    del x

