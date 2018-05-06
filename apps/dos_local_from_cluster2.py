#!/usr/bin/env pytriqs
from ClusterDMFT.cdmft import CDmft
from ClusterDMFT.evaluation.analytical_continuation import pade_tr as pade
from pytriqs.gf.local import BlockGf, GfImFreq, GfReFreq
from pytriqs.plot.mpl_interface import oplot
from numpy import pi
from matplotlib import pyplot as plt, cm
import sys

freq = int(sys.argv[1])
max_w = int(sys.argv[2])
n_graphs = len(sys.argv) -2

for n, arch in enumerate(sys.argv[3:len(sys.argv)]):
    x = CDmft(archive = arch)
    g = x.load('G_c_iw')
    g_w = pade(g, pade_n_omega_n = freq, pade_eta = 0.05, dos_n_points = 1200, dos_window = (-max_w, max_w), clip_threshold = 0)
    oplot(g_w, RI = 'S', name = arch[0:-3], color = cm.jet(n /float(n_graphs - 1)))
plt.gca().set_ylabel('$A(\omega)$')
plt.gca().set_ylim(bottom=0)
plt.savefig('dos'+str(freq)+'.png', dpi = 300)
plt.close()
print 'dos'+str(freq)+'.png ready'
del x

