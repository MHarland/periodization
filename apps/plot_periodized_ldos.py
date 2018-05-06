#!/usr/bin/env pytriqs
from ClusterDMFT.periodization import Periodization
from matplotlib import pyplot as plt
from pytriqs.plot.mpl_interface import oplot
import sys

arch = sys.argv[1]

lat = Periodization(archive = arch)
oplot(_tr_g_lat_pade([lat.get_g_lat_loc()])[0], RI = 'S', name = 'local_DOS')
plt.savefig('plot.png')
