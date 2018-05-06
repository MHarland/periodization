#!/usr/bin/env pytriqs
import sys
from numpy import array, save, empty
from ClusterDMFT.periodization.periodization import PeriodizationBase as Periodization, g_k_to_imshow_data

path = [[0,0,0],[.5,0,.5],[.5,.25,.75],[3/8.,3/8.,.75],[.5,.5,.5],[0,0,0]]
path_labels = ['$\Gamma$', 'X', 'W', 'K', 'L', '$\Gamma$']

for nr, arch in enumerate(sys.argv[1:]):
    lat = Periodization(archive = arch)
    x, y, z, k_ticks = g_k_to_imshow_data(lat.get_tr_g_lat_pade(), path, lat.bz_grid)
    k_ticks_indices = array([k_ticks[i][0] for i in range(len(k_ticks))])
    data = array([x, y, z, k_ticks_indices])
    save('dkw_' + arch[:-3] + '.npy', data)
    #for dat, datstr in zip(data, ['x', 'y', 'z', 'kt']):
    #    save('dkw' +   + '_'+ arch[:-3] + '.npy', dat)
