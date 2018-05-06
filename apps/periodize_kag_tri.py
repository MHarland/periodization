#!/usr/bin/env pytriqs
from numpy import sin, cos, pi
from ClusterDMFT.cellular_dmft import CDmft
from ClusterDMFT.periodization import Periodization
from ClusterDMFT.lattice.superlattices import kag_tri
import sys

lat = kag_tri()
n_kpts = int(sys.argv[1])
for arg in sys.argv[2:len(sys.argv)]:
    print 'periodizing ' + arg + '...'
    x = CDmft(archive = arg)
    t = [[-1]]
    lat = Periodization([[1, 0, 0], [cos(pi / 3), sin(pi / 3), 0]], [[0, 0, 0], [1, 0, 0], [cos(pi / 3), sin(pi / 3), 0]], lat.get_h_eff(), n_kpts, [[0, 0, 0], [.5, 0, 0], [.5, .5, 0]])
    lat.set_all(x.load('Sigma_c_iw'))
    lat.write_to_disk(x.parameters['archive'])
    del lat
    print arg + ' ready'
