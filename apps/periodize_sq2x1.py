#!/usr/bin/env pytriqs
from ClusterDMFT.cellular_dmft import CDmft
from ClusterDMFT.periodization import Periodization
from numpy import sqrt
import sys

n_kpts = int(sys.argv[1])
for arg in sys.argv[2:len(sys.argv)]:
    print 'periodizing ' + arg + '...'
    x = CDmft(archive = arg)
    lat = Periodization([[1, 0, 0], [0, 1, 0]], [[0, 0, 0]], {(1, 0) : [[-1]], (-1, 0) : [[-1]], (0, 1) : [[-1]], (0, -1) : [[-1]]}, n_kpts, [[0, 0, 0], [1, 0, 0]])
    lat.set_all(x.load('Sigma_c_iw'))
    lat.write_to_disk(x.parameters['archive'])
    del lat
    print arg + ' ready'
