#!/usr/bin/env pytriqs
from ClusterDMFT.cdmft import CDmft
import sys
from matplotlib import pyplot as plt, cm
from numpy.linalg import eigvals
from numpy import zeros, sqrt

for n, arch in enumerate(sys.argv[1:]):
    c = CDmft(archive = arch)
    eps_ab = c.load('eps')
    grid = c.load('rbz_grid')
    n_k = eps_ab.shape[0]
    n_K = eps_ab.shape[1]
    eps_K = zeros(eps_ab.shape)
    for k in range(n_k):
        e = eigvals(eps_ab[k, :, :])
        for K in range(e.shape[0]):
            eps_K[k, K, K] = e[K].real
    for K in range(n_K):
        plt.hist2d(grid[:, 0], grid[:, 1], bins = int(sqrt(n_k)), weights = eps_K[:, K, K])
        print eps_K[:,K,K]
        plt.colorbar()
        plt.xlabel('$k_x$')
        plt.ylabel('$k_y$')
        plt.title('$\epsilon_' + str(K) + '(k)$')
        plt.savefig(arch[0:-3] + '_eps' + str(K) + '.png', dpi = 300)
        plt.close()
