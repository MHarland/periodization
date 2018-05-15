import itertools as itt, numpy as np
from pytriqs.gf.local import BlockGf, GfImFreq, iOmega_n, inverse
from pytriqs.utility import mpi

from periodization.mpiLists import scatter_list, allgather_list
from periodization.generic import LatticeSelfenergy as LatticeSelfenergyGen, LatticeGreensfunction as LatticeGreensfunctionGen


class LatticeGreensfunction(LatticeGreensfunctionGen, LatticeSelfenergyGen):
    def __init__(self, blocknames, blockindices, r, hopping_r, nk, g_r, weights_r, *args, **kwargs):
        LatticeSelfenergyGen.__init__(self, blocknames, blockindices, r, hopping_r, nk, *args, **kwargs)
        self.wr = weights_r
        self.gr_in = g_r
        self.gk = [
            BlockGf(
                name_block_generator = [
                    (bn, GfImFreq(beta = g_r[0].beta, indices = bi)) for bn, bi in zip(blocknames, blockindices)
                ]
            )
            for k in self.k]
        self.gr = {}
        self.ginvr = {}

    def periodize(self):
        gkpercore = scatter_list(self.gk)
        kpercore = scatter_list(self.k)
        for gk, k in itt.izip(gkpercore, kpercore):
            for gr, r, wr in itt.izip(self.gr_in, self.r, self.wr):
                for s, b in gk:
                    gk[s] += wr * np.exp(complex(0, 2*np.pi * k.dot(r))) * gr[s]
        self.gk = allgather_list(gkpercore)

class LatticeSelfenergy(LatticeSelfenergyGen):
    def __init__(self, lattice_greensfunction, mu):
        self.sigma_k = [gk.copy() for gk in lattice_greensfunction.gk]
        self.k = lattice_greensfunction.k
        self.wk = lattice_greensfunction.wk
        self.hk = lattice_greensfunction.hk
        self._calculate_sigma_k(lattice_greensfunction, mu)

    def _calculate_sigma_k(self, glat, mu):
        sigkpercore = scatter_list(self.sigma_k)
        gkpercore = scatter_list(glat.gk)
        hkpercore = scatter_list(glat.hk)
        for hk, sigk, gk in itt.izip(hkpercore, sigkpercore, gkpercore):
                for s, b in sigk:
                    sigk[s] << iOmega_n + mu - hk[s] - inverse(gk[s])
        self.sigma_k = allgather_list(sigkpercore)
