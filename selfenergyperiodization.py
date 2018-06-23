import itertools as itt, numpy as np
from pytriqs.gf.local import BlockGf, GfImFreq, iOmega_n, inverse
from pytriqs.utility import mpi

from periodization.mpiLists import scatter_list, allgather_list
from periodization.generic import LatticeSelfenergy as LatticeSelfenergyGen, LatticeGreensfunction as LatticeGreensfunctionGen


class LatticeSelfenergy(LatticeSelfenergyGen):
    def __init__(self, blocknames, blockindices, r, hopping_r, nk, sigma_r, weights_r, *args, **kwargs):
        LatticeSelfenergyGen.__init__(self, blocknames, blockindices, r, hopping_r, nk, *args, **kwargs)
        self.wr = weights_r
        self.sigr = sigma_r
        self.sigma_k = [
            BlockGf(
                name_block_generator = [
                    (bn, GfImFreq(mesh = sigma_r[0].mesh, indices = bi)) for bn, bi in zip(blocknames, blockindices)
                    ]
            )
            for k in self.k]

    def periodize(self):
        sigkpercore = scatter_list(self.sigma_k)
        kpercore = scatter_list(self.k)
        for sigk, k in itt.izip(sigkpercore, kpercore):
            for sigr, r, wr in itt.izip(self.sigr, self.r, self.wr):
                for s, b in sigk:
                    sigk[s] << sigk[s] + wr * np.exp(complex(0, 2*np.pi * k.dot(r))) * sigr[s]
        self.sigma_k = allgather_list(sigkpercore)


class LatticeGreensfunction(LatticeGreensfunctionGen):
    def __init__(self, lattice_selfenergy, mu):
        self.mu = mu
        self.k = lattice_selfenergy.k
        self.wk = lattice_selfenergy.wk
        self.hk = lattice_selfenergy.hk
        self.gk = [sig.copy() for sig in lattice_selfenergy.sigma_k]
        self.gr = {}
        self.ginvr = {}
        self._calculate_gk(lattice_selfenergy.sigma_k, mu)

    def _calculate_gk(self, lattice_selfenergy, mu):
        hkpercore = scatter_list(self.hk)
        sigkpercore = scatter_list(lattice_selfenergy)
        gkpercore = scatter_list(self.gk)
        for hk, sigk, gk in itt.izip(hkpercore, sigkpercore, gkpercore):
            for bn, b in gk:
                gk[bn] << inverse((iOmega_n + mu) - hk[bn] - sigk[bn])
        self.gk = allgather_list(gkpercore)
