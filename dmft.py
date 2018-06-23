import itertools as itt, numpy as np
from pytriqs.gf.local import BlockGf, GfImFreq, iOmega_n, inverse
from pytriqs.utility import mpi

from periodization.mpiLists import scatter_list, allgather_list
from periodization.generic import LatticeSelfenergy as LatticeSelfenergyGen, LatticeGreensfunction as LatticeGreensfunctionGen
from greensfunctionperiodization import LatticeSelfenergy


class LatticeGreensfunction(LatticeGreensfunctionGen, LatticeSelfenergyGen):
    def __init__(self, blocknames, blockindices, r, hopping_r, nk, selfenergy, mu, weights_r, *args, **kwargs):
        if 'gk_on_the_fly' in kwargs:
            gk_on_the_fly = kwargs.pop('gk_on_the_fly')
        else:
            gk_on_the_fly = False
        LatticeSelfenergyGen.__init__(self, blocknames, blockindices, r, hopping_r, nk, *args, **kwargs)
        self.wr = weights_r
        self.gr = {}
        self.ginvr = {}
        if not gk_on_the_fly:
            self.gk = [
                BlockGf(
                    name_block_generator = [
                        (bn, GfImFreq(mesh = selfenergy.mesh, indices = bi)) for bn, bi in zip(blocknames, blockindices)
                    ]
                )
                for k in self.k]
            self._calculate_gk(selfenergy, mu)
        else:
            self.gk = GkOnTheFly(blocknames, blockindices, selfenergy, mu, self.hk)

    def _calculate_gk(self, selfenergy, mu):
        hkpercore = scatter_list(self.hk)
        gkpercore = scatter_list(self.gk)
        for hk, gk in itt.izip(hkpercore, gkpercore):
            for bn, b in gk:
                gk[bn] << inverse((iOmega_n + mu) - hk[bn] - selfenergy[bn])
        self.gk = allgather_list(gkpercore)

class GkOnTheFly:
    def __init__(self, blocknames, blockindices, selfenergy, mu, hk):
        self.hk = hk
        self.selfenergy = selfenergy
        self.mu = mu
        self.g_ki = BlockGf(name_block_generator = [(bn, GfImFreq(mesh = selfenergy.mesh, indices = bi)) for bn, bi in zip(blocknames, blockindices)])
    
    def __getitem__(self, i_k):
        for bn, b in self.g_ki:
            b << inverse((iOmega_n + self.mu) - self.hk[i_k][bn] - self.selfenergy[bn])
        return self.g_ki
