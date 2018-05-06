import itertools as itt, numpy as np
from pytriqs.gf.local import BlockGf, GfImFreq, iOmega_n, inverse
from pytriqs.utility import mpi

from periodization.mpiLists import scatter_list, allgather_list
from periodization.generic import LatticeSelfenergy as LatticeSelfenergyGen


class LatticeSelfenergy(LatticeSelfenergyGen):
    """
    delta and g are Anderson impurity quantities in real space an on Matsubara frequencies
    r is the lattice vector of the full lattice
    single orbital only
    """
    def __init__(self, blocknames, blockindices, r, hopping_r, nk, g, delta, nsites, verbose = False):
        LatticeSelfenergyGen.__init__(self, blocknames, blockindices, r, hopping_r, nk, verbose)
        self.delta = delta
        self.g = g
        assert self.nk % nsites == 0, "nk mod nsites != 0"
        self.nsites = nsites
        self.nblocks = self.nk / self.nsites
        self.sigma_k = [
            BlockGf(
                name_block_generator = [
                    (bn, GfImFreq(beta = g.beta, indices = range(i*self.nsites, (i+1)*self.nsites))) for bn, bi in zip(blocknames, blockindices)
                    ]
            )
            for i in self.nblocks]

    def calculate(self):
        pass

    def _calc_costfunction(self):
        pass

    def _calc_derivative(self):
        pass

    def _perform_update(self):
        pass
