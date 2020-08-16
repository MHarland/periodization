import itertools as itt
import numpy as np
from pytriqs.gf import BlockGf, GfImFreq, iOmega_n, inverse
from pytriqs.utility import mpi

from periodization.mpiLists import scatter_list, allgather_list
from periodization.generic import LatticeSelfenergy as LatticeSelfenergyGen
from periodization.selfenergyperiodization import LatticeGreensfunction


class LatticeSelfenergy(LatticeSelfenergyGen):
    def __init__(self, blocknames, blockindices, r, hopping_r, nk, m_r, weights_r, mu, *args, **kwargs):
        LatticeSelfenergyGen.__init__(
            self, blocknames, blockindices, r, hopping_r, nk, *args, **kwargs)
        self.wr = weights_r
        self.mr = m_r
        self.mu = mu
        self.sigma_k = [
            BlockGf(
                name_block_generator=[
                    (bn, GfImFreq(beta=m_r[0].mesh.beta, indices=bi)) for bn, bi in zip(blocknames, blockindices)
                ], make_copies=False
            )
            for k in self.k]

    def periodize(self):
        sigkpercore = scatter_list(self.sigma_k)
        kpercore = scatter_list(self.k)
        mk = self.sigma_k[0].copy()
        for sigk, k in itt.izip(sigkpercore, kpercore):
            mk.zero()
            for mr, r, wr in itt.izip(self.mr, self.r, self.wr):
                for s, b in sigk:
                    mk[s] << mk[s] + wr * \
                        np.exp(complex(0, 2*np.pi * k.dot(r))) * mr[s]
            for s, b in sigk:
                sigk[s] << (iOmega_n + self.mu) - inverse(mk[s])
        self.sigma_k = allgather_list(sigkpercore)
