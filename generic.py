import itertools as itt, numpy as np
from pytriqs.gf.local import BlockGf, GfReFreq, GfImFreq, inverse
from pytriqs.utility import mpi

from periodization.mpiLists import scatter_list, allgather_list
from periodization.utility import trace_giw, trace_h



class LatticeSelfenergy:
    """
    r is in the basis of the lattice vectors
    k is in the basis of the reciprocal lattice vectors
    input parameters are lists with order defined by translations r
    """
    def __init__(self, blocknames, blockindices, r, hopping_r, nk, hopping_k = None, k_mesh = None, wk = None, verbose = False):
        self.verbose = verbose
        self.r = np.array(r)
        self.d = len(self.r[0])
        self.nk = nk
        if k_mesh is None:
            self.k = np.array([k for k in itt.product(*[np.linspace(-.5, .5, nk, False)]*self.d)])
        else:
            self.k = k_mesh
        self.ik = range(len(self.k))
        if wk is None:
            self.wk = [1./(self.nk**self.d) for k in self.k]
        else:
            self.wk = wk
        if hopping_k is None:
            self.hr = [{bn: np.array(b, dtype = complex) for bn, b in hr.items()} for hr in hopping_r]
            self.hk = [{bn: 0 for bn in hr.keys()} for k in self.k]
            self.hk = self._transform_hopping()
        else:
            assert hopping_r is None, "gets either hopping_r or hopping_k"
            self.hk = hopping_k

    def _transform_hopping(self):
        hkpercore = scatter_list(self.hk)
        kpercore = scatter_list(self.k)
        for hk, k in itt.izip(hkpercore, kpercore):
            for hr, r in itt.izip(self.hr, self.r):
                for b in hr.keys():
                    hk[b] += np.exp(complex(0, 2*np.pi * k.dot(r))) * hr[b]
        return allgather_list(hkpercore)

class LatticeGreensfunction:
    def calculate_real_space_at(self, r1, r2, label):
        r12 = np.array(r1) - np.array(r2)
        gtmp = self.gk[0].copy()
        gtmp.zero()
        gtmp2 = gtmp.copy()
        ikspercore = scatter_list(range(len(self.k)))
        for i_k in ikspercore:
            k = self.k[i_k]
            wk = self.wk[i_k]
            gk = self.gk[i_k]
            for s, b in gtmp:
                b << b + gk[s] * wk * np.exp(complex(0, -2*np.pi * k.dot(r12)))
        for s, b in gtmp2:
            b << mpi.all_reduce(mpi.world, gtmp[s], lambda x, y: x + y)
        self.gr[label] = gtmp2

    def __getitem__(self, index):
        if not(index in self.gr.keys()):
            self.calculate_real_space_at(index[0], index[1], index)
        return self.gr[index]

    def inverse_real_space_at(self, r1, r2):
        label = (r1, r2)
        if not label in self.ginvr.keys():
            r12 = np.array(r1) - np.array(r2)
            gtmp = self.gk[0].copy()
            gtmp.zero()
            gtmp2 = gtmp.copy()
            wkpercore = scatter_list(self.wk)
            gkpercore = scatter_list(self.gk)
            kpercore = scatter_list(self.k)
            for k, wk, gk in itt.izip(kpercore, wkpercore, gkpercore):
                for s, b in gtmp:
                    b << b + inverse(gk[s]) * wk * np.exp(complex(0, -2*np.pi * k.dot(r12)))
            for s, b in gtmp2:
                b << mpi.all_reduce(mpi.world, gtmp[s], lambda x, y: x + y)
            self.ginvr[label] = gtmp2
        return self.ginvr[label]

class LocalLatticeGreensfunction(BlockGf):
    def __init__(self, lattice_greensfunction):
        glat = lattice_greensfunction
        gtmp = glat.gk[0].copy()
        gtmp.zero()
        BlockGf.__init__(self, name_block_generator = [(s, b) for s, b in gtmp], make_copies = True)
        wkpercore = scatter_list(glat.wk)
        gkpercore = scatter_list(glat.gk)
        for wk, gk in itt.izip(wkpercore, gkpercore):#itt.izip(*[mpi.slice_array(a) for a in [wkpercore, gkpercore]]):
            for s, b in gtmp:
                b << b + gk[s] * wk
        for s, b in self:
            b << mpi.all_reduce(mpi.world, gtmp[s], lambda x, y: x + y)


class Bandstructure:
    """ traces the orbitals! """
    def __init__(self, lattice_greensfunction, path, pade_n_omega_n = 20, pade_eta = 5*10**(-2), dos_n_points = 500, dos_window = (-10, 10), k_stepwidth = 1e-4):
        glat = lattice_greensfunction
        self.path = np.array(path)
        if hasattr(glat, 'mu'):
            self.mu = glat.mu
        else:
            self.mu = None
        kinds, self.pathinds = self._kinds(glat.k, self.path, k_stepwidth)
        self.k = np.array(glat.k[kinds])
        gw = GfReFreq(indices = [0], window = dos_window, n_points = dos_n_points)
        self.w = np.array([w.real for w in gw.mesh])
        akwpercore = []
        hkpercore = []
        kindspercore = scatter_list(kinds)
        for kind in kindspercore:
            hkpercore.append(trace_h(glat.hk[kind]))
            gkiw = trace_giw(glat.gk[kind])
            gw = GfReFreq(indices = [0], window = dos_window, n_points = dos_n_points)
            gw.set_from_pade(gkiw, n_points = pade_n_omega_n, freq_offset = pade_eta)
            akwpercore.append(-1*gw.data[:,0,0].imag/np.pi)
        self.akw = np.array(allgather_list(akwpercore))
        self.hk = np.array(allgather_list(hkpercore))
        self.k_enumeration = range(len(self.k))
        self.akw_extent = [self.k_enumeration[0], self.k_enumeration[-1], self.w.min(), self.w.max()]
        
    def _kinds(self, glatmesh, path, scan_mesh_stepwidth):
        kinds = []
        pathinds = [0]
        accum_segment_len = 0
        for iseg in range(len(path)-1):
            a, b = path[iseg], path[iseg + 1]
            norm = np.linalg.norm(a - b)
            abunit = (b - a) / norm
            istep = 0
            while istep*scan_mesh_stepwidth <= norm:
                k = a + istep * scan_mesh_stepwidth * abunit
                ind_candidate = self._index_closest_k(k, glatmesh)
                if len(kinds) > 0:
                    if ind_candidate != kinds[-1]:
                        kinds.append(ind_candidate)
                else:
                    kinds.append(ind_candidate)
                istep += 1
            pathinds.append(len(kinds)-1)
        return np.array(kinds), np.array(pathinds)
        
    def _index_closest_k(self, q, kmesh, minnorm = 1e5):
        """finds the index of the k of kmesh closest to q"""
        #np.argmin(np.norm(kmesh - q))
        ind = None
        for i, k in enumerate(kmesh):
            d = np.linalg.norm(k - q)
            if d < minnorm:
                ind = i
                minnorm = d
        return ind
