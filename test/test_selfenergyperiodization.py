import unittest, os, numpy as np, h5py
from pytriqs.gf.local import BlockGf, GfImFreq, GfImTime
from maxent.bryanToTRIQS import MaximumEntropy
from bethe.h5interface import Storage

from periodization.selfenergyperiodization import LatticeSelfenergy, LatticeGreensfunction
from periodization.generic import LocalLatticeGreensfunction, Bandstructure
from periodization.utility import plot_giw, save_plot, plot, plot_bandstructure


class TestLatticeSelfenergy(unittest.TestCase):

    def test_LatticeSelfenergy_full(self):
        sto = Storage('cdmftdimerchain.h5')
        sigmaimp = sto.load('se_imp_iw')
        gimp = sto.load('g_imp_iw')
        mu = sto.load('mu')
        
        spins = ['up', 'dn']
        sig_loc = BlockGf(name_block_generator = [(s, GfImFreq(beta = sigmaimp.beta, indices = [0])) for s in spins])

        sig_loc['up'][0,0] << (sigmaimp['up-+'][0,0]+sigmaimp['up--'][0,0]+sigmaimp['dn-+'][0,0]+sigmaimp['dn--'][0,0]) * .25
        #sig_loc['up'][0,0] << (sigmaimp['up'][0,0]+sigmaimp['up'][1,1]+sigmaimp['dn'][0,0]+sigmaimp['dn'][1,1]) * .25
        #sig_loc['up'][0,0] << (sigmaimp['up'][0,0]+sigmaimp['up'][1,1]+sigmaimp['dn'][0,0]+sigmaimp['dn'][1,1]) * .25
        sig_loc['dn'] << sig_loc['up']
        sig_nn = BlockGf(name_block_generator = [(s, GfImFreq(beta = sigmaimp.beta, indices = [0])) for s in spins])
        sig_nn['up'][0,0] << (sigmaimp['up-+'][0,0]-sigmaimp['up--'][0,0]+sigmaimp['dn-+'][0,0]-sigmaimp['dn--'][0,0]) * .25
        #sig_nn['up'][0,0] << (sigmaimp['up'][0,0]-sigmaimp['up'][1,1]+sigmaimp['dn'][0,0]-sigmaimp['dn'][1,1]) * .25
        #sig_nn['up'][0,0] << (sigmaimp['up'][0,1]+sigmaimp['up'][1,0]+sigmaimp['dn'][0,1]+sigmaimp['dn'][1,0]) * .25
        sig_nn['dn'] << sig_nn['up']

        blocknames = spins
        blockinds = [[0]]*2
        translations = [[0.], [-1.], [1.]] #r
        sigma_r = [sig_loc, sig_nn, sig_nn]
        #weights_r = [1, .5, .5]
        weights_r = [1.]*3
        hopping_r = [{s:[[t]] for s in spins} for t in [0, -1, -1]]
        nk = 32

        slat = LatticeSelfenergy(blocknames, blockinds, translations, hopping_r, nk, sigma_r, weights_r)
        eps = [val[0, 0].real for h in slat.hk for key, val in h.items() if key == 'up']
        plot(slat.k, eps)
        save_plot('epsk_sep.pdf')

        plot(np.array([iw.imag for iw in gimp.mesh])[1025:1050], .5*(gimp['up-+'].data[1025:1050,0,0]+gimp['up--'].data[1025:1050,0,0]).imag, label = 'imp')
        #plot(np.array([iw.imag for iw in gimp.mesh])[1025:1050], .5*(gimp['up'].data[1025:1050,0,0]+gimp['up'].data[1025:1050,0,0]).imag, label = 'imp')
        #for nk in [8, 16, 32, 64]:#, 128, 256]:
        for nk in [256]:
            slat = LatticeSelfenergy(blocknames, blockinds, translations, hopping_r, nk, sigma_r, weights_r)
            slat.periodize()
            glat = LatticeGreensfunction(slat, mu)
            glatloc = LocalLatticeGreensfunction(glat)
            plot_giw(glatloc, label = nk)
        save_plot('gimp_vs_glatloc_sep.pdf', legend = True)
        
        if False:
            maxent_pars = {'ntau': 500,'nomega': 1000,'bandwidth': 30,'sigma': 0.003}
            gtau = BlockGf(name_block_generator = [(s, GfImTime(indices = [i for i in b.indices], beta = glatloc.beta, n_points = len(glatloc.mesh)*2))for s, b in glatloc])
            for bn, b in glatloc:
                gtau[bn].set_from_inverse_fourier(b)
            maxent = MaximumEntropy(gtau, maxent_pars['ntau'])
            maxent.calculateTotDOS(maxent_pars['nomega'], maxent_pars['bandwidth'], maxent_pars['sigma'])
            plot(maxent.getOmegaMesh(), maxent.getTotDOS())
            save_plot('maxent_sep.pdf')

        if True:
            kpath = [[-.5],[0]]
            n_omega_n = 29
            eta = .01
            n_omega = 301
            omega_window = (-10, 10)
            bands = Bandstructure(glat, kpath, n_omega_n, eta, n_omega, omega_window)
            plot_bandstructure(bands)
            save_plot('bandstructure_sep.pdf')

        glat.calculate_real_space_at([0], [1], '01')
        glat.calculate_real_space_at([0], [2], '02')
        glat.calculate_real_space_at([0], [3], '03')
        glat.calculate_real_space_at([0], [4], '04')
        glat.calculate_real_space_at([0], [5], '05')
        plot(np.array([iw.imag for iw in gimp.mesh])[1025:1050], .5*(gimp['up-+'].data[1025:1050,0,0]-gimp['up--'].data[1025:1050,0,0]).real, label = 'imp')
        plot_giw(glat['01'], label = 'lat01', imag = False)
        plot_giw(glat['02'], label = 'lat02', imag = False)
        plot_giw(glat['03'], label = 'lat03', imag = False)
        plot_giw(glat['04'], label = 'lat04', imag = False, ls = '--')
        plot_giw(glat['05'], label = 'lat05', imag = False)
        save_plot('glat01vsgimp01_sep.pdf', legend = True)
