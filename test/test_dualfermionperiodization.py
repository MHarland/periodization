import unittest, os, numpy as np, h5py
from bethe.h5interface import Storage
from bethe.transformation import MatrixTransformation


class TestDFPeriodization(unittest.TestCase):

    def test_LatticeSelfenergy_full(self):
        sto = Storage('cdmftdimerchain.h5')
        delta_tau = sto.load('delta_tau')
        gimp = sto.load('g_imp_iw')
        delta = gimp.copy()
        for s, b in delta:
            b.set_from_fourier(delta_tau[s])
        mu = sto.load('mu')
        spins = ['up', 'dn']
        sitestruct = [[s, range(2)] for s in spins]
        momstruct = [['up-+',[0]],['up--',[0]],['dn-+',[0]],['dn--',[0]]]
        p = 1/np.sqrt(2)*np.array([[1,1],[1,-1]])
        u = {s:p for s in spins}
        transf = MatrixTransformation(sitestruct, u, momstruct)
        delta = transf.backtransform_g(delta)
        gimp = transf.backtransform_g(gimp)
